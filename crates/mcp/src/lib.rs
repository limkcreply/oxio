//! MCP client: spawn a configured MCP server over stdio, run the handshake, and
//! expose each of its advertised tools as an oxio `Tool`. rmcp provides the JSON-RPC
//! transport and initialize handshake; this module is the proxy `Tool` and registry
//! wiring for user-declared servers.

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use oxio_core::{Ctx, Result, Tool, ToolKind, ToolOutput, ToolSpec};
use std::collections::HashMap;
use std::path::PathBuf;

use std::future::Future;

use http::{HeaderName, HeaderValue};
use rmcp::model::{CallToolRequestParams, ElicitRequestParams, ElicitResult, ElicitationAction};
use rmcp::service::{MaybeSendFuture, NotificationContext, Peer, RequestContext, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{ClientHandler, RoleClient, ServiceExt};
use serde_json::{json, Map, Value};
use tokio::process::Command;

pub mod oauth;

/// Publishes a server's freshly-discovered proxy tools back to the kernel's live tool
/// registry. Supplied by the caller (which owns the registry handle + permission gate),
/// so this crate mutates nothing itself - it just hands over the rebuilt tools.
pub type ToolPublish = Arc<dyn Fn(Vec<Arc<dyn Tool>>) + Send + Sync>;

/// Routes a server ELICITATION (a server-initiated request for user input, per the MCP
/// elicitation spec) to the UI and returns the user's answer (`None` = declined/cancelled).
/// Supplied by the caller - the UI owns terminal I/O - so this crate stays UI-agnostic.
/// Async: given the prompt message, returns a boxed future of the optional answer.
pub type Elicit = Arc<
    dyn Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
        + Send
        + Sync,
>;

/// The first property name in an elicitation form schema (best-effort single-field fill),
/// read structurally so we don't depend on the schema type's internals.
fn first_schema_field(schema: &rmcp::model::ElicitationSchema) -> Option<String> {
    serde_json::to_value(schema)
        .ok()?
        .get("properties")?
        .as_object()?
        .keys()
        .next()
        .cloned()
}

/// Keeps each connected `RunningService` alive for the session (its `Peer`, embedded in
/// every proxy tool, is only valid while the service lives). Session-lifetime by design -
/// the connection should last until the process exits.
fn service_anchor() -> &'static Mutex<Vec<RunningService<RoleClient, OxioClient>>> {
    static A: OnceLock<Mutex<Vec<RunningService<RoleClient, OxioClient>>>> = OnceLock::new();
    A.get_or_init(|| Mutex::new(Vec::new()))
}

/// oxio's MCP client handler. On a `*_list_changed` notification it RE-DISCOVERS the
/// server's tools/resources/prompts (via the notification's peer) and republishes them to
/// the kernel's live registry through `on_change` - so a mid-session toolset change takes
/// effect on the next turn, no reconnect. With no `on_change`, it just logs.
/// (MCP **Roots** is deliberately NOT advertised - the spec deprecated it in SEP-2577 and
/// slated it for removal, so a shipping product should not build on it.)
#[derive(Clone)]
struct OxioClient {
    server: String,
    on_change: Option<ToolPublish>,
    elicit: Option<Elicit>,
}

impl OxioClient {
    fn new(server: &str, on_change: Option<ToolPublish>, elicit: Option<Elicit>) -> Self {
        Self {
            server: server.to_string(),
            on_change,
            elicit,
        }
    }
}

/// Re-discover a server's proxies and hand them to `cb` (or log if none). Shared by all
/// three `*_list_changed` handlers, since the registry swap replaces the whole `<server>__`
/// set at once.
async fn republish(server: String, cb: Option<ToolPublish>, peer: Peer<RoleClient>) {
    let Some(cb) = cb else {
        eprintln!("mcp {server}: list changed (no live-update handler; refresh on reconnect)");
        return;
    };
    match discover_proxies(&peer, &server).await {
        Ok(tools) => cb(tools),
        Err(e) => eprintln!("mcp {server}: re-list after change failed: {e}"),
    }
}

impl ClientHandler for OxioClient {
    /// Advertise the elicitation capability so servers know they may request user input -
    /// only when a UI callback is present (else we'd claim support we can't fulfil).
    fn get_info(&self) -> rmcp::model::ClientInfo {
        let mut info = rmcp::model::ClientInfo::default();
        if self.elicit.is_some() {
            info.capabilities.elicitation = Some(rmcp::model::ElicitationCapability::default());
        }
        info
    }
    fn on_tool_list_changed(
        &self,
        ctx: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + MaybeSendFuture + '_ {
        republish(self.server.clone(), self.on_change.clone(), ctx.peer)
    }
    fn on_resource_list_changed(
        &self,
        ctx: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + MaybeSendFuture + '_ {
        republish(self.server.clone(), self.on_change.clone(), ctx.peer)
    }
    fn on_prompt_list_changed(
        &self,
        ctx: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + MaybeSendFuture + '_ {
        republish(self.server.clone(), self.on_change.clone(), ctx.peer)
    }

    /// MCP elicitation: the server asks the user for input. Route the form message to the
    /// UI via the `elicit` callback and return the answer under the schema's first field
    /// (best-effort single-field fill). URL elicitation, or no UI (non-interactive), declines
    /// cleanly - the server's operation continues.
    fn create_elicitation(
        &self,
        request: ElicitRequestParams,
        _ctx: RequestContext<RoleClient>,
    ) -> impl Future<Output = std::result::Result<ElicitResult, rmcp::ErrorData>> + MaybeSendFuture + '_
    {
        let elicit = self.elicit.clone();
        async move {
            let (message, field) = match &request {
                ElicitRequestParams::FormElicitationParams {
                    message,
                    requested_schema,
                    ..
                } => (message.clone(), first_schema_field(requested_schema)),
                _ => return Ok(ElicitResult::new(ElicitationAction::Decline)),
            };
            let Some(elicit) = elicit else {
                return Ok(ElicitResult::new(ElicitationAction::Decline));
            };
            match elicit(message).await {
                Some(ans) => {
                    let field = field.unwrap_or_else(|| "value".to_string());
                    Ok(ElicitResult::new(ElicitationAction::Accept)
                        .with_content(json!({ field: ans })))
                }
                None => Ok(ElicitResult::new(ElicitationAction::Decline)),
            }
        }
    }
}

/// Build the proxy `Tool`s for a server from a live `Peer`: one per advertised tool, plus
/// a `read_resource` tool if it has resources and a `prompt` tool if it has prompts.
/// Used at connect AND on re-discovery, so both paths produce an identical tool set.
async fn discover_proxies(
    peer: &Peer<RoleClient>,
    name: &str,
) -> std::result::Result<Vec<Arc<dyn Tool>>, String> {
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    let listed = peer.list_all_tools().await.map_err(|e| e.to_string())?;
    for t in listed {
        let input_schema =
            serde_json::to_value(&*t.input_schema).unwrap_or_else(|_| json!({"type": "object"}));
        tools.push(Arc::new(McpTool {
            peer: peer.clone(),
            remote_name: t.name.to_string(),
            advertised_name: format!("{name}__{}", t.name),
            description: t.description.map(|d| d.to_string()).unwrap_or_default(),
            input_schema,
        }));
    }
    if let Ok(resources) = peer.list_all_resources().await {
        if !resources.is_empty() {
            let catalog: Vec<(String, String)> = resources
                .iter()
                .map(|r| (r.uri.clone(), r.name.clone()))
                .collect();
            tools.push(Arc::new(McpResourceTool {
                peer: peer.clone(),
                advertised_name: format!("{name}__read_resource"),
                catalog,
            }));
        }
    }
    if let Ok(prompts) = peer.list_all_prompts().await {
        if !prompts.is_empty() {
            let catalog: Vec<(String, String)> = prompts
                .iter()
                .map(|p| (p.name.clone(), p.description.clone().unwrap_or_default()))
                .collect();
            tools.push(Arc::new(McpPromptTool {
                peer: peer.clone(),
                advertised_name: format!("{name}__prompt"),
                catalog,
            }));
        }
    }
    Ok(tools)
}

/// A configured MCP server. Set EITHER `url` (remote, streamable-HTTP - the primary
/// MCP model: a server living on the network) OR `command`+`args`+`env` (stdio: a
/// locally-spawned subprocess).
#[derive(Clone, Debug, Default)]
pub struct McpServerCfg {
    pub command: String,
    pub args: Vec<String>,
    /// Environment variables set on the spawned server process (server config,
    /// e.g. a Qdrant RAG server's `QDRANT_URL` / `COLLECTION_NAME`). Stdio only.
    pub env: Vec<(String, String)>,
    /// Remote streamable-HTTP endpoint (e.g. `http://host.local:8000/mcp`). When
    /// set, oxio connects over the network instead of spawning a subprocess.
    pub url: Option<String>,
    /// Remote auth: static HTTP headers sent on every request (name, value). Non-secret.
    pub headers: Vec<(String, String)>,
    /// Remote auth: (header-name, ENV_VAR) pairs - the value is read from the environment
    /// at connect, so secrets (bearer tokens/API keys) stay out of config/disk.
    /// Empty/missing env vars are skipped.
    pub env_headers: Vec<(String, String)>,
    /// Remote auth mode. `"oauth"` → attach a stored OAuth bearer token at connect
    /// (obtained via `oauth::login`); anything else (incl. `None`) → header/no auth.
    pub auth: Option<String>,
    /// OAuth scopes to request at login (`auth = "oauth"`).
    pub oauth_scopes: Vec<String>,
    /// Pre-registered OAuth client id (omit → dynamic client registration).
    pub oauth_client_id: Option<String>,
    /// Where this server's OAuth token is persisted/read (`<config>/oauth/<name>.json`).
    /// The caller supplies it so this crate needs no `config` dependency.
    pub oauth_token_path: Option<PathBuf>,
}

/// The result of connecting to one MCP server: its advertised tools plus the
/// server-level `instructions` brief from the `initialize` handshake (the digital
/// contract's how-to-use half). Both come from the automated handshake - the client
/// authors nothing about the server; it relays what the server self-advertises.
pub struct Loaded {
    pub tools: Vec<Arc<dyn Tool>>,
    /// Server-provided `instructions` (from `initialize`), if any. A vendor-quality
    /// client injects this into the model context (Claude Code does exactly this);
    /// oxio relays it verbatim and maintains none of it.
    pub instructions: Option<String>,
}

/// Connect to a server, run the handshake, and return a proxy `Tool` for each of
/// its advertised tools plus the server's `instructions`. The server name prefixes
/// each tool so multiple servers cannot collide.
///
/// `on_change` is the live-update hook: when the server later sends
/// `tools/list_changed` (or resources/prompts), the handler re-discovers and calls
/// it to republish into the kernel's registry. Pass `None` to only log such changes.
pub async fn load_server(
    name: &str,
    cfg: &McpServerCfg,
    on_change: Option<ToolPublish>,
    elicit: Option<Elicit>,
) -> std::result::Result<Loaded, String> {
    // REMOTE (streamable-HTTP) when a url is set - the server lives on the network;
    // else STDIO - spawn the command as a local subprocess. Both yield the same
    // `RunningService`, so discovery + proxy wiring below is shared.
    let service: RunningService<RoleClient, OxioClient> = if let Some(url) = &cfg.url {
        // Auth: build the request headers (static + env-resolved) and attach them to the
        // transport config. Secrets come from env vars, never from disk.
        // `from_config` uses rmcp's default reqwest client.
        let mut headers = build_headers(&cfg.headers, &cfg.env_headers);
        // OAuth mode: attach a fresh bearer from the stored token (auto-refreshed).
        // Absent token → actionable error rather than a silent unauthenticated connect.
        if cfg.auth.as_deref() == Some("oauth") {
            let path = cfg
                .oauth_token_path
                .clone()
                .ok_or_else(|| format!("{name}: auth=oauth but no token path configured"))?;
            let bearer = oauth::resolve_bearer(url, &path)
                .await
                .map_err(|e| format!("{name}: {e}"))?;
            if let Ok(v) = HeaderValue::from_str(&format!("Bearer {bearer}")) {
                headers.insert(HeaderName::from_static("authorization"), v);
            }
        }
        let mut config = StreamableHttpClientTransportConfig::with_uri(url.clone());
        config.custom_headers = headers;
        let transport = StreamableHttpClientTransport::from_config(config);
        OxioClient::new(name, on_change.clone(), elicit.clone())
            .serve(transport)
            .await
            .map_err(|e| e.to_string())?
    } else {
        let args = cfg.args.clone();
        let envs = cfg.env.clone();
        let transport = TokioChildProcess::new(Command::new(&cfg.command).configure(|c| {
            for a in &args {
                c.arg(a);
            }
            for (k, v) in &envs {
                c.env(k, v);
            }
        }))
        .map_err(|e| e.to_string())?;
        OxioClient::new(name, on_change.clone(), elicit.clone())
            .serve(transport)
            .await
            .map_err(|e| e.to_string())?
    };

    // The server-level brief from the initialize handshake - relayed verbatim, never
    // authored or maintained here. Absent for servers that don't provide one.
    let instructions = service.peer_info().and_then(|i| i.instructions.clone());

    // Discover the initial tool set through the live peer, then ANCHOR the service for
    // the session so the peer (embedded in every proxy) stays valid. Discovery uses the
    // same `discover_proxies` the list_changed handler uses - one code path, no drift.
    let peer = service.peer().clone();
    let tools = discover_proxies(&peer, name).await?;
    eprintln!("mcp {name}: {} tools/resources/prompts", tools.len());
    service_anchor()
        .lock()
        .expect("service anchor")
        .push(service);
    Ok(Loaded {
        tools,
        instructions,
    })
}

/// Build the HTTP header map for a remote MCP server: static headers first, then
/// env-resolved ones (value read from the named env var at connect - secrets stay out
/// of config). Invalid names/values and empty/missing env vars are skipped, never fatal.
fn build_headers(
    static_headers: &[(String, String)],
    env_headers: &[(String, String)],
) -> HashMap<HeaderName, HeaderValue> {
    let mut headers: HashMap<HeaderName, HeaderValue> = HashMap::new();
    let mut put = |name: &str, value: &str| match (
        HeaderName::from_bytes(name.as_bytes()),
        HeaderValue::from_str(value),
    ) {
        (Ok(n), Ok(v)) => {
            headers.insert(n, v);
        }
        _ => eprintln!("mcp: skipping invalid header `{name}`"),
    };
    for (name, value) in static_headers {
        put(name, value);
    }
    for (name, env_var) in env_headers {
        match std::env::var(env_var) {
            Ok(v) if !v.trim().is_empty() => put(name, &v),
            _ => {} // missing/empty env var → header simply not set
        }
    }
    headers
}

/// A oxio `Tool` that forwards to a remote MCP tool over the server's live peer.
struct McpTool {
    peer: Peer<RoleClient>,
    remote_name: String,
    advertised_name: String,
    description: String,
    input_schema: Value,
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.advertised_name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }
    fn kind(&self) -> ToolKind {
        // Remote tools may do anything; treat as effectful so `safety::Guarded`
        // gates them behind the permission prompt.
        ToolKind::Write
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let arguments: Option<Map<String, Value>> = match input {
            Value::Object(m) => Some(m),
            Value::Null => None,
            other => {
                let mut m = Map::new();
                m.insert("input".to_string(), other);
                Some(m)
            }
        };
        let param = match arguments {
            Some(map) => CallToolRequestParams::new(self.remote_name.clone()).with_arguments(map),
            None => CallToolRequestParams::new(self.remote_name.clone()),
        };
        match self.peer.call_tool(param).await {
            Ok(res) => Ok(ToolOutput::ok(render_result(&res))),
            Err(e) => Ok(ToolOutput::error(format!(
                "mcp {}: {e}",
                self.advertised_name
            ))),
        }
    }
}

/// Proxy for an MCP server's RESOURCES: one Read tool that reads a resource by URI.
struct McpResourceTool {
    peer: Peer<RoleClient>,
    advertised_name: String,
    catalog: Vec<(String, String)>, // (uri, name)
}

#[async_trait]
impl Tool for McpResourceTool {
    fn spec(&self) -> ToolSpec {
        let list: String = self
            .catalog
            .iter()
            .map(|(uri, name)| format!("\n- {uri} - {name}"))
            .collect();
        ToolSpec {
            name: self.advertised_name.clone(),
            description: format!(
                "Read a document/resource published by this MCP server. Pass the `uri` of one \
                 of its available resources:{list}"
            ),
            input_schema: json!({
                "type": "object",
                "properties": { "uri": { "type": "string", "description": "the resource URI to read" } },
                "required": ["uri"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let uri = match input.get("uri").and_then(|v| v.as_str()) {
            Some(u) => u.to_string(),
            None => return Ok(ToolOutput::error("read_resource: missing 'uri'")),
        };
        let param = rmcp::model::ReadResourceRequestParams::new(uri);
        match self.peer.read_resource(param).await {
            Ok(res) => {
                let mut out = String::new();
                for c in &res.contents {
                    if let rmcp::model::ResourceContents::TextResourceContents { text, .. } = c {
                        out.push_str(text);
                        out.push('\n');
                    }
                }
                Ok(ToolOutput::ok(if out.trim().is_empty() {
                    format!("{res:?}")
                } else {
                    out.trim_end().to_string()
                }))
            }
            Err(e) => Ok(ToolOutput::error(format!("read_resource: {e}"))),
        }
    }
}

/// Proxy for an MCP server's PROMPTS: one Read tool that lists them (no `name`) or
/// fetches a rendered prompt (`name` + optional `arguments`).
struct McpPromptTool {
    peer: Peer<RoleClient>,
    advertised_name: String,
    catalog: Vec<(String, String)>, // (name, description)
}

#[async_trait]
impl Tool for McpPromptTool {
    fn spec(&self) -> ToolSpec {
        let list: String = self
            .catalog
            .iter()
            .map(|(name, desc)| format!("\n- {name}: {desc}"))
            .collect();
        ToolSpec {
            name: self.advertised_name.clone(),
            description: format!(
                "Fetch a prompt template published by this MCP server. Omit `name` to list them, \
                 or pass a `name` (+ optional `arguments` object) to get the rendered prompt. \
                 Available prompts:{list}"
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "the prompt to fetch (omit to list)" },
                    "arguments": { "type": "object", "description": "template arguments" }
                }
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let name = match input.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                let listing: String = self
                    .catalog
                    .iter()
                    .map(|(n, d)| format!("- {n}: {d}\n"))
                    .collect();
                return Ok(ToolOutput::ok(format!("Available prompts:\n{listing}")));
            }
        };
        let mut param = rmcp::model::GetPromptRequestParams::new(name);
        if let Some(args) = input.get("arguments").and_then(|v| v.as_object()).cloned() {
            param = param.with_arguments(args);
        }
        match self.peer.get_prompt(param).await {
            Ok(res) => {
                let mut out = String::new();
                for m in &res.messages {
                    if let Some(t) = m.content.as_text() {
                        out.push_str(&t.text);
                        out.push('\n');
                    }
                }
                Ok(ToolOutput::ok(if out.trim().is_empty() {
                    format!("{res:?}")
                } else {
                    out.trim_end().to_string()
                }))
            }
            Err(e) => Ok(ToolOutput::error(format!("get_prompt: {e}"))),
        }
    }
}

/// Flatten an MCP tool result to text: concatenate text content blocks, else Debug.
fn render_result(res: &rmcp::model::CallToolResult) -> String {
    let mut out = String::new();
    for c in &res.content {
        if let Some(t) = c.as_text() {
            out.push_str(&t.text);
            out.push('\n');
        }
    }
    if out.trim().is_empty() {
        format!("{res:?}")
    } else {
        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::build_headers;
    use http::HeaderName;

    #[test]
    fn static_headers_are_set() {
        let h = build_headers(&[("X-Client".into(), "oxio".into())], &[]);
        assert_eq!(h.get(&HeaderName::from_static("x-client")).unwrap(), "oxio");
    }

    #[test]
    fn env_header_resolves_from_environment_and_keeps_secret_out_of_config() {
        // SAFETY: single-threaded test; unique var name.
        unsafe { std::env::set_var("OXIO_TEST_MCP_TOKEN", "Bearer s3cret") };
        let h = build_headers(
            &[],
            &[("Authorization".into(), "OXIO_TEST_MCP_TOKEN".into())],
        );
        assert_eq!(
            h.get(&HeaderName::from_static("authorization")).unwrap(),
            "Bearer s3cret"
        );
        unsafe { std::env::remove_var("OXIO_TEST_MCP_TOKEN") };
    }

    #[test]
    fn missing_env_var_is_skipped_not_fatal() {
        let h = build_headers(
            &[],
            &[("Authorization".into(), "OXIO_TEST_DEFINITELY_UNSET".into())],
        );
        assert!(h.is_empty());
    }

    #[test]
    fn invalid_header_name_is_skipped() {
        let h = build_headers(
            &[("bad name".into(), "v".into()), ("ok".into(), "v".into())],
            &[],
        );
        assert_eq!(h.len(), 1);
    }
}
