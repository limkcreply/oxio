//! oxio frontend: assemble a `Kernel` from config, and render streamed events
//! to the terminal (thinking dimmed, answer normal). The interactive REPL lands
//! here next; for now this powers the one-shot path.

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use auth::{ApiKeyAuth, Auth, NoAuth};
use oxio_core::{
    Content, Ctx, Hook, Kernel, Message, NoticeLevel, NullSink, Observer, Params, Provider,
    Registry, Request, Result as CoreResult, Role, StreamEvent, StreamSink, Tool, ToolKind,
    ToolOutput, ToolSpec, Transformer, TurnState,
};
// ratatui frontend (feature-gated; `--no-default-features` falls back to the line REPL).
#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "tui")]
pub use tui::run_tui;

/// Interactive entry point: the ratatui TUI when built with `tui` (the default),
/// otherwise the legacy line REPL. `main` calls this and stays feature-agnostic.
pub async fn run_interactive(
    cfg: &Config,
    continue_session: bool,
    resume_pick: bool,
) -> anyhow::Result<()> {
    #[cfg(feature = "tui")]
    {
        run_tui(cfg, continue_session, resume_pick).await
    }
    #[cfg(not(feature = "tui"))]
    {
        repl(cfg, continue_session, resume_pick).await
    }
}

use std::collections::BTreeMap;

use config::{AuthKind, Config, ProviderCfg, RateCfg, WireApi};
use providers::{
    AnthropicAdapter, ChainProvider, ChatCompletionsAdapter, ResponsesAdapter, Sampling,
    SwappableProvider,
};
use safety::{
    AllowAll, ApprovalRequest, Approver, Checkpointing, Decision, DenyAll, Guarded, LoopGuard,
    SnapshotStore,
};
use serde_json::{json, Value};

/// Default system prompt. Baseline is OpenAI's open-source Codex CLI system prompt
/// (Apache-2.0), adapted for oxio.
const DEFAULT_SYSTEM_PROMPT: &str = include_str!("default_system_prompt.md");

/// Markers wrapping an image path in the submitted turn text. The composer emits
/// `⟦img:/abs/path⟧` for each dropped/pasted image; [`ImageExpander`] rewrites these
/// into `Content::Image` blocks at PreTurn. Chosen to never collide with a slash
/// command (starts with '⟦', not '/') or with normal prose.
pub const IMAGE_MARKER_OPEN: &str = "⟦img:";
pub const IMAGE_MARKER_CLOSE: &str = "⟧";

/// PreTurn transformer: expand `⟦img:PATH⟧` markers in the just-submitted user
/// message into base64 `Content::Image` blocks so the primary model sees the image
/// inline.
/// The wire adapters already serialize `Content::Image`; a text-only model ignores
/// it. A read failure is folded in as visible text, never silently dropped.
struct ImageExpander;

#[async_trait]
impl Transformer for ImageExpander {
    async fn transform(&self, _hook: Hook, state: &mut TurnState) -> CoreResult<()> {
        use base64::Engine as _;
        let Some(msg) = state
            .messages
            .iter_mut()
            .rev()
            .find(|m| m.role == Role::User)
        else {
            return Ok(());
        };
        let text = msg.as_text();
        if !text.contains(IMAGE_MARKER_OPEN) {
            return Ok(());
        }
        let mut cleaned = String::new();
        let mut images: Vec<Content> = Vec::new();
        let mut rest = text.as_str();
        while let Some(start) = rest.find(IMAGE_MARKER_OPEN) {
            cleaned.push_str(&rest[..start]);
            let after = &rest[start + IMAGE_MARKER_OPEN.len()..];
            let Some(end) = after.find(IMAGE_MARKER_CLOSE) else {
                cleaned.push_str(&rest[start..]); // unterminated marker: keep literally
                rest = "";
                break;
            };
            let path = &after[..end];
            match std::fs::read(path) {
                Ok(bytes) => images.push(Content::Image {
                    media_type: image_media_type(path).to_string(),
                    data: base64::engine::general_purpose::STANDARD.encode(&bytes),
                }),
                Err(e) => cleaned.push_str(&format!("[image attach failed: {path}: {e}]")),
            }
            rest = &after[end + IMAGE_MARKER_CLOSE.len()..];
        }
        cleaned.push_str(rest);
        let cleaned = cleaned.trim().to_string();

        let mut content: Vec<Content> = Vec::new();
        if !cleaned.is_empty() {
            content.push(Content::Text { text: cleaned });
        }
        content.extend(images);
        if content.is_empty() {
            content.push(Content::Text {
                text: String::new(),
            });
        }
        msg.content = content;
        Ok(())
    }
}

/// Injects the system prompt at the start of a turn (a `Transformer` at PreTurn). The
/// effective prompt is `text` + the current MCP-server brief (`mcp_brief`, filled as
/// servers connect - including in the background), so late-connecting servers' `instructions`
/// still land. Kept as a SINGLE system message (rebuilt each turn) - adapter-safe (some
/// wire formats take only the first system message) and always current.
struct SystemPrompt {
    text: String,
    mcp_brief: Option<Arc<std::sync::Mutex<String>>>,
}

#[async_trait]
impl Transformer for SystemPrompt {
    async fn transform(&self, _hook: Hook, state: &mut TurnState) -> CoreResult<()> {
        let brief = self
            .mcp_brief
            .as_ref()
            .and_then(|b| b.lock().ok().map(|s| s.clone()))
            .unwrap_or_default();
        let full = if brief.trim().is_empty() {
            self.text.clone()
        } else {
            format!(
                "{}\n\n# MCP Server Instructions\n\nConnected MCP servers provided these instructions for using their tools:\n{brief}",
                self.text
            )
        };
        // Ensure exactly one system message, at the front, equal to the current `full`.
        match state.messages.iter_mut().find(|m| m.role == Role::System) {
            Some(m) => *m = Message::text(Role::System, full),
            None => state.messages.insert(0, Message::text(Role::System, full)),
        }
        Ok(())
    }
}

/// Session-varying environment the model needs at start: date, timezone/region,
/// working directory, OS/arch, and git branch. Without this the model has no idea
/// what "today" is and must shell out to `date`. Computed once at session start.
fn env_context() -> String {
    let now = std::process::Command::new("date")
        .arg("+%A, %d %B %Y %H:%M %z (%Z)")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(String::from))
        .unwrap_or_default();
    let branch = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let mut s = String::from("<environment>\n");
    if !now.is_empty() {
        s.push_str(&format!("Current date/time: {now}\n"));
    }
    s.push_str(&format!(
        "Operating system: {} ({})\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    if !cwd.is_empty() {
        // Where we are AND that it's not a boundary - one clause, not a sermon.
        s.push_str(&format!(
            "Working directory: {cwd} (you can also read and search parent directories and anywhere on this machine, not only here)\n"
        ));
    }
    if let Some(b) = branch {
        s.push_str(&format!("Git branch: {b}\n"));
    }
    // Factual memory pointer (all models): where durable memory lives + that it persists and
    // is auto-loaded, so the model knows it HAS memory and where to update it. Not coaching.
    s.push_str(&format!(
        "Durable memory: {} (project) - persists across sessions, auto-loaded above; save/update via the memory tool.\n",
        memory::memory_path().display()
    ));
    s.push_str("</environment>");
    s
}

/// Discover project instruction files and concatenate them least-specific first.
/// Sources: a user-global `<config-dir>/AGENTS.md`, then `AGENTS.md` / `OXIO.md`
/// from the git root down to the cwd (nearest = most specific = last, so it wins).
/// Bounded to the git root so we never read unrelated parent files. `AGENTS.md` is the
/// cross-tool convention; `OXIO.md` is oxio's own.
fn load_project_instructions() -> Option<String> {
    use std::path::Path;
    let cwd = std::env::current_dir().ok()?;
    // Directory chain: git root … cwd (root-most first). If no git root, just cwd.
    let git_root = cwd.ancestors().find(|d| d.join(".git").exists());
    let mut chain: Vec<PathBuf> = match git_root {
        Some(root) => {
            let mut v: Vec<PathBuf> = cwd
                .ancestors()
                .take_while(|d| *d != root)
                .map(Path::to_path_buf)
                .collect();
            v.push(root.to_path_buf());
            v.reverse();
            v
        }
        None => vec![cwd.clone()],
    };
    // User-global instructions are least specific → first.
    if let Some(dir) = config::config_path().parent() {
        chain.insert(0, dir.to_path_buf());
    }

    let mut out = String::new();
    let mut seen = std::collections::HashSet::new();
    for dir in &chain {
        for name in ["AGENTS.md", "OXIO.md"] {
            let p = dir.join(name);
            if !seen.insert(p.clone()) {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&p) {
                let t = text.trim();
                if !t.is_empty() {
                    if !out.is_empty() {
                        out.push_str("\n\n");
                    }
                    out.push_str(&format!("# Project instructions: {}\n{t}", p.display()));
                }
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Split a `SKILL.md` into its YAML-ish frontmatter map and its markdown body.
/// Lightweight `key: value` parser (no YAML dep) - SKILL.md frontmatter is flat.
/// Returns (frontmatter, body); body is the whole text when there is no frontmatter.
fn parse_frontmatter(text: &str) -> (std::collections::BTreeMap<String, String>, String) {
    use std::collections::BTreeMap;
    let t = text.strip_prefix('\u{feff}').unwrap_or(text).trim_start();
    if let Some(rest) = t.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            let mut fm = BTreeMap::new();
            for line in rest[..end].lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let v = v.trim().trim_matches('"').trim_matches('\'').trim();
                    fm.insert(k.trim().to_lowercase(), v.to_string());
                }
            }
            let body = rest[end + 4..]
                .trim_start_matches(['\n', '\r'])
                .trim_start();
            return (fm, body.to_string());
        }
    }
    (BTreeMap::new(), text.to_string())
}

/// Discover on-disk skills following the SKILL.md convention (oxio flavour): a
/// `skills/<name>/SKILL.md` under the user config dir, and `.oxio/skills/<name>/SKILL.md`
/// in the project (cwd + git root). Each skill's frontmatter gives `name`/`description`
/// (shown always); the body is the instructions (loaded on demand via the `skill` tool).
/// Bundled files in the skill dir are noted so the model can read them progressively.
fn discover_file_skills() -> std::collections::BTreeMap<String, config::SkillCfg> {
    let mut bases: Vec<PathBuf> = Vec::new();
    if let Some(cfgdir) = config::config_path().parent() {
        bases.push(cfgdir.join("skills"));
    }
    if let Ok(cwd) = std::env::current_dir() {
        bases.push(cwd.join(".oxio/skills"));
        if let Some(root) = cwd.ancestors().find(|d| d.join(".git").exists()) {
            bases.push(root.join(".oxio/skills"));
        }
    }
    discover_skills_in(&bases)
}

/// SKILL.md discovery over an explicit list of base dirs (testable core of
/// [`discover_file_skills`]).
fn discover_skills_in(bases: &[PathBuf]) -> std::collections::BTreeMap<String, config::SkillCfg> {
    use std::collections::BTreeMap;
    let mut skills: BTreeMap<String, config::SkillCfg> = BTreeMap::new();
    for base in bases {
        let Ok(entries) = std::fs::read_dir(base) else {
            continue;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            let Ok(text) = std::fs::read_to_string(dir.join("SKILL.md")) else {
                continue;
            };
            let (fm, body) = parse_frontmatter(&text);
            let name = fm.get("name").cloned().unwrap_or_else(|| {
                dir.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            });
            let mut instructions = body;
            let bundled: Vec<String> = std::fs::read_dir(&dir)
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.file_name().to_string_lossy().to_string())
                        .filter(|n| n != "SKILL.md")
                        .collect()
                })
                .unwrap_or_default();
            if !bundled.is_empty() {
                instructions.push_str(&format!(
                    "\n\n---\nBundled files in this skill directory ({}): {}. Read any you need with the file tools.",
                    dir.display(),
                    bundled.join(", ")
                ));
            }
            // First base wins per name (user config dir before project); config-inline
            // skills override these later at the registration site.
            skills.entry(name).or_insert(config::SkillCfg {
                description: fm.get("description").cloned(),
                instructions,
            });
        }
    }
    skills
}

/// Where a remote server's OAuth token is persisted: `<config dir>/oauth/<server>.json`.
/// Shared by kernel-build (connect, reads it) and the `login` command (writes it) so
/// both agree on the location.
pub fn oauth_token_path(server_name: &str) -> PathBuf {
    let dir = config::config_path()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    dir.join("oauth").join(format!("{server_name}.json"))
}

/// `oxio login <name>`: run the interactive OAuth flow for a configured remote MCP
/// server (`auth = "oauth"`) and persist the token. Errors are actionable.
#[cfg(feature = "mcp")]
pub async fn mcp_login(cfg: &Config, name: &str) -> anyhow::Result<()> {
    let sc = cfg
        .mcp
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("no MCP server '{name}' in config"))?;
    if sc.auth.as_deref() != Some("oauth") {
        anyhow::bail!("MCP server '{name}' is not configured for OAuth (set `auth = \"oauth\"`)");
    }
    let url = sc
        .url
        .clone()
        .ok_or_else(|| anyhow::anyhow!("MCP server '{name}' has no `url`"))?;
    let scopes = sc
        .oauth
        .as_ref()
        .map(|o| o.scopes.clone())
        .unwrap_or_default();
    let client_id = sc.oauth.as_ref().and_then(|o| o.client_id.clone());
    mcp::oauth::login(
        name,
        &url,
        &scopes,
        client_id.as_deref(),
        &oauth_token_path(name),
    )
    .await
    .map_err(|e| anyhow::anyhow!(e))
}

/// Fallback when the binary is built without MCP support.
#[cfg(not(feature = "mcp"))]
pub async fn mcp_login(_cfg: &Config, _name: &str) -> anyhow::Result<()> {
    anyhow::bail!("this build has no MCP support (rebuild with `--features mcp`)")
}

/// The session's effective skills: on-disk SKILL.md skills merged with config-inline
/// `[skills.<name>]` (config-inline overrides a same-named file skill). Shared by the
/// kernel builder and `doctor` so both see the identical set.
pub fn resolve_skills(cfg: &Config) -> std::collections::BTreeMap<String, config::SkillCfg> {
    let mut skills = discover_file_skills();
    for (name, s) in &cfg.skills {
        skills.insert(name.clone(), s.clone());
    }
    skills
}

/// Append each bound skill's instructions to an agent's base system prompt - the oxio
/// form of a sub-agent's `skills: [...]` binding. Unknown names are skipped with a warning
/// (never fatal). `agent` names the agent only for the warning.
fn bind_agent_skills(
    base: String,
    names: &[String],
    skills: &std::collections::BTreeMap<String, config::SkillCfg>,
    agent: &str,
) -> String {
    let mut out = base;
    for n in names {
        match skills.get(n) {
            Some(s) => out.push_str(&format!("\n\n# Skill: {n}\n{}", s.instructions)),
            None => eprintln!("agent '{agent}': skill '{n}' not found - skipped"),
        }
    }
    out
}

/// Runs user-configured shell hooks on lifecycle events (`[[hooks]]` in config).
/// Observational and fire-and-forget: a hook can never block or break a turn.
/// Context is passed via env (`OXIO_HOOK_EVENT`, `OXIO_MODEL`).
struct HookRunner {
    hooks: Vec<config::HookCfg>,
}

impl HookRunner {
    fn event_name(hook: Hook) -> Option<&'static str> {
        Some(match hook {
            Hook::PreTurn => "pre_turn",
            Hook::PostTurn => "post_turn",
            Hook::PreModel => "pre_model",
            Hook::PostModel => "post_model",
            Hook::PreTool => "pre_tool",
            Hook::PostTool => "post_tool",
            Hook::OnError => "on_error",
            _ => return None,
        })
    }
}

#[async_trait]
impl Observer for HookRunner {
    async fn on_event(&self, hook: Hook, state: &TurnState) {
        let Some(event) = Self::event_name(hook) else {
            return;
        };
        for h in &self.hooks {
            if h.on == event {
                let _ = tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(&h.run)
                    .env("OXIO_HOOK_EVENT", event)
                    .env("OXIO_MODEL", &state.model)
                    .spawn();
            }
        }
    }
}

/// Resolve the active provider CHAIN (primary + configured fallbacks) from config,
/// composed into one `Provider`, plus the primary model id (for display / token
/// budget). Reused by `build_kernel` and by live `/model` endpoint swaps.
pub async fn resolve_chain(cfg: &Config) -> anyhow::Result<(Arc<dyn Provider>, String)> {
    let (pname, ppc) = cfg.primary().ok_or_else(|| {
        anyhow::anyhow!("no primary provider configured - see `oxio config show`")
    })?;
    let mut providers: Vec<Arc<dyn Provider>> = vec![provider_from(pname, ppc).await?];
    for name in &cfg.defaults.fallback {
        if let Some(pc) = cfg.providers.get(name) {
            providers.push(provider_from(name, pc).await?);
        }
    }
    let model = ppc.model.clone().unwrap_or_default();
    let chain: Arc<dyn Provider> = if providers.len() == 1 {
        providers.pop().unwrap()
    } else {
        Arc::new(ChainProvider::new(providers))
    };
    Ok((chain, model))
}

/// Session knobs the interactive UI mutates at runtime WITHOUT a kernel rebuild.
/// Held by the run loop; the modules below read them each turn.
pub struct LiveControls {
    /// The session's provider slot - repoint for a live `/model` machine/model switch.
    pub provider: Arc<SwappableProvider>,
    /// Thinking mode, read by `ThinkingToggle` each turn: 0 = auto (model default,
    /// send nothing), 1 = force on, 2 = force off. `/think` writes it.
    pub thinking: Arc<std::sync::atomic::AtomicU8>,
    /// MCP server connection outcomes at startup: `(name, Ok(tool_count) | Err(reason))`.
    /// The UI renders these as connected/offline status lines.
    pub mcp_status: Vec<(String, Result<usize, String>)>,
    /// Runtime MCP control for the `/mcp` command: connect/disconnect servers live.
    #[cfg(feature = "mcp")]
    pub mcp: Option<McpConnector>,
}

/// `PreModel` transformer that applies the runtime thinking toggle to the turn's
/// params. Auto (0) leaves `None` so the model keeps its own default (no flag sent).
struct ThinkingToggle {
    flag: Arc<std::sync::atomic::AtomicU8>,
}

#[async_trait]
impl oxio_core::Transformer for ThinkingToggle {
    async fn transform(
        &self,
        hook: Hook,
        state: &mut oxio_core::TurnState,
    ) -> oxio_core::Result<()> {
        if hook == Hook::PreModel {
            match self.flag.load(std::sync::atomic::Ordering::Relaxed) {
                1 => state.params.thinking = Some(true),
                2 => state.params.thinking = Some(false),
                _ => {}
            }
        }
        Ok(())
    }
}

/// Reload config from disk and resolve the active provider chain - the live
/// `/model` path, run AFTER `connect_cmd` has saved the mutated config. Returns
/// the rebuilt chain plus the freshly-loaded `Config` (so the caller can read the
/// new machine's model for display) so a single disk read serves both.
pub async fn resolve_chain_reload() -> anyhow::Result<(Arc<dyn Provider>, Config)> {
    let cfg = config::load()?;
    let (chain, _model) = resolve_chain(&cfg).await?;
    Ok((chain, cfg))
}

/// Build a `Kernel` from config: primary provider, plus the configured fallback
/// chain (wrapped as a composing `ChainProvider`) when present. Async so each
/// provider's credential can be resolved through the `Auth` seam. `approver`
/// backs the permission gate that fronts write/exec tools. The returned
/// `SwappableProvider` is the session's provider slot - the UI repoints it on
/// `/model` for a live endpoint switch without a kernel rebuild.
/// Reconcile the PRIMARY provider's pinned model against what its server actually serves,
/// at startup - so a stale pin (the box swapped models) self-heals on launch, not only on
/// `/model`. Bounded (3s) and conservative: if the probe times out or returns nothing we
/// can't verify staleness, so we leave the pin untouched (unreachable ≠ gone). Returns
/// `(old, new)` when it changed the pin. Only the primary is probed (one call), never all.
pub async fn reconcile_primary_model(cfg: &mut Config) -> Option<String> {
    let (name, pc) = cfg.primary()?;
    let (name, url, pinned) = (
        name.to_string(),
        pc.base_url.clone()?,
        pc.model.clone().unwrap_or_default(),
    );
    let served = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        providers::list_models(&url),
    )
    .await
    .ok()?;
    if served.is_empty() {
        return None; // couldn't verify - leave the pin untouched
    }
    if pinned.is_empty() {
        // No model pinned → adopt the first served (a model is required to chat).
        let new = served.into_iter().next()?;
        if let Some(p) = cfg.providers.get_mut(&name) {
            p.model = Some(new.clone());
        }
        let _ = config::save(cfg);
        return Some(format!("no primary model set - using '{new}'"));
    }
    if served.iter().any(|m| m == &pinned) {
        return None; // pin is served → all good
    }
    // Pin isn't in /v1/models. That is NOT proof it's stale - many routers accept ALIASES
    // (e.g. a router alias) that never appear in the model list. So DO NOT auto-change it:
    // just hint, per the conservative rule. The user picks a real model with /model if wrong.
    Some(format!(
        "primary model '{pinned}' isn't in {name}'s model list - may be a valid alias; if wrong, pick one with /model <model>"
    ))
}

/// Map a stored MCP config entry to the wire config the `mcp` crate consumes.
#[cfg(feature = "mcp")]
fn mcp_wire_cfg(name: &str, sc: &config::McpServerCfg) -> mcp::McpServerCfg {
    mcp::McpServerCfg {
        command: sc.command.clone().unwrap_or_default(),
        args: sc.args.clone(),
        env: sc.env.clone().into_iter().collect(),
        url: sc.url.clone(),
        headers: sc.headers.clone().into_iter().collect(),
        env_headers: sc.env_headers.clone().into_iter().collect(),
        auth: sc.auth.clone(),
        oauth_scopes: sc
            .oauth
            .as_ref()
            .map(|o| o.scopes.clone())
            .unwrap_or_default(),
        oauth_client_id: sc.oauth.as_ref().and_then(|o| o.client_id.clone()),
        oauth_token_path: Some(oauth_token_path(name)),
    }
}

/// Connect ONE MCP server and hot-add its (gated) tools to the live registry under the
/// `<name>__` prefix, appending its instructions brief. Shared by startup (deferred + sync)
/// and the runtime `/mcp` command. Returns the tool count, or the connect error.
#[cfg(feature = "mcp")]
async fn mcp_connect_one(
    tools_handle: &oxio_core::SharedTools,
    approver: &Arc<dyn Approver>,
    elicit: Option<mcp::Elicit>,
    brief: &Arc<std::sync::Mutex<String>>,
    name: &str,
    scfg: &mcp::McpServerCfg,
) -> Result<usize, String> {
    let on_change: mcp::ToolPublish = {
        let handle = tools_handle.clone();
        let ap = approver.clone();
        let prefix = format!("{name}__");
        Arc::new(move |fresh: Vec<Arc<dyn oxio_core::Tool>>| {
            let gated: Vec<Arc<dyn oxio_core::Tool>> = fresh
                .into_iter()
                .map(|t| Guarded::wrap(t, ap.clone()))
                .collect();
            oxio_core::Registry::swap_server_tools(&handle, &prefix, gated);
        })
    };
    let loaded = mcp::load_server(name, scfg, Some(on_change), elicit).await?;
    let n = loaded.tools.len();
    if let Some(ins) = loaded.instructions.filter(|s| !s.trim().is_empty()) {
        if let Ok(mut b) = brief.lock() {
            b.push_str(&format!("\n## {name}\n{ins}\n"));
        }
    }
    let gated: Vec<Arc<dyn oxio_core::Tool>> = loaded
        .tools
        .into_iter()
        .map(|t| Guarded::wrap(t, approver.clone()))
        .collect();
    oxio_core::Registry::swap_server_tools(tools_handle, &format!("{name}__"), gated);
    Ok(n)
}

/// Runtime MCP control handed to the TUI so `/mcp` can connect/disconnect servers live,
/// using the same registry + gate + brief as startup.
#[cfg(feature = "mcp")]
#[derive(Clone)]
pub struct McpConnector {
    tools_handle: oxio_core::SharedTools,
    approver: Arc<dyn Approver>,
    elicit: Option<mcp::Elicit>,
    brief: Arc<std::sync::Mutex<String>>,
}

#[cfg(feature = "mcp")]
impl McpConnector {
    /// Connect a server from its stored config, hot-adding its tools. Returns the tool count.
    pub async fn connect(&self, name: &str, sc: &config::McpServerCfg) -> Result<usize, String> {
        let wire = mcp_wire_cfg(name, sc);
        mcp_connect_one(
            &self.tools_handle,
            &self.approver,
            self.elicit.clone(),
            &self.brief,
            name,
            &wire,
        )
        .await
    }
    /// Drop a server's tools from the live registry (disconnect).
    pub fn disconnect(&self, name: &str) {
        oxio_core::Registry::swap_server_tools(
            &self.tools_handle,
            &format!("{name}__"),
            Vec::new(),
        );
    }
}

/// Connect each configured MCP server in the BACKGROUND and hot-add its tools to the live
/// registry as it lands - the standard async-connect pattern (the UI renders immediately;
/// tools appear when ready). Each outcome streams to `status` for the UI to show in place.
/// Spawned detached by `build_kernel` for the TUI path.
#[cfg(feature = "mcp")]
async fn connect_mcp_deferred(
    servers: Vec<(String, mcp::McpServerCfg)>,
    approver: Arc<dyn Approver>,
    tools_handle: oxio_core::SharedTools,
    brief: Arc<std::sync::Mutex<String>>,
    elicit: Option<mcp::Elicit>,
    status: tokio::sync::mpsc::UnboundedSender<(String, Result<usize, String>)>,
) {
    for (name, scfg) in servers {
        let outcome = mcp_connect_one(
            &tools_handle,
            &approver,
            elicit.clone(),
            &brief,
            &name,
            &scfg,
        )
        .await;
        let _ = status.send((name, outcome));
    }
}

pub async fn build_kernel(
    cfg: &Config,
    approver: Arc<dyn Approver>,
    asker: Option<AskSender>,
    // When Some (the TUI), MCP servers connect in the BACKGROUND: build_kernel returns
    // immediately (input box renders now, not after every handshake), and each server's
    // outcome is streamed here as it lands. When None (doctor / one-shot / line REPL),
    // MCP connects synchronously so those paths have the full result before they run.
    mcp_status_tx: Option<tokio::sync::mpsc::UnboundedSender<(String, Result<usize, String>)>>,
) -> anyhow::Result<(
    Kernel,
    Arc<session::SessionStore>,
    Arc<context::Compactor>,
    Arc<SnapshotStore>,
    LiveControls,
)> {
    let snapshots = SnapshotStore::new();
    let (pname, ppc) = cfg.primary().ok_or_else(|| {
        anyhow::anyhow!("no primary provider configured - see `oxio config show`")
    })?;

    let (chain, model) = resolve_chain(cfg).await?;
    // Wrap the chain in a swappable handle so `/model` can repoint the session
    // at another endpoint LIVE (next turn) without rebuilding the kernel or losing
    // session/transcript. Every downstream consumer that defaults to "primary"
    // (compactor, swarm, agents, view_image) is handed THIS, so they follow the swap.
    let swap = Arc::new(SwappableProvider::new(chain));
    let provider: Arc<dyn Provider> = swap.clone();
    let mut reg = Registry::new();
    reg.add_provider(provider.clone());
    // Runtime thinking toggle (auto/on/off), applied to each turn's params by a
    // PreModel transformer; the UI flips it via `/think`. See `LiveControls`.
    let thinking = Arc::new(std::sync::atomic::AtomicU8::new(0));
    reg.add_transformer(
        Hook::PreModel,
        Arc::new(ThinkingToggle {
            flag: thinking.clone(),
        }),
    );

    // Active profile = {prompt, tool set, context ceiling} - the two-scales
    // mechanism (swapped with the model; harness identical across profiles).
    let profile = cfg.profiles.get(&cfg.defaults.profile);
    let mut system_prompt = profile
        .and_then(|p| p.prompt.clone())
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
    // Project instruction files (AGENTS.md / OXIO.md hierarchy) appended so the
    // agent knows this project's conventions.
    if let Some(proj) = load_project_instructions() {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(&proj);
    }
    // Session-varying environment (date, timezone/region, cwd, os, git) - the model
    // is otherwise blind to "today". Appended AFTER the static prompt so the stable
    // prefix stays cache-friendly.
    system_prompt.push_str("\n\n");
    system_prompt.push_str(&env_context());
    let profile_tools: Vec<String> = profile.map(|p| p.tools.clone()).unwrap_or_default();

    // Built-in tools, each fronted by the permission gate. `Guarded::wrap` is a
    // no-op for read tools and gates write/exec tools. The profile's tool set
    // (when non-empty) restricts which are exposed. Innermost is the corrective-hint
    // layer (config-driven; only augments error output) so it wraps the tool's own
    // result before checkpointing/approval.
    let hints_enabled = cfg.defaults.tool_hints.unwrap_or(true);
    for t in tools::builtin() {
        if profile_tools.is_empty() || profile_tools.iter().any(|n| n == &t.spec().name) {
            let t = tools::Hinting::wrap(t, &cfg.hints, hints_enabled);
            // Checkpoint file-writing tools (undo), then gate on permission.
            reg.add_tool(Guarded::wrap(
                Checkpointing::wrap(t, snapshots.clone()),
                approver.clone(),
            ));
        }
    }
    // Configured MCP servers: connect (automated `initialize` handshake), discover each
    // remote tool, register it as a gated proxy, and collect the server's `instructions`
    // brief. Open registry - ANY user-declared server, no allowlist/policing; execution is
    // gated per-call by `Guarded`, so the user controls actions, not which servers connect.
    // A server that fails to connect is recorded (shown offline at startup) and skipped,
    // never a hard failure. Runs BEFORE the prompt is finalized so a discovered brief can
    // be injected into it.
    // Server `instructions` briefs accumulate here as MCP servers connect (synchronously
    // now, or in the background). `SystemPrompt` injects the CURRENT brief each turn, so a
    // server that connects after the first turn still gets its brief into context.
    let mcp_brief = Arc::new(std::sync::Mutex::new(String::new()));
    // MCP status: populated synchronously for the None (doctor/one-shot/REPL) path; for the
    // TUI (deferred) path it streams over `mcp_status_tx` instead and this stays empty.
    #[allow(unused_mut)]
    let mut mcp_status: Vec<(String, Result<usize, String>)> = Vec::new();
    // Runtime MCP control (built inside the mcp block where the pieces are in scope), moved
    // into LiveControls so `/mcp` can connect/disconnect servers live. Assigned once in the
    // block below (which always runs under the mcp feature).
    #[cfg(feature = "mcp")]
    let mcp_conn: Option<McpConnector>;
    #[cfg(feature = "mcp")]
    {
        // Elicitation: a server-initiated request for user input is routed to the SAME ask
        // seam the model's ask_user_question uses (a channel to the event loop), so it never
        // blocks. No UI (non-interactive) → None → the client declines cleanly.
        let elicit: Option<mcp::Elicit> = asker.clone().map(|asker| {
            let e: mcp::Elicit = Arc::new(move |msg: String| {
                let asker = asker.clone();
                Box::pin(async move {
                    let (tx, rx) = tokio::sync::oneshot::channel();
                    if asker
                        .send(AskRequest {
                            question: msg,
                            options: Vec::new(),
                            resp: tx,
                        })
                        .is_err()
                    {
                        return None;
                    }
                    rx.await.ok().flatten()
                })
                    as std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
            });
            e
        });
        // One transport-config snapshot, shared by both paths.
        let servers: Vec<(String, mcp::McpServerCfg)> = cfg
            .mcp
            .iter()
            .map(|(name, sc)| (name.clone(), mcp_wire_cfg(name, sc)))
            .collect();

        if let Some(tx) = mcp_status_tx {
            // DEFERRED (TUI): connect in the background so the input box renders immediately,
            // instead of blocking on every handshake. Tools hot-add to the live registry as
            // each server lands (same `<name>__` prefix + gate); status streams to `tx`.
            if !servers.is_empty() {
                tokio::spawn(connect_mcp_deferred(
                    servers,
                    approver.clone(),
                    reg.tools_handle(),
                    mcp_brief.clone(),
                    elicit.clone(),
                    tx,
                ));
            }
        } else {
            // SYNCHRONOUS: doctor/one-shot/REPL want the full result (and the injected
            // server-instructions brief) before running.
            for (name, scfg) in &servers {
                let on_change: mcp::ToolPublish = {
                    let handle = reg.tools_handle();
                    let approver = approver.clone();
                    let prefix = format!("{name}__");
                    Arc::new(move |fresh: Vec<Arc<dyn oxio_core::Tool>>| {
                        let gated: Vec<Arc<dyn oxio_core::Tool>> = fresh
                            .into_iter()
                            .map(|t| Guarded::wrap(t, approver.clone()))
                            .collect();
                        oxio_core::Registry::swap_server_tools(&handle, &prefix, gated);
                    })
                };
                match mcp::load_server(name, scfg, Some(on_change), elicit.clone()).await {
                    Ok(loaded) => {
                        mcp_status.push((name.clone(), Ok(loaded.tools.len())));
                        if let Some(ins) = loaded.instructions.filter(|s| !s.trim().is_empty()) {
                            if let Ok(mut b) = mcp_brief.lock() {
                                b.push_str(&format!("\n## {name}\n{ins}\n"));
                            }
                        }
                        for t in loaded.tools {
                            reg.add_tool(Guarded::wrap(t, approver.clone()));
                        }
                    }
                    Err(e) => mcp_status.push((name.clone(), Err(e))),
                }
            }
        }
        mcp_conn = Some(McpConnector {
            tools_handle: reg.tools_handle(),
            approver: approver.clone(),
            elicit,
            brief: mcp_brief.clone(),
        });
    }
    #[cfg(not(feature = "mcp"))]
    let _ = mcp_status_tx;

    // System prompt (profile override, else the default) injected at turn start.
    reg.add_transformer(
        Hook::PreTurn,
        Arc::new(SystemPrompt {
            text: system_prompt.clone(),
            mcp_brief: Some(mcp_brief.clone()),
        }),
    );
    // Expand ⟦img:PATH⟧ markers (dropped/pasted images) into inline Content::Image.
    reg.add_transformer(Hook::PreTurn, Arc::new(ImageExpander));
    // Cheap deterministic loop guard: reset per turn (PreTurn), check each model
    // step (PostModel). No model call.
    let loop_guard = Arc::new(LoopGuard::default());
    reg.add_transformer(Hook::PreTurn, loop_guard.clone());
    reg.add_transformer(Hook::PostModel, loop_guard);
    // Auto-compaction: layered token-budget compaction at PreModel (fires at 90%
    // of the model's context window; summarizes via the primary provider).
    // Context ceiling precedence: profile override → per-MODEL `[models.<id>]`
    // (context is a model property, not the machine's) → provider-level →
    // safe default.
    let mut context_window = profile
        .and_then(|p| p.context_ceiling)
        .map(|c| c as usize)
        .or_else(|| cfg.models.get(&model).and_then(|m| m.context_window))
        .or(ppc.context_window);
    // Nothing configured it: ask the endpoint what the model's window actually is. Local
    // servers report it (vLLM `max_model_len`, etc.); closed cloud vendors usually don't,
    // so this stays None there and drops to the default. Reads only what the server says.
    if context_window.is_none() {
        if let Some(base) = ppc.base_url.as_deref() {
            context_window = providers::model_max_context(base, &model).await;
        }
    }
    // Still unknown (e.g. a cloud vendor that doesn't advertise it): leave it `None` -
    // no fabricated ceiling. The user sets it with `/model ctx <n>` for that model.
    let compactor = Arc::new(context::Compactor::new(
        provider.clone(),
        model.clone(),
        context_window,
    ));
    reg.add_transformer(Hook::PreModel, compactor.clone());
    // Multi-turn session history: inject prior turns at PreTurn, capture at PostTurn.
    // Auto-logs every turn verbatim to a fresh per-session transcript so `--continue`
    // can restore losslessly; `repl` overrides the path when resuming.
    let session_path = session::new_session_path();
    let session_meta = session::SessionMeta {
        id: session_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("session")
            .to_string(),
        cwd: std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_default(),
        model: model.clone(),
        created_ms: session::now_ms(),
    };
    let session = session::SessionStore::new_logging(session_path, session_meta);
    reg.add_transformer(Hook::PreTurn, session.clone());
    reg.add_transformer(Hook::PostTurn, session.clone());
    // Durable memory: inject stored facts into context each turn (extraction is
    // triggered from the REPL: /remember, and auto on exit when enabled).
    reg.add_transformer(Hook::PreTurn, Arc::new(memory::MemoryInjector));
    // Interactive clarify tool lives here (UI layer owns terminal I/O), not in
    // the tools crate. kind=Read; non-TTY runs get a clean "no user" result.
    reg.add_tool(Arc::new(AskUser { asker }));

    // Sub-agent: delegates a subtask to a fresh kernel run (a sub-agent is just
    // another kernel run). Its model defaults to the main model but can be a
    // SEPARATE provider (smaller/faster) via `agent_provider`. Its tools default
    // to FULL (read+write+exec so it can code/fix bugs); `agent_tools="read"`
    // restricts to read-only. Writes stay gated by `Guarded`, and the `agent`
    // tool is excluded from its set - no recursion.
    let (agent_provider, agent_model): (Arc<dyn Provider>, String) =
        match cfg.defaults.agent_provider.as_deref() {
            Some(an) if an != pname => match cfg.providers.get(an) {
                // Same as vision: a bad key on the optional agent provider degrades to
                // the primary, never a startup crash.
                Some(apc) => match provider_from(an, apc).await {
                    Ok(p) => (p, apc.model.clone().unwrap_or_default()),
                    Err(e) => {
                        eprintln!("agent_provider '{an}': {e} - using primary for the agent");
                        (provider.clone(), model.clone())
                    }
                },
                None => {
                    eprintln!("agent_provider '{an}' not found - using primary for the agent");
                    (provider.clone(), model.clone())
                }
            },
            _ => (provider.clone(), model.clone()),
        };
    let agent_read_only = matches!(
        cfg.defaults.agent_tools.as_deref(),
        Some("read") | Some("readonly") | Some("read_only")
    );
    let sub_tools: Vec<Arc<dyn Tool>> = tools::builtin()
        .into_iter()
        .filter(|t| !agent_read_only || t.kind() == ToolKind::Read)
        .map(|t| Guarded::wrap(t, approver.clone()))
        .collect();
    // swarm: parallel multi-agent orchestration. Sub-agents are READ-ONLY (safe
    // to run concurrently - no write conflicts), bounded to 4 at a time.
    let swarm_tools: Vec<Arc<dyn Tool>> = tools::builtin()
        .into_iter()
        .filter(|t| t.kind() == ToolKind::Read)
        .map(|t| Guarded::wrap(t, approver.clone()))
        .collect();
    reg.add_tool(Arc::new(Swarm {
        provider: provider.clone(),
        model: model.clone(),
        tools: swarm_tools,
        system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        max_concurrent: 4,
    }));

    reg.add_tool(Arc::new(Agent {
        name: "agent".into(),
        description:
            "Delegate a self-contained subtask to a sub-agent with its own fresh context and \
tools; returns its final answer. Good for isolating a large exploration, or handing off a codeable \
subtask (the sub-agent can edit and run commands; writes stay permission-gated). Put ALL needed \
context in `task`."
                .into(),
        provider: agent_provider,
        model: agent_model,
        tools: sub_tools,
        system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
    }));

    // Named, individually-configured sub-agents from `[agents.<name>]` - each a
    // delegating tool `agent_<name>` with its own model / tools / permission /
    // prompt.
    // Resolve the session's skills once - shared by agent skill-binding (below) and the
    // `skill` tool (further down).
    let session_skills = resolve_skills(cfg);
    for (aname, acfg) in &cfg.agents {
        let (aprov, amodel): (Arc<dyn Provider>, String) = match acfg.provider.as_deref() {
            Some(pn) if pn != pname => match cfg.providers.get(pn) {
                Some(pc) => (
                    provider_from(pn, pc).await?,
                    pc.model.clone().unwrap_or_default(),
                ),
                None => {
                    eprintln!("agent '{aname}': provider '{pn}' not found - using primary");
                    (provider.clone(), model.clone())
                }
            },
            _ => (provider.clone(), model.clone()),
        };
        let perm = acfg.permission.as_deref().unwrap_or("prompt");
        let aapprover: Arc<dyn Approver> = match perm {
            "accept" | "accept_edits" | "auto" => Arc::new(AllowAll),
            "deny" => Arc::new(DenyAll),
            _ => approver.clone(),
        };
        let read_only = perm == "read_only"
            || matches!(acfg.tools.as_deref(), Some(t) if t.iter().any(|x| x == "read"));
        let atools: Vec<Arc<dyn Tool>> = tools::builtin()
            .into_iter()
            .filter(|t| {
                if read_only && t.kind() != ToolKind::Read {
                    return false;
                }
                match &acfg.tools {
                    None => true,
                    Some(list) if list.iter().any(|x| x == "all" || x == "read") => true,
                    Some(list) => list.iter().any(|x| x == &t.spec().name),
                }
            })
            .map(|t| Guarded::wrap(t, aapprover.clone()))
            .collect();
        // Bind this agent's declared skills: append each resolved skill's instructions to
        // its system prompt, so the role always loads its domain skills (the oxio
        // equivalent of a vendor sub-agent's `skills: [...]` frontmatter).
        let base_prompt = acfg
            .prompt
            .clone()
            .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
        let asys = bind_agent_skills(
            base_prompt,
            acfg.skills.as_deref().unwrap_or(&[]),
            &session_skills,
            aname,
        );
        reg.add_tool(Arc::new(Agent {
            name: format!("agent_{aname}"),
            description: acfg.description.clone().unwrap_or_else(|| {
                format!("Delegate a self-contained subtask to the '{aname}' sub-agent.")
            }),
            provider: aprov,
            model: amodel,
            tools: atools,
            system_prompt: asys,
        }));
    }

    // view_image: routes to a vision provider (`vision_provider`, default primary
    // so a multimodal local model is reused). Sends image(s) as a user message,
    // returns the model's answer.
    let (vision_provider, vision_model): (Arc<dyn Provider>, String) =
        match cfg.defaults.vision_provider.as_deref() {
            Some(vn) if vn != pname => match cfg.providers.get(vn) {
                // A missing/invalid key on the OPTIONAL vision provider must NOT brick
                // startup - degrade to the primary for view_image and warn instead.
                Some(vpc) => match provider_from(vn, vpc).await {
                    Ok(p) => (p, vpc.model.clone().unwrap_or_default()),
                    Err(e) => {
                        eprintln!("vision_provider '{vn}': {e} - using primary for view_image");
                        (provider.clone(), model.clone())
                    }
                },
                None => {
                    eprintln!("vision_provider '{vn}' not found - using primary for view_image");
                    (provider.clone(), model.clone())
                }
            },
            _ => (provider.clone(), model.clone()),
        };
    reg.add_tool(Arc::new(ViewImage {
        provider: vision_provider,
        model: vision_model,
    }));

    // Reusable instruction packs → the `skill` tool (loads instructions on demand).
    // Two sources merged: on-disk SKILL.md skills (oxio flavour) and config-inline
    // `[skills.<name>]`; config-inline overrides a same-named file skill. Registered
    // only when at least one skill exists.
    if !session_skills.is_empty() {
        reg.add_tool(Arc::new(SkillTool {
            skills: session_skills,
        }));
    }

    // Language servers → the `lsp` diagnostics tool (kind=Read). Empty config =
    // no tool registered (zero cost).
    #[cfg(feature = "lsp")]
    if !cfg.lsp.is_empty() {
        let servers = cfg
            .lsp
            .iter()
            .map(|(ext, c)| {
                (
                    ext.clone(),
                    lsp::LspServerCfg {
                        command: c.command.clone(),
                        args: c.args.clone(),
                    },
                )
            })
            .collect();
        reg.add_tool(Arc::new(lsp::LspTool { servers }));
    }

    // User shell hooks (observational) fired on lifecycle events.
    if !cfg.hooks.is_empty() {
        reg.add_observer(Arc::new(HookRunner {
            hooks: cfg.hooks.clone(),
        }));
    }

    Ok((
        Kernel::new(reg, model),
        session,
        compactor,
        snapshots,
        LiveControls {
            provider: swap,
            thinking,
            mcp_status,
            #[cfg(feature = "mcp")]
            mcp: mcp_conn,
        },
    ))
}

/// Interactive permission prompt. Asks on the terminal before a write/exec tool
/// runs; if stdin is not a TTY (piped / headless) it blocks by policy, so an
/// unattended run can never silently perform a gated action.
pub struct StdinApprover;

#[async_trait]
impl Approver for StdinApprover {
    async fn approve(&self, req: &ApprovalRequest) -> Decision {
        // No TTY = no human at the prompt, so a refusal here is a policy Block,
        // not a live "No" (the two drive different model behavior in safety).
        if !std::io::stdin().is_terminal() {
            return Decision::Block;
        }
        print!("\nallow {} ? [y]es / [N]o: ", req.summary);
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            return Decision::Block;
        }
        // A human typed at the prompt; anything that is not yes is a live "No".
        match line.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => Decision::Once,
            _ => Decision::Deny,
        }
    }
}

/// Ask the user one clarifying question and return their typed answer. Lives in
/// the UI layer because it owns terminal I/O. Interactive only: a non-TTY run
/// returns a clean result telling the model to proceed on its own judgement.
/// A question routed from the `ask_user_question` tool to the TUI event loop, which
/// displays it and sends the user's typed answer back over `resp` (`None` = no answer /
/// cancelled). Mirrors the approval-modal channel so the tool NEVER does a blocking
/// stdin read in the TUI (which deadlocks against crossterm's raw-mode input).
pub struct AskRequest {
    pub question: String,
    pub options: Vec<String>,
    pub resp: tokio::sync::oneshot::Sender<Option<String>>,
}
/// Channel the TUI installs so `ask_user_question` routes through the event loop.
pub type AskSender = tokio::sync::mpsc::UnboundedSender<AskRequest>;

pub struct AskUser {
    /// When set (TUI), route the question over this channel and `await` the answer - no
    /// blocking stdin read, and cancellable. `None` (line REPL / one-shot) → stdin fallback.
    pub asker: Option<AskSender>,
}

#[async_trait]
impl Tool for AskUser {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "ask_user_question".into(),
            description: "Ask the user ONE clarifying question and return their typed answer. Use \
sparingly, only when genuinely blocked: either you are missing information you need to proceed \
correctly (e.g. which of several cases applies to their situation), or it is a decision only the user \
can make. Do NOT use it for anything you can determine yourself or infer from the conversation - never \
ask the obvious. Optionally pass `options` as suggested choices."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "the question to ask" },
                    "options": { "type": "array", "items": { "type": "string" }, "description": "optional suggested choices" }
                },
                "required": ["question"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> CoreResult<ToolOutput> {
        let question = match input.get("question").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return Ok(ToolOutput::error("ask_user_question: missing 'question'")),
        };
        // TUI path: route the request → UI → response over a channel - send the question to
        // the event loop and `await` the answer. This never
        // touches stdin - a blocking stdin read deadlocks against crossterm's raw-mode input
        // and can't be cancelled (the hang). The `await` is async, so the kernel's guard()
        // cancels it on Esc/timeout.
        if let Some(tx) = &self.asker {
            let options: Vec<String> = input
                .get("options")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            if tx
                .send(AskRequest {
                    question: question.to_string(),
                    options,
                    resp: resp_tx,
                })
                .is_err()
            {
                return Ok(ToolOutput::error("ask_user_question: UI unavailable"));
            }
            return match resp_rx.await {
                Ok(Some(ans)) if !ans.trim().is_empty() => {
                    Ok(ToolOutput::ok(format!("user answered: {ans}")))
                }
                Ok(_) => Ok(ToolOutput::ok("(user gave no answer)")),
                Err(_) => Ok(ToolOutput::error(
                    "ask_user_question: UI closed before answering",
                )),
            };
        }
        if !std::io::stdin().is_terminal() {
            return Ok(ToolOutput::error(
                "ask_user_question: no interactive user (non-TTY); proceed with your best judgement",
            ));
        }
        println!("\n{question}");
        if let Some(opts) = input.get("options").and_then(|v| v.as_array()) {
            for (i, o) in opts.iter().filter_map(|v| v.as_str()).enumerate() {
                println!("  {}. {o}", i + 1);
            }
        }
        print!("> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            return Ok(ToolOutput::error("ask_user_question: failed to read input"));
        }
        let ans = line.trim();
        if ans.is_empty() {
            return Ok(ToolOutput::ok("(user gave no answer)"));
        }
        Ok(ToolOutput::ok(format!("user answered: {ans}")))
    }
}

fn image_media_type(path: &str) -> &'static str {
    match std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}

/// View one or more local images: reads each path, converts on the machine to a
/// base64 data URL, sends them with a question
/// to the configured vision provider, and returns its answer. Routing lives here
/// (`vision_provider`, default = primary → reuse a multimodal local model).
pub struct ViewImage {
    provider: Arc<dyn Provider>,
    model: String,
}

#[async_trait]
impl Tool for ViewImage {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "view_image".into(),
            description: "View local image(s): pass `paths` (one or more image files) and an optional \
`question`. The images are sent to the vision model together; returns its answer. Needs a multimodal \
provider (configure `vision_provider`, or point the primary at a vision model)."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "paths": { "type": "array", "items": { "type": "string" }, "description": "image file paths" },
                    "question": { "type": "string", "description": "what to ask about the image(s)" }
                },
                "required": ["paths"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> CoreResult<ToolOutput> {
        let paths: Vec<String> = match input.get("paths").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            None => {
                return Ok(ToolOutput::error(
                    "view_image: 'paths' must be an array of image file paths",
                ))
            }
        };
        if paths.is_empty() {
            return Ok(ToolOutput::error("view_image: 'paths' is empty"));
        }
        let question = input
            .get("question")
            .and_then(|v| v.as_str())
            .unwrap_or("Describe these image(s) in detail.");
        match describe_images(&self.provider, &self.model, &paths, question, ctx).await {
            Ok(text) => Ok(ToolOutput::ok(text)),
            Err(e) => Ok(ToolOutput::error(format!("view_image: {e}"))),
        }
    }
}

/// Send local image(s) to a (multimodal) provider and return its text answer.
/// Shared by the `view_image` tool and the composer's vision fallback: reads each
/// file, base64-encodes on the machine, and asks `question`.
pub async fn describe_images(
    provider: &Arc<dyn Provider>,
    model: &str,
    paths: &[String],
    question: &str,
    ctx: &Ctx,
) -> anyhow::Result<String> {
    use base64::Engine as _;
    let mut content = vec![Content::Text {
        text: question.to_string(),
    }];
    for p in paths {
        let bytes = std::fs::read(p).map_err(|e| anyhow::anyhow!("read {p}: {e}"))?;
        content.push(Content::Image {
            media_type: image_media_type(p).to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(&bytes),
        });
    }
    let req = Request {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content,
            tool_calls: vec![],
            tool_call_id: None,
        }],
        tools: vec![],
        params: Params::default(),
    };
    let resp = provider
        .complete(req, ctx)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(resp.message.as_text())
}

/// Resolve the cloud/vision fallback: `(provider, model, name)` - but ONLY when a
/// DISTINCT `defaults.vision_provider` is configured (so it is a real fallback, not
/// the text-only primary). Returns `None` otherwise, so the caller can tell the user
/// to configure one instead of silently re-failing against the same model.
pub async fn vision_fallback(cfg: &Config) -> Option<(Arc<dyn Provider>, String, String)> {
    let name = cfg.defaults.vision_provider.as_deref()?;
    if name == cfg.defaults.primary {
        return None;
    }
    let pc = cfg.providers.get(name)?;
    let provider = provider_from(name, pc).await.ok()?;
    Some((
        provider,
        pc.model.clone().unwrap_or_default(),
        name.to_string(),
    ))
}

/// Resolve the PRIMARY provider `(provider, model)` - the local model, reused to generate
/// the ghost next-prompt suggestion. Separate handle from the kernel's (a cheap HTTP client
/// against the same endpoint) so a suggestion never contends with the live turn.
pub async fn primary_provider(cfg: &Config) -> Option<(Arc<dyn Provider>, String)> {
    let (name, pc) = cfg.primary()?;
    let provider = provider_from(name, pc).await.ok()?;
    Some((provider, pc.model.clone().unwrap_or_default()))
}

/// Predict the user's NEXT prompt from the recent conversation, for the ghost-text hint
/// (Tab to accept). Reuses the user's own model - a cheap one-shot completion, no tools,
/// nothing added to the transcript. `None` if the model returns nothing usable (never a
/// fabricated placeholder). Kept short: first line only, length-capped.
pub async fn suggest_next(
    provider: &Arc<dyn Provider>,
    model: &str,
    convo: &str,
    ctx: &Ctx,
) -> Option<String> {
    let prompt = format!(
        "You are predicting the user's NEXT message in this coding session. Given the conversation \
below, write the single most likely next thing the user would type. Reply with ONLY that message - \
no quotes, no preamble, no explanation, at most one short line.\n\nConversation:\n{convo}\n\nNext message:"
    );
    let req = Request {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text { text: prompt }],
            tool_calls: vec![],
            tool_call_id: None,
        }],
        tools: vec![],
        params: Params::default(),
    };
    let resp = provider.complete(req, ctx).await.ok()?;
    let line = resp.message.as_text();
    let line = line
        .trim()
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('"')
        .trim();
    if line.is_empty() || line.chars().count() > 160 {
        None
    } else {
        Some(line.to_string())
    }
}

/// Loads a named skill's instructions on demand (reusable instruction pack).
/// Lives in the UI layer (built from config). kind=Read.
pub struct SkillTool {
    skills: std::collections::BTreeMap<String, config::SkillCfg>,
}

#[async_trait]
impl Tool for SkillTool {
    fn spec(&self) -> ToolSpec {
        let mut avail = String::new();
        for (name, s) in &self.skills {
            avail.push_str(&format!(
                "\n- {name}: {}",
                s.description.clone().unwrap_or_default()
            ));
        }
        ToolSpec {
            name: "skill".into(),
            description: format!(
                "Load a named skill's instructions on demand (call BEFORE the relevant work). Available skills:{avail}"
            ),
            input_schema: json!({
                "type": "object",
                "properties": { "name": { "type": "string", "description": "the skill to load" } },
                "required": ["name"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> CoreResult<ToolOutput> {
        let name = match input.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return Ok(ToolOutput::error("skill: missing 'name'")),
        };
        match self.skills.get(name) {
            Some(s) => Ok(ToolOutput::ok(s.instructions.clone())),
            None => {
                let names: Vec<&str> = self.skills.keys().map(|s| s.as_str()).collect();
                Ok(ToolOutput::error(format!(
                    "skill: no skill '{name}' (have: {})",
                    names.join(", ")
                )))
            }
        }
    }
}

/// Forwards a sub-agent's events to the parent UI's progress channel, TAGGED with
/// the agent's name, so a spawned agent is visible live (its tool activity,
/// start/done) instead of a silent black box. Only NOTICES are forwarded - the
/// sub-agent's own token stream would flood the main transcript, and its answer
/// returns via `run_sub_kernel`'s result. The per-agent activity line surfaces its
/// tool use without flooding the transcript.
struct AgentSink {
    inner: Arc<dyn StreamSink>,
    name: String,
}

#[async_trait]
impl StreamSink for AgentSink {
    async fn send(&self, ev: StreamEvent) {
        if let StreamEvent::Notice { level, text } = ev {
            self.inner
                .send(StreamEvent::Notice {
                    level,
                    text: format!("[{}] {text}", self.name),
                })
                .await;
        }
        // TextDelta / ThinkingDelta / ToolCallDelta / Done: dropped - the sub-agent's
        // answer is returned to the caller, not streamed into the parent transcript.
    }
}

/// Run one task in a fresh kernel (given tools + system prompt + loop guard) and
/// return its final text. Shared by `agent` (one run) and `swarmflow` (many
/// parallel runs). Takes an owned `Ctx` so it can be spawned concurrently.
/// `agent_name` tags the live activity forwarded to the UI (via `ctx.progress`).
async fn run_sub_kernel(
    provider: Arc<dyn Provider>,
    model: String,
    tools: Vec<Arc<dyn Tool>>,
    system_prompt: String,
    task: String,
    ctx: Ctx,
    agent_name: &str,
) -> CoreResult<String> {
    let mut reg = Registry::new();
    reg.add_provider(provider);
    for t in tools {
        reg.add_tool(t);
    }
    reg.add_transformer(
        Hook::PreTurn,
        Arc::new(SystemPrompt {
            text: system_prompt,
            mcp_brief: None,
        }),
    );
    let lg = Arc::new(LoopGuard::default());
    reg.add_transformer(Hook::PreTurn, lg.clone());
    reg.add_transformer(Hook::PostModel, lg);
    let kernel = Kernel::new(reg, model);

    // If the parent attached a progress channel, surface this agent's activity
    // live (tagged by name) + a start/done line with tokens & elapsed - exposing
    // the per-agent metrics a statusline/script can render. Else run silent.
    let sink: Box<dyn StreamSink> = match &ctx.progress {
        Some(p) => Box::new(AgentSink {
            inner: p.clone(),
            name: agent_name.to_string(),
        }),
        None => Box::new(NullSink),
    };
    let started = std::time::Instant::now();
    sink.send(StreamEvent::Notice {
        level: NoticeLevel::Progress,
        text: "started".into(),
    })
    .await;
    let result = kernel.run_turn(task, &ctx, &*sink).await;
    match result {
        Ok(o) => {
            let secs = started.elapsed().as_secs_f64();
            let tok = o.usage.input_tokens + o.usage.output_tokens;
            sink.send(StreamEvent::Notice {
                level: NoticeLevel::Progress,
                text: format!("done · {tok} tok · {secs:.1}s"),
            })
            .await;
            Ok(o.text)
        }
        Err(e) => {
            sink.send(StreamEvent::Notice {
                level: NoticeLevel::Warn,
                text: format!("failed: {e}"),
            })
            .await;
            Err(e)
        }
    }
}

const MEMORY_EXTRACTION_PROMPT: &str = "You extract DURABLE, reusable facts from a coding session for \
future sessions: user preferences, decisions, conventions, project constraints, and gotchas. Output ONE \
fact per line, terse and self-contained. Only strong, reusable facts - skip transient chatter. If there \
is nothing durable worth keeping, output nothing.";

/// Extract durable memories from a transcript via a read-only sub-agent and append
/// each to the memory store. Returns how many were stored (tier-2 extraction).
async fn extract_memories(
    provider: Arc<dyn Provider>,
    model: String,
    transcript: Vec<Message>,
) -> usize {
    let mut text = String::new();
    for m in &transcript {
        let t = m.as_text();
        if !t.is_empty() {
            text.push_str(&format!("[{:?}] {}\n", m.role, t));
        }
    }
    if text.trim().is_empty() {
        return 0;
    }
    let out = match run_sub_kernel(
        provider,
        model,
        vec![],
        MEMORY_EXTRACTION_PROMPT.to_string(),
        text,
        Ctx::default(),
        "memory",
    )
    .await
    {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let mut n = 0;
    for line in out.lines() {
        let fact = line.trim().trim_start_matches(['-', '*', '•']).trim();
        if fact.len() >= 3
            && !fact.eq_ignore_ascii_case("none")
            && memory::append_memory(fact).is_ok()
        {
            n += 1;
        }
    }
    n
}

const MEMORY_CONSOLIDATION_PROMPT: &str = "You consolidate a list of durable memory facts. Merge \
duplicates and near-duplicates, drop anything obsolete or contradicted (keep the newer), and tighten \
wording. Output the clean set, ONE fact per line, and nothing else.";

/// Run consolidation only once the store grows past this (cheapest-first gating).
const CONSOLIDATE_THRESHOLD: usize = 20;

/// Tier-3 consolidation: dedup/merge/prune the memory store via a read-only
/// sub-agent, rewriting it. Returns (before, after) counts.
async fn consolidate_memories(provider: Arc<dyn Provider>, model: String) -> (usize, usize) {
    let facts = memory::load_memories();
    let before = facts.len();
    if before < 2 {
        return (before, before);
    }
    let input = facts
        .iter()
        .map(|f| format!("- {f}"))
        .collect::<Vec<_>>()
        .join("\n");
    let out = match run_sub_kernel(
        provider,
        model,
        vec![],
        MEMORY_CONSOLIDATION_PROMPT.to_string(),
        input,
        Ctx::default(),
        "memory",
    )
    .await
    {
        Ok(s) => s,
        Err(_) => return (before, before),
    };
    let cleaned: Vec<String> = out
        .lines()
        .map(|l| {
            l.trim()
                .trim_start_matches(['-', '*', '•'])
                .trim()
                .to_string()
        })
        .filter(|f| f.len() >= 3)
        .collect();
    if cleaned.is_empty() {
        return (before, before);
    }
    let after = cleaned.len();
    let _ = memory::replace_memories(&cleaned);
    (before, after)
}

/// swarm: deterministic PARALLEL multi-agent orchestration on top of `agent`.
/// Runs a list of subtasks as isolated read-only sub-agents concurrently (bounded),
/// aggregates, with an optional synthesis pass. Reliable on any model - a plain
/// task list, no model-authored script. Read-only sub-agents keep parallel runs conflict-free.
pub struct Swarm {
    provider: Arc<dyn Provider>,
    model: String,
    tools: Vec<Arc<dyn Tool>>,
    system_prompt: String,
    max_concurrent: usize,
}

#[async_trait]
impl Tool for Swarm {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "swarm".into(),
            description: "Run several subtasks IN PARALLEL as isolated sub-agents (each a fresh context, \
read-only tools), then aggregate. Pass `tasks` (array of self-contained subtasks) and optional \
`synthesize` (bool) to merge them into one answer. Use for breadth: review/audit/research across many \
files at once, beyond a single context."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tasks": { "type": "array", "items": { "type": "string" }, "description": "self-contained subtasks to run in parallel" },
                    "synthesize": { "type": "boolean", "description": "merge the results into one answer" }
                },
                "required": ["tasks"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> CoreResult<ToolOutput> {
        let tasks: Vec<String> = input
            .get("tasks")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if tasks.is_empty() {
            return Ok(ToolOutput::error(
                "swarmflow: 'tasks' must be a non-empty array of subtasks",
            ));
        }
        let synthesize = input
            .get("synthesize")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut results: Vec<(usize, String)> = Vec::new();
        let mut base = 0usize;
        for batch in tasks.chunks(self.max_concurrent.max(1)) {
            let mut set = tokio::task::JoinSet::new();
            for (i, task) in batch.iter().enumerate() {
                let gi = base + i;
                let (p, m, tls, sp, c, t) = (
                    self.provider.clone(),
                    self.model.clone(),
                    self.tools.clone(),
                    self.system_prompt.clone(),
                    ctx.clone(),
                    task.clone(),
                );
                let label = format!("swarm#{}", gi + 1);
                set.spawn(async move {
                    let r = run_sub_kernel(p, m, tls, sp, t, c, &label)
                        .await
                        .unwrap_or_else(|e| format!("[error: {e}]"));
                    (gi, r)
                });
            }
            while let Some(joined) = set.join_next().await {
                if let Ok(pair) = joined {
                    results.push(pair);
                }
            }
            base += batch.len();
        }
        results.sort_by_key(|(i, _)| *i);
        let mut combined = String::new();
        for (i, r) in &results {
            combined.push_str(&format!(
                "=== task {} ({}) ===\n{}\n\n",
                i + 1,
                tasks[*i],
                r
            ));
        }
        let combined = combined.trim_end().to_string();

        if synthesize {
            let synth = format!("Synthesize these parallel sub-agent results into one coherent answer:\n\n{combined}");
            return match run_sub_kernel(
                self.provider.clone(),
                self.model.clone(),
                self.tools.clone(),
                self.system_prompt.clone(),
                synth,
                ctx.clone(),
                "swarm:synthesis",
            )
            .await
            {
                Ok(s) => Ok(ToolOutput::ok(s)),
                Err(e) => Ok(ToolOutput::error(format!("swarmflow: synthesis: {e}"))),
            };
        }
        Ok(ToolOutput::ok(combined))
    }
}

/// Sub-agent tool: runs a delegated subtask in a fresh kernel with read-only
/// tools and its own transcript, returning the sub-agent's final answer. Reuses
/// the frozen kernel - no separate orchestration engine.
pub struct Agent {
    name: String,
    description: String,
    provider: Arc<dyn Provider>,
    model: String,
    tools: Vec<Arc<dyn Tool>>,
    system_prompt: String,
}

#[async_trait]
impl Tool for Agent {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: json!({
                "type": "object",
                "properties": { "task": { "type": "string", "description": "the self-contained subtask" } },
                "required": ["task"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> CoreResult<ToolOutput> {
        let task = match input.get("task").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => return Ok(ToolOutput::error("agent: missing 'task'")),
        };
        match run_sub_kernel(
            self.provider.clone(),
            self.model.clone(),
            self.tools.clone(),
            self.system_prompt.clone(),
            task,
            ctx.clone(),
            &self.name,
        )
        .await
        {
            Ok(text) => Ok(ToolOutput::ok(text)),
            Err(e) => Ok(ToolOutput::error(format!("agent: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctx_status_formats_percent_and_k_tokens() {
        assert_eq!(ctx_status(13000, Some(32000)), "40% · 13k/32k");
        assert_eq!(ctx_status(0, Some(32000)), "0% · 0/32k");
        assert_eq!(ctx_status(500, Some(4000)), "12% · 500/4k");
        // Unknown window: no fabricated ceiling or percentage, and no divide-by-zero.
        assert_eq!(
            ctx_status(100, None),
            "100 used · window unset (/model ctx <n>)"
        );
        assert_eq!(
            ctx_status(100, Some(0)),
            "100 used · window unset (/model ctx <n>)"
        );
    }

    #[test]
    fn cost_from_config_pricing_or_none() {
        let mut p = BTreeMap::new();
        p.insert(
            "gemma".to_string(),
            RateCfg {
                input: 0.10,
                output: 0.30,
            },
        );
        // Priced model: 1M in + 1M out at gemma rate = $0.40.
        let c = cost_usd(1_000_000, 1_000_000, &p, "gemma-3-27b").expect("priced");
        assert!((c - 0.40).abs() < 1e-9, "got {c}");
        // Unpriced model → None (no fabricated figure), and empty config → None.
        assert!(cost_usd(1_000_000, 0, &p, "some-unknown-model").is_none());
        assert!(cost_usd(1_000_000, 0, &BTreeMap::new(), "gemma-3").is_none());
    }

    #[tokio::test]
    async fn ask_user_non_tty_returns_clean_result() {
        // Under `cargo test` stdin is not a terminal.
        let out = AskUser { asker: None }
            .call(json!({ "question": "which db?" }), &Ctx::default())
            .await
            .unwrap();
        assert!(
            out.is_error && out.content.contains("no interactive user"),
            "got {:?}",
            out.content
        );
    }

    #[tokio::test]
    async fn ask_user_missing_question_errors() {
        let out = AskUser { asker: None }
            .call(json!({}), &Ctx::default())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("missing 'question'"));
    }

    struct StubVision;
    #[async_trait]
    impl Provider for StubVision {
        fn name(&self) -> &str {
            "stub-vision"
        }
        async fn complete(&self, _req: Request, _ctx: &Ctx) -> CoreResult<oxio_core::Response> {
            Ok(oxio_core::Response {
                message: Message::text(Role::Assistant, "SEEN"),
                usage: Default::default(),
                stop_reason: oxio_core::StopReason::EndTurn,
            })
        }
    }

    #[tokio::test]
    async fn view_image_validates_and_sends_to_vision_provider() {
        let vi = ViewImage {
            provider: Arc::new(StubVision),
            model: "m".into(),
        };
        assert!(
            vi.call(json!({}), &Ctx::default()).await.unwrap().is_error,
            "missing paths"
        );
        assert!(
            vi.call(json!({ "paths": [] }), &Ctx::default())
                .await
                .unwrap()
                .is_error,
            "empty paths"
        );

        let f = std::env::temp_dir().join("oxio_vi_test.png");
        std::fs::write(&f, [0u8, 1, 2, 3]).unwrap();
        let out = vi
            .call(
                json!({ "paths": [f.to_string_lossy()], "question": "what?" }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(
            !out.is_error && out.content.contains("SEEN"),
            "got {:?}",
            out.content
        );
        let _ = std::fs::remove_file(&f);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn swarm_fans_out_tasks_and_aggregates() {
        let s = Swarm {
            provider: Arc::new(StubVision),
            model: "m".into(),
            tools: vec![],
            system_prompt: "sp".into(),
            max_concurrent: 2,
        };
        assert!(
            s.call(json!({ "tasks": [] }), &Ctx::default())
                .await
                .unwrap()
                .is_error,
            "empty tasks"
        );
        let out = s
            .call(json!({ "tasks": ["a", "b", "c"] }), &Ctx::default())
            .await
            .unwrap();
        assert!(!out.is_error, "got {:?}", out.content);
        for label in ["task 1", "task 2", "task 3"] {
            assert!(
                out.content.contains(label),
                "aggregated {label}; got {:?}",
                out.content
            );
        }
        assert!(out.content.contains("SEEN"), "sub-agent result present");
    }
}

/// Resolve a provider's credential through the auth seam, then build the wire
/// adapter. Official vendor sign-in is delegated later; today: none / api-key.
/// Compact token count: `12k` above 1000, raw below.
fn fmt_tokens(n: usize) -> String {
    if n >= 1000 {
        format!("{}k", (n + 500) / 1000)
    } else {
        n.to_string()
    }
}

/// Context-used status: `41% · 13k/32k` when the window is known; when it is unknown we
/// show the used count and a hint to set it, never a fabricated ceiling or percentage.
fn ctx_status(used: usize, window: Option<usize>) -> String {
    match window {
        Some(w) if w > 0 => {
            let pct = used.saturating_mul(100) / w;
            format!("{pct}% · {}/{}", fmt_tokens(used), fmt_tokens(w))
        }
        _ => format!("{} used · window unset (/model ctx <n>)", fmt_tokens(used)),
    }
}

/// Whether to emit ANSI color: off when `NO_COLOR` is set or stdout is not a TTY
/// (piped/redirected). Memoized. Honors https://no-color.org.
fn color_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal())
}

/// The landing banner: logo + status, color-gated.
fn print_banner(provider: &str, model: &str, profile: &str, resumed: usize) {
    let c = color_enabled();
    let (accent, dim, rst) = if c {
        ("\x1b[38;5;39m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let logo = [r"  ┌─┐╲ ╱│┌─┐", r"  │ │ ╳ ││ │", r"  └─┘╱ ╲│└─┘"];
    println!();
    for line in logo {
        println!("{accent}{line}{rst}");
    }
    println!();
    println!("  {dim}model{rst}  {model}");
    println!("  {dim}via{rst}    {provider}  {dim}·{rst}  profile:{profile}");
    if resumed > 0 {
        println!("  {dim}resumed{rst} {resumed} messages restored");
    }
    println!("  {dim}help{rst}   /help  /undo  /compact  /resume  /quit");
    println!();
}

/// (input, output) USD per 1M tokens for `model`, looked up from the user's
/// `[pricing.*]` config (key = a substring of the model name). Pricing is a config
/// layer, NOT hardcoded - models and prices churn, and a stale baked-in number is
/// worse than none. `None` = unpriced → we show no dollar figure, never a fabricated one.
fn model_rate(pricing: &BTreeMap<String, RateCfg>, model: &str) -> Option<(f64, f64)> {
    let m = model.to_lowercase();
    pricing
        .iter()
        .find(|(k, _)| !k.is_empty() && m.contains(&k.to_lowercase()))
        .map(|(_, r)| (r.input, r.output))
}

/// USD for the token counts, or `None` if the model has no configured price.
fn cost_usd(
    input_tok: u64,
    output_tok: u64,
    pricing: &BTreeMap<String, RateCfg>,
    model: &str,
) -> Option<f64> {
    let (ri, ro) = model_rate(pricing, model)?;
    Some((input_tok as f64 * ri + output_tok as f64 * ro) / 1_000_000.0)
}

async fn provider_from(name: &str, pc: &ProviderCfg) -> anyhow::Result<Arc<dyn Provider>> {
    let auth: Arc<dyn Auth> = match pc.auth {
        AuthKind::None => Arc::new(NoAuth),
        AuthKind::ApiKey => {
            let key = auth::api_key_from_env(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "provider '{name}' needs an API key - set env OXIO_{}_API_KEY",
                    name.to_uppercase().replace('-', "_")
                )
            })?;
            Arc::new(ApiKeyAuth::new(key))
        }
        AuthKind::Oauth => anyhow::bail!(
            "provider '{name}': OAuth sign-in not wired yet - use auth = \"api_key\" or \"none\" for now"
        ),
    };
    let bearer = auth.bearer().await?;

    match pc.wire_api {
        WireApi::ChatCompletions => Ok(Arc::new(
            ChatCompletionsAdapter::new(
                name,
                pc.base_url.clone().unwrap_or_default(),
                pc.model.clone().unwrap_or_default(),
                bearer,
            )
            .with_sampling(Sampling {
                temperature: pc.temperature,
                top_p: pc.top_p,
                max_tokens: pc.max_tokens,
                reasoning_effort: pc.reasoning_effort.clone(),
            }),
        )),
        WireApi::Responses => Ok(Arc::new(
            ResponsesAdapter::new(
                name,
                pc.base_url.clone().unwrap_or_default(),
                pc.model.clone().unwrap_or_default(),
                bearer,
            )
            .with_sampling(Sampling {
                temperature: pc.temperature,
                top_p: pc.top_p,
                max_tokens: pc.max_tokens,
                reasoning_effort: pc.reasoning_effort.clone(),
            }),
        )),
        WireApi::Anthropic => Ok(Arc::new(
            AnthropicAdapter::new(
                name,
                pc.base_url.clone().unwrap_or_default(),
                pc.model.clone().unwrap_or_default(),
                bearer,
            )
            .with_sampling(Sampling {
                temperature: pc.temperature,
                top_p: pc.top_p,
                max_tokens: pc.max_tokens,
                reasoning_effort: pc.reasoning_effort.clone(),
            }),
        )),
    }
}

/// Streams events to stdout: thinking dimmed, answer normal.
pub struct StdoutSink;

#[async_trait]
impl StreamSink for StdoutSink {
    async fn send(&self, ev: StreamEvent) {
        let c = color_enabled();
        match ev {
            StreamEvent::ThinkingDelta(t) => {
                if c {
                    print!("\x1b[2m{t}\x1b[0m");
                } else {
                    print!("{t}");
                }
                let _ = std::io::stdout().flush();
            }
            StreamEvent::TextDelta(t) => {
                print!("{t}");
                let _ = std::io::stdout().flush();
            }
            // Out-of-band runtime notices (tool dispatch, compaction, fallback) so
            // the terminal never sits dead while work happens. Styled by level.
            StreamEvent::Notice { level, text } => {
                match (c, level) {
                    (true, NoticeLevel::Warn) => println!("\x1b[33m{text}\x1b[0m"),
                    (true, _) => println!("\x1b[2m{text}\x1b[0m"),
                    (false, _) => println!("{text}"),
                }
                let _ = std::io::stdout().flush();
            }
            _ => {}
        }
    }
}

/// One-shot: run a single prompt and stream the answer to stdout.
pub async fn run_once(cfg: &Config, prompt: &str) -> anyhow::Result<()> {
    let (kernel, _session, _compactor, _snapshots, _ctl) =
        build_kernel(cfg, Arc::new(StdinApprover), None, None).await?;
    let ctx = Ctx::default();
    kernel.run_turn(prompt, &ctx, &StdoutSink).await?;
    println!();
    Ok(())
}

/// Read one trimmed line from stdin (onboarding is pre-TUI, line-based). Empty on error.
fn prompt_line(label: &str) -> String {
    print!("{label}");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return String::new();
    }
    line.trim().to_string()
}

/// Sanitize an endpoint label into a config key (lowercase, no spaces/dots).
fn provider_key(label: &str) -> String {
    label.to_lowercase().replace([' ', '.'], "-")
}

/// A local OpenAI-compatible provider (discovered or custom) - auth=none.
fn local_provider(base_url: String, model: Option<String>) -> ProviderCfg {
    ProviderCfg {
        wire_api: WireApi::ChatCompletions,
        base_url: Some(base_url),
        model,
        auth: AuthKind::None,
        temperature: None,
        top_p: None,
        max_tokens: None,
        reasoning_effort: None,
        context_window: None,
        vision: None,
    }
}

/// Known cloud vendors: (key, label, base_url, wire_api). API-key auth is the ready
/// path; per-vendor OAuth is the staged `auth` work (only MCP-server OAuth exists today).
pub(crate) fn cloud_vendors() -> Vec<(&'static str, &'static str, &'static str, WireApi)> {
    vec![
        (
            "openai",
            "OpenAI",
            "https://api.openai.com/v1",
            WireApi::ChatCompletions,
        ),
        (
            "anthropic",
            "Anthropic",
            "https://api.anthropic.com",
            WireApi::Anthropic,
        ),
        (
            "gemini",
            "Gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            WireApi::ChatCompletions,
        ),
        (
            "grok",
            "Grok (xAI)",
            "https://api.x.ai/v1",
            WireApi::ChatCompletions,
        ),
    ]
}

/// A raw-mode, arrow-key SCROLLING single-select picker. Shows a fixed window of `window`
/// rows; the highlighted row is always kept on screen, so a long list (20 Ollama models +
/// 10 on another box) never floods the terminal - the cursor scrolls the window and a
/// "N more ↓" marker shows there's more. Returns the chosen index, or `None` on Esc/q/Ctrl-C.
/// Used by first-run onboarding (pre-TUI); `/model` uses the TUI's own picker in-session.
fn pick_scroll(prompt: &str, rows: &[String], window: usize) -> Option<usize> {
    use crossterm::{
        cursor,
        event::{self, Event, KeyCode, KeyModifiers},
        execute,
        style::{Attribute, Print, SetAttribute},
        terminal::{self, Clear, ClearType},
    };
    if rows.is_empty() {
        return None;
    }
    let win = window.clamp(1, rows.len());
    let region = win + 1; // list rows + one status line
    let mut sel = 0usize;
    let mut off = 0usize; // index of the first visible row

    let mut out = std::io::stdout();
    if terminal::enable_raw_mode().is_err() {
        return None;
    }
    let _ = execute!(out, cursor::Hide);
    let _ = execute!(out, Print(format!("{prompt}\r\n")));
    for _ in 0..region {
        let _ = execute!(out, Print("\r\n")); // reserve the region so MoveUp lands correctly
    }

    let result = loop {
        if sel < off {
            off = sel;
        }
        if sel >= off + win {
            off = sel + 1 - win;
        }
        let _ = execute!(out, cursor::MoveToColumn(0), cursor::MoveUp(region as u16));
        for i in 0..win {
            let _ = execute!(out, Clear(ClearType::CurrentLine));
            let idx = off + i;
            if idx < rows.len() {
                let marker = if idx == sel { "▸ " } else { "  " };
                if idx == sel {
                    let _ = execute!(out, SetAttribute(Attribute::Reverse));
                }
                let _ = execute!(out, Print(format!("{marker}{}", rows[idx])));
                if idx == sel {
                    let _ = execute!(out, SetAttribute(Attribute::Reset));
                }
            }
            let _ = execute!(out, Print("\r\n"));
        }
        let _ = execute!(out, Clear(ClearType::CurrentLine));
        let up = if off > 0 { " · ↑ more" } else { "" };
        let down = if off + win < rows.len() {
            format!(" · {} more ↓", rows.len() - (off + win))
        } else {
            String::new()
        };
        let _ = execute!(
            out,
            Print(format!(
                "  ↑/↓ move · enter select · esc skip{up}{down}\r\n"
            ))
        );
        let _ = std::io::Write::flush(&mut out);

        match event::read() {
            Ok(Event::Key(k)) => match k.code {
                KeyCode::Up => sel = sel.saturating_sub(1),
                KeyCode::Down if sel + 1 < rows.len() => sel += 1,
                KeyCode::Enter => break Some(sel),
                KeyCode::Esc | KeyCode::Char('q') => break None,
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => break None,
                _ => {}
            },
            Ok(_) => {}
            Err(_) => break None,
        }
    };

    let _ = execute!(out, cursor::Show);
    let _ = terminal::disable_raw_mode();
    let _ = execute!(out, Print("\r\n"));
    result
}

/// One selectable onboarding row: either a concrete (endpoint, model) to connect, or one
/// of the tail actions (add custom / add cloud / skip).
enum OnboardPick {
    /// A scanned model on a scanned endpoint - connect straight to it.
    Model {
        label: String,
        base_url: String,
        model: String,
    },
    AddCustom,
    AddCloud,
    Skip,
}

/// First-run onboarding. Scans the famous local servers and lists EVERY model each one
/// serves (an endpoint like Ollama has no "active" model - it hot-loads per request - so we
/// never guess one; the user arrow-picks). Always appends the real mechanisms - add a custom
/// endpoint (any URL) or a cloud vendor - so it never dead-ends, whether 0, 1, or 30 models
/// are found. Registration stays open forever via `/model`; this is just the first touch.
pub async fn onboard() -> anyhow::Result<Option<Config>> {
    if !std::io::stdin().is_terminal() {
        return Ok(None);
    }
    println!("\n  welcome to oxio. let's get you connected.\n");
    print!("scanning local servers (Ollama, LM Studio, llama.cpp, vLLM, Jan, text-gen-webui)… ");
    let _ = std::io::stdout().flush();
    let found = providers::scan_local().await;
    let total: usize = found.iter().map(|e| e.models.len()).sum();
    println!(
        "{}",
        if total == 0 {
            "none found.".to_string()
        } else {
            format!("found {total} model(s):")
        }
    );

    // Flatten every (endpoint, model) into a selectable row, then the tail actions.
    let mut picks: Vec<OnboardPick> = Vec::new();
    let mut rows: Vec<String> = Vec::new();
    for e in &found {
        for m in &e.models {
            rows.push(format!("{:<14} {}", e.label, m));
            picks.push(OnboardPick::Model {
                label: e.label.clone(),
                base_url: e.base_url.clone(),
                model: m.clone(),
            });
        }
    }
    rows.push("+ add a custom endpoint   (any URL - custom port or remote box)".to_string());
    picks.push(OnboardPick::AddCustom);
    rows.push("+ add a cloud provider    (OpenAI / Anthropic / Gemini / Grok)".to_string());
    picks.push(OnboardPick::AddCloud);
    rows.push("skip for now".to_string());
    picks.push(OnboardPick::Skip);

    let chosen = match pick_scroll("connect to:", &rows, 8) {
        Some(i) => &picks[i],
        None => {
            println!(
                "skipped - add anytime with `/model add <name> <url>` or `oxio provider add`."
            );
            return Ok(None);
        }
    };

    match chosen {
        OnboardPick::Skip => {
            println!(
                "skipped - add anytime with `/model add <name> <url>` or `oxio provider add`."
            );
            Ok(None)
        }
        OnboardPick::AddCustom => onboard_custom().await,
        OnboardPick::AddCloud => onboard_cloud(),
        OnboardPick::Model {
            label,
            base_url,
            model,
        } => {
            let name = provider_key(label);
            // Load the existing config and ADD to it - onboarding never wipes prior setup
            // (mcp, agents, profiles). A true first run has no file and this yields a default.
            let mut cfg = config::load().unwrap_or_default();
            cfg.providers.insert(
                name.clone(),
                local_provider(base_url.clone(), Some(model.clone())),
            );
            cfg.defaults.primary = name.clone();
            offer_mcp_setup(&mut cfg);
            config::save(&cfg)?;
            println!("\nusing '{name}' · {model}. saved. entering oxio…\n");
            Ok(Some(cfg))
        }
    }
}

/// After a model is configured in onboarding, offer to add ONE MCP server. Optional -
/// "no" (default) goes straight to the CLI. Adds to `cfg.mcp`; it connects when oxio starts.
/// More servers are added later with `/mcp add`.
fn offer_mcp_setup(cfg: &mut Config) {
    let ans = prompt_line("\n  connect an MCP server now? [y/N]: ");
    if !matches!(ans.trim().to_lowercase().as_str(), "y" | "yes") {
        return;
    }
    let name = prompt_line("  mcp name (e.g. filesystem): ");
    if name.is_empty() {
        println!("  skipped.");
        return;
    }
    let spec = prompt_line("  url (http…/mcp) OR command (e.g. npx -y @modelcontextprotocol/server-filesystem /path): ");
    if spec.is_empty() {
        println!("  skipped.");
        return;
    }
    let entry = if spec.starts_with("http://") || spec.starts_with("https://") {
        config::McpServerCfg {
            url: Some(spec),
            ..Default::default()
        }
    } else {
        let mut toks = spec.split_whitespace();
        let command = toks.next().unwrap_or_default().to_string();
        let args: Vec<String> = toks.map(|s| s.to_string()).collect();
        config::McpServerCfg {
            command: Some(command),
            args,
            ..Default::default()
        }
    };
    cfg.mcp.insert(name.clone(), entry);
    println!("  added mcp '{name}' - it connects when oxio starts. Add more later with /mcp add.");
}

/// Onboarding branch: add a user-defined endpoint (any OpenAI-compatible URL). The
/// code-free mechanism for custom-port local servers and remote boxes. After the URL, it
/// PROBES the endpoint's `/v1/models` and lets the user arrow-pick a model - no typing a
/// model id from memory. An unreachable endpoint is NOT a failure (the "set up the CLI now,
/// the server later" case): the endpoint is saved with no model pinned, to be chosen via
/// `/model` once the server is up.
async fn onboard_custom() -> anyhow::Result<Option<Config>> {
    let name = prompt_line("  name (e.g. my-local-llm): ");
    if name.is_empty() {
        println!("no name - skipped.");
        return Ok(None);
    }
    let url = prompt_line("  base URL (e.g. http://localhost:9000/v1): ");
    if url.is_empty() {
        println!("no URL - skipped.");
        return Ok(None);
    }
    print!("  probing {url} … ");
    let _ = std::io::stdout().flush();
    let models = providers::list_models(&url).await;
    let model = if models.is_empty() {
        println!("unreachable (or serves no models).");
        println!("  endpoint saved with no model - pick one with /model once the server is up.");
        None
    } else {
        println!("found {} model(s).", models.len());
        match pick_scroll("model:", &models, 8) {
            Some(i) => Some(models[i].clone()),
            None => {
                println!("  no model picked - endpoint saved; pick one later with /model.");
                None
            }
        }
    };
    // Load the existing config and ADD to it - onboarding never wipes prior setup (mcp,
    // agents, profiles). On a true first run there is no file and this yields a default.
    let mut cfg = config::load().unwrap_or_default();
    cfg.providers
        .insert(name.clone(), local_provider(url, model.clone()));
    cfg.defaults.primary = name.clone();
    offer_mcp_setup(&mut cfg);
    config::save(&cfg)?;
    match &model {
        Some(m) => println!("\nadded '{name}' · {m}. saved. entering oxio…\n"),
        None => println!("\nadded '{name}' (no model yet). saved. entering oxio…\n"),
    }
    Ok(Some(cfg))
}

/// Onboarding branch: add a cloud vendor. API-key auth (the ready path); the key is
/// written to `<config>/.env` (0600) as `OXIO_<NAME>_API_KEY`, never into config.toml
/// (secrets stay out of the config file). Per-vendor OAuth lands later per the auth plan.
fn onboard_cloud() -> anyhow::Result<Option<Config>> {
    let vendors = cloud_vendors();
    for (i, (_, label, url, _)) in vendors.iter().enumerate() {
        println!("  [{}] {label}  ({url})", i + 1);
    }
    let pick = prompt_line(&format!("  vendor [1-{}]: ", vendors.len()));
    let Some(idx) = pick
        .parse::<usize>()
        .ok()
        .filter(|n| *n >= 1 && *n <= vendors.len())
    else {
        println!("no vendor - skipped.");
        return Ok(None);
    };
    let (key_name, label, base_url, wire) = vendors[idx - 1];
    let model = prompt_line("  model (e.g. gpt-4.1 / claude-sonnet-4 / blank to set later): ");
    let api_key = prompt_line(&format!(
        "  {label} API key (blank = set env OXIO_{}_API_KEY later): ",
        key_name.to_uppercase()
    ));

    // Load the existing config and ADD to it - onboarding never wipes prior setup (mcp,
    // agents, profiles). On a true first run there is no file and this yields a default.
    let mut cfg = config::load().unwrap_or_default();
    cfg.providers.insert(
        key_name.to_string(),
        ProviderCfg {
            wire_api: wire,
            base_url: Some(base_url.to_string()),
            model: if model.is_empty() { None } else { Some(model) },
            auth: AuthKind::ApiKey,
            temperature: None,
            top_p: None,
            max_tokens: None,
            reasoning_effort: None,
            context_window: None,
            vision: None,
        },
    );
    cfg.defaults.primary = key_name.to_string();
    offer_mcp_setup(&mut cfg);
    config::save(&cfg)?;

    if !api_key.is_empty() {
        // Persist the key to <config>/.env (0600) - loaded at startup, kept out of config.toml.
        let var = format!("OXIO_{}_API_KEY", key_name.to_uppercase());
        // Apply the key to THIS process immediately - startup already ran the .env load,
        // so without this the freshly-onboarded provider would error until a restart.
        std::env::set_var(&var, &api_key);
        if let Some(dir) = config::config_path().parent() {
            let env_path = dir.join(".env");
            let mut body = std::fs::read_to_string(&env_path).unwrap_or_default();
            if !body.ends_with('\n') && !body.is_empty() {
                body.push('\n');
            }
            body.push_str(&format!("{var}={api_key}\n"));
            if std::fs::write(&env_path, body).is_ok() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ =
                        std::fs::set_permissions(&env_path, std::fs::Permissions::from_mode(0o600));
                }
                println!("  key saved to {} (0600).", env_path.display());
            }
        }
    } else {
        println!(
            "  set the key before chatting: export OXIO_{}_API_KEY=…",
            key_name.to_uppercase()
        );
    }
    println!("\nadded '{key_name}'. saved. entering oxio…\n");
    Ok(Some(cfg))
}

/// `--resume`: list past sessions (newest first) and let the user pick one to
/// reload. Returns the chosen transcript path, or `None` to start fresh.
fn pick_session() -> Option<PathBuf> {
    let sessions = session::list_sessions();
    if sessions.is_empty() {
        println!(
            "no past sessions found in {}",
            session::sessions_dir().display()
        );
        return None;
    }
    if !std::io::stdin().is_terminal() {
        return None;
    }
    println!("past sessions:");
    for (i, s) in sessions.iter().enumerate().take(20) {
        let model = s.meta.as_ref().map(|m| m.model.as_str()).unwrap_or("?");
        let preview = if s.preview.is_empty() {
            "(empty)"
        } else {
            &s.preview
        };
        println!(
            "  [{}] {} msgs · {} · {}",
            i + 1,
            s.message_count,
            model,
            preview
        );
    }
    print!("resume which? [1-{}] / [q]uit: ", sessions.len().min(20));
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return None;
    }
    let choice = line.trim();
    if choice.eq_ignore_ascii_case("q") {
        return None;
    }
    choice
        .parse::<usize>()
        .ok()
        .filter(|n| *n >= 1 && *n <= sessions.len())
        .map(|n| sessions[n - 1].path.clone())
}

/// Load a user-defined slash command: `.oxio/commands/<name>.md` (project) or
/// `<config-dir>/commands/<name>.md` (global). `$ARGUMENTS` in the file is replaced
/// with the text after the command name; the result becomes the turn's prompt.
fn load_custom_command(name: &str, full: &str) -> Option<String> {
    if name.is_empty() {
        return None;
    }
    let args = full.strip_prefix(name).map(str::trim).unwrap_or("");
    let mut candidates = vec![PathBuf::from(format!(".oxio/commands/{name}.md"))];
    if let Some(dir) = config::config_path().parent() {
        candidates.push(dir.join("commands").join(format!("{name}.md")));
    }
    for p in candidates {
        if let Ok(text) = std::fs::read_to_string(&p) {
            return Some(text.replace("$ARGUMENTS", args));
        }
    }
    None
}

/// Interactive REPL: reedline line editing + history, streaming answers, and
/// Ctrl-C cancels the in-flight turn (Ctrl-C/Ctrl-D at the prompt exits).
pub async fn repl(cfg: &Config, continue_session: bool, resume_pick: bool) -> anyhow::Result<()> {
    use reedline::{DefaultPrompt, DefaultPromptSegment, Reedline, Signal};

    let (kernel, session, compactor, snapshots, _ctl) =
        build_kernel(cfg, Arc::new(StdinApprover), None, None).await?;
    // `--continue` reloads the latest session; `--resume` shows a picker. Both
    // restore the chosen transcript in full (lossless) and keep appending to it.
    let mut resumed = 0usize;
    let resume_path = if resume_pick {
        pick_session()
    } else if continue_session {
        session::latest_session()
    } else {
        None
    };
    if let Some(path) = resume_path {
        session.continue_from(&path);
        resumed = session.len();
    }
    let (pname, pc) = cfg.primary().ok_or_else(|| {
        anyhow::anyhow!("no primary provider configured - see `oxio config show`")
    })?;
    let active_profile = if cfg.defaults.profile.is_empty() {
        "default"
    } else {
        &cfg.defaults.profile
    };
    print_banner(
        pname,
        &pc.model.clone().unwrap_or_default(),
        active_profile,
        resumed,
    );

    let mut editor = Reedline::create();
    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("oxio".to_string()),
        DefaultPromptSegment::Empty,
    );

    // Running token totals → cost. A provider with no API key is local, so its
    // dollar figure is what running locally SAVED vs a comparable hosted rate.
    let cost_model = pc.model.clone().unwrap_or_default();
    let cost_label = if pc.auth == AuthKind::None {
        "saved"
    } else {
        "cost"
    };
    let (mut total_in, mut total_out) = (0u64, 0u64);

    loop {
        match editor.read_line(&prompt) {
            Ok(Signal::Success(buffer)) => {
                let input = buffer.trim();
                if input.is_empty() {
                    continue;
                }
                // Resolve this line into the turn's prompt. Builtin slash commands
                // handle themselves (continue/break); a custom command file expands
                // into a prompt; a plain line is the prompt as typed.
                let turn_prompt: String = if let Some(cmd) = input.strip_prefix('/') {
                    match cmd.split_whitespace().next().unwrap_or("") {
                        // Manual compaction of the session history (needs async + the handles).
                        "compact" => {
                            if session.is_empty() {
                                println!("(nothing to compact)");
                            } else {
                                let n = session.len();
                                match compactor.compact_history(session.snapshot()).await {
                                    Ok(c) => {
                                        session.replace(c);
                                        println!("compacted session ({n} messages summarized)");
                                    }
                                    Err(e) => println!("[compact failed: {e}]"),
                                }
                            }
                            continue;
                        }
                        // Undo the most recent file-changing tool call.
                        "undo" => {
                            match snapshots.undo() {
                                Some(msg) => println!("{msg}"),
                                None => println!("(nothing to undo)"),
                            }
                            continue;
                        }
                        // Clear the screen AND the conversation history.
                        "clear" => {
                            session.clear();
                            print!("\x1b[2J\x1b[H");
                            println!("(session cleared)");
                            continue;
                        }
                        // Extract durable memories from this session now (tier 2).
                        "remember" => {
                            match provider_from(pname, pc).await {
                                Ok(prov) => {
                                    let model = pc.model.clone().unwrap_or_default();
                                    let n = extract_memories(
                                        prov.clone(),
                                        model.clone(),
                                        session.snapshot(),
                                    )
                                    .await;
                                    let mut msg = format!("remembered {n} durable fact(s)");
                                    if memory::load_memories().len() >= CONSOLIDATE_THRESHOLD {
                                        let (b, a) = consolidate_memories(prov, model).await;
                                        if a < b {
                                            msg.push_str(&format!("; consolidated {b}→{a}"));
                                        }
                                    }
                                    println!("{msg}");
                                }
                                Err(e) => println!("[remember failed: {e}]"),
                            }
                            continue;
                        }
                        name => {
                            // A user-defined command file expands into a prompt and
                            // runs as a turn; otherwise fall back to builtin slashes.
                            match load_custom_command(name, cmd) {
                                Some(expanded) => expanded,
                                None => {
                                    if handle_slash(cmd, cfg) {
                                        break;
                                    }
                                    continue;
                                }
                            }
                        }
                    }
                } else {
                    input.to_string()
                };
                // Ctrl-C during a turn cancels it (via the frozen core's Ctx).
                let ctx = Ctx::default();
                let token = ctx.cancel.clone();
                let canceller = tokio::spawn(async move {
                    if tokio::signal::ctrl_c().await.is_ok() {
                        token.cancel();
                    }
                });
                match kernel.run_turn(&turn_prompt, &ctx, &StdoutSink).await {
                    Ok(outcome) => {
                        total_in += outcome.usage.input_tokens;
                        total_out += outcome.usage.output_tokens;
                        println!();
                    }
                    Err(e) => println!("\n[{e}]"),
                }
                canceller.abort();
                // Live status: window fill, plus running cost when the model is priced
                // in config (unpriced → tokens only, never a fabricated dollar figure).
                let used = context::estimate_tokens(&session.snapshot());
                let ctx_line = ctx_status(used, compactor.context_window());
                let full = match cost_usd(total_in, total_out, &cfg.pricing, &cost_model) {
                    Some(usd) => format!("ctx {ctx_line} · {cost_label} ${usd:.4}"),
                    None => format!("ctx {ctx_line}"),
                };
                if color_enabled() {
                    println!("\x1b[2m{full}\x1b[0m");
                } else {
                    println!("{full}");
                }
            }
            Ok(Signal::CtrlC) | Ok(Signal::CtrlD) => {
                println!("bye");
                break;
            }
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        }
    }
    // Auto-memory (off by default): extract durable facts from the session on exit.
    if cfg.defaults.auto_memory && !session.is_empty() {
        if let Ok(prov) = provider_from(pname, pc).await {
            let model = pc.model.clone().unwrap_or_default();
            let n = extract_memories(prov.clone(), model.clone(), session.snapshot()).await;
            if n > 0 {
                println!("[auto-memory: remembered {n} fact(s)]");
            }
            if memory::load_memories().len() >= CONSOLIDATE_THRESHOLD {
                let (b, a) = consolidate_memories(prov, model).await;
                if a < b {
                    println!("[auto-memory: consolidated {b}→{a}]");
                }
            }
        }
    }
    Ok(())
}

/// Handle a `/command`. Returns true to quit. Commands whose backing module is
/// not built yet say so plainly rather than pretend.
/// Names of user-defined slash commands discovered in `.oxio/commands/` and the
/// global commands dir, for `/help` discoverability.
fn list_custom_commands() -> Vec<String> {
    let mut dirs = vec![PathBuf::from(".oxio/commands")];
    if let Some(d) = config::config_path().parent() {
        dirs.push(d.join("commands"));
    }
    let mut names = Vec::new();
    for dir in dirs {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "md") {
                    if let Some(n) = p.file_stem().and_then(|s| s.to_str()).map(String::from) {
                        if !names.contains(&n) {
                            names.push(n);
                        }
                    }
                }
            }
        }
    }
    names.sort();
    names
}

fn handle_slash(cmd: &str, cfg: &Config) -> bool {
    let mut parts = cmd.split_whitespace();
    match parts.next().unwrap_or("") {
        "quit" | "q" | "exit" => return true,
        "help" | "h" => {
            println!(
                "commands: /help  /quit  /clear  /compact  /remember  /undo  /model  /provider"
            );
            println!("  /compact summarize session · /remember extract memory · /undo revert last file edit · /clear reset");
            let custom = list_custom_commands();
            if !custom.is_empty() {
                let list: Vec<String> = custom.iter().map(|n| format!("/{n}")).collect();
                println!("custom (from .oxio/commands): {}", list.join("  "));
            }
        }
        "clear" => print!("\x1b[2J\x1b[H"),
        "model" => {
            if let Some((n, p)) = cfg.primary() {
                println!(
                    "model: {} (provider {n})",
                    p.model.clone().unwrap_or_default()
                );
            }
        }
        "provider" => {
            println!("primary: {}", cfg.defaults.primary);
            for name in cfg.providers.keys() {
                println!("  - {name}");
            }
        }
        other @ ("login" | "logout" | "compact" | "tools" | "profile") => {
            println!("/{other}: lands with a later step");
        }
        other => println!("unknown command: /{other} (try /help)"),
    }
    false
}

#[cfg(test)]
mod skill_tests {
    use super::{discover_skills_in, parse_frontmatter};

    #[test]
    fn agent_skill_binding_appends_known_skips_unknown() {
        use super::bind_agent_skills;
        let mut skills = std::collections::BTreeMap::new();
        skills.insert(
            "mobile".to_string(),
            config::SkillCfg {
                description: None,
                instructions: "USE SWIFTUI".to_string(),
            },
        );
        let out = bind_agent_skills(
            "BASE PROMPT".to_string(),
            &["mobile".to_string(), "ghost".to_string()],
            &skills,
            "builder",
        );
        assert!(out.starts_with("BASE PROMPT"), "base prompt preserved");
        assert!(
            out.contains("# Skill: mobile\nUSE SWIFTUI"),
            "known skill appended"
        );
        assert!(
            !out.contains("ghost"),
            "unknown skill skipped, not injected"
        );
    }

    #[test]
    fn frontmatter_splits_meta_and_body() {
        let (fm, body) = parse_frontmatter("---\nname: x\ndescription: does y\n---\nBODY HERE");
        assert_eq!(fm.get("name").unwrap(), "x");
        assert_eq!(fm.get("description").unwrap(), "does y");
        assert_eq!(body, "BODY HERE");
    }

    #[test]
    fn no_frontmatter_is_all_body() {
        let (fm, body) = parse_frontmatter("just instructions");
        assert!(fm.is_empty());
        assert_eq!(body, "just instructions");
    }

    #[test]
    fn discovers_skill_md_with_bundled_file_note() {
        let tmp = std::env::temp_dir().join(format!("oxio-skilltest-{}", std::process::id()));
        let sdir = tmp.join("greet");
        std::fs::create_dir_all(&sdir).unwrap();
        std::fs::write(
            sdir.join("SKILL.md"),
            "---\nname: greet\ndescription: say hi\n---\nSay hello.",
        )
        .unwrap();
        std::fs::write(sdir.join("template.txt"), "hi").unwrap();
        let found = discover_skills_in(std::slice::from_ref(&tmp));
        let s = found.get("greet").expect("greet skill discovered");
        assert_eq!(s.description.as_deref(), Some("say hi"));
        assert!(s.instructions.contains("Say hello."));
        assert!(
            s.instructions.contains("template.txt"),
            "bundled file noted"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }
}
