//! oxio configuration: the full TOML schema (declared complete up front),
//! plus load/save and XDG path resolution. Configuration is data, never code -
//! editable by hand or by `oxio config`/`provider` commands, always in sync.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Current config schema version. Bump ONLY for a breaking change (a renamed/removed
/// field, or one whose meaning changed) and add a matching step in [`migrate`]. Purely
/// additive fields need no bump - serde defaults already read older files. A config
/// written before versioning existed has no `version` key and parses as 0, triggering
/// migration on first load by a newer binary.
pub const SCHEMA_VERSION: u32 = 1;

/// The whole config. Every field is declared now; unfilled ones are simply
/// unused until their vertical lands.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// Schema version this config was written with. Absent (old files) → 0 → migrated
    /// on load. Every `save` stamps [`SCHEMA_VERSION`], so a persisted config is current.
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub defaults: Defaults,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderCfg>,
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfileCfg>,
    /// Per-MODEL attributes, keyed by model id. `[models.<id>]`. Context length and
    /// capabilities (vision, thinking) belong to the model, not the machine that
    /// serves it - a machine can serve many models with different windows. Merged in
    /// by model id, so the same id means the same capabilities on any endpoint.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelCfg>,
    /// External MCP servers to connect to (stdio). Each becomes a set of gated
    /// tools. `[mcp.<name>]` with `command` + `args`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, McpServerCfg>,
    /// Named, individually-configured sub-agents. `[agents.<name>]`; each becomes
    /// a delegating tool with its own model / tools / permission / prompt.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub agents: BTreeMap<String, AgentCfg>,
    /// Language servers per file extension. `[lsp.<ext>]` with `command` + `args`;
    /// powers the `lsp` diagnostics tool.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub lsp: BTreeMap<String, LspServerCfg>,
    /// Reusable instruction packs loaded on demand by the `skill` tool.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub skills: BTreeMap<String, SkillCfg>,
    /// User shell hooks fired on lifecycle events. `[[hooks]]` with `on` + `run`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hooks: Vec<HookCfg>,
    /// Token pricing, user-maintained (NOT hardcoded - models and prices churn).
    /// `[pricing.<key>]` with `input` / `output` in USD per 1M tokens; `<key>` is a
    /// substring matched against the model name. No entry → no dollar figure shown.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub pricing: BTreeMap<String, RateCfg>,
    /// User-configurable status line template. A
    /// format string over live telemetry placeholders: `{time} {model} {provider}
    /// {profile} {cwd} {ctx_pct} {ctx_bar} {in} {out} {cost} {cost_label}`. Unset →
    /// a sensible default. NOT hardcoded in the renderer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub statusline: Option<String>,
    /// Corrective-hint layer: per-tool text appended to a tool's ERROR output to steer
    /// the model's retry back to correct usage. `[hints.<tool>]` overrides the built-in
    /// default for that tool; add an entry to hint any tool. Toggle via
    /// `[defaults] tool_hints`. Only fires on error, never on success.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub hints: BTreeMap<String, String>,
}

/// USD per 1M tokens for a model (a `[pricing.<key>]` entry).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RateCfg {
    pub input: f64,
    pub output: f64,
}

/// A user shell hook: run `run` (via `sh -c`) when lifecycle event `on` fires.
/// Events: `pre_turn` / `post_turn` / `pre_model` / `post_model` / `pre_tool` /
/// `post_tool` / `on_error`. Context is passed via env (`OXIO_HOOK_EVENT`,
/// `OXIO_MODEL`). Observational (non-blocking); the blocking/deny variant is a later refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookCfg {
    pub on: String,
    pub run: String,
}

/// A configured MCP server, set under `[mcp.<your-name>]` - the name is yours to
/// choose and is all the UI ever shows. Two transports (set exactly one):
/// `url` = REMOTE (a streamable-HTTP endpoint of a server on the LAN/cloud, the
/// primary MCP model); or `command`+`args`+`env` = STDIO (a local subprocess oxio
/// spawns). `env` matches the vendor `.mcp.json` shape for server-side config. Kept
/// as plain config data here; the `mcp` crate consumes it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerCfg {
    /// STDIO transport: the command to spawn. Omit for a remote (`url`) server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables set on the spawned server process. `[mcp.<name>.env]`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// REMOTE transport: streamable-HTTP endpoint (e.g. `http://host.local:8000/mcp`).
    /// Set this instead of `command` for a server that lives on the network.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// REMOTE auth - static HTTP headers sent on every request (`[mcp.<name>.headers]`,
    /// e.g. a non-secret `X-Client = "oxio"`). For SECRETS use `env_headers` so the
    /// value never lands in this file.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// REMOTE auth - secret-safe headers: maps a header NAME to an ENVIRONMENT VARIABLE
    /// whose value is read at connect time (`[mcp.<name>.env_headers]`, e.g.
    /// `Authorization = "GITHUB_MCP_TOKEN"` where that env var holds `Bearer ghp_…`).
    /// The token stays in the environment/keychain, never in the config. Empty/missing
    /// env vars are skipped. This is how you connect to a vendor MCP that needs a key.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env_headers: BTreeMap<String, String>,
    /// REMOTE auth mode - the shape of authentication the server requires. The MCP
    /// client dispatches on this so it can face any commercial server:
    /// `"none"` (default; open, or auth carried entirely by `headers`/`env_headers`),
    /// or `"oauth"` (interactive OAuth 2.0 sign-in - browser/device flow with dynamic
    /// client registration and token refresh, per the MCP auth spec). Header-token
    /// auth needs no mode; it is just `env_headers`. Kept a string (not a closed enum)
    /// so new shapes are additive and never a breaking config change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<String>,
    /// OAuth tuning (only read when `auth = "oauth"`): `[mcp.<name>.oauth]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth: Option<OAuthCfg>,
}

/// OAuth 2.0 settings for a remote MCP server (`[mcp.<name>.oauth]`). All optional:
/// with none set, the client uses dynamic client registration + the server's advertised
/// metadata (the common MCP case). Tokens are stored/refreshed by the client, never here.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OAuthCfg {
    /// Requested scopes. Empty = the server's default scopes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
    /// Pre-registered client id (omit to use dynamic client registration).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// Loopback redirect port for the authorization callback (omit = ephemeral).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callback_port: Option<u16>,
}

/// A reusable instruction pack loaded on demand by the `skill` tool
/// (`[skills.<name>]`). Keeps capability-specific guidance out of the always-on
/// system prompt until the model asks for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCfg {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub instructions: String,
}

/// A configured language server (stdio) for a file extension (`[lsp.<ext>]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerCfg {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

/// A named, individually-configured sub-agent (`[agents.<name>]`). Each becomes a
/// delegating tool with its own model, tools, permission, and prompt.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCfg {
    /// What the sub-agent is for (shown to the model in the tool description).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Provider name for this agent's model; defaults to primary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Tool allowlist: exact names, or `["all"]` / `["read"]` shorthands.
    /// Omitted = all tools (subject to `permission`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// Permission mode for this agent's writes: prompt | accept | deny | read_only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission: Option<String>,
    /// System-prompt override; defaults to the baseline coding-agent prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Skills bound to this agent, by name. Their instructions are appended to this
    /// agent's system prompt at startup - the oxio equivalent of a vendor sub-agent
    /// declaring `skills: [...]` in frontmatter, so a role (e.g. `mobile`) always loads
    /// its domain skills. Names resolve against the session's skills (on-disk SKILL.md
    /// + `[skills.<name>]`); an unknown name is skipped with a warning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Defaults {
    /// Primary provider name (must key into `providers`).
    #[serde(default)]
    pub primary: String,
    /// Fallback chain, in order, tried when the primary fails.
    #[serde(default)]
    pub fallback: Vec<String>,
    /// Active profile name.
    #[serde(default)]
    pub profile: String,
    /// Provider name for sub-agents (`agent` tool); keys into `providers`.
    /// Defaults to `primary` when unset. Lets a smaller/faster model do delegated
    /// work without occupying the main model or its context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_provider: Option<String>,
    /// Which tools a sub-agent gets: `"all"` (default - read+write+exec, so it can
    /// code / fix bugs; writes still permission-gated) or `"read"` (read-only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_tools: Option<String>,
    /// Auto-extract durable memory from a session on exit (off by default, like
    /// Gemini). Manual extraction is always available via `/remember`.
    #[serde(default)]
    pub auto_memory: bool,
    /// Provider for the `view_image` tool (a vision model); keys into `providers`.
    /// Defaults to primary, so a multimodal local model is simply reused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_provider: Option<String>,
    /// Write-tool approval mode: `prompt` (ask each time), `accept` (auto-allow), or
    /// `deny`. Drives which approver the interactive session uses; shown live in the
    /// status line. Empty → prompt.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub approval: String,
    /// Corrective-hint layer toggle. Default on (`None` = on): a tool's error output is
    /// augmented with a per-tool usage hint to guide the model's retry. Set
    /// `tool_hints = false` for a strong model that needs no coaching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_hints: Option<bool>,
    /// Ghost next-prompt suggestion (Tab to accept), generated by the primary model after
    /// each turn. Default OFF (`None` = off): it costs an extra model call per turn, so it
    /// is opt-in. `suggestions = true` (or `/suggest` in-session) turns it on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggestions: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCfg {
    pub wire_api: WireApi,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub auth: AuthKind,
    // Sampling.
    // Standard models: temperature/top_p/max_tokens. Reasoning models: reasoning_effort
    // (temperature/top_p then omitted). All optional - omitted knobs use model defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Model context window in tokens; drives compaction's 90% trigger. Defaults
    /// to a safe value when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    /// Whether this model can see images. `Some(true)` → attach images inline;
    /// `Some(false)` → route images straight to `defaults.vision_provider` (no wasted
    /// blind call); `None` (unknown) → try inline, fall back on error. Local coder
    /// models are typically `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
}

/// Per-model attributes (`[models.<id>]`), merged in by the active model id. These
/// belong to the MODEL, not the machine: one endpoint can serve models with
/// different context windows and capabilities. Any field unset falls back to the
/// provider-level value, then a safe default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCfg {
    /// Context window in tokens; drives compaction's 90% trigger. Overrides the
    /// provider-level `context_window` for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    /// Whether this model can see images (overrides the provider-level `vision`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
    /// Thinking/reasoning default for this model: `auto` | `on` | `off`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireApi {
    ChatCompletions,
    Responses,
    Anthropic,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthKind {
    #[default]
    None,
    ApiKey,
    Oauth,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileCfg {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_ceiling: Option<u64>,
}

/// `~/.config/oxio/config.toml` (XDG), or `./oxio.toml` as a last resort.
pub fn config_path() -> PathBuf {
    match dirs::config_dir() {
        Some(dir) => dir.join("oxio").join("config.toml"),
        None => PathBuf::from("oxio.toml"),
    }
}

/// Load the config from the default path (empty config if the file is absent),
/// migrating it in place if it was written by an older schema. This is the caring
/// upgrade path: a returning user's config is read, brought current if needed, and
/// the original is BACKED UP before any rewrite - never silently clobbered.
pub fn load() -> anyhow::Result<Config> {
    let path = config_path();
    if !path.exists() {
        return Ok(Config::default());
    }
    let mut cfg: Config = toml::from_str(&std::fs::read_to_string(&path)?)?;
    if cfg.version < SCHEMA_VERSION {
        let from = cfg.version;
        migrate(&mut cfg);
        // Back the original up BEFORE rewriting so an upgrade can never lose an older
        // install's config. A persist failure (e.g. read-only dir) is non-fatal - the
        // migrated config is still used for this run.
        let backup = path.with_extension(format!("toml.v{from}.bak"));
        let _ = std::fs::copy(&path, &backup);
        match save_to(&path, &cfg) {
            Ok(()) => eprintln!("config: migrated schema v{from} → v{SCHEMA_VERSION} (backup: {})", backup.display()),
            Err(e) => eprintln!("config: migrated schema v{from} → v{SCHEMA_VERSION} in memory; could not persist ({e})"),
        }
    }
    Ok(cfg)
}

/// Parse a config from an explicit path WITHOUT migration side effects (empty config
/// if absent). Used by tooling/tests that want the file as-written.
pub fn load_from(path: &Path) -> anyhow::Result<Config> {
    if !path.exists() {
        return Ok(Config::default());
    }
    let text = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&text)?)
}

/// Bring a parsed config up to [`SCHEMA_VERSION`], stepwise. Each step transforms in
/// place and bumps `version`. Additive fields need NO step (serde defaults read older
/// files); add a step here ONLY for a breaking change (rename/remove/semantic change).
fn migrate(cfg: &mut Config) {
    // v0 → v1: versioning introduced. v0 files are additive-compatible with v1, so
    // this only stamps the version - no field moves.
    if cfg.version == 0 {
        cfg.version = 1;
    }
    // Future breaking change, e.g.:
    // if cfg.version == 1 { /* move/rename fields */ cfg.version = 2; }
    cfg.version = SCHEMA_VERSION; // guarantee current even if a step were missed
}

/// Persist to the default path, creating parent dirs.
pub fn save(cfg: &Config) -> anyhow::Result<()> {
    save_to(&config_path(), cfg)
}

pub fn save_to(path: &Path, cfg: &Config) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Every persisted config is stamped current, so a file on disk is never left at an
    // older/absent version once written by this binary.
    let mut cfg = cfg.clone();
    cfg.version = SCHEMA_VERSION;
    std::fs::write(path, toml::to_string_pretty(&cfg)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
impl Config {
    /// DEV-ONLY: a ready-to-run default pointing at a GENERIC local endpoint
    /// (`localhost:11434`, the common local-LLM port) so developers iterate fast.
    /// A maintainer's personal rig stays out of source - override at runtime with
    /// `OXIO_DEV_SEED_URL` / `OXIO_DEV_SEED_MODEL`. MUST NOT be used on the
    /// production first-run path (prod uses onboarding, never a hardcoded endpoint);
    /// gated in the binary behind debug builds / `OXIO_DEV`.
    pub fn seed_local() -> Self {
        let base_url = std::env::var("OXIO_DEV_SEED_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
        let model = std::env::var("OXIO_DEV_SEED_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
        let mut providers = BTreeMap::new();
        providers.insert(
            "local".to_string(),
            ProviderCfg {
                wire_api: WireApi::ChatCompletions,
                base_url: Some(base_url),
                model: Some(model),
                auth: AuthKind::None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                reasoning_effort: None,
                context_window: None,
                vision: None,
            },
        );
        Config {
            version: SCHEMA_VERSION,
            defaults: Defaults {
                primary: "local".into(),
                fallback: vec![],
                profile: "full".into(),
                agent_provider: None,
                agent_tools: None,
                vision_provider: None,
                auto_memory: false,
                approval: String::new(),
                tool_hints: None,
                suggestions: None,
            },
            providers,
            profiles: BTreeMap::new(),
            models: BTreeMap::new(),
            mcp: BTreeMap::new(),
            agents: BTreeMap::new(),
            lsp: BTreeMap::new(),
            skills: BTreeMap::new(),
            hooks: Vec::new(),
            pricing: BTreeMap::new(),
            statusline: None,
            hints: BTreeMap::new(),
        }
    }

    pub fn primary(&self) -> Option<(&str, &ProviderCfg)> {
        let name = self.defaults.primary.as_str();
        if let Some(p) = self.providers.get(name) {
            return Some((name, p));
        }
        // `defaults.primary` is unset or dangling (e.g. the active machine was removed,
        // leaving the pointer stale). Fall back to the first configured provider so the
        // session stays usable - `primary()` returns None ONLY when there are genuinely
        // no providers (a true first run), so onboarding never fires over a real config.
        self.providers.iter().next().map(|(n, p)| (n.as_str(), p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_toml() {
        let cfg = Config::seed_local();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back.defaults.primary, "local");
        let (name, p) = back.primary().unwrap();
        assert_eq!(name, "local");
        assert_eq!(p.wire_api, WireApi::ChatCompletions);
        assert!(p.model.is_some());
    }

    #[test]
    fn empty_when_absent() {
        let cfg = load_from(Path::new("/nonexistent/oxio/x.toml")).unwrap();
        assert!(cfg.providers.is_empty());
        assert!(cfg.primary().is_none());
    }

    #[test]
    fn old_config_without_version_parses_as_zero() {
        // A config written before schema versioning has no `version` key.
        let cfg: Config = toml::from_str("[defaults]\nprimary = \"x\"\n").unwrap();
        assert_eq!(
            cfg.version, 0,
            "absent version parses as 0 → will migrate on load"
        );
    }

    #[test]
    fn migrate_brings_old_config_current() {
        let mut cfg = Config {
            version: 0,
            ..Default::default()
        };
        migrate(&mut cfg);
        assert_eq!(cfg.version, SCHEMA_VERSION);
    }

    #[test]
    fn save_stamps_current_schema_version() {
        // Even saving a version-0 config writes it at the current schema version, so a
        // file on disk is never left stale once this binary has written it.
        let mut path = std::env::temp_dir();
        path.push(format!("oxio-migtest-{}.toml", std::process::id()));
        let old = Config {
            version: 0,
            ..Default::default()
        };
        save_to(&path, &old).unwrap();
        let back = load_from(&path).unwrap();
        assert_eq!(back.version, SCHEMA_VERSION);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_then_load(/* uses a temp path under the OS temp dir */) {
        let mut path = std::env::temp_dir();
        path.push(format!("oxio-cfgtest-{}.toml", std::process::id()));
        let cfg = Config::seed_local();
        save_to(&path, &cfg).unwrap();
        let back = load_from(&path).unwrap();
        assert_eq!(back.defaults.primary, "local");
        let _ = std::fs::remove_file(&path);
    }
}
