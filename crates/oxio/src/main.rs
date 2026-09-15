//! oxio - local-first agentic CLI entry point. Full command surface declared;
//! the local vertical (one-shot, config, doctor) is filled. Commands whose module
//! hasn't landed report cleanly as unavailable - never faked.

use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use config::{AuthKind, Config, ProviderCfg, WireApi};
use oxio_core::{Ctx, NullSink};

#[derive(Parser)]
#[command(
    name = "oxio",
    version,
    about = "Local-first agentic CLI",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    /// One-shot prompt (non-interactive). Omit for the interactive REPL.
    prompt: Vec<String>,
    /// Resume the most recent session, reloading its full transcript losslessly.
    #[arg(long = "continue")]
    continue_session: bool,
    /// Pick a past session to resume from a list (full lossless restore).
    #[arg(long = "resume")]
    resume: bool,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show or locate configuration
    Config {
        #[command(subcommand)]
        action: ConfigCmd,
    },
    /// Health check: config + provider endpoints
    Doctor,
    /// Interactive REPL
    Repl,
    /// Manage saved sessions: list / rm
    Session {
        #[command(subcommand)]
        action: SessionCmd,
    },
    /// Manage providers: add / list / use / remove
    Provider {
        #[command(subcommand)]
        action: ProviderCmd,
    },
    /// Set the primary provider's model
    Model { model: String },
    /// OAuth sign-in to a configured MCP server (`[mcp.<name>]` with `auth = "oauth"`);
    /// opens the browser, then stores the token for future connects.
    Login {
        /// name of the MCP server (the `[mcp.<name>]` key)
        vendor: String,
    },
    /// Manage model profiles: list / use
    Profile {
        #[command(subcommand)]
        action: ProfileCmd,
    },
}

#[derive(Subcommand)]
enum ProfileCmd {
    /// List configured profiles (* = active)
    List,
    /// Set the active profile
    Use { name: String },
}

#[derive(Subcommand)]
enum SessionCmd {
    /// List saved sessions (newest first)
    List,
    /// Delete a saved session by its list number
    Rm { index: usize },
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Print the config file path
    Path,
    /// Print the effective config
    Show,
}

#[derive(Subcommand)]
enum ProviderCmd {
    /// Add or update a provider
    Add {
        name: String,
        /// wire_api: chat_completions | responses | anthropic
        #[arg(long, default_value = "chat_completions")]
        wire: String,
        #[arg(long)]
        base_url: Option<String>,
        #[arg(long)]
        model: Option<String>,
        /// auth: none | api_key | oauth
        #[arg(long, default_value = "none")]
        auth: String,
        /// sampling: temperature (standard models)
        #[arg(long)]
        temperature: Option<f32>,
        /// sampling: top_p / nucleus (standard models)
        #[arg(long)]
        top_p: Option<f32>,
        /// sampling: max output tokens
        #[arg(long)]
        max_tokens: Option<u32>,
        /// reasoning models: effort (minimal|low|medium|high) - omits temperature/top_p
        #[arg(long)]
        reasoning_effort: Option<String>,
        /// model context window in tokens (drives compaction's 90% trigger)
        #[arg(long)]
        context_window: Option<usize>,
        /// whether this model can see images (true → inline; false → route images to vision_provider)
        #[arg(long)]
        vision: Option<bool>,
    },
    /// List configured providers (* = primary)
    List,
    /// Set the primary provider
    Use { name: String },
    /// Remove a provider
    Remove { name: String },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    load_dotenv();
    let mut cfg = ensure_config()?;

    match cli.cmd {
        Some(Cmd::Config { action }) => match action {
            ConfigCmd::Path => println!("{}", config::config_path().display()),
            ConfigCmd::Show => print!("{}", toml::to_string_pretty(&cfg)?),
        },
        Some(Cmd::Doctor) => doctor(&cfg).await?,
        Some(Cmd::Repl) => ui::run_interactive(&cfg, cli.continue_session, cli.resume).await?,
        Some(Cmd::Session { action }) => {
            let sessions = session::list_sessions();
            match action {
                SessionCmd::List => {
                    if sessions.is_empty() {
                        println!("no saved sessions in {}", session::sessions_dir().display());
                    }
                    for (i, s) in sessions.iter().enumerate() {
                        let model = s.meta.as_ref().map(|m| m.model.as_str()).unwrap_or("?");
                        let preview = if s.preview.is_empty() {
                            "(empty)"
                        } else {
                            &s.preview
                        };
                        println!(
                            "[{}] {} msgs · {} · {}",
                            i + 1,
                            s.message_count,
                            model,
                            preview
                        );
                    }
                }
                SessionCmd::Rm { index } => match sessions.get(index.wrapping_sub(1)) {
                    Some(s) => {
                        std::fs::remove_file(&s.path)?;
                        println!("removed session [{index}] ({})", s.path.display());
                    }
                    None => anyhow::bail!("no session [{index}] - see `oxio session list`"),
                },
            }
        }
        Some(Cmd::Provider { action }) => provider_cmd(&mut cfg, action)?,
        Some(Cmd::Model { model }) => {
            let pname = cfg.defaults.primary.clone();
            match cfg.providers.get_mut(&pname) {
                Some(pc) => {
                    pc.model = Some(model.clone());
                    config::save(&cfg)?;
                    println!("{pname} model = {model}");
                }
                None => anyhow::bail!(
                    "no primary provider - add one with `oxio provider add <name> ...`"
                ),
            }
        }
        Some(Cmd::Profile { action }) => match action {
            ProfileCmd::List => {
                if cfg.profiles.is_empty() {
                    println!("(no profiles - add `[profiles.<name>]` in config: prompt / tools / context_ceiling)");
                }
                for name in cfg.profiles.keys() {
                    let star = if *name == cfg.defaults.profile {
                        "*"
                    } else {
                        " "
                    };
                    println!("{star} {name}");
                }
            }
            ProfileCmd::Use { name } => {
                if !cfg.profiles.contains_key(&name) {
                    anyhow::bail!("no such profile '{name}' - see `oxio profile list`");
                }
                cfg.defaults.profile = name.clone();
                config::save(&cfg)?;
                println!("active profile = {name}");
            }
        },
        Some(Cmd::Login { vendor }) => ui::mcp_login(&cfg, &vendor).await?,
        None => {
            if cli.prompt.is_empty() {
                // Onboarding fires ONLY on a genuinely empty config (no providers at all) -
                // a true first run. Providers present but no valid primary is NOT a first
                // run (e.g. the active machine was removed, dangling `defaults.primary`);
                // repair it by adopting the first provider, loudly, rather than re-onboarding.
                if cfg.providers.is_empty() {
                    if let Some(c) = ui::onboard().await? {
                        cfg = c;
                    }
                } else if cfg.primary().is_none() {
                    // unreachable given primary()'s fallback, but keep the repair explicit:
                    if let Some(name) = cfg.providers.keys().next().cloned() {
                        eprintln!("primary was unset - using '{name}'");
                        cfg.defaults.primary = name;
                        let _ = config::save(&cfg);
                    }
                }
                // Verify the primary-model pin before launching - bounded, conservative.
                // NEVER auto-rewrite a working pin: a pin absent from /v1/models may be a
                // valid router alias, so this only hints (or fills an unset model).
                if let Some(hint) = ui::reconcile_primary_model(&mut cfg).await {
                    eprintln!("{hint}");
                }
                ui::run_interactive(&cfg, cli.continue_session, cli.resume).await?;
            } else {
                ui::run_once(&cfg, &cli.prompt.join(" ")).await?;
            }
        }
    }
    Ok(())
}

/// Load `.env` into the process environment without overwriting variables already set.
fn load_dotenv() {
    // First hit wins per key, so the config dir (oxio's env home) wins over a cwd .env:
    // config-dir PRIVATE/.env, config-dir .env, cwd PRIVATE/.env, cwd .env.
    let mut files = Vec::new();
    if let Some(dir) = config::config_path().parent() {
        files.push(dir.join("PRIVATE").join(".env"));
        files.push(dir.join(".env"));
    }
    files.push(std::path::PathBuf::from("PRIVATE/.env"));
    files.push(std::path::PathBuf::from(".env"));
    for path in files {
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            // Accept a leading `export ` (common .env style) so it doesn't become part
            // of the variable name and silently fail to load.
            let k = k
                .trim()
                .strip_prefix("export ")
                .map(str::trim)
                .unwrap_or_else(|| k.trim());
            let mut v = v.trim();
            if v.len() >= 2
                && ((v.starts_with('"') && v.ends_with('"'))
                    || (v.starts_with('\'') && v.ends_with('\'')))
            {
                v = &v[1..v.len() - 1];
            }
            if !k.is_empty() && std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
    }
}

fn parse_wire(s: &str) -> anyhow::Result<WireApi> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "chat_completions" | "chat" | "openai" | "oai" => WireApi::ChatCompletions,
        "responses" => WireApi::Responses,
        "anthropic" | "claude" => WireApi::Anthropic,
        other => anyhow::bail!("unknown wire_api '{other}' (chat_completions|responses|anthropic)"),
    })
}

fn parse_auth(s: &str) -> anyhow::Result<AuthKind> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "none" => AuthKind::None,
        "api_key" | "apikey" | "key" => AuthKind::ApiKey,
        "oauth" => AuthKind::Oauth,
        other => anyhow::bail!("unknown auth '{other}' (none|api_key|oauth)"),
    })
}

/// Mutate + persist provider config from the CLI, so the primary provider/model
/// is set by the user, never a hardcode.
fn provider_cmd(cfg: &mut Config, action: ProviderCmd) -> anyhow::Result<()> {
    match action {
        ProviderCmd::Add {
            name,
            wire,
            base_url,
            model,
            auth,
            temperature,
            top_p,
            max_tokens,
            reasoning_effort,
            context_window,
            vision,
        } => {
            let pc = ProviderCfg {
                wire_api: parse_wire(&wire)?,
                base_url,
                model,
                auth: parse_auth(&auth)?,
                temperature,
                top_p,
                max_tokens,
                reasoning_effort,
                context_window,
                vision,
            };
            cfg.providers.insert(name.clone(), pc);
            let now_primary = cfg.defaults.primary.is_empty();
            if now_primary {
                cfg.defaults.primary = name.clone();
            }
            config::save(cfg)?;
            println!(
                "added provider '{name}'{}",
                if now_primary { " (set as primary)" } else { "" }
            );
        }
        ProviderCmd::List => {
            if cfg.providers.is_empty() {
                println!("(no providers - add one with `oxio provider add <name> ...`)");
            }
            for (n, pc) in &cfg.providers {
                let star = if *n == cfg.defaults.primary { "*" } else { " " };
                println!(
                    "{star} {n} [{:?}] {} {}",
                    pc.wire_api,
                    pc.base_url.clone().unwrap_or_default(),
                    pc.model.clone().unwrap_or_default()
                );
            }
        }
        ProviderCmd::Use { name } => {
            if !cfg.providers.contains_key(&name) {
                anyhow::bail!("no such provider '{name}' - see `oxio provider list`");
            }
            cfg.defaults.primary = name.clone();
            config::save(cfg)?;
            println!("primary = {name}");
        }
        ProviderCmd::Remove { name } => {
            if cfg.providers.remove(&name).is_none() {
                anyhow::bail!("no such provider '{name}'");
            }
            if cfg.defaults.primary == name {
                cfg.defaults.primary.clear();
            }
            config::save(cfg)?;
            println!("removed '{name}'");
        }
    }
    Ok(())
}

/// Load config. First-run behavior is split by track and MUST NOT leak:
/// - DEV (debug build or `OXIO_DEV` set): seed a working local default so we
///   iterate fast. This uses a hardcoded endpoint and is dev-only.
/// - PRODUCTION (release, no `OXIO_DEV`): NO hardcoded endpoint. First run
///   yields an empty config; onboarding (detect endpoints / wizard) lands later.
fn ensure_config() -> anyhow::Result<Config> {
    let path = config::config_path();
    if path.exists() {
        return config::load();
    }
    let dev = cfg!(debug_assertions) || std::env::var_os("OXIO_DEV").is_some();
    if dev {
        let cfg = Config::seed_local();
        config::save(&cfg)?;
        eprintln!(
            "[dev] seeded default config at {} (primary: local)",
            path.display()
        );
        Ok(cfg)
    } else {
        // Prod first-run: no config, no hardcoded endpoint. Commands needing a
        // provider report cleanly; onboarding wizard is the declared next step.
        Ok(Config::default())
    }
}

/// Health check: report the primary and ping it with a bounded deadline.
async fn doctor(cfg: &Config) -> anyhow::Result<()> {
    println!("config: {}", config::config_path().display());
    match cfg.primary() {
        None => println!("primary: (none configured)"),
        Some((name, pc)) => {
            println!(
                "primary: {name} [{:?}] {}",
                pc.wire_api,
                pc.base_url.clone().unwrap_or_default()
            );
            // build_kernel runs each configured MCP server's FULL connect lifecycle:
            // open the gate (initialize handshake) → discover tools → capture the
            // server's instructions. Results land in `ctl.mcp_status`.
            let (kernel, _session, _compactor, _snapshots, ctl) =
                ui::build_kernel(cfg, Arc::new(ui::StdinApprover), None, None).await?;
            let ctx = Ctx::default().with_deadline(Duration::from_secs(20));
            match kernel.run_turn("ping", &ctx, &NullSink).await {
                Ok(o) => println!("  OK - model replied ({} chars)", o.text.len()),
                Err(e) => println!("  FAIL - {e}"),
            }
            // MCP servers: report the handshake result per configured server. Empty =
            // none configured. The connections tear down when this process exits (the
            // kernel - and the clients it holds - drop at end of `doctor`).
            if cfg.mcp.is_empty() {
                println!("mcp: (none configured)");
            } else {
                println!("mcp servers:");
                for (name, status) in &ctl.mcp_status {
                    match status {
                        Ok(n) => {
                            println!("  {name} ● connected - handshake OK, {n} tool(s) discovered")
                        }
                        Err(e) => println!("  {name} ✗ {e}"),
                    }
                }
            }
            // Skills: on-disk SKILL.md (oxio paths) + config-inline, merged.
            let skills = ui::resolve_skills(cfg);
            if skills.is_empty() {
                println!("skills: (none)");
            } else {
                println!("skills: {} discovered", skills.len());
                for (name, s) in &skills {
                    let d = s.description.clone().unwrap_or_default();
                    println!("  {name} - {d}");
                }
            }
            // Agents: named `[agents.<name>]` sub-agents, each with its bound skills
            // (the agent->skill map), tool scope, and permission mode.
            if cfg.agents.is_empty() {
                println!("agents: (none)");
            } else {
                println!("agents: {} configured", cfg.agents.len());
                for (name, a) in &cfg.agents {
                    let sk = match a.skills.as_ref() {
                        Some(s) if !s.is_empty() => s.join(", "),
                        _ => "-".to_string(),
                    };
                    let tools = match &a.tools {
                        None => "all".to_string(),
                        Some(t) => t.join(", "),
                    };
                    let perm = a.permission.as_deref().unwrap_or("prompt");
                    println!("  {name} - skills: [{sk}] · tools: {tools} · perm: {perm}");
                }
            }
        }
    }
    Ok(())
}
