//! The oxio TUI - a full-screen ratatui frontend (from zero, not the legacy
//! line REPL): an async event loop over
//! (a) crossterm key events, (b) the model stream, and (c) a frame tick that
//! drives the live in-turn indicator. Regions: transcript · indicator · framed
//! composer · persistent bottom status bar.
//!
//! This is Phases 1–2 of `UIUX-BUILD.md`: app shell + turn streaming + the live
//! `working… (elapsed · ↓tokens)` indicator + the status bar (model · ctx% bar ·
//! tokens · cost · cwd). Markdown/diff render, themes, and the composer popups are
//! later phases; nothing here is faked.

use std::io::Stdout;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{
    DisableBracketedPaste, DisableFocusChange, EnableBracketedPaste, EnableFocusChange, Event,
    EventStream, KeyCode, KeyEventKind, KeyModifiers,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport};

use std::collections::BTreeMap;

use tokio_util::sync::CancellationToken;

use oxio_core::{Ctx, Kernel, NoticeLevel, StreamEvent, StreamSink};

use crate::{IMAGE_MARKER_CLOSE, IMAGE_MARKER_OPEN};
use config::{Config, RateCfg};
use safety::{AllowAll, ApprovalRequest, Approver, Decision, DenyAll, SnapshotStore};
use tokio::sync::oneshot;

use crate::cost_usd;

/// A pending approval request handed from a tool call to the UI loop.
type ApprovalMsg = (ApprovalRequest, oneshot::Sender<Decision>);

/// Approver that runs inside the tool call but delegates the y/a/N decision to the
/// TUI event loop over a channel - because a raw-mode TUI can't do a blocking
/// `stdin().read_line` like `StdinApprover`. Sends the request, awaits the reply.
struct TuiApprover {
    tx: tokio::sync::mpsc::UnboundedSender<ApprovalMsg>,
}

#[async_trait::async_trait]
impl Approver for TuiApprover {
    async fn approve(&self, req: &ApprovalRequest) -> Decision {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self.tx.send((req.clone(), resp_tx)).is_err() {
            return Decision::Deny; // UI gone
        }
        resp_rx.await.unwrap_or(Decision::Deny)
    }
}

type Term = Terminal<CrosstermBackend<Stdout>>;

/// A finished transcript line with a role-derived style.
struct Row {
    text: String,
    style: Style,
}

/// The sink handed to `run_turn`: forwards every stream event to the UI loop.
struct TuiSink {
    tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
}

#[async_trait::async_trait]
impl StreamSink for TuiSink {
    async fn send(&self, ev: StreamEvent) {
        let _ = self.tx.send(ev);
    }
}

/// Word-wrap `text` to `width` columns for the transcript: greedy, hard-breaking a
/// word longer than `width`. Char-count based (matches ratatui for the ASCII/BMP
/// content the transcript holds; a wide-char line clips rather than wraps). Wrapping
/// here - instead of Paragraph's own wrap - makes the rendered row count EXACT so the
/// scroll offset never drifts into blank space. Always returns at least one row.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    let mut w = 0usize;
    for word in text.split_inclusive(' ') {
        let wlen = word.chars().count();
        if wlen > width {
            // Hard-break a word longer than the pane: flush, then chunk by width.
            if w > 0 {
                out.push(std::mem::take(&mut line));
                w = 0;
            }
            for ch in word.chars() {
                if w == width {
                    out.push(std::mem::take(&mut line));
                    w = 0;
                }
                line.push(ch);
                w += 1;
            }
        } else {
            if w + wlen > width && w > 0 {
                out.push(std::mem::take(&mut line));
                w = 0;
            }
            line.push_str(word);
            w += wlen;
        }
    }
    out.push(line);
    out
}

/// Elapsed seconds, compact: `0s`, `59s`, `1m 00s`, `1h 02m 03s`.
fn fmt_elapsed(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!(
            "{}h {:02}m {:02}s",
            secs / 3600,
            (secs % 3600) / 60,
            secs % 60
        )
    }
}

// A comet tracing a figure-8 (vertical infinity) through one braille cell: each frame is
// the leading dot plus the previous two positions, so a short tail trails along the path.
const SPINNER: [&str; 8] = ["⠣", "⠋", "⠙", "⠜", "⡔", "⣄", "⣠", "⢢"];

/// Height of the inline viewport (the drawn composer block at the bottom). The
/// conversation itself lives in the terminal's native scrollback, so this stays small.
/// Fixed for now - dynamic height would need a custom terminal.
const VIEWPORT_H: u16 = 8;

/// Route attached images through the cloud vision provider: describe them, fold the
/// description into the user's text, and re-run the turn text-only against the local
/// model. Used both proactively (local model declared no vision) and reactively (it
/// errored on the image). Announced via notices, never silent.
async fn vision_route(
    kernel: &Arc<Kernel>,
    sink: &TuiSink,
    vp: &Arc<dyn oxio_core::Provider>,
    vm: &str,
    images: &[String],
    base_text: &str,
    ctx: &Ctx,
) {
    let q = "Describe these image(s) in detail for a coding assistant.";
    match crate::describe_images(vp, vm, images, q, ctx).await {
        Ok(desc) => {
            let folded = format!(
                "{base_text}\n\n[Vision model {vm} description of the attached image(s):]\n{desc}"
            );
            if let Err(e) = kernel.run_turn(folded, ctx, sink).await {
                sink.send(StreamEvent::Notice {
                    level: NoticeLevel::Warn,
                    text: format!("vision fallback re-run failed: {e}"),
                })
                .await;
            }
        }
        Err(e) => {
            sink.send(StreamEvent::Notice {
                level: NoticeLevel::Warn,
                text: format!("vision provider failed: {e}"),
            })
            .await;
        }
    }
}

/// Set the terminal tab/window title (OSC 0). Out-of-band from the ratatui grid so
/// it is safe to emit mid-session; lets the tab show working/idle.
fn set_terminal_title(title: &str) {
    use std::io::Write;
    let mut out = std::io::stdout();
    let _ = write!(out, "\x1b]0;{title}\x07");
    let _ = out.flush();
}

/// Image extensions oxio recognizes when a file is dropped/pasted into the
/// composer. Matches the wire adapters' supported media types.
const IMAGE_EXTS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// A paste at/above either bound collapses to a `[pasted #N +M lines]` chip instead of
/// flooding the inline composer; below it, the text inserts inline as normal.
const PASTE_COLLAPSE_LINES: usize = 6;
const PASTE_COLLAPSE_BYTES: usize = 800;

/// Whether a paste is large enough to collapse to a chip (vs insert inline).
fn should_collapse_paste(s: &str) -> bool {
    s.lines().count().max(1) >= PASTE_COLLAPSE_LINES || s.len() >= PASTE_COLLAPSE_BYTES
}

/// Normalize a pasted/dropped path the way a terminal delivers it, then confirm it
/// is an image that exists. Strip wrapping quotes,
/// resolve a `file://` URL, and un-escape a single backslash-escaped path (macOS
/// drag-drop escapes spaces as `\ `). Returns the absolute path string if - and only
/// if - the result is one existing file with an image extension; otherwise `None`,
/// so ordinary pasted text falls through untouched.
/// Parse ONE filesystem path out of dropped/pasted/typed text - strips wrapping quotes, a
/// `file://` URL, and shell-escaped spaces. Returns the candidate only if the WHOLE token
/// is a single path (not a phrase); existence + kind are the caller's to check. Shared by
/// the paste-time image capture and the submit-time file intake.
fn parse_dropped_path(pasted: &str) -> Option<String> {
    let s = pasted.trim();
    if s.is_empty() {
        return None;
    }
    // Strip a single pair of wrapping quotes.
    let unquoted = s
        .strip_prefix('"')
        .and_then(|x| x.strip_suffix('"'))
        .or_else(|| s.strip_prefix('\'').and_then(|x| x.strip_suffix('\'')))
        .unwrap_or(s);
    // file:// URL → path (decode %20 → space only; that covers the common case).
    let candidate = if let Some(rest) = unquoted.strip_prefix("file://") {
        // Drop an optional host segment before the leading '/'.
        let path = rest.find('/').map(|i| &rest[i..]).unwrap_or(rest);
        path.replace("%20", " ")
    } else {
        // A single shell-escaped path: turn "\ " back into " ". Only accept it if the
        // token has no unescaped whitespace (i.e. it really is one path, not a phrase).
        let mut out = String::with_capacity(unquoted.len());
        let mut chars = unquoted.chars().peekable();
        let mut multi_token = false;
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(&n) = chars.peek() {
                    out.push(n);
                    chars.next();
                    continue;
                }
            }
            if c.is_whitespace() {
                multi_token = true;
                break;
            }
            out.push(c);
        }
        if multi_token {
            return None;
        }
        out
    };
    Some(candidate)
}

/// A dropped/pasted single IMAGE file path (paste-time attach). `None` if the token isn't
/// an existing image file.
fn dropped_image_path(pasted: &str) -> Option<String> {
    let candidate = parse_dropped_path(pasted)?;
    let path = std::path::Path::new(&candidate);
    let is_image = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| IMAGE_EXTS.contains(&e.as_str()));
    if is_image && path.is_file() {
        Some(candidate)
    } else {
        None
    }
}

/// Below this, a shared file's full content is inlined directly (convenient one-shot).
/// Above it, we present a MANIFEST instead and let the AI read what it needs - you can't
/// dump a 100MB workbook/pdf into the turn and hope the model digests it.
const INTAKE_MAX_BYTES: usize = 32 * 1024;

/// Build the turn text for a proactively-read file. Small file → inline the whole thing.
/// Large file → a MANIFEST: its structure (spreadsheet sheet list, or line/size counts),
/// a short head preview, and how to read specific parts with `read_file`. The AI then
/// chooses what to pull, instead of drowning in (or truncating) a huge dump.
fn inject_file(path: &str, content: &str, user_msg: &str) -> String {
    if content.len() <= INTAKE_MAX_BYTES {
        return format!(
            "The user shared the file {path}. Its contents:\n\n{content}\n\n---\n\nUser message: {user_msg}"
        );
    }
    let lines = content.lines().count();
    let kb = content.len() / 1024;
    // Spreadsheet structure lines (`# Workbook:` / `# Sheet: X (N rows)`) if present.
    let structure: Vec<&str> = content
        .lines()
        .filter(|l| l.starts_with("# Workbook:") || l.starts_with("# Sheet:"))
        .collect();
    let structure_block = if structure.is_empty() {
        String::new()
    } else {
        format!("Structure:\n{}\n\n", structure.join("\n"))
    };
    let head: String = content.lines().take(30).collect::<Vec<_>>().join("\n");
    format!(
        "The user shared {path} - it is large ({lines} lines, ~{kb} KB), so it is NOT inlined \
in full.\n\n{structure_block}First 30 lines:\n\n{head}\n\n[Read specific parts with read_file \
on {path}: use `offset`/`limit` for a line range, or `sheet=\"<name>\"` for one spreadsheet \
sheet.]\n\n---\n\nUser message: {user_msg}"
    )
}

/// Compact text of the last few user/assistant turns, fed to the ghost-suggestion model.
/// Skips system/tool messages; caps each message so the prompt stays small.
fn recent_convo(msgs: &[oxio_core::Message]) -> String {
    use oxio_core::Role;
    let mut out = String::new();
    for m in msgs
        .iter()
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let who = match m.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            _ => continue,
        };
        let text = m.as_text();
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        let capped: String = text.chars().take(500).collect();
        out.push_str(who);
        out.push_str(": ");
        out.push_str(&capped);
        out.push('\n');
    }
    out.trim().to_string()
}

/// Resolve a `/model` selector - a 1-based number or a provider name - to a name.
fn resolve_provider(cfg: &Config, sel: &str) -> Option<String> {
    if let Ok(n) = sel.parse::<usize>() {
        cfg.providers.keys().nth(n.saturating_sub(1)).cloned()
    } else if cfg.providers.contains_key(sel) {
        Some(sel.to_string())
    } else {
        None
    }
}

/// What the `/model` handler must do LIVE after `connect_cmd` has updated config. The config
/// mutation + save already happened synchronously in `connect_cmd`; these variants carry the async
/// follow-up (rebuild the provider chain, probe an endpoint) that needs the run loop.
enum ConnectAction {
    /// Nothing further - just show the lines.
    None,
    /// Switch the live session to this machine (endpoint) - its `model` is the machine default.
    SwitchMachine(String),
    /// Switch the live session's ACTIVE machine to this model id (same base_url).
    SwitchModel(String),
    /// Probe this base_url's `/v1/models` and list what the machine serves.
    ListModels(String),
    /// Refresh: probe known local defaults for NEW servers to add, and re-probe each
    /// configured endpoint to refresh its model + flag stale ones (hint, never auto-remove).
    Scan,
    /// A just-added endpoint (by name) with no model yet: probe it and open the model picker
    /// so the user confirms a real model and the session reloads onto it - no typed guessing.
    /// If the endpoint is unreachable, say so (endpoint stays saved; pick a model later).
    ProbePick(String),
}

/// Build the in-session connect picker: every model on every CONFIGURED endpoint (probed
/// live) plus any famous local server not yet in config (marked "new"). A configured
/// endpoint's pinned model is always offered even if `/v1/models` omits it - that pin may be
/// a valid router alias (e.g. a router alias), so absence from the list ≠ gone. Unreachable
/// endpoints are skipped (nothing to connect to). `None` if there is nothing to pick.
async fn build_connect_pick(current: &str, current_model: &str) -> Option<PendingPick> {
    let cfg = config::load().ok()?;
    let mut rows: Vec<String> = Vec::new();
    let mut targets: Vec<PickTarget> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (name, pc) in &cfg.providers {
        let Some(url) = pc.base_url.clone() else {
            continue;
        };
        seen.insert(url.trim_end_matches('/').to_string());
        let mut models = providers::list_models(&url).await;
        // Always keep a set pin available (may be a router alias absent from /v1/models).
        if let Some(m) = pc.model.clone() {
            if !m.is_empty() && !models.contains(&m) {
                models.insert(0, m);
            }
        }
        for m in models {
            let live = if name == current && m == current_model {
                "→ "
            } else {
                "  "
            };
            rows.push(format!("{live}{name:<14} {m}"));
            targets.push(PickTarget::Connect {
                name: name.clone(),
                base_url: url.clone(),
                model: m,
                add: false,
            });
        }
    }

    for e in providers::scan_local().await {
        if seen.contains(e.base_url.trim_end_matches('/')) {
            continue;
        }
        let name = crate::provider_key(&e.label);
        for m in e.models {
            rows.push(format!("  {:<14} {m}  (new)", e.label));
            targets.push(PickTarget::Connect {
                name: name.clone(),
                base_url: e.base_url.clone(),
                model: m,
                add: true,
            });
        }
    }

    if targets.is_empty() {
        return None;
    }
    Some(PendingPick {
        title: "connect".to_string(),
        rows,
        targets,
        sel: 0,
        off: 0,
    })
}

/// Probe ONE just-added endpoint and build a model picker scoped to it - the confirm step
/// after `/model add`. `None` if the endpoint is unreachable or serves no models (caller
/// reports that; the endpoint stays saved so a model can be picked once the server is up).
async fn build_endpoint_pick(name: &str) -> Option<PendingPick> {
    let cfg = config::load().ok()?;
    let pc = cfg.providers.get(name)?;
    let url = pc.base_url.clone()?;
    let models = providers::list_models(&url).await;
    if models.is_empty() {
        return None;
    }
    let rows: Vec<String> = models.iter().map(|m| format!("  {name:<14} {m}")).collect();
    let targets: Vec<PickTarget> = models
        .into_iter()
        .map(|m| PickTarget::Connect {
            name: name.to_string(),
            base_url: url.clone(),
            model: m,
            add: false,
        })
        .collect();
    Some(PendingPick {
        title: format!("model on {name}"),
        rows,
        targets,
        sel: 0,
        off: 0,
    })
}

/// `/model` - in-session endpoint manager: list / switch machine / switch model / add / remove /
/// models, mirroring the CLI `provider` verbs over the SAME config (load + save). One machine
/// (endpoint) can serve many models: a bare arg that is NOT a machine name is treated as a model id
/// and applied to the active machine. Config mutation + save happen here; the returned
/// `ConnectAction` carries the async follow-up the run loop performs (rebuild + live swap, or probe).
/// `current` is the machine this running session is on.
fn connect_cmd(rest: &str, current: &str) -> (Vec<String>, ConnectAction) {
    let mut cfg = match config::load() {
        Ok(c) => c,
        Err(e) => {
            return (
                vec![format!("connect: cannot read config: {e}")],
                ConnectAction::None,
            )
        }
    };
    let mut parts = rest.split_whitespace();
    match parts.next() {
        None | Some("list") => {
            if cfg.providers.is_empty() {
                return (
                    vec!["no endpoints - add one with /model add <name> <url> [model]".into()],
                    ConnectAction::None,
                );
            }
            let mut out = vec!["endpoints (* = default at launch, → = active now):".to_string()];
            for (i, (n, pc)) in cfg.providers.iter().enumerate() {
                let star = if *n == cfg.defaults.primary { "*" } else { " " };
                let live = if n == current { "→" } else { " " };
                out.push(format!(
                    "  {star}{live} {}) {n}  {}  {}",
                    i + 1,
                    pc.base_url.clone().unwrap_or_default(),
                    pc.model.clone().unwrap_or_default()
                ));
            }
            out.push(format!("  active now: {current} · /model (no args) → arrow-pick a model · /model <machine>/<model> to switch"));
            out.push("  /model models - list served · /model ctx <tokens> - context window".into());
            out.push("  /model add <name> <url> [model] - add any endpoint · /model remove <name>".into());
            out.push("  /model cloud [vendor] - add a cloud provider · /model scan - find new + refresh".into());
            (out, ConnectAction::None)
        }
        // Probe the ACTIVE machine's /v1/models and list what it serves.
        Some("models") => match cfg.providers.get(current).and_then(|pc| pc.base_url.clone()) {
            Some(url) => (vec![], ConnectAction::ListModels(url)),
            None => (vec![format!("connect: active machine '{current}' has no base_url")], ConnectAction::None),
        },
        // Refresh: discover new famous-default servers + re-probe configured endpoints.
        Some("scan") => (vec![], ConnectAction::Scan),
        // Add a cloud vendor (api-key preset). `/model cloud` lists; `/model cloud <name>`
        // adds it (auth=api_key); the key goes in the env/`.env`, never config.toml. Per-vendor
        // OAuth is staged (only MCP-server OAuth exists today) - key auth is the ready path.
        Some("cloud") => {
            let vendors = crate::cloud_vendors();
            match parts.next() {
                None => {
                    let mut out = vec!["cloud vendors - add with /model cloud <name>:".to_string()];
                    for (k, label, url, _) in &vendors {
                        out.push(format!("  {k} - {label} ({url})"));
                    }
                    out.push("  then set the key: export OXIO_<NAME>_API_KEY=…  (or add it to <config>/.env)".into());
                    (out, ConnectAction::None)
                }
                Some(v) => match vendors.iter().find(|(k, ..)| *k == v) {
                    Some((k, label, url, wire)) => {
                        cfg.providers.insert(
                            k.to_string(),
                            config::ProviderCfg {
                                wire_api: *wire,
                                base_url: Some(url.to_string()),
                                model: None,
                                auth: config::AuthKind::ApiKey,
                                temperature: None,
                                top_p: None,
                                max_tokens: None,
                                reasoning_effort: None,
                                context_window: None,
                                vision: None,
                            },
                        );
                        match config::save(&cfg) {
                            Ok(()) => (
                                vec![format!(
                                    "added cloud '{k}' ({label}) · set OXIO_{}_API_KEY, then /model {k} to use",
                                    k.to_uppercase()
                                )],
                                ConnectAction::None,
                            ),
                            Err(e) => (vec![format!("connect: save failed: {e}")], ConnectAction::None),
                        }
                    }
                    None => (vec![format!("unknown vendor '{v}' - /model cloud to list")], ConnectAction::None),
                },
            }
        }
        // Set the ACTIVE MODEL's context window (context belongs to the model, not the
        // machine - `[models.<id>]`). Applies next launch: the compactor's ceiling is
        // fixed when the kernel builds.
        Some("ctx") => {
            let arg = parts.next();
            let model = cfg.providers.get(current).and_then(|pc| pc.model.clone()).unwrap_or_default();
            if model.is_empty() {
                return (vec![format!("connect: active machine '{current}' has no model set - /model <model> first")], ConnectAction::None);
            }
            match arg {
                Some(n) => match n.replace('_', "").parse::<usize>() {
                    Ok(tokens) if tokens >= 1024 => {
                        cfg.models.entry(model.clone()).or_default().context_window = Some(tokens);
                        match config::save(&cfg) {
                            Ok(()) => (
                                vec![format!("model '{model}' context_window = {tokens} (applies next launch - restart to use it)")],
                                ConnectAction::None,
                            ),
                            Err(e) => (vec![format!("connect: save failed: {e}")], ConnectAction::None),
                        }
                    }
                    _ => (vec!["usage: /model ctx <tokens> (e.g. 262144 for 256k)".into()], ConnectAction::None),
                },
                None => {
                    let cur = cfg
                        .models
                        .get(&model)
                        .and_then(|m| m.context_window)
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "unset (default 32768)".into());
                    (vec![format!("model '{model}' context_window = {cur} · /model ctx <tokens>")], ConnectAction::None)
                }
            }
        }
        Some("add") => {
            let (name, url, model) = (parts.next(), parts.next(), parts.next());
            match (name, url) {
                (Some(name), Some(url)) => {
                    let pc = config::ProviderCfg {
                        wire_api: config::WireApi::ChatCompletions,
                        base_url: Some(url.to_string()),
                        model: model.map(String::from),
                        auth: config::AuthKind::None,
                        temperature: None,
                        top_p: None,
                        max_tokens: None,
                        reasoning_effort: None,
                        context_window: None,
                        vision: None,
                    };
                    let has_model = model.is_some();
                    cfg.providers.insert(name.to_string(), pc);
                    if cfg.defaults.primary.is_empty() {
                        cfg.defaults.primary = name.to_string();
                    }
                    match config::save(&cfg) {
                        // Model given explicitly → activate + live-reload straight away.
                        // No model → probe the endpoint and open the model picker (confirm a
                        // real model, then reload) instead of leaving a blind, unverified entry.
                        Ok(()) if has_model => (
                            vec![format!("added endpoint '{name}' → {url}")],
                            ConnectAction::SwitchMachine(name.to_string()),
                        ),
                        Ok(()) => (
                            vec![format!("added endpoint '{name}' → {url} · probing for models…")],
                            ConnectAction::ProbePick(name.to_string()),
                        ),
                        Err(e) => (vec![format!("connect: save failed: {e}")], ConnectAction::None),
                    }
                }
                _ => (vec!["usage: /model add <name> <url> [model]".into()], ConnectAction::None),
            }
        }
        Some("remove") => match resolve_provider(&cfg, parts.next().unwrap_or("")) {
            // Refuse to remove the ACTIVE machine - doing so leaves the live session
            // pointing at a machine that no longer exists, and every later /model fails
            // with "no machine '<it>'". Switch away first, then remove.
            Some(n) if n == current => (
                vec![format!("'{n}' is the active machine - switch to another first (/model <machine>), then remove it")],
                ConnectAction::None,
            ),
            Some(n) => {
                cfg.providers.remove(&n);
                if cfg.defaults.primary == n {
                    cfg.defaults.primary.clear();
                }
                let _ = config::save(&cfg);
                (vec![format!("removed endpoint '{n}'")], ConnectAction::None)
            }
            None => (vec!["no such endpoint - /model to list".into()], ConnectAction::None),
        },
        Some(sel) => match resolve_provider(&cfg, sel) {
            // Arg is a MACHINE name → switch machine (persist new default + live swap).
            Some(n) if n == current => (vec![format!("already on machine '{n}'")], ConnectAction::None),
            Some(n) => {
                cfg.defaults.primary = n.clone();
                match config::save(&cfg) {
                    Ok(()) => (vec![], ConnectAction::SwitchMachine(n)),
                    Err(e) => (vec![format!("connect: save failed: {e}")], ConnectAction::None),
                }
            }
            // Arg is NOT a machine → treat it as a MODEL id on the active machine: pin it as that
            // machine's model, persist, and hand it back for a live rebuild+swap.
            None => match cfg.providers.get_mut(current) {
                Some(pc) => {
                    if pc.model.as_deref() == Some(sel) {
                        return (vec![format!("already on model '{sel}'")], ConnectAction::None);
                    }
                    pc.model = Some(sel.to_string());
                    match config::save(&cfg) {
                        Ok(()) => (vec![], ConnectAction::SwitchModel(sel.to_string())),
                        Err(e) => (vec![format!("connect: save failed: {e}")], ConnectAction::None),
                    }
                }
                None => (vec![format!("no machine '{current}' in config - /model to list")], ConnectAction::None),
            },
        },
    }
}

/// A left→right terracotta shimmer over `text`: a bright band sweeps across the
/// characters (a gentle travelling wave, not a whole-word blink). `tick` advances
/// ~10/s; the band moves ~0.5 char/tick, so it reads as a slow shimmer.
fn shimmer_spans(text: &str, tick: usize) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let span = chars.len() as f32 + 6.0;
    let pos = (tick as f32 * 0.5) % span;
    chars
        .into_iter()
        .enumerate()
        .map(|(i, ch)| {
            let t = (1.0 - ((i as f32 - pos).abs() / 4.0)).clamp(0.0, 1.0);
            let lerp = |a: f32, b: f32| (a + (b - a) * t) as u8;
            let col = Color::Rgb(lerp(92.0, 168.0), lerp(36.0, 80.0), lerp(30.0, 64.0));
            Span::styled(ch.to_string(), Style::default().fg(col))
        })
        .collect()
}

/// Theme accent - terracotta. Used for the logo, the composer prompt, and the base
/// of the working shimmer, so the whole UI shares one signature colour.
// The ONE fixed brand colour. Everything else uses the terminal's own foreground
// (via `Style::default()` / the DIM modifier) so text stays readable in BOTH light
// and dark terminals - hardcoding cream would vanish on a light theme, off-black on
// a dark one. Terracotta #7A3028 is dark enough to read on a light/cream background
// yet saturated enough to show on a dark one, so it works either way.
const THEME: Color = Color::Rgb(122, 48, 40); // akite deep terracotta #7A3028 (accent/logo)
fn theme() -> Style {
    Style::default().fg(THEME)
}

// User-prompt highlight - a SUBTLE terracotta wash, so the user's turn is distinct from
// the AI's plain output. Terminals have no real alpha, so "terracotta transparent ~80%" is
// approximated as the accent #7A3028 blended ~20% over a dark base → a low, warm wash
// rather than an opaque pastel. The bar sets BOTH colours (a bg is the one place the
// "lean on the terminal default fg" rule can't hold): the dark wash + warm-cream text pair
// is self-contained and reads in light and dark terminals. The `❯` gutter stays.
const USER_BG: Color = Color::Rgb(45, 30, 29); // #7A3028 at ~20% over a dark base - subtle wash
const USER_FG: Color = Color::Rgb(232, 222, 214); // warm cream, readable on the wash
fn user_hl() -> Style {
    Style::default().bg(USER_BG).fg(USER_FG)
}

/// An in-session arrow-key connect picker - the `/model` equivalent of onboarding's
/// scrolling list. Rendered as a bordered panel over the composer; ↑/↓ scroll a fixed
/// window (long lists don't flood), Enter connects live, Esc cancels.
struct PendingPick {
    title: String,
    rows: Vec<String>, // display lines, parallel to `targets`
    targets: Vec<PickTarget>,
    sel: usize,
    off: usize, // index of the first visible row (scroll offset)
}

/// What accepting a picked row does.
enum PickTarget {
    /// Switch the live session to this endpoint + model. `add` carries a provider entry to
    /// insert into config first, for a freshly scanned box not yet registered.
    Connect {
        name: String,
        base_url: String,
        model: String,
        add: bool,
    },
}

struct App {
    rows: Vec<Row>,
    input: String,
    cursor: usize,           // byte offset of the caret within `input`
    history: Vec<String>,    // submitted prompts, oldest first
    hist_idx: Option<usize>, // current recall position (None = live input)
    // In-flight turn: streamed answer + thinking, start time, output-char count.
    live: String,
    think: String,
    working: Option<Instant>,
    out_chars: usize,
    tick: usize,
    // pending write-approval: the reply channel + a human summary of the request
    pending: Option<oneshot::Sender<Decision>>,
    pending_summary: String,
    // pending ask_user_question: the reply channel the next composer submit answers (Esc
    // = no answer). The question + options are pushed to scrollback on receipt; ↑/↓ cycle
    // the options into the composer (Enter accepts, or type free-text for "Other").
    pending_ask: Option<oneshot::Sender<Option<String>>>,
    pending_ask_options: Vec<String>,
    pending_ask_sel: Option<usize>,
    // in-session connect picker (`/model` with no args): arrow-select an endpoint+model
    // to switch to live. Rendered as a panel over the composer; None = not picking.
    pending_pick: Option<PendingPick>,
    // images dropped/pasted into the composer, attached to the next submitted turn
    pending_images: Vec<String>,
    // long pastes collapsed to a `[pasted #N +M lines]` chip: (marker, full text). The
    // composer + transcript show the chip; the marker expands to full text at submit.
    pending_pastes: Vec<(String, String)>,
    model: String,
    reasoning: String,
    provider: String,
    profile: String,
    cwd: String,
    git_branch: String,
    project: String,
    host: String,
    session_id: String,
    approval: &'static str,
    tz_offset: i64,
    statusline: Option<String>,
    cost_label: &'static str,
    pricing: BTreeMap<String, RateCfg>,
    ctx_used: usize,
    ctx_window: Option<usize>,
    total_in: u64,
    total_out: u64,
    // session-wide stats for the exit summary
    started: Instant,
    turns: u32,
    tool_calls: u32,
    // type-ahead: messages composed while a turn is streaming, latched FIFO to auto-send
    // when the turn finishes. Up recalls the
    // last one into the composer to edit/hold.
    queued: std::collections::VecDeque<String>,
    // unsent composer text, stashed when browsing history so arrowing back down past
    // the newest entry restores what you were typing (not a blank line).
    draft: Option<String>,
    // ghost next-prompt suggestion (dimmed in the composer, Tab to accept); generated by
    // the primary model after a turn when `suggest` is on. None = nothing to show.
    ghost: Option<String>,
    suggest: bool,
    quit: bool,
}

impl App {
    fn dim() -> Style {
        Style::default().add_modifier(Modifier::DIM)
    }

    fn push(&mut self, text: impl Into<String>, style: Style) {
        self.rows.push(Row {
            text: text.into(),
            style,
        });
    }

    /// Fold the current streamed buffers (thinking + answer) into the transcript.
    /// Called at each model-step boundary; does NOT clear `working` - the turn may
    /// continue with tool calls or further model steps. The turn is settled only when
    /// the spawned task signals completion (done_rx).
    fn commit_stream(&mut self) {
        // `take` moves the buffers out so we don't hold an immutable borrow of
        // `self` while `push` borrows it mutably.
        let think = std::mem::take(&mut self.think);
        for l in think.lines() {
            self.push(l.to_string(), Self::dim());
        }
        let live = std::mem::take(&mut self.live);
        for l in live.lines() {
            self.push(l.to_string(), Style::default());
        }
    }
}

/// Persisted prompt-history file (config dir) - cross-session up/down recall, like a
/// shell's history, surviving restarts in the same or a new terminal.
fn prompt_history_path() -> Option<std::path::PathBuf> {
    config::config_path()
        .parent()
        .map(|d| d.join("prompt_history"))
}

/// Load persisted prompts (oldest first), newlines un-escaped, capped to the last 1000.
fn load_prompt_history() -> Vec<String> {
    let Some(p) = prompt_history_path() else {
        return Vec::new();
    };
    let Ok(body) = std::fs::read_to_string(&p) else {
        return Vec::new();
    };
    let mut v: Vec<String> = body
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.replace("\\n", "\n"))
        .collect();
    let n = v.len();
    if n > 1000 {
        v.drain(0..n - 1000);
    }
    v
}

/// Append one submitted prompt to the history file (newlines escaped so each prompt is
/// one line). Best-effort - a failed write never affects the turn.
fn append_prompt_history(text: &str) {
    let Some(p) = prompt_history_path() else {
        return;
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)
    {
        use std::io::Write;
        let _ = f.write_all(format!("{}\n", text.replace('\n', "\\n")).as_bytes());
    }
}

/// Run the oxio TUI. Falls back to the caller on setup failure.
pub async fn run_tui(
    cfg: &Config,
    continue_session: bool,
    resume_pick: bool,
) -> anyhow::Result<()> {
    // Approval mode is config-driven. Interactive mode uses a TUI modal (not
    // stdin), routed to the event loop over this channel.
    let (appr_tx, appr_rx) = tokio::sync::mpsc::unbounded_channel::<ApprovalMsg>();
    let approver: Arc<dyn Approver> = match cfg.defaults.approval.as_str() {
        "accept" => Arc::new(AllowAll),
        "deny" => Arc::new(DenyAll),
        _ => Arc::new(TuiApprover { tx: appr_tx }),
    };
    let approval_mode = approver.mode();
    // ask_user_question routes through this channel to the event loop (never a blocking
    // stdin read, which deadlocks against the TUI's raw-mode input).
    let (ask_tx, ask_rx) = tokio::sync::mpsc::unbounded_channel::<crate::AskRequest>();
    // MCP servers connect in the background (input box renders now, not after handshakes);
    // each server's outcome streams over this channel and is shown in place.
    let (mcp_tx, mcp_rx) =
        tokio::sync::mpsc::unbounded_channel::<(String, Result<usize, String>)>();
    let (kernel, session, compactor, snapshots, ctl) =
        crate::build_kernel(cfg, approver, Some(ask_tx), Some(mcp_tx)).await?;
    let kernel = Arc::new(kernel);

    // Cloud/vision fallback: (provider, model, name) when a distinct vision_provider
    // is configured, else None. Used when the local model can't handle an image.
    let vision_fb = crate::vision_fallback(cfg).await;

    // Ghost next-prompt suggestion: reuse the primary model (a separate cheap handle,
    // just an HTTP client - no network). Always resolved so `/suggest` can toggle it at
    // runtime; whether it actually GENERATES is gated by `app.suggest` at turn-end.
    let suggest_provider = crate::primary_provider(cfg).await;

    // Resume (same as the line REPL).
    if resume_pick {
        if let Some(p) = crate::pick_session() {
            session.continue_from(&p);
        }
    } else if continue_session {
        if let Some(p) = session::latest_session() {
            session.continue_from(&p);
        }
    }

    let (pname, pc) = cfg.primary().ok_or_else(|| {
        anyhow::anyhow!("no primary provider configured - see `oxio config show`")
    })?;

    let mut app = App {
        rows: Vec::new(),
        input: String::new(),
        cursor: 0,
        history: load_prompt_history(),
        hist_idx: None,
        live: String::new(),
        think: String::new(),
        working: None,
        out_chars: 0,
        tick: 0,
        pending: None,
        pending_ask: None,
        pending_ask_options: Vec::new(),
        pending_ask_sel: None,
        pending_pick: None,
        pending_summary: String::new(),
        pending_images: Vec::new(),
        pending_pastes: Vec::new(),
        model: pc.model.clone().unwrap_or_default(),
        reasoning: pc.reasoning_effort.clone().unwrap_or_default(),
        provider: pname.to_string(),
        profile: if cfg.defaults.profile.is_empty() {
            "default".to_string()
        } else {
            cfg.defaults.profile.clone()
        },
        cwd: std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_default(),
        git_branch: git_branch(),
        project: git_root(),
        host: hostname(),
        approval: approval_mode, // derived from the live approver, not hardcoded
        tz_offset: tz_offset_secs(),
        session_id: session
            .transcript_path()
            .and_then(|p| p.file_stem().and_then(|s| s.to_str()).map(String::from))
            .unwrap_or_default(),
        statusline: cfg.statusline.clone(),
        cost_label: if pc.auth == config::AuthKind::None {
            "saved"
        } else {
            "cost"
        },
        pricing: cfg.pricing.clone(),
        ctx_used: 0,
        ctx_window: compactor.context_window(),
        total_in: 0,
        total_out: 0,
        started: Instant::now(),
        turns: 0,
        tool_calls: 0,
        queued: std::collections::VecDeque::new(),
        draft: None,
        ghost: None,
        suggest: cfg.defaults.suggestions.unwrap_or(false),
        quit: false,
    };
    // Resume vs fresh: on --continue/--resume with restored history, REPLAY the prior
    // transcript into the visible history so it reads as if never broken, and skip the
    // big landing. Fresh start → branded logo + info block.
    let restored = session.snapshot();
    if !restored.is_empty() {
        use oxio_core::Role;
        app.push(
            format!("  oxio · resumed session ({} messages)", restored.len()),
            App::dim(),
        );
        for m in &restored {
            let text = m.as_text();
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            match m.role {
                Role::User => {
                    app.push(String::new(), Style::default());
                    app.push(format!("❯ {}", text.replace('\n', " ")), user_hl());
                    app.push(String::new(), Style::default());
                }
                Role::Assistant => {
                    for line in text.lines() {
                        app.push(line.to_string(), Style::default());
                    }
                    app.push(String::new(), Style::default());
                }
                _ => {} // skip system/tool messages in the visual replay
            }
        }
        app.push("─── resumed; continue below ───".to_string(), App::dim());
        app.ctx_used = context::estimate_tokens(&restored);
    } else {
        // Landing: branded logo + info block so the screen is never a blank void.
        let accent = theme();
        let profile = if cfg.defaults.profile.is_empty() {
            "default"
        } else {
            &cfg.defaults.profile
        };
        let cwd = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_default();
        app.push("", Style::default());
        for l in ["  ┌─┐╲ ╱│┌─┐", "  │ │ ╳ ││ │", "  └─┘╱ ╲│└─┘"]
        {
            app.push(l.to_string(), accent);
        }
        app.push("  github.com/limkcreply/oxio".to_string(), App::dim());
        app.push("".to_string(), Style::default());
        let info = Style::default(); // terminal default fg - adapts to light/dark, no bold
        app.push(format!("  model  {}", app.model), info);
        app.push(format!("  via    {pname}  ·  profile:{profile}"), info);
        app.push(format!("  cwd    {cwd}"), info);
        // MCP servers connect in the BACKGROUND now (the input box is already usable) - show
        // a connecting line; each server's result lands live below as it resolves (mcp_rx).
        let _ = &ctl.mcp_status; // (populated only on the synchronous doctor/one-shot path)
        if !cfg.mcp.is_empty() {
            app.push(
                format!(
                    "  mcp    connecting to {} server(s) in the background…",
                    cfg.mcp.len()
                ),
                App::dim(),
            );
        }
        app.push("".to_string(), Style::default());
        app.push(
            "  /help for commands  ·  Ctrl-C to quit".to_string(),
            App::dim(),
        );
    }

    // Terminal setup. INLINE viewport (NOT alt-screen): the conversation is written to
    // the terminal's NATIVE scrollback via insert_before, so native wheel/two-finger
    // scroll AND text selection work. Only the composer block lives in the drawn viewport.
    let mut stdout = std::io::stdout();
    enable_raw_mode()?;
    crossterm::execute!(stdout, EnableBracketedPaste, EnableFocusChange)?;
    set_terminal_title("oxio");
    let mut term = Terminal::with_options(
        CrosstermBackend::new(stdout),
        TerminalOptions {
            viewport: Viewport::Inline(VIEWPORT_H),
        },
    )?;

    let primary_vision = cfg.primary().and_then(|(_, pc)| pc.vision);
    let res = event_loop(
        &mut term,
        &mut app,
        &kernel,
        &session,
        &compactor,
        &snapshots,
        appr_rx,
        ask_rx,
        mcp_rx,
        vision_fb,
        primary_vision,
        suggest_provider,
        &ctl,
    )
    .await;

    // Teardown - best-effort: run every step even if one hiccups (a transient inline
    // cursor-position query must never skip cleanup or abort before the summary). Clear
    // the inline viewport BEFORE disabling raw mode so a late DSR reply is still consumed
    // in raw mode rather than leaking `\x1b[..R` into the shell.
    let _ = term.clear(); // drop the inline viewport so the shell prompt returns clean
    let _ = term.show_cursor();
    let _ = crossterm::execute!(
        term.backend_mut(),
        DisableBracketedPaste,
        DisableFocusChange
    );
    // Drain any pending input (e.g. a late cursor-position reply) while STILL in raw mode,
    // so it can't spill into the shell as a literal `\x1b[..R`. EventStream is already
    // dropped, so a plain crossterm poll/read owns stdin here.
    while crossterm::event::poll(std::time::Duration::from_millis(0)).unwrap_or(false) {
        let _ = crossterm::event::read();
    }
    let _ = disable_raw_mode();
    set_terminal_title("");
    print_exit_summary(&app);
    res
}

/// On quit, print a session summary to the normal terminal (alt screen already
/// left): id, model, turns, tool calls, tokens, cost, wall time - drawn from stats
/// oxio already tracks. Uses a
/// raw ANSI escape (ratatui styling no longer applies) and honours NO_COLOR.
fn print_exit_summary(app: &App) {
    if app.turns == 0 {
        return; // nothing happened - no noise on an immediate quit
    }
    let secs = app.started.elapsed().as_secs();
    let cost = cost_usd(app.total_in, app.total_out, &app.pricing, &app.model);
    let (c0, c1) = if std::env::var_os("NO_COLOR").is_none() {
        ("\x1b[38;2;122;48;40m", "\x1b[0m") // terracotta #7A3028
    } else {
        ("", "")
    };
    let row = |k: &str, v: String| println!("  {k:<9}{v}");
    println!("\n{c0}  oxio · session summary{c1}");
    if !app.session_id.is_empty() {
        row("session", app.session_id.clone());
    }
    row("model", app.model.clone());
    row(
        "turns",
        format!("{} · {} tool calls", app.turns, app.tool_calls),
    );
    row(
        "tokens",
        format!("↑{} ↓{}", fmt_k(app.total_in), fmt_k(app.total_out)),
    );
    match cost {
        Some(c) => row("cost", format!("${c:.4}")),
        None => row("cost", format!("$0 ({})", app.cost_label)),
    }
    row("time", fmt_elapsed(secs));
    println!();
}

// The interactive loop genuinely needs all of these collaborators (kernel, session,
// compactor, snapshots, approval + vision routing); grouping them into a struct would
// add indirection without clarity.
#[allow(clippy::too_many_arguments)]
async fn event_loop(
    term: &mut Term,
    app: &mut App,
    kernel: &Arc<Kernel>,
    session: &Arc<session::SessionStore>,
    compactor: &Arc<context::Compactor>,
    snapshots: &Arc<SnapshotStore>,
    mut appr_rx: tokio::sync::mpsc::UnboundedReceiver<ApprovalMsg>,
    mut ask_rx: tokio::sync::mpsc::UnboundedReceiver<crate::AskRequest>,
    mut mcp_rx: tokio::sync::mpsc::UnboundedReceiver<(String, Result<usize, String>)>,
    vision_fb: Option<(Arc<dyn oxio_core::Provider>, String, String)>,
    primary_vision: Option<bool>,
    suggest_provider: Option<(Arc<dyn oxio_core::Provider>, String)>,
    ctl: &crate::LiveControls,
) -> anyhow::Result<()> {
    let mut keys = EventStream::new();
    let mut ticker = tokio::time::interval(Duration::from_millis(100));
    // Per-turn stream channel + cancel token (rebuilt each turn).
    let (_init_tx, mut sink_rx) = tokio::sync::mpsc::unbounded_channel::<StreamEvent>();
    let mut cancel: Option<CancellationToken> = None;
    // Turn-completion signal: the spawned turn task sends `()` once `run_turn` truly
    // returns (after ALL model steps + tool calls). Only then do we settle the working
    // indicator - a per-step `Done` must NOT stop it, or the spinner dies mid-turn
    // (e.g. during a tool call) and the screen looks frozen.
    let (done_tx, mut done_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    // Ghost next-prompt suggestion: a spawned task after each turn computes it off the UI
    // thread and sends the text here; the loop parks it in `app.ghost` for Tab-to-accept.
    let (ghost_tx, mut ghost_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    // Terminal focus (for the bell-when-away signal). Assumed focused until the
    // terminal reports otherwise; terminals without focus reporting never beep.
    let mut focused = true;
    // Tab-title state - written via the ratatui backend (single writer) only on
    // transitions, so the OSC never interleaves with insert_before (which leaked a
    // stray '[' in Terminal.app's stricter OSC parser).
    let mut titled_working = false;

    // The single submit path: echo the prompt,
    // attach any dropped images, spawn the turn, and return the new stream receiver +
    // cancel token. Called from Enter (idle) AND from the type-ahead queue flush.
    let start_turn = |app: &mut App,
                      text: String|
     -> (
        tokio::sync::mpsc::UnboundedReceiver<StreamEvent>,
        CancellationToken,
    ) {
        if app.history.last() != Some(&text) {
            app.history.push(text.clone());
            append_prompt_history(&text); // persist for cross-session recall
        }
        app.hist_idx = None;
        // Blank above the prompt (top space) and below it (bottom space).
        app.push(String::new(), Style::default());
        app.push(format!("❯ {text}"), user_hl());
        app.push(String::new(), Style::default());
        let mut images: Vec<String> = app.pending_images.drain(..).collect();
        let mut turn_text = text.clone();
        // Expand collapsed-paste chips back to full text for the MODEL; the transcript echo
        // (`❯ {text}`) keeps the compact `[pasted #N +M lines]` chip, so the screen stays clean.
        for (marker, full) in app.pending_pastes.drain(..) {
            turn_text = turn_text.replace(&marker, &full);
        }
        // Agentic file intake: if the whole message is a file path (drag-drop often arrives
        // as typed text, so it misses the paste-time image capture), ACT on it - don't make
        // the model ask "shall I open this?". Image → vision route (below); pdf/docx/xlsx →
        // extract the text locally and hand it straight to the model.
        if let Some(fpath) = parse_dropped_path(&text) {
            let p = std::path::Path::new(&fpath);
            if p.is_file() {
                let is_img = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .is_some_and(|e| IMAGE_EXTS.contains(&e.as_str()));
                if is_img {
                    if !images.contains(&fpath) {
                        images.push(fpath.clone());
                    }
                } else if let Some(kind) = tools::docs::DocKind::from_path(p) {
                    match tools::docs::extract(p, kind) {
                        Ok(ex) if ex.scanned_pdf => {
                            app.push(
                                format!("  ⎿ {fpath} is a scanned PDF (no text layer) - needs a vision model"),
                                App::dim(),
                            );
                        }
                        Ok(ex) => {
                            turn_text = inject_file(&fpath, &ex.text, &text);
                            app.push(format!("  ⎿ read {fpath}"), App::dim());
                        }
                        Err(e) => {
                            app.push(format!("  ⎿ could not read {fpath}: {e}"), App::dim());
                        }
                    }
                } else {
                    // Any other file is plain text (csv, txt, md, json, code) - read and inject
                    // it too, so a dropped text file is acted on, not left for the model to
                    // refuse. Non-UTF-8 (a binary we don't extract) is left to the model.
                    match std::fs::read_to_string(p) {
                        Ok(content) => {
                            turn_text = inject_file(&fpath, &content, &text);
                            app.push(format!("  ⎿ read {fpath}"), App::dim());
                        }
                        Err(_) => {
                            app.push(
                                format!("  ⎿ {fpath} isn't UTF-8 text - leaving it for the model"),
                                App::dim(),
                            );
                        }
                    }
                }
            }
        }
        for path in &images {
            turn_text.push_str(&format!("\n{IMAGE_MARKER_OPEN}{path}{IMAGE_MARKER_CLOSE}"));
        }
        app.working = Some(Instant::now());
        app.tick = 0;
        app.out_chars = 0;
        app.turns += 1;
        // NOTE: title/bell are written via the ratatui backend in the event loop (single
        // writer) - a raw OSC write here would interleave with insert_before in the inline
        // model and leave a stray character.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        // Attach the UI channel as the turn's progress sink so a spawned sub-agent's
        // activity (tagged by name) forwards here live instead of running silent.
        let ctx = Ctx::default().with_progress(Arc::new(TuiSink { tx: tx.clone() }));
        let cancel_tok = ctx.cancel.clone();
        let k = kernel.clone();
        let vfb = vision_fb.clone();
        let pv = primary_vision;
        let base_text = text;
        let turn_done = done_tx.clone();
        tokio::spawn(async move {
            let sink = TuiSink { tx };
            // Whole turn (every model step + tool call) runs here; `working` stays lit
            // until it finishes, then `turn_done` settles it. `return` exits the block.
            async {
                let has_images = !images.is_empty();
                if has_images && pv == Some(false) {
                    match &vfb {
                        Some((vp, vm, vname)) => {
                            sink.send(StreamEvent::Notice { level: NoticeLevel::Info,
                                text: format!("local model has no vision - reading image via {vname} ({vm})…") }).await;
                            vision_route(&k, &sink, vp, vm, &images, &base_text, &ctx).await;
                        }
                        None => {
                            sink.send(StreamEvent::Notice { level: NoticeLevel::Warn,
                                text: "local model can't see images and no vision_provider is configured - sending text only".into() }).await;
                            if let Err(e) = k.run_turn(base_text.clone(), &ctx, &sink).await {
                                sink.send(StreamEvent::Notice { level: NoticeLevel::Warn, text: format!("turn failed: {e}") }).await;
                            }
                        }
                    }
                    return;
                }
                if let Err(e) = k.run_turn(turn_text, &ctx, &sink).await {
                    match (has_images, &vfb) {
                        (true, Some((vp, vm, vname))) => {
                            sink.send(StreamEvent::Notice { level: NoticeLevel::Warn,
                                text: format!("local model failed on the image ({e}) - routing to {vname} ({vm})…") }).await;
                            vision_route(&k, &sink, vp, vm, &images, &base_text, &ctx).await;
                        }
                        (true, None) => {
                            sink.send(StreamEvent::Notice { level: NoticeLevel::Warn,
                                text: format!("local model failed on the image ({e}); no vision_provider configured to fall back to") }).await;
                        }
                        _ => {
                            sink.send(StreamEvent::Notice { level: NoticeLevel::Warn, text: format!("turn failed: {e}") }).await;
                        }
                    }
                }
            }
            .await;
            let _ = turn_done.send(());
        });
        (rx, cancel_tok)
    };

    loop {
        // Sync the tab title on transition, through the SAME backend writer as the frame
        // (never a separate raw stdout handle - that interleaves and leaks a '['). ASCII
        // only, to avoid any OSC multibyte edge in older terminal parsers.
        let now_working = app.working.is_some();
        if now_working != titled_working {
            let title = if now_working {
                "oxio - working"
            } else {
                "oxio"
            };
            let _ = crossterm::execute!(term.backend_mut(), crossterm::terminal::SetTitle(title));
            titled_working = now_working;
        }
        // Flush committed transcript lines into the terminal's NATIVE scrollback (above
        // the inline viewport). This is the whole point of the model: history lives in
        // real scrollback, so native scroll + selection work. Then clear the buffer.
        if !app.rows.is_empty() {
            let width = term.get_frame().area().width.max(1) as usize;
            let mut wrapped: Vec<Line> = Vec::new();
            for r in &app.rows {
                // A row with a background (the user-prompt bar) is padded to full width
                // so the highlight spans the pane, not just the glyphs. Plain rows aren't.
                let pad = r.style.bg.is_some();
                for mut piece in wrap_text(&r.text, width) {
                    if pad {
                        let len = piece.chars().count();
                        if len < width {
                            piece.push_str(&" ".repeat(width - len));
                        }
                    }
                    wrapped.push(Line::styled(piece, r.style));
                }
            }
            app.rows.clear();
            let h = wrapped.len().min(u16::MAX as usize) as u16;
            if h > 0 {
                let _ = term.insert_before(h, move |buf: &mut Buffer| {
                    let area = buf.area;
                    Paragraph::new(wrapped).render(area, buf);
                });
            }
        }
        // A transient terminal hiccup - ratatui's inline-viewport cursor-position query
        // (ESC[6n) racing the async EventStream on a resize - must NOT kill the session.
        // Skip the frame; the next tick redraws, and staying in the loop lets EventStream
        // absorb the stray DSR reply instead of it leaking to the shell after teardown.
        // (Propagating it did exactly that: aborted the turn + leaked `7;4R`.)
        let _ = term.draw(|f| draw(f, app));
        if app.quit {
            return Ok(());
        }

        tokio::select! {
            _ = ticker.tick() => {
                if app.working.is_some() { app.tick = app.tick.wrapping_add(1); }
            }
            maybe_key = keys.next() => {
                if matches!(maybe_key.as_ref(), Some(Ok(Event::FocusGained))) {
                    focused = true;
                } else if matches!(maybe_key.as_ref(), Some(Ok(Event::FocusLost))) {
                    focused = false;
                } else if let Some(Ok(Event::Paste(s))) = maybe_key.as_ref() {
                    if app.working.is_none() {
                        // A dropped/pasted image path is attached to the next turn, not
                        // inserted as text - this also keeps a leading-'/' path out of
                        // the slash-command parser.
                        if let Some(path) = dropped_image_path(s) {
                            let name = std::path::Path::new(&path)
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or(&path)
                                .to_string();
                            app.pending_images.push(path);
                            let marker = format!("[image: {name}] ");
                            app.input.insert_str(app.cursor, &marker);
                            app.cursor += marker.len();
                        } else {
                            let lines = s.lines().count().max(1);
                            if should_collapse_paste(s) {
                                // Long paste → collapse to a chip: stash the
                                // full text, show `[pasted #N +M lines]` in the composer, expand
                                // at submit. Keeps the small inline composer from flooding.
                                let n = app.pending_pastes.len() + 1;
                                let marker = format!("[pasted #{n} +{lines} lines]");
                                app.pending_pastes.push((marker.clone(), s.to_string()));
                                let chip = format!("{marker} ");
                                app.input.insert_str(app.cursor, &chip);
                                app.cursor += chip.len();
                            } else {
                                app.input.insert_str(app.cursor, s);
                                app.cursor += s.len();
                            }
                        }
                    }
                } else if let Some(Ok(Event::Key(k))) = maybe_key {
                    if k.kind == KeyEventKind::Release { continue; }
                    // A pending write-approval: ONLY y / s / a / n / Esc decide. Stray keys
                    // are IGNORED - never an accidental reject (VS Code turns two-finger
                    // scroll into arrow keys). Scrolling itself is native (scrollback).
                    if app.pending.is_some() {
                        match k.code {
                            // Every write asks each time - a single yes allows this call only.
                            KeyCode::Char('y') | KeyCode::Char('Y') => {
                                if let Some(r) = app.pending.take() { let _ = r.send(Decision::Once); }
                            }
                            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                                if let Some(r) = app.pending.take() { let _ = r.send(Decision::Deny); }
                            }
                            _ => {} // ignore stray input (arrows/scroll) - do not decide
                        }
                        continue;
                    }
                    match (k.code, k.modifiers) {
                        // Connect picker up: navigation keys act, Enter connects, Esc cancels,
                        // everything else is swallowed so stray keys don't leak into the composer.
                        (KeyCode::Up, _) if app.pending_pick.is_some() => {
                            if let Some(p) = app.pending_pick.as_mut() { p.sel = p.sel.saturating_sub(1); }
                        }
                        (KeyCode::Down, _) if app.pending_pick.is_some() => {
                            if let Some(p) = app.pending_pick.as_mut() {
                                if p.sel + 1 < p.rows.len() { p.sel += 1; }
                            }
                        }
                        (KeyCode::Enter, _) if app.pending_pick.is_some() => {
                            let p = app.pending_pick.take().unwrap();
                            if let Some(PickTarget::Connect { name, base_url, model, add }) =
                                p.targets.into_iter().nth(p.sel)
                            {
                                if let Ok(mut cfg) = config::load() {
                                    if add && !cfg.providers.contains_key(&name) {
                                        cfg.providers.insert(name.clone(), crate::local_provider(base_url.clone(), Some(model.clone())));
                                    } else if let Some(pc) = cfg.providers.get_mut(&name) {
                                        pc.model = Some(model.clone());
                                    }
                                    if cfg.defaults.primary.is_empty() { cfg.defaults.primary = name.clone(); }
                                    let _ = config::save(&cfg);
                                }
                                match crate::resolve_chain_reload().await {
                                    Ok((chain, _)) => {
                                        ctl.provider.set(chain);
                                        app.provider = name.clone();
                                        app.model = model.clone();
                                        app.push(format!("connected → {name} · {model}"), App::dim());
                                    }
                                    Err(e) => app.push(
                                        format!("connect: cannot connect to '{name}': {e} (config saved; restart to apply)"),
                                        App::dim(),
                                    ),
                                }
                            }
                            continue;
                        }
                        (KeyCode::Esc, _) if app.pending_pick.is_some() => {
                            app.pending_pick = None;
                            app.push("  (connect cancelled)".to_string(), App::dim());
                        }
                        (_, _) if app.pending_pick.is_some() => {}
                        (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                            if let Some(c) = &cancel { c.cancel(); } else { app.quit = true; }
                        }
                        // Tab accepts the ghost next-prompt suggestion into the composer.
                        (KeyCode::Tab, _) if app.ghost.is_some() && app.input.is_empty() => {
                            app.input = app.ghost.take().unwrap_or_default();
                            app.cursor = app.input.chars().count();
                        }
                        (KeyCode::Esc, _) => {
                            // A pending ask_user_question? Esc declines it (answer = None)
                            // and lets the turn continue on the model's own judgement.
                            if let Some(r) = app.pending_ask.take() {
                                let _ = r.send(None);
                                app.pending_ask_options.clear();
                                app.pending_ask_sel = None;
                                app.push("  (question dismissed)".to_string(), App::dim());
                            } else if app.ghost.is_some() {
                                // A pending ghost? Esc dismisses it first (not the turn).
                                app.ghost = None;
                            } else if let Some(c) = &cancel {
                                c.cancel();
                            }
                        }
                        (KeyCode::Enter, KeyModifiers::ALT) => {
                            app.input.insert(app.cursor, '\n'); // multiline: Alt+Enter = newline
                            app.cursor += 1;
                        }
                        (KeyCode::Enter, _) => {
                            let text = app.input.trim().to_string();
                            // Answering a pending ask_user_question: the composer submit IS
                            // the answer, routed back to the waiting tool - NOT a new turn.
                            if let Some(r) = app.pending_ask.take() {
                                app.input.clear();
                                app.cursor = 0;
                                app.hist_idx = None;
                                app.pending_ask_options.clear();
                                app.pending_ask_sel = None;
                                if text.is_empty() {
                                    let _ = r.send(None);
                                } else {
                                    app.push(format!("  ↳ {text}"), App::dim());
                                    let _ = r.send(Some(text));
                                }
                                continue;
                            }
                            if text.is_empty() { continue; }
                            // Slash commands run in the TUI, not sent to the model. A
                            // command token never contains '/', so an absolute path like
                            // /Users/…/x.png falls through to the model instead of being
                            // mis-parsed as an unknown command.
                            let is_slash_cmd = text.strip_prefix('/').is_some_and(|c| {
                                !c.split_whitespace().next().unwrap_or("").contains('/')
                            });
                            // Type-ahead: while a turn streams, latch a normal message to
                            // auto-send when it finishes. Up
                            // recalls it. Slash commands still run immediately, mid-turn.
                            if app.working.is_some() && !is_slash_cmd {
                                app.queued.push_back(text);
                                app.input.clear();
                                app.cursor = 0;
                                app.hist_idx = None;
                                continue;
                            }
                            if let Some(cmd) = text.strip_prefix('/').filter(|_| is_slash_cmd) {
                                app.input.clear();
                                app.cursor = 0;
                                app.hist_idx = None;
                                match cmd.split_whitespace().next().unwrap_or("") {
                                    "quit" | "q" | "exit" => app.quit = true,
                                    "clear" => {
                                        session.clear();
                                        app.rows.clear();
                                        app.push("(session cleared)", App::dim());
                                    }
                                    "imageq" | "imageQ" => {
                                        let arg =
                                            cmd.split_whitespace().nth(1).unwrap_or("").to_lowercase();
                                        match arg.as_str() {
                                            "low" | "medium" | "high" => {
                                                std::env::set_var("OXIO_IMAGE_QUALITY", &arg);
                                                app.push(format!("image quality → {arg}"), App::dim());
                                            }
                                            "" => {
                                                let cur = std::env::var("OXIO_IMAGE_QUALITY")
                                                    .unwrap_or_else(|_| "medium (default)".into());
                                                app.push(
                                                    format!("image quality: {cur} · usage: /imageQ low|medium|high"),
                                                    App::dim(),
                                                );
                                            }
                                            _ => app.push("usage: /imageQ low|medium|high", App::dim()),
                                        }
                                    }
                                    "suggest" => {
                                        app.suggest = !app.suggest;
                                        app.ghost = None;
                                        let note = if suggest_provider.is_none() {
                                            "next-prompt suggestions on, but no primary provider resolved"
                                        } else if app.suggest {
                                            "next-prompt suggestions on (Tab to accept the ghost hint)"
                                        } else {
                                            "next-prompt suggestions off"
                                        };
                                        app.push(note, App::dim());
                                    }
                                    "remember" => {
                                        // /remember [global] <fact> - YOU set the scope, no model
                                        // guessing. First token "global" → user-wide; else project.
                                        let rest = cmd.split_once(char::is_whitespace).map(|(_, r)| r).unwrap_or("").trim();
                                        let (global, fact) = match rest.strip_prefix("global") {
                                            Some(f) if f.is_empty() || f.starts_with(char::is_whitespace) => (true, f.trim()),
                                            _ => (false, rest),
                                        };
                                        if fact.is_empty() {
                                            app.push("usage: /remember [global] <fact>  (global = all projects; default = this project)", App::dim());
                                        } else {
                                            let res = if global {
                                                memory::append_global_memory(fact)
                                            } else {
                                                memory::append_memory(fact)
                                            };
                                            match res {
                                                Ok(()) => app.push(
                                                    format!("remembered ({}): {fact}", if global { "global" } else { "project" }),
                                                    App::dim(),
                                                ),
                                                Err(e) => app.push(format!("remember failed: {e}"), App::dim()),
                                            }
                                        }
                                    }
                                    "model" => {
                                        let rest = cmd.split_once(char::is_whitespace).map(|(_, r)| r).unwrap_or("").trim();
                                        // Bare `/model` → the arrow-key picker (same UX as first-run
                                        // onboarding): scan every configured endpoint + famous locals and
                                        // list every model to arrow-select. Sub-verbs stay text-driven.
                                        if rest.is_empty() {
                                            app.push("scanning endpoints…".to_string(), App::dim());
                                            match build_connect_pick(&app.provider, &app.model).await {
                                                Some(p) => app.pending_pick = Some(p),
                                                None => app.push(
                                                    "no reachable endpoints - /model add <name> <url> [model] or /model cloud".to_string(),
                                                    App::dim(),
                                                ),
                                            }
                                            continue;
                                        }
                                        let (lines, action) = connect_cmd(rest, &app.provider);
                                        for line in lines {
                                            app.push(line, App::dim());
                                        }
                                        // Live rebuilds repoint the session's provider slot from the
                                        // now-saved config; the next turn uses it, transcript untouched.
                                        // (The compactor keeps the boot endpoint's context ceiling for
                                        // its 90% trigger - a minor budgeting imprecision, not a wire
                                        // fault: each adapter carries its own model id, so inference is
                                        // correct against the swapped machine/model immediately.)
                                        match action {
                                            ConnectAction::None => {}
                                            ConnectAction::ListModels(url) => {
                                                app.push(format!("probing {url} …"), App::dim());
                                                let models = providers::list_models(&url).await;
                                                if models.is_empty() {
                                                    app.push(
                                                        "  (no models advertised, or endpoint unreachable)".to_string(),
                                                        App::dim(),
                                                    );
                                                } else {
                                                    app.push(format!("models on {} (→ = current):", app.provider), App::dim());
                                                    for m in &models {
                                                        let live = if *m == app.model { "→" } else { " " };
                                                        app.push(format!("  {live} {m}"), App::dim());
                                                    }
                                                    app.push("  /model <model> to switch".to_string(), App::dim());
                                                }
                                            }
                                            ConnectAction::Scan => {
                                                let cfg2 = config::load().unwrap_or_default();
                                                // 1) New famous-default servers not already in config → offer to add.
                                                app.push("scanning local defaults for new servers…".to_string(), App::dim());
                                                let mut any_new = false;
                                                for (label, base) in providers::KNOWN_LOCAL_ENDPOINTS {
                                                    if cfg2.providers.values().any(|pc| pc.base_url.as_deref() == Some(*base)) {
                                                        continue;
                                                    }
                                                    let models = providers::list_models(base).await;
                                                    if let Some(m) = models.first() {
                                                        any_new = true;
                                                        let sugg = label.to_lowercase().replace([' ', '.'], "-");
                                                        app.push(format!("  + {label} ({base}) serves '{m}' - add: /model add {sugg} {base}"), App::dim());
                                                    }
                                                }
                                                if !any_new {
                                                    app.push("  no new local servers found.".to_string(), App::dim());
                                                }
                                                // 2) Re-probe each configured endpoint; refresh + flag stale.
                                                // Conservative: unreachable is NOT proof of gone - hint, never auto-remove.
                                                app.push("refreshing configured endpoints…".to_string(), App::dim());
                                                for (name, pc) in &cfg2.providers {
                                                    let Some(url) = pc.base_url.clone() else { continue };
                                                    let served = providers::list_models(&url).await;
                                                    let pinned = pc.model.clone().unwrap_or_default();
                                                    if served.is_empty() {
                                                        app.push(format!("  {name} ○ unreachable - transient? if truly gone: /model remove {name}"), App::dim());
                                                    } else if pinned.is_empty() || served.iter().any(|m| m == &pinned) {
                                                        app.push(format!("  {name} ● ok ({} model(s))", served.len()), App::dim());
                                                    } else {
                                                        app.push(format!("  {name} ⚠ pinned '{pinned}' no longer served - /model {name} to adopt, or /model remove {name}"), App::dim());
                                                    }
                                                }
                                            }
                                            ConnectAction::SwitchMachine(name) => {
                                                // Verify the pinned model against what the machine serves
                                                // (probe /v1/models) - but NEVER auto-rewrite a working pin.
                                                // A pin absent from /v1/models may be a valid router ALIAS
                                                // (e.g. a router alias), so absence ≠ stale: only adopt when
                                                // NO model is pinned (a model is required to chat); otherwise
                                                // just hint. Probe empty (server down) → leave the pin alone.
                                                if let Ok(mut cfg2) = config::load() {
                                                    if let Some(url) = cfg2.providers.get(&name).and_then(|pc| pc.base_url.clone()) {
                                                        let served = providers::list_models(&url).await;
                                                        let pinned = cfg2.providers.get(&name).and_then(|pc| pc.model.clone()).unwrap_or_default();
                                                        if !served.is_empty() && pinned.is_empty() {
                                                            if let Some(first) = served.into_iter().next() {
                                                                if let Some(pc) = cfg2.providers.get_mut(&name) {
                                                                    pc.model = Some(first.clone());
                                                                }
                                                                let _ = config::save(&cfg2);
                                                                app.push(format!("no model pinned - using '{first}'"), App::dim());
                                                            }
                                                        } else if !served.is_empty() && !served.iter().any(|m| m == &pinned) {
                                                            app.push(format!("note: '{pinned}' isn't in {name}'s model list - may be a valid alias; if wrong, /model <model>"), App::dim());
                                                        }
                                                    }
                                                }
                                                match crate::resolve_chain_reload().await {
                                                    Ok((chain, cfg2)) => {
                                                        ctl.provider.set(chain);
                                                        app.provider = name.clone();
                                                        if let Some(pc) = cfg2.providers.get(&name) {
                                                            app.model = pc.model.clone().unwrap_or_default();
                                                        }
                                                        if app.model.is_empty() {
                                                            app.push(
                                                                format!("switched machine → {name} · NO MODEL (endpoint served none; set one with /model <model>)"),
                                                                App::dim(),
                                                            );
                                                        } else {
                                                            app.push(format!("switched machine → {name} · {}", app.model), App::dim());
                                                        }
                                                    }
                                                    Err(e) => app.push(
                                                        format!("connect: cannot build '{name}': {e} (config saved; restart to apply)"),
                                                        App::dim(),
                                                    ),
                                                }
                                            }
                                            ConnectAction::SwitchModel(id) => {
                                                match crate::resolve_chain_reload().await {
                                                    Ok((chain, _cfg2)) => {
                                                        ctl.provider.set(chain);
                                                        app.model = id.clone();
                                                        app.push(format!("switched model → {id} (on {})", app.provider), App::dim());
                                                    }
                                                    Err(e) => app.push(
                                                        format!("connect: cannot switch to model '{id}': {e} (config saved; restart to apply)"),
                                                        App::dim(),
                                                    ),
                                                }
                                            }
                                            ConnectAction::ProbePick(name) => {
                                                // Confirm step after `/model add`: probe the endpoint and
                                                // open the model picker; Enter connects + reloads live. If it
                                                // can't be reached, say so - the endpoint stays saved.
                                                match build_endpoint_pick(&name).await {
                                                    Some(p) => app.pending_pick = Some(p),
                                                    None => app.push(
                                                        format!("  '{name}' unreachable or serves no models - endpoint saved; run /model once it's up to pick a model"),
                                                        App::dim(),
                                                    ),
                                                }
                                            }
                                        }
                                    }
                                    #[cfg(feature = "mcp")]
                                    "mcp" => {
                                        let rest = cmd.split_once(char::is_whitespace).map(|(_, r)| r.trim()).unwrap_or("");
                                        let mut parts = rest.split_whitespace();
                                        match parts.next() {
                                            None | Some("list") => match config::load() {
                                                Ok(cfg) if !cfg.mcp.is_empty() => {
                                                    app.push("mcp servers:".to_string(), App::dim());
                                                    for (n, sc) in &cfg.mcp {
                                                        let target = sc.url.clone().or_else(|| sc.command.clone()).unwrap_or_default();
                                                        app.push(format!("  {n}  {target}"), App::dim());
                                                    }
                                                    app.push("  /mcp add <name> <url | command…> · /mcp connect <name> · /mcp remove <name>".to_string(), App::dim());
                                                }
                                                _ => app.push("no mcp servers - /mcp add <name> <url | command…>".to_string(), App::dim()),
                                            },
                                            Some("add") => {
                                                let name = parts.next().unwrap_or("").to_string();
                                                let spec: Vec<String> = parts.map(|s| s.to_string()).collect();
                                                if name.is_empty() || spec.is_empty() {
                                                    app.push("usage: /mcp add <name> <url | command [args…]>".to_string(), App::dim());
                                                } else {
                                                    // URL first token = remote; otherwise command + args = stdio.
                                                    let entry = if spec[0].starts_with("http://") || spec[0].starts_with("https://") {
                                                        config::McpServerCfg { url: Some(spec[0].clone()), ..Default::default() }
                                                    } else {
                                                        config::McpServerCfg { command: Some(spec[0].clone()), args: spec[1..].to_vec(), ..Default::default() }
                                                    };
                                                    if let Ok(mut cfg) = config::load() {
                                                        cfg.mcp.insert(name.clone(), entry.clone());
                                                        let _ = config::save(&cfg);
                                                    }
                                                    if let Some(conn) = &ctl.mcp {
                                                        app.push(format!("connecting mcp '{name}'…"), App::dim());
                                                        match conn.connect(&name, &entry).await {
                                                            Ok(n) => app.push(format!("mcp {name} ● connected ({n} tool(s)) · saved"), App::dim()),
                                                            Err(e) => app.push(format!("mcp {name} ○ saved, but connect failed: {e}"), App::dim()),
                                                        }
                                                    }
                                                }
                                            }
                                            Some("connect") => {
                                                let name = parts.next().unwrap_or("").to_string();
                                                match config::load().ok().and_then(|c| c.mcp.get(&name).cloned()) {
                                                    Some(sc) => {
                                                        if let Some(conn) = &ctl.mcp {
                                                            app.push(format!("connecting mcp '{name}'…"), App::dim());
                                                            match conn.connect(&name, &sc).await {
                                                                Ok(n) => app.push(format!("mcp {name} ● connected ({n} tool(s))"), App::dim()),
                                                                Err(e) => app.push(format!("mcp {name} ○ failed: {e}"), App::dim()),
                                                            }
                                                        }
                                                    }
                                                    None => app.push(format!("no mcp '{name}' - /mcp add <name> <url | command…>"), App::dim()),
                                                }
                                            }
                                            Some("remove") | Some("rm") => {
                                                let name = parts.next().unwrap_or("").to_string();
                                                if let Ok(mut cfg) = config::load() {
                                                    if cfg.mcp.remove(&name).is_some() {
                                                        let _ = config::save(&cfg);
                                                        if let Some(conn) = &ctl.mcp {
                                                            conn.disconnect(&name);
                                                        }
                                                        app.push(format!("removed mcp '{name}' (disconnected)"), App::dim());
                                                    } else {
                                                        app.push(format!("no mcp '{name}'"), App::dim());
                                                    }
                                                }
                                            }
                                            Some(other) => app.push(format!("unknown: /mcp {other} - list | add | connect | remove"), App::dim()),
                                        }
                                    }
                                    "think" => {
                                        use std::sync::atomic::Ordering;
                                        let rest = cmd.split_once(char::is_whitespace).map(|(_, r)| r.trim()).unwrap_or("");
                                        let cur = ctl.thinking.load(Ordering::Relaxed);
                                        let label = |v: u8| match v { 1 => "on", 2 => "off", _ => "auto (model default)" };
                                        match rest {
                                            "on" => { ctl.thinking.store(1, Ordering::Relaxed); app.push("thinking → on (sends enable_thinking=true)".to_string(), App::dim()); }
                                            "off" => { ctl.thinking.store(2, Ordering::Relaxed); app.push("thinking → off (sends enable_thinking=false)".to_string(), App::dim()); }
                                            "auto" => { ctl.thinking.store(0, Ordering::Relaxed); app.push("thinking → auto (no flag sent; model decides)".to_string(), App::dim()); }
                                            "" => app.push(format!("thinking is {} · /think on|off|auto", label(cur)), App::dim()),
                                            _ => app.push("usage: /think on|off|auto".to_string(), App::dim()),
                                        }
                                    }
                                    "help" | "h" => {
                                        // Per-command education: each command with a
                                        // one-line "what it delivers". Commands that take arguments print
                                        // their own usage when run with no/invalid args.
                                        for line in [
                                            "commands - type one; those with options print usage when run bare:",
                                            "  /model      arrow-pick a model to connect (bare /model); sub-verbs: list/add/remove/cloud/scan/models/ctx",
                                            "  /mcp        manage MCP servers: /mcp (list) · /mcp add <name> <url|command…> · /mcp connect/remove <name>",
                                            "  /think        toggle model thinking/reasoning: /think on | off | auto",
                                            "  /remember     save a durable fact to memory: /remember <fact>  (·/remember global for user-wide)",
                                            "  /compact      summarize the session now to reclaim context budget",
                                            "  /undo         revert the last file write a tool made",
                                            "  /open         reopen the last rendered page in the browser: /open [title]",
                                            "  /suggest      toggle ghost next-prompt suggestions (Tab accepts)",
                                            "  /imageQ       set generated-image quality: /imageQ low | medium | high",
                                            "  /clear        clear the screen and scrollback",
                                            "  /help         this list      /quit  exit oxio",
                                        ] {
                                            app.push(line.to_string(), App::dim());
                                        }
                                        app.push(
                                            "keys: Alt+Enter newline · Up/Down history · Ctrl-A/E/U/K/W edit · Esc interrupt · Tab accept suggestion",
                                            App::dim(),
                                        );
                                        app.push("source & issues: github.com/limkcreply/oxio", App::dim());
                                    }
                                    "undo" => {
                                        let m = snapshots.undo().unwrap_or_else(|| "(nothing to undo)".into());
                                        app.push(m, App::dim());
                                    }
                                    "compact" => {
                                        if session.is_empty() {
                                            app.push("(nothing to compact)", App::dim());
                                        } else {
                                            let n = session.len();
                                            match compactor.compact_history(session.snapshot()).await {
                                                Ok(c) => {
                                                    session.replace(c);
                                                    app.push(format!("compacted session ({n} messages)"), App::dim());
                                                }
                                                Err(e) => app.push(format!("[compact failed: {e}]"), App::dim()),
                                            }
                                        }
                                    }
                                    "open" => {
                                        // Local mirror of a reopen-last-artifact keybind: re-open a
                                        // render_page artifact in the browser (no re-render). Bare
                                        // `/open` reopens the most recent; `/open <title>` matches one.
                                        let arg = cmd.split_once(char::is_whitespace).map(|(_, r)| r.trim().to_lowercase()).unwrap_or_default();
                                        let target = if arg.is_empty() {
                                            tools::latest_artifact()
                                        } else {
                                            tools::artifact_list().into_iter().find(|(t, _)| t.contains(&arg))
                                        };
                                        match target {
                                            Some((title, url)) => {
                                                tools::open_url(&url);
                                                app.push(format!("opened {title} → {url}"), App::dim());
                                            }
                                            None => {
                                                let list = tools::artifact_list();
                                                if list.is_empty() {
                                                    app.push("no local page is open - render one first, then /open reopens it", App::dim());
                                                } else {
                                                    let names: Vec<String> = list.iter().map(|(t, _)| t.clone()).collect();
                                                    app.push(format!("open pages: {} · /open <title>", names.join(", ")), App::dim());
                                                }
                                            }
                                        }
                                    }
                                    other => app.push(format!("unknown command: /{other} (try /help)"), App::dim()),
                                }
                                continue;
                            }
                            app.input.clear();
                            app.cursor = 0;
                            let (rx, tok) = start_turn(app, text);
                            sink_rx = rx;
                            cancel = Some(tok);
                        }
                        // Answering an ask_user_question with options: ↑/↓ cycle the choices
                        // into the composer (Enter accepts, or type free-text for "Other").
                        (KeyCode::Up, _) if app.pending_ask.is_some() && !app.pending_ask_options.is_empty() => {
                            let n = app.pending_ask_options.len();
                            let sel = match app.pending_ask_sel { None => n - 1, Some(0) => 0, Some(i) => i - 1 };
                            app.pending_ask_sel = Some(sel);
                            app.input = app.pending_ask_options[sel].clone();
                            app.cursor = app.input.chars().count();
                        }
                        (KeyCode::Down, _) if app.pending_ask.is_some() && !app.pending_ask_options.is_empty() => {
                            let n = app.pending_ask_options.len();
                            let sel = match app.pending_ask_sel { None => 0, Some(i) if i + 1 < n => i + 1, Some(i) => i };
                            app.pending_ask_sel = Some(sel);
                            app.input = app.pending_ask_options[sel].clone();
                            app.cursor = app.input.chars().count();
                        }
                        (KeyCode::Up, _) if app.input.is_empty() && !app.queued.is_empty() => {
                            // Recall the last latched (queued) message into the composer to
                            // edit it or hold it back.
                            if let Some(last) = app.queued.pop_back() {
                                app.input = last;
                                app.cursor = app.input.len();
                            }
                        }
                        (KeyCode::Up, _) if !app.history.is_empty() => {
                            let i = match app.hist_idx {
                                Some(0) => 0,
                                Some(i) => i - 1,
                                None => {
                                    // Entering history: stash the unsent draft so Down restores it.
                                    app.draft = Some(app.input.clone());
                                    app.history.len() - 1
                                }
                            };
                            app.hist_idx = Some(i);
                            app.input = app.history[i].clone();
                            app.cursor = app.input.len();
                        }
                        (KeyCode::Down, _) => match app.hist_idx {
                            Some(i) if i + 1 < app.history.len() => {
                                app.hist_idx = Some(i + 1);
                                app.input = app.history[i + 1].clone();
                                app.cursor = app.input.len();
                            }
                            Some(_) => {
                                // Past the newest entry → restore the stashed draft, not blank.
                                app.hist_idx = None;
                                app.input = app.draft.take().unwrap_or_default();
                                app.cursor = app.input.len();
                            }
                            None => {}
                        },
                        (KeyCode::Left, _) => { app.cursor = prev_boundary(&app.input, app.cursor); }
                        (KeyCode::Right, _) => { app.cursor = next_boundary(&app.input, app.cursor); }
                        (KeyCode::Home, _) | (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
                            app.cursor = 0;
                        }
                        (KeyCode::End, _) | (KeyCode::Char('e'), KeyModifiers::CONTROL) => {
                            app.cursor = app.input.len();
                        }
                        (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                            app.input.replace_range(..app.cursor, ""); // clear to line start
                            app.cursor = 0;
                        }
                        (KeyCode::Char('k'), KeyModifiers::CONTROL) => {
                            app.input.truncate(app.cursor); // clear to line end
                        }
                        (KeyCode::Char('w'), KeyModifiers::CONTROL) if app.cursor > 0 => {
                            let p = prev_word(&app.input, app.cursor); // delete previous word
                            app.input.replace_range(p..app.cursor, "");
                            app.cursor = p;
                        }
                        (KeyCode::Backspace, _) if app.cursor > 0 => {
                            app.ghost = None; // editing dismisses the suggestion
                            let prev = prev_boundary(&app.input, app.cursor);
                            app.input.replace_range(prev..app.cursor, "");
                            app.cursor = prev;
                        }
                        (KeyCode::Char(ch), _) => {
                            app.ghost = None; // typing dismisses the suggestion
                            app.input.insert(app.cursor, ch);
                            app.cursor += ch.len_utf8();
                        }
                        _ => {}
                    }
                } else if maybe_key.is_none() {
                    app.quit = true;
                }
            }
            Some(ev) = sink_rx.recv() => {
                match ev {
                    StreamEvent::TextDelta(t) => {
                        app.out_chars += t.len();
                        app.live.push_str(&t);
                        // Flush completed lines to scrollback; keep the trailing partial.
                        while let Some(nl) = app.live.find('\n') {
                            let line = app.live[..nl].to_string();
                            app.push(line, Style::default());
                            app.live.replace_range(..=nl, "");
                        }
                    }
                    StreamEvent::ThinkingDelta(t) => {
                        app.think.push_str(&t);
                        while let Some(nl) = app.think.find('\n') {
                            let line = app.think[..nl].to_string();
                            app.push(line, App::dim());
                            app.think.replace_range(..=nl, "");
                        }
                    }
                    StreamEvent::Notice { level, text } => {
                        // Count tool dispatches (kernel emits "→ <tool> …") for the summary.
                        if text.starts_with('\u{2192}') {
                            app.tool_calls += 1;
                        }
                        // Bold ONLY tool labels (→ dispatch, ✓/✗ result) - they're the
                        // scannable anchors. Everything else stays plain/dim (no bold fatigue).
                        let t = text.trim_start();
                        let is_tool_label = text.starts_with('\u{2192}')
                            || t.starts_with('\u{2713}')
                            || t.starts_with('\u{2717}');
                        let style = if is_tool_label {
                            Style::default().add_modifier(Modifier::BOLD)
                        } else if matches!(level, NoticeLevel::Warn) {
                            Style::default().fg(Color::Yellow)
                        } else {
                            App::dim()
                        };
                        app.push(text, style);
                    }
                    StreamEvent::Done { usage, .. } => {
                        // A model STEP finished: bank usage and commit the streamed text
                        // to the transcript, but keep `working` LIT - the turn may still
                        // run tool calls / more model steps. Settling happens on done_rx.
                        app.total_in += usage.input_tokens;
                        app.total_out += usage.output_tokens;
                        app.commit_stream();
                    }
                    _ => {}
                }
            }
            Some(()) = done_rx.recv() => {
                // The WHOLE turn finished (all steps + tool calls). Now settle: fold any
                // trailing text, stop the indicator, refresh status, idle the tab, bell.
                app.commit_stream();
                app.working = None;
                app.out_chars = 0;
                cancel = None;
                app.ctx_used = context::estimate_tokens(&session.snapshot());
                app.git_branch = git_branch(); // refresh once per turn (cached, not per-frame)
                app.project = git_root();
                // Title returns to idle via the loop-top sync (working → false). Ring the
                // bell through the backend (single writer) if the user has switched away.
                if !focused {
                    let _ = crossterm::execute!(term.backend_mut(), crossterm::style::Print("\u{7}"));
                }
                // Type-ahead: auto-send the next message queued during the turn.
                if let Some(next) = app.queued.pop_front() {
                    let (rx, tok) = start_turn(app, next);
                    sink_rx = rx;
                    cancel = Some(tok);
                } else if app.suggest {
                    // Ghost next-prompt suggestion: predict off the UI thread with the
                    // primary model, then park it in app.ghost via ghost_tx (Tab accepts).
                    if let Some((prov, model)) = &suggest_provider {
                        let convo = recent_convo(&session.snapshot());
                        if !convo.is_empty() {
                            let (prov, model, gtx) = (prov.clone(), model.clone(), ghost_tx.clone());
                            tokio::spawn(async move {
                                let ctx = Ctx::default();
                                if let Some(s) = crate::suggest_next(&prov, &model, &convo, &ctx).await {
                                    let _ = gtx.send(s);
                                }
                            });
                        }
                    }
                }
            }
            Some(g) = ghost_rx.recv() => {
                // Only show the suggestion if the user hasn't started typing in the meantime.
                if app.input.is_empty() && app.working.is_none() {
                    app.ghost = Some(g);
                }
            }
            Some((req, resp)) = appr_rx.recv() => {
                // Show a colored diff preview of the pending edit so the user SEES which
                // lines change (red removed, green added) before allowing - not blind.
                let diff = tools::diff_preview(&req.tool, &req.input);
                if !diff.is_empty() {
                    app.push(String::new(), Style::default());
                    let shown = diff.len().min(60);
                    for (kind, line) in diff.iter().take(shown) {
                        let style = match kind {
                            '-' => Style::default().fg(Color::Red),
                            '+' => Style::default().fg(Color::Green),
                            '#' => Style::default().add_modifier(Modifier::BOLD),
                            _ => App::dim(),
                        };
                        let prefix = match kind {
                            '-' | '+' => *kind,
                            _ => ' ',
                        };
                        app.push(format!("{prefix} {line}"), style);
                    }
                    if diff.len() > shown {
                        app.push(format!("  … {} more lines", diff.len() - shown), App::dim());
                    }
                }
                app.pending = Some(resp);
                app.pending_summary = req.summary;
            }
            // ask_user_question: the tool routes its question here (never a blocking stdin
            // read). Show the question + any options; the next composer submit answers it.
            Some(req) = ask_rx.recv() => {
                app.push(String::new(), Style::default());
                app.push(format!("  ? {}", req.question), Style::default().add_modifier(Modifier::BOLD));
                for (i, o) in req.options.iter().enumerate() {
                    app.push(format!("    {}. {o}", i + 1), App::dim());
                }
                let nav = if req.options.is_empty() { "" } else { "↑/↓ pick · " };
                app.push(format!("  {nav}type your answer and press Enter · Esc to skip"), App::dim());
                app.pending_ask = Some(req.resp);
                app.pending_ask_options = req.options;
                app.pending_ask_sel = None;
            }
            // Background MCP connect results - stream in as each server resolves, so the
            // input box was usable immediately and status fills in place.
            Some((name, res)) = mcp_rx.recv() => {
                match res {
                    Ok(n) => app.push(format!("  mcp    {name}  ● connected ({n} tool(s))"), App::dim()),
                    Err(e) => app.push(format!("  mcp    {name}  ○ offline - {}", e.lines().next().unwrap_or("unreachable")), App::dim()),
                }
            }
        }
    }
}

/// One frame of the INLINE viewport: in-turn indicator · framed composer · status bar.
/// History is NOT drawn here - it lives in the terminal's native scrollback (flushed
/// via `insert_before` in the event loop), so native scroll + selection work.
fn draw(f: &mut ratatui::Frame, app: &App) {
    let area = f.area(); // the inline viewport (VIEWPORT_H rows)
                         // Composer grows with line count, bounded by the viewport (indicator + status take
                         // one row each). Fixed viewport height is a known limit; dynamic height is a follow-up.
    let input_rows = app.input.split('\n').count().max(1) as u16;
    let max_input = area.height.saturating_sub(2).max(3);
    // While the connect picker is up, the middle region becomes the list panel (windowed),
    // sized to the visible window + borders, bounded by the viewport.
    let input_h = if let Some(p) = &app.pending_pick {
        let win = p.rows.len().clamp(1, 8) as u16;
        (win + 2).clamp(3, max_input)
    } else {
        (input_rows + 2).clamp(3, max_input)
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),       // in-turn indicator / approval / queued
            Constraint::Length(input_h), // framed composer
            Constraint::Length(1),       // status bar
        ])
        .split(area);

    // Indicator: approval prompt > working shimmer (+ queued count) > queued count > blank.
    const WORDS: [&str; 6] = [
        "Working",
        "Thinking",
        "Reasoning",
        "Crunching",
        "Cooking",
        "Churning",
    ];
    let indicator = if app.pending.is_some() {
        // Truncate the summary to the viewport width so the options are NEVER clipped off
        // the right edge (a long command must not hide the y/s/a/n choices). Compact opts.
        let prefix = "⚠ allow ";
        let opts = "  [y/n]";
        let width = area.width as usize;
        let budget = width.saturating_sub(prefix.chars().count() + 1 + opts.chars().count());
        let full = app.pending_summary.chars().count();
        let mut summary: String = app.pending_summary.chars().take(budget.max(6)).collect();
        if full > summary.chars().count() {
            summary.push('…');
        }
        Line::styled(
            format!("{prefix}{summary}?{opts}"),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
    } else if let Some(started) = app.working {
        let sp = SPINNER[app.tick % SPINNER.len()];
        let secs = started.elapsed().as_secs();
        let word = WORDS[(secs as usize / 6) % WORDS.len()]; // cycle every 6s (calm)
        let toks = app.out_chars / 4;
        let mut spans = shimmer_spans(&format!("{sp} {word}…"), app.tick);
        spans.push(Span::styled(
            format!(
                " ({} · ↓{toks} tokens · Esc to interrupt)",
                fmt_elapsed(secs)
            ),
            App::dim(),
        ));
        if !app.queued.is_empty() {
            spans.push(Span::styled(
                format!(" · {} queued", app.queued.len()),
                App::dim(),
            ));
        }
        Line::from(spans)
    } else if !app.queued.is_empty() {
        Line::styled(format!("⏎ {} queued", app.queued.len()), App::dim())
    } else {
        Line::raw("")
    };
    f.render_widget(Paragraph::new(indicator), chunks[0]);

    // Framed composer; empty shows a dim placeholder. First line gets the "❯ " prompt,
    // continuation lines a matching 2-space indent.
    let accent = theme();
    let input_lines: Vec<Line> = if app.input.is_empty() {
        if let Some(g) = &app.ghost {
            // Ghost next-prompt suggestion: dimmed after the prompt, with a Tab hint.
            vec![Line::from(vec![
                Span::styled("❯ ", accent),
                Span::styled(g.clone(), App::dim()),
                Span::styled("  (Tab)", Style::default().fg(Color::DarkGray)),
            ])]
        } else {
            vec![Line::from(vec![
                Span::styled("❯ ", accent),
                Span::styled("send a message…", App::dim()),
            ])]
        }
    } else {
        app.input
            .split('\n')
            .enumerate()
            .map(|(i, l)| {
                let prefix = if i == 0 { "❯ " } else { "  " };
                Line::from(vec![Span::styled(prefix, accent), Span::raw(l.to_string())])
            })
            .collect()
    };
    let input_area = chunks[1];
    if let Some(p) = &app.pending_pick {
        // Connect picker: a windowed, scrolling, highlighted list in place of the composer.
        let win = (input_area.height.saturating_sub(2) as usize).max(1);
        let off = p.off.min(p.sel);
        let off = if p.sel >= off + win {
            p.sel + 1 - win
        } else {
            off
        };
        let mut lines: Vec<Line> = Vec::new();
        for i in 0..win {
            let idx = off + i;
            if idx >= p.rows.len() {
                break;
            }
            let selected = idx == p.sel;
            let marker = if selected { "▸ " } else { "  " };
            let style = if selected {
                user_hl()
            } else {
                Style::default()
            };
            lines.push(Line::styled(format!("{marker}{}", p.rows[idx]), style));
        }
        let more_up = off > 0;
        let more_dn = off + win < p.rows.len();
        let mut title = format!(" {} - ↑/↓ move · enter connect · esc cancel ", p.title);
        if more_up || more_dn {
            let n = if more_dn {
                p.rows.len() - (off + win)
            } else {
                0
            };
            title = format!(
                " {} - ↑/↓ · enter · esc{}{} ",
                p.title,
                if more_up { " · ↑more" } else { "" },
                if more_dn {
                    format!(" · {n}↓")
                } else {
                    String::new()
                }
            );
        }
        f.render_widget(
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title)),
            input_area,
        );
    } else {
        f.render_widget(
            Paragraph::new(input_lines).block(Block::default().borders(Borders::ALL)),
            input_area,
        );
        // Cursor: row = newlines before the caret; col = chars in the current line before it.
        let before = &app.input[..app.cursor.min(app.input.len())];
        let row = before.matches('\n').count() as u16;
        let col = before.rsplit('\n').next().unwrap_or("").chars().count() as u16;
        let max_x = input_area.x + input_area.width.saturating_sub(2);
        let max_y = input_area.y + input_area.height.saturating_sub(2);
        let cx = (input_area.x + 1 + 2 + col).min(max_x);
        let cy = (input_area.y + 1 + row).min(max_y);
        f.set_cursor_position((cx, cy));
    }

    // Status bar.
    f.render_widget(Paragraph::new(status_line(app)), chunks[2]);
}

/// Config-driven status line: expand the user's
/// `[statusline]` template over live telemetry, or a sensible default. NOT hardcoded.
/// Placeholders: {time} {model} {provider} {profile} {cwd} {ctx_pct} {ctx_bar}
/// {in} {out} {cost} {cost_label}.
const DEFAULT_STATUSLINE: &str = "{time} · {model} · {ctx_bar} {ctx_pct} · {cost}";

fn status_line(app: &App) -> Line<'static> {
    let bar_w = 12usize;
    // Window may be unknown (None): show a fresh bar, "?" percent, and used-only count -
    // never a fabricated ceiling or percentage.
    let known = app.ctx_window.filter(|w| *w > 0);
    let pct = known
        .map(|w| app.ctx_used.saturating_mul(100) / w)
        .unwrap_or(0);
    let pct_str = if known.is_some() {
        format!("{pct}%")
    } else {
        "?".to_string()
    };
    let filled = if known.is_some() {
        (pct * bar_w / 100).min(bar_w)
    } else {
        0
    };
    let bar = format!("[{}{}]", "=".repeat(filled), " ".repeat(bar_w - filled));
    let ctx_num = match known {
        Some(w) => format!("{}/{}", fmt_k(app.ctx_used as u64), fmt_k(w as u64)),
        None => format!("{} tok", fmt_k(app.ctx_used as u64)),
    };
    let window_str = match app.ctx_window {
        Some(w) => fmt_k(w as u64),
        None => "unset".to_string(),
    };
    // {cost} is a ready segment: "saved $x" when priced, else the token count.
    let cost = match cost_usd(app.total_in, app.total_out, &app.pricing, &app.model) {
        Some(v) => format!("{} ${v:.4}", app.cost_label),
        None => format!("{} tok", fmt_k(app.total_out)),
    };
    // Default (no custom template): per-field colours - time grey, model white, bar
    // green, ctx% cyan, tokens red, cost mustard, project purple.
    if app.statusline.is_none() {
        let mustard = Color::Rgb(200, 160, 60);
        let purple = Color::Rgb(180, 120, 220);
        let sep = || Span::styled(" · ", App::dim());
        let cost_span = match cost_usd(app.total_in, app.total_out, &app.pricing, &app.model) {
            Some(v) => Span::styled(
                format!("{} ${v:.4}", app.cost_label),
                Style::default().fg(mustard),
            ),
            None => Span::styled(
                format!("{} tok", fmt_k(app.total_out)),
                Style::default().fg(Color::Red),
            ),
        };
        let mut spans = vec![
            Span::raw(" "),
            Span::styled(clock(app.tz_offset), Style::default().fg(Color::DarkGray)),
            sep(),
            Span::styled(app.model.clone(), Style::default().fg(Color::White)),
            sep(),
            Span::styled(bar.clone(), Style::default().fg(Color::Green)),
            Span::raw(" "),
            Span::styled(pct_str.clone(), Style::default().fg(Color::Cyan)),
            sep(),
            Span::styled(
                // Context USED / window - same numerator as the bar/pct above, so they
                // agree. (total_out is output-only and belongs in {cost}, not here.)
                ctx_num.clone(),
                Style::default().fg(Color::Red),
            ),
            sep(),
            cost_span,
        ];
        // Git branch (blue) - only when in a repo; empty otherwise.
        if !app.git_branch.is_empty() {
            spans.push(sep());
            spans.push(Span::styled(
                app.git_branch.clone(),
                Style::default().fg(Color::Blue),
            ));
        }
        spans.push(sep());
        spans.push(Span::styled(
            app.project.clone(),
            Style::default().fg(purple),
        ));
        return Line::from(spans);
    }
    let remaining = if known.is_some() {
        format!("{}%", 100usize.saturating_sub(pct))
    } else {
        "?".to_string()
    };
    let reasoning = if app.reasoning.is_empty() {
        "-"
    } else {
        app.reasoning.as_str()
    };
    let tmpl = app.statusline.as_deref().unwrap_or(DEFAULT_STATUSLINE);
    // Each placeholder is filled from live state; unpriced/absent degrade cleanly.
    let s = tmpl
        .replace("{time}", &clock(app.tz_offset))
        .replace("{model}", &app.model)
        .replace("{reasoning}", reasoning) // reasoning-level
        .replace("{provider}", &app.provider)
        .replace("{profile}", &app.profile) // ≈ permission-profile
        .replace("{cwd}", &app.cwd)
        .replace("{git_branch}", &app.git_branch) // git-branch
        .replace("{project}", &app.project) // project-root
        .replace("{approval}", app.approval) // approval-mode
        .replace("{host}", &app.host) // hostname
        .replace("{session}", &app.session_id) // thread-id
        .replace("{version}", env!("CARGO_PKG_VERSION")) // version
        .replace("{ctx_pct}", &pct_str) // context-used
        .replace("{ctx_remaining}", &remaining) // context-remaining
        .replace("{ctx_window}", &window_str) // context-window
        .replace("{ctx_bar}", &bar)
        .replace("{in}", &fmt_k(app.total_in)) // input-tokens
        .replace("{out}", &fmt_k(app.total_out)) // output-tokens
        .replace("{total}", &fmt_k(app.total_in + app.total_out)) // total-tokens-used
        .replace("{cost}", &cost) // thread dollar cost
        .replace("{cost_label}", app.cost_label);
    Line::styled(format!(" {s} "), App::dim())
}

/// Compact token count: `12k` above 1000, raw below.
fn fmt_k(n: u64) -> String {
    if n >= 1000 {
        format!("{}k", (n + 500) / 1000)
    } else {
        n.to_string()
    }
}

/// Current git branch (cached by the caller - run at startup + on turn-end, never
/// per frame). Empty if not a repo.
fn git_branch() -> String {
    std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Previous UTF-8 char boundary before byte offset `i` (0 if none).
fn prev_boundary(s: &str, i: usize) -> usize {
    s[..i]
        .char_indices()
        .next_back()
        .map(|(b, _)| b)
        .unwrap_or(0)
}

/// Next UTF-8 char boundary after byte offset `i` (unchanged at end).
fn next_boundary(s: &str, i: usize) -> usize {
    s[i..].chars().next().map(|c| i + c.len_utf8()).unwrap_or(i)
}

/// Start of the word before byte offset `i` (skip trailing spaces, then a word).
fn prev_word(s: &str, i: usize) -> usize {
    let trimmed = s[..i].trim_end_matches(' ');
    trimmed.rfind(' ').map(|p| p + 1).unwrap_or(0)
}

/// Git project root (basename), cached by the caller. Empty if not a repo.
fn git_root() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            let full = String::from_utf8_lossy(&o.stdout).trim().to_string();
            full.rsplit('/').next().unwrap_or(&full).to_string()
        })
        .unwrap_or_default()
}

/// Machine hostname (env, else the `hostname` command). Cached by the caller.
fn hostname() -> String {
    if let Ok(h) = std::env::var("HOSTNAME") {
        if !h.is_empty() {
            return h;
        }
    }
    std::process::Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Local timezone offset in seconds, from `date +%z` (e.g. `+1000` → 36000).
/// Computed once at startup; std gives no local offset without a dep.
fn tz_offset_secs() -> i64 {
    let out = std::process::Command::new("date").arg("+%z").output().ok();
    let z = out
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    // z like "+1000" / "-0530"
    if z.len() == 5 {
        if let (Ok(h), Ok(m)) = (z[1..3].parse::<i64>(), z[3..5].parse::<i64>()) {
            let mag = h * 3600 + m * 60;
            return if z.starts_with('-') { -mag } else { mag };
        }
    }
    0
}

/// Local wall-clock HH:MM (real local time, not UTC), applying the cached offset.
fn clock(offset_secs: i64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let utc = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let local = (utc + offset_secs).rem_euclid(86400);
    format!("{:02}:{:02}", local / 3600, (local % 3600) / 60)
}

#[cfg(test)]
mod tests {
    use super::{dropped_image_path, should_collapse_paste, wrap_text};

    #[test]
    fn long_paste_collapses_short_paste_inlines() {
        // Short paste (few lines, small) inserts inline - no chip.
        assert!(!should_collapse_paste("one line"));
        assert!(!should_collapse_paste("a\nb\nc"));
        // Long paste (>= 6 lines OR >= 800 bytes) collapses to a chip.
        assert!(should_collapse_paste("a\nb\nc\nd\ne\nf"));
        assert!(should_collapse_paste(&"x".repeat(900)));
    }

    #[test]
    fn paste_marker_expands_back_to_full_text() {
        // The submit-time expansion is a plain marker→full replace; a chip round-trips.
        let full = "line1\nline2\nline3\nline4\nline5\nline6";
        let marker = "[pasted #1 +6 lines]";
        let composer = format!("check this {marker} please");
        let expanded = composer.replace(marker, full);
        assert!(expanded.contains("line6") && !expanded.contains("[pasted #1"));
    }

    #[test]
    fn wrap_text_produces_exact_bounded_rows() {
        assert_eq!(wrap_text("", 10), vec![""]); // empty → one empty row
        assert_eq!(wrap_text("short", 10), vec!["short"]);
        // greedy word wrap: each row within width
        for row in wrap_text("aa bb cc dd ee", 5) {
            assert!(row.chars().count() <= 5, "row over width: {row:?}");
        }
        // a word longer than width is hard-broken, never dropped
        assert_eq!(wrap_text("abcdefgh", 3), vec!["abc", "def", "gh"]);
    }

    /// Write a real (tiny) file with the given name in a temp dir; return its path.
    fn touch(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("oxio_ddtest_{name}"));
        std::fs::write(&p, b"x").unwrap();
        p
    }

    #[test]
    fn plain_text_is_not_a_path() {
        assert_eq!(dropped_image_path("hello world"), None);
        assert_eq!(dropped_image_path("/help"), None); // a slash command, not an image
    }

    #[test]
    fn existing_image_is_detected() {
        let p = touch("a.png");
        let s = p.to_str().unwrap();
        assert_eq!(dropped_image_path(s).as_deref(), Some(s));
        // Quoted and whitespace-wrapped variants normalize to the same path.
        assert_eq!(dropped_image_path(&format!("\"{s}\"")).as_deref(), Some(s));
        assert_eq!(dropped_image_path(&format!("  {s}  ")).as_deref(), Some(s));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn file_url_and_escaped_spaces() {
        let p = touch("with space.png");
        let s = p.to_str().unwrap();
        // macOS drag-drop escapes spaces as "\ ".
        let escaped = s.replace(' ', "\\ ");
        assert_eq!(dropped_image_path(&escaped).as_deref(), Some(s));
        // file:// URL with %20-encoded space.
        let url = format!("file://{}", s.replace(' ', "%20"));
        assert_eq!(dropped_image_path(&url).as_deref(), Some(s));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn non_image_and_missing_and_multitoken_rejected() {
        let txt = touch("note.txt");
        assert_eq!(dropped_image_path(txt.to_str().unwrap()), None); // wrong extension
        std::fs::remove_file(&txt).ok();
        assert_eq!(dropped_image_path("/no/such/file.png"), None); // does not exist
                                                                   // Two real space-separated tokens is a phrase, not one path.
        let p = touch("b.png");
        let two = format!("{} extra", p.to_str().unwrap());
        assert_eq!(dropped_image_path(&two), None);
        std::fs::remove_file(&p).ok();
    }
}
