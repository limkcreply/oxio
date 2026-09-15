//! Safety rails, built against the frozen core's real seams:
//!
//! - **Write serialization is already free.** The kernel dispatches a message's
//!   tool calls sequentially (kernel.rs `for call in message.tool_calls`), so
//!   writes cannot interleave within a turn. We add no redundant serializer.
//! - **The permission gate is composition, not a kernel hook.** The kernel has
//!   no veto seam in the tool loop (a `PreTool` transformer can mutate state but
//!   cannot stop `tool.call()`). So the gate is a composing `Tool` wrapper
//!   (`Guarded`), the same pattern as `providers::ChainProvider`. It gates only
//!   `Write`-kind tools; `Read` tools pass through at zero cost.
//! - **Loop safety is a cheap deterministic counter.** It aborts fast on
//!   identical repeats with NO model call, complementing the kernel's turn cap
//!   (`max_tool_iterations`). The repeat threshold is labeled below.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use oxio_core::{
    Ctx, Hook, OxioError, Result, Role, Tool, ToolKind, ToolOutput, ToolSpec, Transformer,
    TurnState,
};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Permission gate
// ---------------------------------------------------------------------------

/// What the user (or a headless policy) decides for a gated tool call.
///
/// `Deny` and `Block` differ by WHO refused, which must drive different
/// model-facing behavior: a live human "No" means stop and wait for that human; a policy
/// refusal with no human present means try a reasonable, non-malicious alternative
/// or report the limitation. Collapsing them is the bug that made the model chase
/// workarounds after an interactive "No".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Allow this one call only.
    Once,
    /// Allow this call and every later call of the same tool this session.
    Session,
    /// Allow always: grant for this session AND persist to `.oxio/configs.local.toml`
    /// so it never re-prompts in this project. The first `Always` creates that file.
    Always,
    /// A live human refused this call at the prompt. Stop and wait for them.
    Deny,
    /// A policy refused it and no human is present (headless / rule / non-TTY).
    /// The model may find a reasonable non-malicious alternative or report back.
    Block,
}

/// The action awaiting approval, handed to an [`Approver`].
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub tool: String,
    pub summary: String,
    /// Raw tool input, so the UI can render a diff/preview before the user approves.
    pub input: Value,
}

/// The seam the gate calls before a write/exec tool runs. The UI implements an
/// interactive prompt; headless callers implement a policy (deny / allow).
#[async_trait]
pub trait Approver: Send + Sync {
    async fn approve(&self, req: &ApprovalRequest) -> Decision;
    /// Short label for the active approval mode, shown live in the status line
    /// (derived, never hardcoded by the UI). Override per approver.
    fn mode(&self) -> &'static str {
        "prompt"
    }
}

/// Safe non-interactive default: refuse every gated call. Right for headless /
/// non-TTY runs where there is no user to ask.
pub struct DenyAll;

#[async_trait]
impl Approver for DenyAll {
    async fn approve(&self, _req: &ApprovalRequest) -> Decision {
        // No human present, so this is a policy refusal, not a live "No".
        Decision::Block
    }
    fn mode(&self) -> &'static str {
        "deny"
    }
}

/// Non-interactive allow-all - for tests and an explicit `--yes` policy only.
/// NOT a production default.
pub struct AllowAll;

#[async_trait]
impl Approver for AllowAll {
    async fn approve(&self, _req: &ApprovalRequest) -> Decision {
        Decision::Session
    }
    fn mode(&self) -> &'static str {
        "accept"
    }
}

// Model-facing denial text, split by who refused: a live human "No" (stop and wait)
// vs a policy denial with no human present (try a legitimate alternative or escalate).

/// Shown when a live human refuses the call at the prompt: stop and wait.
const REJECT_MESSAGE: &str = "The user declined this tool call. Nothing was changed (for a file edit, no write happened). Stop and wait for the user's instructions before taking any further action.";

/// The policy-denial rule: try legitimate alternatives, never circumvent, escalate if essential.
const DENIAL_WORKAROUND_GUIDANCE: &str = "You may pursue this goal with other tools that are genuinely suited to it, but do not try to circumvent this denial - do not repurpose unrelated capabilities to get around it. If the blocked capability is truly required, stop and explain to the user what you were trying to do and why you need it, and let the user decide how to proceed.";

/// Policy-denial message naming the blocked tool.
fn auto_reject_message(tool: &str) -> String {
    format!("Permission to use {tool} has been denied. {DENIAL_WORKAROUND_GUIDANCE}")
}

/// Wraps a tool with a permission gate. `Read` tools pass straight through; only
/// `Write`-kind tools are gated, and a `Session` grant is cached so the user is
/// asked once per tool per session, not once per call.
pub struct Guarded {
    inner: Arc<dyn Tool>,
    approver: Arc<dyn Approver>,
}

impl Guarded {
    pub fn new(inner: Arc<dyn Tool>, approver: Arc<dyn Approver>) -> Self {
        Guarded { inner, approver }
    }

    /// Wrap only if the tool writes; read tools are returned unwrapped, so
    /// wrapping the whole built-in set is a zero-cost no-op for read-only tools
    /// and automatically gates write tools when they land.
    pub fn wrap(inner: Arc<dyn Tool>, approver: Arc<dyn Approver>) -> Arc<dyn Tool> {
        if inner.kind() == ToolKind::Write {
            Arc::new(Guarded::new(inner, approver))
        } else {
            inner
        }
    }
}

#[async_trait]
impl Tool for Guarded {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }
    fn kind(&self) -> ToolKind {
        self.inner.kind()
    }
    fn summarize(&self, input: &Value) -> String {
        self.inner.summarize(input)
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> Result<ToolOutput> {
        let name = self.inner.spec().name;
        // EVERY write asks - no blanket grant suppresses it. Create/edit/delete, our own
        // artifact or the user's file: all identical, all prompt. (A cached "always allow"
        // is exactly what silently let an unexpected write through, so it is gone.)
        let req = ApprovalRequest {
            tool: name.clone(),
            // The tool's own human-readable summary (e.g. "Run cargo build"), same as the
            // transcript label - not a raw-JSON dump.
            summary: self.inner.summarize(&input),
            input: input.clone(),
        };
        match self.approver.approve(&req).await {
            // Live human "No": stop and wait.
            Decision::Deny => return Ok(ToolOutput::error(REJECT_MESSAGE.to_string())),
            // Policy denial, no human present: try a legitimate alternative or escalate.
            Decision::Block => return Ok(ToolOutput::error(auto_reject_message(&name))),
            // Any yes allows THIS call only; the next write asks again.
            Decision::Once | Decision::Session | Decision::Always => {}
        }
        self.inner.call(input, ctx).await
    }
}

// ---------------------------------------------------------------------------
// Checkpoint / undo - snapshot files before a write tool runs, restore on /undo.
// ---------------------------------------------------------------------------

/// One checkpoint: the files a single write tool call was about to change, with
/// their PRIOR content (`None` = the file did not exist, so undo deletes it).
struct Checkpoint {
    label: String,
    files: Vec<(PathBuf, Option<String>)>,
}

/// LIFO stack of checkpoints. `/undo` pops the most recent and restores it. Only
/// touches files oxio itself snapshotted - never guesses at other files.
#[derive(Default)]
pub struct SnapshotStore {
    stack: Mutex<Vec<Checkpoint>>,
}

impl SnapshotStore {
    pub fn new() -> Arc<Self> {
        Arc::new(SnapshotStore::default())
    }

    pub fn is_empty(&self) -> bool {
        self.stack.lock().unwrap().is_empty()
    }

    fn record(&self, label: String, files: Vec<(PathBuf, Option<String>)>) {
        if !files.is_empty() {
            self.stack.lock().unwrap().push(Checkpoint { label, files });
        }
    }

    /// Restore the most recent checkpoint: rewrite prior content, or delete files
    /// that did not exist before. Returns a human report, or `None` if empty.
    pub fn undo(&self) -> Option<String> {
        let cp = self.stack.lock().unwrap().pop()?;
        let mut restored = 0usize;
        for (path, prior) in &cp.files {
            let ok = match prior {
                Some(content) => std::fs::write(path, content).is_ok(),
                // Only delete files WE recorded as newly created (prior = None).
                None => std::fs::remove_file(path).is_ok() || !path.exists(),
            };
            if ok {
                restored += 1;
            }
        }
        Some(format!(
            "undid {} - {}/{} file(s) restored",
            cp.label,
            restored,
            cp.files.len()
        ))
    }
}

/// The paths a write tool is about to change, extracted from its input. Unknown
/// write tools return empty (undo simply won't cover them - never a wrong guess).
fn affected_paths(tool: &str, input: &Value) -> Vec<PathBuf> {
    match tool {
        "write_file" => input
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| vec![PathBuf::from(p)])
            .unwrap_or_default(),
        "apply_patch" => input
            .get("patch")
            .and_then(|v| v.as_str())
            .map(patch_paths)
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Parse the V4A patch envelope for every file path it touches (Add/Delete/Update,
/// plus a rename destination).
fn patch_paths(patch: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for line in patch.lines() {
        let l = line.trim();
        for marker in [
            "*** Add File: ",
            "*** Delete File: ",
            "*** Update File: ",
            "*** Move to: ",
        ] {
            if let Some(p) = l.strip_prefix(marker) {
                paths.push(PathBuf::from(p.trim()));
            }
        }
    }
    paths
}

/// Wraps a write tool so it snapshots the files it changes BEFORE the write, and
/// keeps the checkpoint only when the write succeeds. Read tools pass through.
pub struct Checkpointing {
    inner: Arc<dyn Tool>,
    store: Arc<SnapshotStore>,
}

impl Checkpointing {
    pub fn wrap(inner: Arc<dyn Tool>, store: Arc<SnapshotStore>) -> Arc<dyn Tool> {
        if inner.kind() == ToolKind::Write {
            Arc::new(Checkpointing { inner, store })
        } else {
            inner
        }
    }
}

#[async_trait]
impl Tool for Checkpointing {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }
    fn kind(&self) -> ToolKind {
        self.inner.kind()
    }
    fn summarize(&self, input: &Value) -> String {
        self.inner.summarize(input)
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> Result<ToolOutput> {
        let name = self.inner.spec().name;
        // Capture prior state BEFORE the write. Skip files that exist but are not
        // readable as text (binary) - safer to not cover them than to guess.
        let mut priors: Vec<(PathBuf, Option<String>)> = Vec::new();
        for p in affected_paths(&name, &input) {
            if p.exists() {
                if let Ok(content) = std::fs::read_to_string(&p) {
                    priors.push((p, Some(content)));
                }
            } else {
                priors.push((p, None));
            }
        }
        let out = self.inner.call(input, ctx).await?;
        if !out.is_error {
            self.store.record(name, priors);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Loop guard
// ---------------------------------------------------------------------------

/// Default repeat threshold. OURS, not a Gemini constant: high enough that a
/// couple of legitimately-identical steps do not trip it, well below the
/// kernel's `max_tool_iterations` cap so it still saves wasted round-trips.
pub const DEFAULT_LOOP_THRESHOLD: usize = 5;

#[derive(Default)]
struct LoopState {
    last: Option<String>,
    count: usize,
}

/// Deterministic loop guard: if the model's latest step (its tool calls, or its
/// text) is byte-identical to the previous step `threshold` times in a row, the
/// turn is aborted with an error. No model call. Register at BOTH `PreTurn`
/// (resets per turn) and `PostModel` (checks each model step).
pub struct LoopGuard {
    threshold: usize,
    state: Mutex<LoopState>,
}

impl LoopGuard {
    pub fn new(threshold: usize) -> Self {
        LoopGuard {
            threshold: threshold.max(2),
            state: Mutex::new(LoopState::default()),
        }
    }
}

impl Default for LoopGuard {
    fn default() -> Self {
        LoopGuard::new(DEFAULT_LOOP_THRESHOLD)
    }
}

/// The signature of the model's latest step, or `None` if there is nothing to
/// compare (e.g. the last message is not the assistant's, or is empty text).
fn signature(state: &TurnState) -> Option<String> {
    let m = state.messages.last()?;
    if m.role != Role::Assistant {
        return None;
    }
    if !m.tool_calls.is_empty() {
        let mut s = String::from("calls:");
        for c in &m.tool_calls {
            s.push_str(&c.name);
            s.push('(');
            s.push_str(&serde_json::to_string(&c.arguments).unwrap_or_default());
            s.push(')');
        }
        Some(s)
    } else {
        let t = m.as_text();
        if t.trim().is_empty() {
            None
        } else {
            Some(format!("text:{t}"))
        }
    }
}

#[async_trait]
impl Transformer for LoopGuard {
    async fn transform(&self, hook: Hook, state: &mut TurnState) -> Result<()> {
        if hook == Hook::PreTurn {
            let mut st = self.state.lock().unwrap();
            st.last = None;
            st.count = 0;
            return Ok(());
        }
        // PostModel (or any check hook): update the repeat counter.
        let sig = match signature(state) {
            Some(s) => s,
            None => return Ok(()),
        };
        let mut st = self.state.lock().unwrap();
        if st.last.as_deref() == Some(sig.as_str()) {
            st.count += 1;
        } else {
            st.last = Some(sig);
            st.count = 1;
        }
        if st.count >= self.threshold {
            let n = st.count;
            return Err(OxioError::Other(format!(
                "loop detected: the model repeated the same step {n} times; aborting turn"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxio_core::{Message, ToolCall, ToolKind, TurnState};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // --- test doubles ---

    struct DummyTool {
        name: &'static str,
        kind: ToolKind,
        runs: AtomicUsize,
    }
    impl DummyTool {
        fn new(name: &'static str, kind: ToolKind) -> Arc<Self> {
            Arc::new(DummyTool {
                name,
                kind,
                runs: AtomicUsize::new(0),
            })
        }
    }
    #[async_trait]
    impl Tool for DummyTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.into(),
                description: "d".into(),
                input_schema: json!({}),
            }
        }
        fn kind(&self) -> ToolKind {
            self.kind
        }
        async fn call(&self, _input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput::ok("ran"))
        }
    }

    /// Counts how many times it was asked; always grants for the session.
    struct CountingApprover {
        asked: AtomicUsize,
    }
    #[async_trait]
    impl Approver for CountingApprover {
        async fn approve(&self, _req: &ApprovalRequest) -> Decision {
            self.asked.fetch_add(1, Ordering::SeqCst);
            Decision::Session
        }
    }

    fn assistant_calls(name: &str, args: Value) -> TurnState {
        let mut msg = Message::text(Role::Assistant, "");
        msg.content.clear();
        msg.tool_calls = vec![ToolCall {
            id: "1".into(),
            name: name.into(),
            arguments: args,
        }];
        TurnState {
            input: String::new(),
            model: String::new(),
            messages: vec![msg],
            tools: vec![],
            params: Default::default(),
            usage: Default::default(),
            last_error: None,
            notices: Vec::new(),
        }
    }

    // --- permission gate ---

    #[tokio::test]
    async fn read_tool_passes_through_ungated() {
        let read = DummyTool::new("read", ToolKind::Read);
        // wrap() must return the SAME tool (unwrapped) for a read tool.
        let wrapped = Guarded::wrap(read.clone(), Arc::new(DenyAll));
        let out = wrapped.call(json!({}), &Ctx::default()).await.unwrap();
        assert!(!out.is_error, "read tool runs even under DenyAll");
        assert_eq!(read.runs.load(Ordering::SeqCst), 1);
    }

    /// Refuses as a live human would at the prompt.
    struct DenyLive;
    #[async_trait]
    impl Approver for DenyLive {
        async fn approve(&self, _req: &ApprovalRequest) -> Decision {
            Decision::Deny
        }
    }

    #[tokio::test]
    async fn policy_block_does_not_run_and_allows_reasonable_alternative() {
        // DenyAll = no human present = policy Block: names the tool and the anti-circumvention rule.
        let write = DummyTool::new("write", ToolKind::Write);
        let wrapped = Guarded::wrap(write.clone(), Arc::new(DenyAll));
        let out = wrapped.call(json!({}), &Ctx::default()).await.unwrap();
        assert!(out.is_error, "blocked write returns an error result");
        assert!(
            out.content
                .contains("Permission to use write has been denied"),
            "names the tool, policy denial"
        );
        assert!(
            out.content.contains("do not try to circumvent this denial"),
            "keeps the anti-circumvention rule"
        );
        assert_eq!(write.runs.load(Ordering::SeqCst), 0, "inner never executed");
    }

    #[tokio::test]
    async fn live_no_stops_and_waits_no_workaround() {
        // A human typed No at the prompt: stop and wait, no workaround.
        let write = DummyTool::new("write", ToolKind::Write);
        let wrapped = Guarded::wrap(write.clone(), Arc::new(DenyLive));
        let out = wrapped.call(json!({}), &Ctx::default()).await.unwrap();
        assert!(out.is_error, "declined write returns an error result");
        assert!(
            out.content.contains("The user declined this tool call"),
            "reject text"
        );
        assert!(
            out.content
                .contains("Stop and wait for the user's instructions"),
            "stop + wait, not a menu"
        );
        assert_eq!(write.runs.load(Ordering::SeqCst), 0, "inner never executed");
    }

    #[tokio::test]
    async fn checkpoint_undo_restores_prior_content() {
        let dir = std::env::temp_dir().join(format!("oxio-undo-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("f.txt");
        std::fs::write(&file, "original").unwrap();

        let store = SnapshotStore::new();
        let wrapped =
            Checkpointing::wrap(DummyTool::new("write_file", ToolKind::Write), store.clone());
        // The wrapped call snapshots the target's prior content before the write.
        let _ = wrapped
            .call(
                json!({ "path": file.to_str().unwrap(), "content": "new" }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        std::fs::write(&file, "modified").unwrap(); // simulate the edit landing
        assert!(store.undo().is_some(), "undo reports a restore");
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "original",
            "prior content restored"
        );
        let _ = std::fs::remove_file(&file);
    }

    #[tokio::test]
    async fn checkpoint_undo_deletes_newly_created_file() {
        let dir = std::env::temp_dir().join(format!("oxio-undo2-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("new.txt");
        let _ = std::fs::remove_file(&file);

        let store = SnapshotStore::new();
        let wrapped =
            Checkpointing::wrap(DummyTool::new("write_file", ToolKind::Write), store.clone());
        let _ = wrapped
            .call(
                json!({ "path": file.to_str().unwrap(), "content": "x" }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        std::fs::write(&file, "created").unwrap(); // the write created it
        store.undo();
        assert!(
            !file.exists(),
            "undo deletes a file that did not exist before the write"
        );
    }

    #[tokio::test]
    async fn write_tool_asks_every_time() {
        let write = DummyTool::new("write", ToolKind::Write);
        let approver = Arc::new(CountingApprover {
            asked: AtomicUsize::new(0),
        });
        let wrapped = Guarded::wrap(write.clone(), approver.clone());
        for _ in 0..3 {
            let out = wrapped.call(json!({}), &Ctx::default()).await.unwrap();
            assert!(!out.is_error);
        }
        // Every write asks - no blanket grant caches it (the must-ask rule).
        assert_eq!(
            approver.asked.load(Ordering::SeqCst),
            3,
            "asked on every write, no caching"
        );
        assert_eq!(write.runs.load(Ordering::SeqCst), 3, "ran all three times");
    }

    // --- loop guard ---

    #[tokio::test]
    async fn loopguard_aborts_after_threshold_identical_steps() {
        let g = LoopGuard::new(3);
        let mut st = assistant_calls("list_dir", json!({"path": "."}));
        // steps 1 and 2 are fine, step 3 trips the threshold.
        assert!(g.transform(Hook::PostModel, &mut st).await.is_ok());
        assert!(g.transform(Hook::PostModel, &mut st).await.is_ok());
        let err = g.transform(Hook::PostModel, &mut st).await.unwrap_err();
        assert!(matches!(err, OxioError::Other(m) if m.contains("loop detected")));
    }

    #[tokio::test]
    async fn loopguard_ignores_varied_steps() {
        let g = LoopGuard::new(3);
        for i in 0..6 {
            let mut st = assistant_calls("list_dir", json!({ "path": format!("dir{i}") }));
            assert!(
                g.transform(Hook::PostModel, &mut st).await.is_ok(),
                "varied args never loop"
            );
        }
    }

    #[tokio::test]
    async fn loopguard_resets_on_preturn() {
        let g = LoopGuard::new(3);
        let mut st = assistant_calls("list_dir", json!({"path": "."}));
        assert!(g.transform(Hook::PostModel, &mut st).await.is_ok());
        assert!(g.transform(Hook::PostModel, &mut st).await.is_ok());
        // A new turn resets the counter, so the same step is fine again.
        let mut pre = st.clone();
        assert!(g.transform(Hook::PreTurn, &mut pre).await.is_ok());
        assert!(
            g.transform(Hook::PostModel, &mut st).await.is_ok(),
            "counter reset by PreTurn"
        );
        assert!(g.transform(Hook::PostModel, &mut st).await.is_ok());
        let err = g.transform(Hook::PostModel, &mut st).await.unwrap_err();
        assert!(matches!(err, OxioError::Other(m) if m.contains("loop detected")));
    }
}
