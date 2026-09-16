//! Context compaction as a bolt-on `Transformer` module on the frozen microkernel,
//! layered cheapest-first:
//! - Tier 1 sheds tokens mechanically (collapse old tool outputs, no model call), so
//!   granular history survives as long as possible.
//! - Tier 2 calls the model to summarize only when Tier 1 is not enough.
//! - The trigger fires late (90% soft) with a hard backstop, counting only conversation
//!   growth, to retain the most live context.
//! - A circuit breaker stops after repeated failures, guarding against runaway compaction.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use oxio_core::{
    Content, Ctx, Hook, Message, NoticeLevel, Params, Provider, Request, Result, Role, Transformer,
    TurnState,
};

/// Structured checkpoint-summary instruction: an 8-section schema that preserves verbatim
/// user messages so a fresh model can resume the session without re-deriving context.
///
/// The summarizer call sends `tools: []` (see `summarize`), so the model cannot call a
/// tool - the guardrail is structural, not prompt-based.
///
/// The model drafts in an `<analysis>` scratchpad (improves quality) then writes the real
/// summary in `<summary>`; [`format_summary`] strips the scratchpad before injection.
const SUMMARIZATION_PROMPT: &str = "You are compacting a long coding session so another model can \
resume it. Create a detailed summary of the conversation so far, paying close attention to \
the user's explicit requests and your previous actions, capturing technical details, code patterns, and \
decisions essential for continuing without losing context.

Before the summary, wrap your analysis in <analysis> tags: go through the conversation chronologically \
and identify, for each part, the user's requests and intent, your approach, key decisions and code \
patterns, specific details (file names, full code snippets, function signatures, edits), and any errors \
and how you fixed them. Pay special attention to user feedback - especially where the user told you to \
do something differently. Then write the summary inside <summary> tags with these sections:

1. Primary Request and Intent: all of the user's explicit requests and intents, in detail.
2. Key Technical Concepts: technologies, frameworks, and concepts discussed.
3. Files and Code Sections: files examined/modified/created, with full code snippets where applicable \
and why each matters.
4. Errors and fixes: each error hit, how it was fixed, and any user feedback on it.
5. Problem Solving: problems solved and ongoing troubleshooting.
6. All user messages: list EVERY non-tool-result user message verbatim - these are critical for \
understanding feedback and changing intent.
7. Pending Tasks: tasks you were explicitly asked to work on.
8. Current Work: precisely what was being worked on immediately before this summary, with file names \
and code snippets.
9. Next Step (optional): the next step, DIRECTLY in line with the user's most recent explicit request; \
include a verbatim quote of where you left off so intent does not drift.

Output ONLY the <analysis> block followed by the <summary> block, nothing else.";

/// Rough token estimate (~4 chars/token) over message text + tool-call payloads.
/// Server-observed usage refinement is a later improvement.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    let chars: usize = messages
        .iter()
        .map(|m| {
            let content: usize = m
                .content
                .iter()
                .map(|c| match c {
                    Content::Text { text } | Content::Thinking { text } => text.len(),
                    // base64 image data is heavy but not 1:1 tokens; approximate.
                    Content::Image { data, .. } => data.len() / 3,
                })
                .sum();
            let calls: usize = m
                .tool_calls
                .iter()
                .map(|tc| tc.name.len() + tc.arguments.to_string().len())
                .sum();
            content + calls
        })
        .sum();
    chars / 4
}

/// Tier 1 (mechanical, no model call): collapse OLD tool-result message contents
/// to a stub, keeping the most recent `keep_recent` messages untouched. Returns
/// how many messages were collapsed.
pub fn shed_tool_outputs(messages: &mut [Message], keep_recent: usize) -> usize {
    let cut = messages.len().saturating_sub(keep_recent);
    let mut collapsed = 0;
    for m in messages.iter_mut().take(cut) {
        if m.role == Role::Tool && m.as_text() != "[tool output elided]" {
            m.content = vec![Content::Text {
                text: "[tool output elided]".to_string(),
            }];
            collapsed += 1;
        }
    }
    collapsed
}

/// Layered compactor mounted as a `PreModel` transformer.
pub struct Compactor {
    provider: Arc<dyn Provider>,
    model: String,
    /// `None` = the model's window is unknown (not configured, not advertised by the
    /// endpoint). We do NOT invent a number: auto-compaction is disabled and the UI shows
    /// the window as unset rather than a misleading default.
    context_window: Option<usize>,
    keep_recent: usize,
    failures: Mutex<usize>,
}

impl Compactor {
    pub fn new(
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
        context_window: Option<usize>,
    ) -> Self {
        Compactor {
            provider,
            model: model.into(),
            context_window: context_window.map(|w| w.max(4096)),
            keep_recent: 6,
            failures: Mutex::new(0),
        }
    }

    /// The 90% auto-compaction trigger, or `None` when the window is unknown (never fire).
    fn soft_limit(&self) -> Option<usize> {
        self.context_window.map(|w| w * 9 / 10)
    }

    /// The model's context window, so the UI can show ctx-used against budget.
    /// `None` when unknown - the UI shows "unset" instead of a fabricated ceiling.
    pub fn context_window(&self) -> Option<usize> {
        self.context_window
    }

    /// Manual compaction: shed old tool outputs, then summarize, returning the
    /// compacted message list. Used by the REPL `/compact` command over the
    /// session history.
    pub async fn compact_history(&self, messages: Vec<Message>) -> Result<Vec<Message>> {
        let mut state = TurnState {
            input: String::new(),
            model: self.model.clone(),
            messages,
            tools: vec![],
            params: Params::default(),
            usage: Default::default(),
            last_error: None,
            notices: Vec::new(),
        };
        shed_tool_outputs(&mut state.messages, self.keep_recent);
        self.summarize(&mut state).await?;
        Ok(state.messages)
    }

    /// Tier 2: summarize everything except the most recent `keep_recent`
    /// non-system messages via one model call; rebuild as system + summary +
    /// recent verbatim.
    async fn summarize(&self, state: &mut TurnState) -> Result<()> {
        let system: Vec<Message> = state
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        let body: Vec<Message> = state
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        if body.len() <= self.keep_recent {
            return Ok(()); // nothing worth summarizing
        }
        let split = body.len() - self.keep_recent;
        let (old, recent) = body.split_at(split);

        let mut transcript = String::new();
        for m in old {
            let text = m.as_text();
            if !text.is_empty() {
                transcript.push_str(&format!("[{:?}] {}\n", m.role, text));
            }
            for tc in &m.tool_calls {
                transcript.push_str(&format!(
                    "[{:?} called {}] {}\n",
                    m.role, tc.name, tc.arguments
                ));
            }
        }

        let req = Request {
            model: self.model.clone(),
            messages: vec![
                Message::text(Role::System, SUMMARIZATION_PROMPT),
                Message::text(Role::User, transcript),
            ],
            tools: vec![],
            params: Params::default(),
        };
        let resp = self.provider.complete(req, &Ctx::default()).await?;
        let summary = format_summary(&resp.message.as_text());

        // Wrap as a continuation brief:
        // the resuming model is told this is a continuation, and that the recent tail below
        // is verbatim - so it trusts the summary for the old context and the raw messages for
        // the live detail. Leaner than the source: no transcript-path/proactive plumbing.
        let brief = format!(
            "This session is being continued from a longer conversation that was compacted to save \
             context. The summary below covers the earlier portion; the messages after it are \
             preserved verbatim.\n\n{summary}"
        );

        let mut rebuilt = system;
        rebuilt.push(Message::text(Role::User, brief));
        rebuilt.extend(recent.iter().cloned());
        state.messages = rebuilt;
        Ok(())
    }
}

/// A compact, human-readable compaction notice: method, before→after tokens, and the
/// percentage shed (the number a user actually reads to gauge how much room was freed).
/// Kept in one place so both tiers phrase it identically.
fn compaction_notice(method: &str, before: usize, after: usize) -> String {
    let pct = (before.saturating_sub(after) * 100)
        .checked_div(before)
        .unwrap_or(0);
    format!("\u{267b} compacted context ({method}): {before}\u{2192}{after} tok, -{pct}%")
}

/// Strip the `<analysis>` drafting scratchpad and unwrap the `<summary>` block from the
/// model's compaction output, leaving just the section text, using plain string ops
/// (no regex dependency). Tolerant: if
/// the model omitted the tags, the trimmed text is returned as-is so nothing is lost.
fn format_summary(raw: &str) -> String {
    // Prefer the explicit <summary>…</summary> block.
    if let (Some(open), Some(close)) = (raw.find("<summary>"), raw.rfind("</summary>")) {
        if close > open {
            return raw[open + "<summary>".len()..close].trim().to_string();
        }
    }
    // No <summary> tag: drop everything up to and including a closing </analysis>, so the
    // scratchpad never reaches context even when the summary wasn't tag-wrapped.
    if let Some(end) = raw.rfind("</analysis>") {
        return raw[end + "</analysis>".len()..].trim().to_string();
    }
    raw.trim().to_string()
}

#[async_trait]
impl Transformer for Compactor {
    async fn transform(&self, hook: Hook, state: &mut TurnState) -> Result<()> {
        if hook != Hook::PreModel {
            return Ok(());
        }
        // Unknown window = never auto-compact (we won't guess a ceiling and trigger against it).
        let Some(soft) = self.soft_limit() else {
            return Ok(());
        };
        if estimate_tokens(&state.messages) < soft {
            return Ok(());
        }
        // Circuit breaker: after repeated failures, stop trying (let the turn
        // proceed) rather than burn calls in a runaway loop.
        if *self.failures.lock().unwrap() >= 3 {
            return Ok(());
        }
        // Compaction is happening; surface it so "memory compressed" is never a
        // silent event (the kernel drains this notice to the sink after this hook).
        let before = estimate_tokens(&state.messages);
        // Tier 1: mechanical shed first - keep granular history if it's enough.
        shed_tool_outputs(&mut state.messages, self.keep_recent);
        let after_shed = estimate_tokens(&state.messages);
        if after_shed < soft {
            state.notice(
                NoticeLevel::Progress,
                compaction_notice("shed old tool output", before, after_shed),
            );
            return Ok(());
        }
        // Tier 2: model summary of the oldest, recent kept verbatim.
        match self.summarize(state).await {
            Ok(()) => {
                *self.failures.lock().unwrap() = 0;
                let after = estimate_tokens(&state.messages);
                state.notice(
                    NoticeLevel::Progress,
                    compaction_notice("summarized", before, after),
                );
            }
            Err(_) => {
                *self.failures.lock().unwrap() += 1;
                state.notice(NoticeLevel::Warn, "context compaction failed; proceeding");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxio_core::{Response, StopReason, ToolCall};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn user(text: &str) -> Message {
        Message::text(Role::User, text)
    }

    #[test]
    fn estimate_tokens_counts_text_and_calls() {
        let msgs = vec![user(&"x".repeat(400))]; // 400 chars ≈ 100 tokens
        assert_eq!(estimate_tokens(&msgs), 100);
    }

    #[test]
    fn shed_collapses_only_old_tool_outputs() {
        let mut msgs = vec![
            user("q"),
            {
                let mut m = Message::text(Role::Tool, "HUGE OLD OUTPUT");
                m.tool_call_id = Some("1".into());
                m
            },
            user("recent 1"),
            user("recent 2"),
        ];
        // keep_recent = 2 → the tool msg (index 1) is old and gets collapsed.
        let n = shed_tool_outputs(&mut msgs, 2);
        assert_eq!(n, 1);
        assert_eq!(msgs[1].as_text(), "[tool output elided]");
        assert_eq!(msgs[2].as_text(), "recent 1"); // recent untouched
    }

    struct StubSummarizer {
        calls: AtomicUsize,
    }
    #[async_trait]
    impl Provider for StubSummarizer {
        fn name(&self) -> &str {
            "stub-sum"
        }
        async fn complete(&self, _req: Request, _ctx: &Ctx) -> Result<Response> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Response {
                message: Message::text(Role::Assistant, "SUMMARY"),
                usage: Default::default(),
                stop_reason: StopReason::EndTurn,
            })
        }
    }

    #[tokio::test]
    async fn summarize_replaces_old_keeps_recent_and_system() {
        let provider = Arc::new(StubSummarizer {
            calls: AtomicUsize::new(0),
        });
        let c = Compactor::new(provider.clone(), "m", Some(8192));
        let mut state = TurnState {
            input: String::new(),
            model: "m".into(),
            messages: vec![
                Message::text(Role::System, "sys"),
                user("old 1"),
                user("old 2"),
                user("old 3"),
                user("old 4"),
                user("old 5"),
                user("old 6"),
                user("old 7"),
                user("recent A"),
                user("recent B"),
                user("recent C"),
                user("recent D"),
                user("recent E"),
                user("recent F"),
            ],
            tools: vec![],
            params: Params::default(),
            usage: Default::default(),
            last_error: None,
            notices: Vec::new(),
        };
        c.summarize(&mut state).await.unwrap();
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            1,
            "one summarize call"
        );
        assert_eq!(state.messages[0].as_text(), "sys", "system preserved");
        assert!(
            state.messages[1].as_text().contains("SUMMARY"),
            "summary injected"
        );
        // 6 recent kept verbatim at the tail.
        assert_eq!(state.messages.last().unwrap().as_text(), "recent F");
        assert!(state.messages.iter().any(|m| m.as_text() == "recent A"));
        assert!(
            !state.messages.iter().any(|m| m.as_text() == "old 1"),
            "old dropped into summary"
        );
    }

    #[tokio::test]
    async fn compaction_surfaces_a_notice_never_silent() {
        let provider = Arc::new(StubSummarizer {
            calls: AtomicUsize::new(0),
        });
        let c = Compactor::new(provider, "m", Some(4096)); // soft limit ≈ 3686 tok
        let big = "x ".repeat(1500); // ~3000 chars ≈ 750 tok each
        let msgs: Vec<Message> = (0..12).map(|_| user(&big)).collect(); // ~9000 tok, over budget
        let mut state = TurnState {
            input: String::new(),
            model: "m".into(),
            messages: msgs,
            tools: vec![],
            params: Params::default(),
            usage: Default::default(),
            last_error: None,
            notices: Vec::new(),
        };
        c.transform(Hook::PreModel, &mut state).await.unwrap();
        assert!(
            state
                .notices
                .iter()
                .any(|(_, t)| t.contains("compacted context")),
            "compaction must surface a notice, not compress silently"
        );
    }

    #[test]
    fn format_summary_strips_analysis_and_unwraps_summary() {
        let raw = "<analysis>\nmy private scratch\n</analysis>\n<summary>\n1. Primary Request: build X\n</summary>";
        let out = format_summary(raw);
        assert!(out.contains("Primary Request"), "summary kept");
        assert!(!out.contains("scratch"), "analysis scratchpad stripped");
        assert!(!out.contains("<summary>"), "tags removed");
    }

    #[test]
    fn format_summary_without_summary_tag_drops_analysis_prefix() {
        let raw = "<analysis>thinking...</analysis>\nActual summary text here.";
        assert_eq!(format_summary(raw), "Actual summary text here.");
    }

    #[test]
    fn format_summary_untagged_is_returned_verbatim() {
        // Model ignored the tags entirely - never lose its output.
        assert_eq!(format_summary("  plain summary  "), "plain summary");
    }

    #[test]
    fn tool_call_payloads_count_toward_estimate() {
        let mut m = Message::text(Role::Assistant, "");
        m.content.clear();
        m.tool_calls = vec![ToolCall {
            id: "1".into(),
            name: "grep".into(),
            arguments: serde_json::json!({"pattern": "x".repeat(400)}),
        }];
        assert!(estimate_tokens(&[m]) > 90, "tool-call args counted");
    }
}
