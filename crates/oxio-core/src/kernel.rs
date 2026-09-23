//! The tiny kernel: a turn state machine + hook dispatch. It owns NO feature
//! logic. At each lifecycle point it runs registered `Transformer`s (which may
//! mutate the turn state) and then notifies `Observer`s (read-only).

use std::future::Future;
use std::sync::Arc;

use futures::StreamExt;

use crate::ctx::Ctx;
use crate::registry::Registry;
use crate::traits::{Provider, StreamSink};
use crate::types::{
    Message, NoticeLevel, OxioError, Params, Request, Result, Role, StopReason, StreamEvent,
    ToolSpec, Usage,
};

/// Cap a tool's output for the activity display: the first `n` lines, each `⎿`-indented, with a
/// `… (+N lines)` marker when more were hidden. Empty content yields an empty string (nothing to
/// show). A blind check/cross hid failures - this makes the result and the error visible.
fn cap_output(content: &str, n: usize) -> String {
    if content.trim().is_empty() {
        return String::new();
    }
    let lines: Vec<&str> = content.lines().collect();
    let mut out: Vec<String> = lines
        .iter()
        .take(n)
        .map(|l| format!("  \u{23BF} {l}"))
        .collect();
    if lines.len() > n {
        out.push(format!("  \u{23BF} … (+{} lines)", lines.len() - n));
    }
    out.join("\n")
}

/// Fixed lifecycle points. Transformers run here (mutate), observers are notified
/// here (read). This set is frozen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hook {
    PreTurn,
    PreModel,
    PostModel,
    PreTool,
    PostTool,
    PreCompact,
    PostCompact,
    OnError,
    OnFallback,
    PostTurn,
}

/// Mutable working state for one turn. Transformers may mutate it; observers get
/// a shared reference. Grows over time without breaking the trait signatures.
#[derive(Debug, Clone)]
pub struct TurnState {
    pub input: String,
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub params: Params,
    pub usage: Usage,
    pub last_error: Option<String>,
    /// Notices a transformer wants surfaced to the user (compaction, memory ops,
    /// fallback). The kernel drains these to the sink after each hook, so a module
    /// with no sink handle can still make its runtime work visible. Additive: the
    /// state grows without touching any trait signature.
    pub notices: Vec<(NoticeLevel, String)>,
}

impl TurnState {
    /// Queue a user-facing notice; drained to the sink by the kernel after the
    /// current hook. Lets a sink-less transformer surface its work.
    pub fn notice(&mut self, level: NoticeLevel, text: impl Into<String>) {
        self.notices.push((level, text.into()));
    }
}

/// The result of a completed turn.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub text: String,
    pub usage: Usage,
    pub messages: Vec<Message>,
}

/// Drives one turn over the registered modules.
pub struct Kernel {
    registry: Registry,
    model: String,
    max_tool_iterations: usize,
}

impl Kernel {
    pub fn new(registry: Registry, model: impl Into<String>) -> Self {
        Kernel {
            registry,
            model: model.into(),
            max_tool_iterations: 16,
        }
    }

    fn tool_specs(&self) -> Vec<ToolSpec> {
        self.registry
            .tools
            .read()
            .expect("tools lock")
            .values()
            .map(|t| t.spec())
            .collect()
    }

    fn primary(&self) -> Result<Arc<dyn Provider>> {
        self.registry
            .providers
            .first()
            .cloned()
            .ok_or(OxioError::NoProvider)
    }

    /// Run transformers registered for `hook`, drain any notices they queued to
    /// the sink, then notify observers.
    async fn at(&self, hook: Hook, state: &mut TurnState, sink: &dyn StreamSink) -> Result<()> {
        for (h, t) in &self.registry.transformers {
            if *h == hook {
                t.transform(hook, state).await?;
            }
        }
        for (level, text) in state.notices.drain(..) {
            sink.send(StreamEvent::Notice { level, text }).await;
        }
        for o in &self.registry.observers {
            o.on_event(hook, state).await;
        }
        Ok(())
    }

    /// Run one model call as a stream, forwarding every event to `sink`, and
    /// return the assembled (message, stop_reason, usage) from the terminal
    /// `Done`. Non-streaming providers get this via the default `stream` impl.
    async fn drive_model(
        &self,
        provider: &dyn Provider,
        req: Request,
        ctx: &Ctx,
        sink: &dyn StreamSink,
    ) -> Result<(Message, StopReason, Usage)> {
        let work = async {
            let mut s = provider.stream(req, ctx).await?;
            let mut done: Option<(Message, StopReason, Usage)> = None;
            while let Some(ev) = s.next().await {
                let ev = ev?;
                sink.send(ev.clone()).await;
                if let StreamEvent::Done {
                    stop_reason,
                    usage,
                    message,
                } = ev
                {
                    done = Some((message, stop_reason, usage));
                }
            }
            done.ok_or_else(|| OxioError::Provider("stream ended without Done".into()))
        };
        // Kernel-enforced cancellation + deadline around the whole model call, so
        // a hung or uncooperative provider cannot wedge the turn.
        guard(ctx, work).await
    }

    /// Drive one turn: Start -> Model -> (Tool dispatch loop) -> End.
    /// `ctx` carries cancellation + deadline; `sink` receives streamed events.
    pub async fn run_turn(
        &self,
        input: impl Into<String>,
        ctx: &Ctx,
        sink: &dyn StreamSink,
    ) -> Result<Outcome> {
        let input = input.into();
        let mut state = TurnState {
            input: input.clone(),
            model: self.model.clone(),
            messages: vec![Message::text(Role::User, input)],
            tools: self.tool_specs(),
            params: Params::default(),
            usage: Usage::default(),
            last_error: None,
            notices: Vec::new(),
        };

        self.at(Hook::PreTurn, &mut state, sink).await?;

        let provider = self.primary()?;

        for _ in 0..self.max_tool_iterations {
            if ctx.is_cancelled() {
                state.last_error = Some("cancelled".into());
                self.at(Hook::OnError, &mut state, sink).await?;
                return Err(OxioError::Cancelled);
            }

            self.at(Hook::PreModel, &mut state, sink).await?;

            let req = Request {
                model: state.model.clone(),
                messages: state.messages.clone(),
                tools: state.tools.clone(),
                params: state.params.clone(),
            };

            let (message, stop_reason, usage) =
                match self.drive_model(&*provider, req, ctx, sink).await {
                    Ok(t) => t,
                    Err(e) => {
                        state.last_error = Some(e.to_string());
                        self.at(Hook::OnError, &mut state, sink).await?;
                        return Err(e);
                    }
                };

            state.usage += usage;
            state.messages.push(message.clone());
            self.at(Hook::PostModel, &mut state, sink).await?;

            let has_calls = !message.tool_calls.is_empty();
            if stop_reason != StopReason::ToolUse || !has_calls {
                break;
            }

            for call in message.tool_calls.clone() {
                self.at(Hook::PreTool, &mut state, sink).await?;
                // Surface the dispatch so the terminal is never dead while a tool
                // runs. Lifecycle visibility (not feature logic): the kernel is the
                // only place holding both the dispatched call and the real sink.
                // Look the tool up once so its own `summarize` humanizes the dispatch line
                // (readable action, not raw arg JSON). Unknown tool → fall back to the name.
                let tool = self.registry.tool(&call.name);
                let summary = match &tool {
                    Some(t) => t.summarize(&call.arguments),
                    None => call.name.clone(),
                };
                sink.send(StreamEvent::Notice {
                    level: NoticeLevel::Progress,
                    text: format!("\u{2192} {summary}"),
                })
                .await;
                // A tool failure (or unknown tool) becomes an error result fed
                // back to the model, NOT a turn abort. Only cancellation/timeout
                // abort the turn.
                let (content, ok) = match tool {
                    None => (format!("tool error: unknown tool '{}'", call.name), false),
                    Some(tool) => match guard(ctx, tool.call(call.arguments.clone(), ctx)).await {
                        Ok(out) => {
                            let ok = !out.is_error;
                            (out.content, ok)
                        }
                        Err(e @ OxioError::Cancelled) | Err(e @ OxioError::Timeout) => {
                            state.last_error = Some(e.to_string());
                            self.at(Hook::OnError, &mut state, sink).await?;
                            return Err(e);
                        }
                        Err(e) => (format!("tool error: {e}"), false),
                    },
                };
                sink.send(StreamEvent::Notice {
                    level: if ok {
                        NoticeLevel::Progress
                    } else {
                        NoticeLevel::Warn
                    },
                    text: format!("  {} {summary}", if ok { "\u{2713}" } else { "\u{2717}" }),
                })
                .await;
                // Surface the tool's output (capped) so a result - and especially a failure - is
                // visible, not a blind check/cross. Errors get more lines than a success preview.
                let preview = cap_output(&content, if ok { 3 } else { 20 });
                if !preview.is_empty() {
                    sink.send(StreamEvent::Notice {
                        level: if ok {
                            NoticeLevel::Progress
                        } else {
                            NoticeLevel::Warn
                        },
                        text: preview,
                    })
                    .await;
                }
                let mut msg = Message::text(Role::Tool, content);
                msg.tool_call_id = Some(call.id.clone());
                state.messages.push(msg);
                self.at(Hook::PostTool, &mut state, sink).await?;
            }
        }

        self.at(Hook::PostTurn, &mut state, sink).await?;

        let text = state
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .map(|m| m.as_text())
            .unwrap_or_default();

        Ok(Outcome {
            text,
            usage: state.usage,
            messages: state.messages,
        })
    }
}

/// Enforce cancellation and an optional deadline around a future. Cancellation
/// wins immediately, an elapsed deadline yields `Timeout`, otherwise the
/// future's own result. This is what makes a hung or uncooperative provider/tool
/// unable to wedge a turn.
async fn guard<T>(ctx: &Ctx, fut: impl Future<Output = Result<T>>) -> Result<T> {
    tokio::pin!(fut);
    let cancelled = ctx.cancel.cancelled();
    tokio::pin!(cancelled);
    match ctx.deadline {
        Some(d) => {
            let sleep = tokio::time::sleep(d);
            tokio::pin!(sleep);
            tokio::select! {
                _ = &mut cancelled => Err(OxioError::Cancelled),
                _ = &mut sleep => Err(OxioError::Timeout),
                r = &mut fut => r,
            }
        }
        None => {
            tokio::select! {
                _ = &mut cancelled => Err(OxioError::Cancelled),
                r = &mut fut => r,
            }
        }
    }
}
