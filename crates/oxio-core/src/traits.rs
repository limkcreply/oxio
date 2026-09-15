//! Frozen mount-point contracts. A feature implements one or more of these and
//! installs them via [`Module::register`]. The kernel knows ONLY these traits -
//! never a concrete vendor, tool, or strategy.

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde_json::Value;

use crate::ctx::Ctx;
use crate::kernel::{Hook, TurnState};
use crate::registry::Registry;
use crate::types::{Request, Response, Result, StreamEvent, ToolOutput, ToolSpec};

/// A model backend: anything that turns a [`Request`] into a [`Response`].
/// Local servers and API vendors are both just providers. Registration order in
/// the [`Registry`] is the fallback chain (first = primary).
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;

    async fn complete(&self, req: Request, ctx: &Ctx) -> Result<Response>;

    /// Default: run `complete` and emit it as one text delta + a `Done` carrying
    /// the assembled message. Real streaming providers override this to emit
    /// granular deltas, ending with a `Done` that carries the full message.
    async fn stream(
        &self,
        req: Request,
        ctx: &Ctx,
    ) -> Result<BoxStream<'static, Result<StreamEvent>>> {
        let resp = self.complete(req, ctx).await?;
        let text = resp.message.as_text();
        let events: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::TextDelta(text)),
            Ok(StreamEvent::Done {
                stop_reason: resp.stop_reason,
                usage: resp.usage,
                message: resp.message,
            }),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

/// Whether a tool only reads or also writes - lets the scheduler serialize
/// writes later without the kernel knowing tool internals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Read,
    Write,
}

/// An action the model can invoke. Receives the control [`Ctx`] so it can honor
/// cancellation and deadlines.
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> Result<ToolOutput>;

    /// A one-line, human-readable summary of a call for the transcript (e.g.
    /// `read_file src/main.rs`), so the UI shows a readable action instead of raw
    /// argument JSON. The kernel calls this for the dispatch line; the default is just
    /// the tool name, and each tool overrides it to surface its key argument. Kept on the
    /// trait (not in the kernel) so tool-specific display knowledge lives in the module,
    /// per the microkernel design.
    fn summarize(&self, _input: &Value) -> String {
        self.spec().name
    }
}

/// Where streamed events go during a turn (the UI, a channel, a log). The kernel
/// forwards every [`StreamEvent`] here as it arrives from the provider.
#[async_trait]
pub trait StreamSink: Send + Sync {
    async fn send(&self, ev: StreamEvent);
}

/// A sink that discards events - for non-streaming callers.
pub struct NullSink;

#[async_trait]
impl StreamSink for NullSink {
    async fn send(&self, _ev: StreamEvent) {}
}

/// Read-only cross-cutting observer (telemetry, logging). Notified at every
/// [`Hook`] with a read view of the turn state.
#[async_trait]
pub trait Observer: Send + Sync {
    async fn on_event(&self, hook: Hook, state: &TurnState);
}

/// Mutates the working [`TurnState`] at a [`Hook`] point - how features affect a
/// turn (compaction, memory injection, session load/save) without the kernel
/// knowing them.
#[async_trait]
pub trait Transformer: Send + Sync {
    async fn transform(&self, hook: Hook, state: &mut TurnState) -> Result<()>;
}

/// A feature bundle. Installs its providers/tools/observers/transformers.
pub trait Module: Send + Sync {
    fn name(&self) -> &str;
    fn register(self: Box<Self>, reg: &mut Registry);
}
