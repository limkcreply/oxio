//! oxio-core - the FROZEN microkernel contract for the oxio agentic CLI.
//!
//! The kernel owns only the turn state machine, the registry, and hook dispatch.
//! Every feature is a separate crate implementing one of the frozen traits in
//! [`traits`] and installing via [`Registry`]. The kernel never learns about a
//! concrete vendor, tool, or strategy. See `HANDOFF.md`.
//!
//! Lifecycle: `Start -> Model -> (Tool dispatch loop) -> End`. At each [`Hook`]
//! the kernel runs `Transformer`s (mutate [`TurnState`]) then notifies
//! `Observer`s (read-only).

pub mod ctx;
pub mod kernel;
pub mod registry;
pub mod testkit;
pub mod traits;
pub mod types;

pub use ctx::Ctx;
pub use kernel::{Hook, Kernel, Outcome, TurnState};
pub use registry::{Registry, SharedTools};
pub use traits::{Module, NullSink, Observer, Provider, StreamSink, Tool, ToolKind, Transformer};
pub use types::{
    Content, Message, NoticeLevel, OxioError, Params, Request, Response, Result, Role, StopReason,
    StreamEvent, ToolCall, ToolOutput, ToolSpec, Usage,
};
