//! Per-call control context: cancellation + deadline + cwd. Threaded through
//! every provider call, tool call, and the turn itself, so any in-flight work is
//! interruptible and time-bounded. Part of the frozen contract.

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::StreamSink;

/// Control context passed to providers, tools, and `run_turn`.
#[derive(Clone)]
pub struct Ctx {
    /// Cancel signal. Providers/tools should `select!` on `cancel.cancelled()`;
    /// the kernel checks it between phases and aborts early.
    pub cancel: CancellationToken,
    /// Optional per-call time budget a provider/tool should enforce.
    pub deadline: Option<Duration>,
    /// Working directory for tools that need it.
    pub cwd: Option<String>,
    /// Optional live-progress channel. A tool doing sub-work (e.g. spawning a
    /// sub-agent) forwards that work's events here so the UI can surface WHICH
    /// agent is doing WHAT, live - instead of a silent black box. `None` = no
    /// forwarding (headless, one-shot, tests). Additive per-call field, same class
    /// as `deadline`/`cwd`; consumers that don't set it are unaffected.
    pub progress: Option<Arc<dyn StreamSink>>,
}

impl Default for Ctx {
    fn default() -> Self {
        Ctx {
            cancel: CancellationToken::new(),
            deadline: None,
            cwd: None,
            progress: None,
        }
    }
}

impl Ctx {
    pub fn new() -> Self {
        Self::default()
    }

    /// True once cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// A child context sharing this cancellation but with its own deadline.
    pub fn with_deadline(&self, deadline: Duration) -> Self {
        Ctx {
            cancel: self.cancel.clone(),
            deadline: Some(deadline),
            cwd: self.cwd.clone(),
            progress: self.progress.clone(),
        }
    }

    /// A child context carrying a live-progress channel (attached by the UI before
    /// a turn so tool sub-work can stream to it).
    pub fn with_progress(&self, progress: Arc<dyn StreamSink>) -> Self {
        Ctx {
            cancel: self.cancel.clone(),
            deadline: self.deadline,
            cwd: self.cwd.clone(),
            progress: Some(progress),
        }
    }
}
