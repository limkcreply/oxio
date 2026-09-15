//! A provider whose inner target can be swapped at runtime, so an interactive
//! session can change endpoints (`/connect`) WITHOUT rebuilding the kernel or
//! losing session/transcript state. The kernel holds this as its single provider;
//! the UI keeps a clone and calls `set()` to repoint it at a new chain.
//!
//! Correctness of the swap rests on one fact of the wire adapters: the request
//! body's model id comes from the adapter's OWN configured model, not from
//! `req.model` (see `chat_completions.rs`). So repointing the inner also repoints
//! which model is actually called - no kernel/model rewrite needed.

use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures::stream::BoxStream;
use oxio_core::{Ctx, Provider, Request, Response, Result, StreamEvent};

pub struct SwappableProvider {
    inner: RwLock<Arc<dyn Provider>>,
}

impl SwappableProvider {
    pub fn new(inner: Arc<dyn Provider>) -> Self {
        Self {
            inner: RwLock::new(inner),
        }
    }

    /// Point the session at a new provider (chain). Takes effect on the NEXT
    /// turn - a turn already in flight keeps the provider it started with.
    pub fn set(&self, inner: Arc<dyn Provider>) {
        *self.inner.write().unwrap() = inner;
    }

    /// Clone the current inner out under a short synchronous lock, then release
    /// the guard BEFORE the caller awaits - a lock is never held across an await.
    fn current(&self) -> Arc<dyn Provider> {
        self.inner.read().unwrap().clone()
    }
}

#[async_trait]
impl Provider for SwappableProvider {
    fn name(&self) -> &str {
        // Fixed label: the live target's own name can't be borrowed out from
        // behind the lock. UI diagnostics use the config name, not this.
        "active"
    }

    async fn complete(&self, req: Request, ctx: &Ctx) -> Result<Response> {
        self.current().complete(req, ctx).await
    }

    async fn stream(
        &self,
        req: Request,
        ctx: &Ctx,
    ) -> Result<BoxStream<'static, Result<StreamEvent>>> {
        self.current().stream(req, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxio_core::{Message, Params, Role, StopReason, Usage};

    struct Fixed(&'static str);
    #[async_trait]
    impl Provider for Fixed {
        fn name(&self) -> &str {
            "fixed"
        }
        async fn complete(&self, _req: Request, _ctx: &Ctx) -> Result<Response> {
            Ok(Response {
                message: Message::text(Role::Assistant, self.0),
                usage: Usage::default(),
                stop_reason: StopReason::EndTurn,
            })
        }
    }

    fn req() -> Request {
        Request {
            model: "m".into(),
            messages: vec![Message::text(Role::User, "hi")],
            tools: vec![],
            params: Params::default(),
        }
    }

    #[tokio::test]
    async fn swap_changes_which_provider_answers() {
        let swap = SwappableProvider::new(Arc::new(Fixed("from first")));
        let r = swap.complete(req(), &Ctx::default()).await.unwrap();
        assert_eq!(r.message.as_text(), "from first");

        swap.set(Arc::new(Fixed("from second")));
        let r = swap.complete(req(), &Ctx::default()).await.unwrap();
        assert_eq!(
            r.message.as_text(),
            "from second",
            "next call uses the swapped-in provider"
        );
    }
}
