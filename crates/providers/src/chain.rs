//! Fallback chain expressed as a composing `Provider`: try providers in order,
//! fall to the next on failure. This is how "local primary + API fallback"
//! (and home-mesh resilience) is realized - the kernel never learns about
//! fallback; it just calls one provider.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use oxio_core::{Ctx, OxioError, Provider, Request, Response, Result, StreamEvent};

pub struct ChainProvider {
    name: String,
    chain: Vec<Arc<dyn Provider>>,
}

impl ChainProvider {
    pub fn new(chain: Vec<Arc<dyn Provider>>) -> Self {
        Self {
            name: "chain".to_string(),
            chain,
        }
    }
}

/// Cancellation and timeout are user/kernel intent, not provider failure - never
/// fall through on them.
fn is_terminal(e: &OxioError) -> bool {
    matches!(e, OxioError::Cancelled | OxioError::Timeout)
}

#[async_trait]
impl Provider for ChainProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, req: Request, ctx: &Ctx) -> Result<Response> {
        let mut last = OxioError::NoProvider;
        for p in &self.chain {
            match p.complete(req.clone(), ctx).await {
                Ok(r) => return Ok(r),
                Err(e) if is_terminal(&e) => return Err(e),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    async fn stream(
        &self,
        req: Request,
        ctx: &Ctx,
    ) -> Result<BoxStream<'static, Result<StreamEvent>>> {
        // Fall through only on failure to ESTABLISH the stream; once a provider
        // is streaming we are committed to it.
        let mut last = OxioError::NoProvider;
        for p in &self.chain {
            match p.stream(req.clone(), ctx).await {
                Ok(s) => return Ok(s),
                Err(e) if is_terminal(&e) => return Err(e),
                Err(e) => last = e,
            }
        }
        Err(last)
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

    enum Kind {
        Down,
        Cancelled,
    }
    struct Err_(Kind);
    #[async_trait]
    impl Provider for Err_ {
        fn name(&self) -> &str {
            "err"
        }
        async fn complete(&self, _req: Request, _ctx: &Ctx) -> Result<Response> {
            Err(match self.0 {
                Kind::Down => OxioError::Provider("boom".into()),
                Kind::Cancelled => OxioError::Cancelled,
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
    async fn falls_through_to_next_on_failure() {
        let chain = ChainProvider::new(vec![
            Arc::new(Err_(Kind::Down)),
            Arc::new(Fixed("from second")),
        ]);
        let r = chain.complete(req(), &Ctx::default()).await.unwrap();
        assert_eq!(r.message.as_text(), "from second");
    }

    #[tokio::test]
    async fn cancellation_does_not_fall_through() {
        let chain = ChainProvider::new(vec![
            Arc::new(Err_(Kind::Cancelled)),
            Arc::new(Fixed("should not reach")),
        ]);
        let e = chain.complete(req(), &Ctx::default()).await.unwrap_err();
        assert!(matches!(e, OxioError::Cancelled));
    }

    #[tokio::test]
    async fn all_fail_returns_last_error() {
        let chain =
            ChainProvider::new(vec![Arc::new(Err_(Kind::Down)), Arc::new(Err_(Kind::Down))]);
        let e = chain.complete(req(), &Ctx::default()).await.unwrap_err();
        assert!(matches!(e, OxioError::Provider(_)));
    }
}
