//! Proof-of-life: one full turn drives Start -> Model -> Tool -> Model -> End,
//! firing hooks in order, through the network-free stub provider + echo tool.

use std::sync::{Arc, Mutex};

use std::time::Duration;

use async_trait::async_trait;
use oxio_core::testkit::{EchoTool, FailTool, SlowProvider, StubProvider};
use oxio_core::traits::{Observer, StreamSink};
use oxio_core::{Ctx, Hook, Kernel, NullSink, OxioError, Registry, StreamEvent, TurnState};

/// Records the order hooks fire in.
struct Recorder(Arc<Mutex<Vec<Hook>>>);

#[async_trait]
impl Observer for Recorder {
    async fn on_event(&self, hook: Hook, _state: &TurnState) {
        self.0.lock().unwrap().push(hook);
    }
}

/// Collects streamed events forwarded by the kernel.
struct CollectSink(Arc<Mutex<Vec<StreamEvent>>>);

#[async_trait]
impl StreamSink for CollectSink {
    async fn send(&self, ev: StreamEvent) {
        self.0.lock().unwrap().push(ev);
    }
}

#[tokio::test]
async fn full_turn_drives_all_phases() {
    let log = Arc::new(Mutex::new(Vec::new()));

    let events = Arc::new(Mutex::new(Vec::new()));

    let mut reg = Registry::new();
    reg.add_provider(Arc::new(StubProvider::new("echo")));
    reg.add_tool(Arc::new(EchoTool));
    reg.add_observer(Arc::new(Recorder(log.clone())));

    let kernel = Kernel::new(reg, "stub-model");
    let ctx = Ctx::default();
    let sink = CollectSink(events.clone());
    let outcome = kernel
        .run_turn("hello", &ctx, &sink)
        .await
        .expect("turn runs");

    // Final assistant text arrives after the tool round-trip.
    assert_eq!(outcome.text, "done: tool result received");
    // Usage accumulates across both model calls (4 + 5).
    assert_eq!(outcome.usage.output_tokens, 9);
    assert_eq!(outcome.usage.input_tokens, 18);

    // The tool result is in the transcript.
    assert!(outcome.messages.iter().any(|m| m.as_text() == "echo: ping"));

    // The streaming path forwarded events to the sink (deltas + a terminal Done).
    let evs = events.lock().unwrap();
    assert!(
        evs.iter().any(|e| matches!(e, StreamEvent::TextDelta(_))),
        "should stream text deltas"
    );
    assert!(
        evs.iter().any(|e| matches!(e, StreamEvent::Done { .. })),
        "should emit a terminal Done"
    );
    // The kernel surfaces tool dispatch as an out-of-band Notice so the terminal is
    // never dead while a tool runs (UI/UX live-surfacing, dim 2/4).
    assert!(
        evs.iter()
            .any(|e| matches!(e, StreamEvent::Notice { text, .. } if text.contains("echo"))),
        "should surface the tool dispatch as a Notice"
    );

    let hooks = log.lock().unwrap().clone();
    for expected in [Hook::PreTurn, Hook::PreTool, Hook::PostTool, Hook::PostTurn] {
        assert!(hooks.contains(&expected), "missing hook {expected:?}");
    }
    // PreModel fires twice: before the tool call, and after the tool result.
    assert!(
        hooks.iter().filter(|h| **h == Hook::PreModel).count() >= 2,
        "model should be called at least twice"
    );
    // PreTurn is first.
    assert_eq!(hooks.first(), Some(&Hook::PreTurn));
}

#[tokio::test]
async fn cancelled_ctx_aborts() {
    let mut reg = Registry::new();
    reg.add_provider(Arc::new(StubProvider::new("echo")));
    reg.add_tool(Arc::new(EchoTool));
    let kernel = Kernel::new(reg, "stub-model");

    let ctx = Ctx::default();
    ctx.cancel.cancel(); // pre-cancel

    let err = kernel.run_turn("hi", &ctx, &NullSink).await.unwrap_err();
    assert!(matches!(err, OxioError::Cancelled));
}

#[tokio::test]
async fn tool_failure_is_fed_back_not_fatal() {
    let mut reg = Registry::new();
    reg.add_provider(Arc::new(StubProvider::new("echo")));
    reg.add_tool(Arc::new(FailTool)); // registered as "echo" - the stub calls it
    let kernel = Kernel::new(reg, "stub-model");

    let outcome = kernel
        .run_turn("hi", &Ctx::default(), &NullSink)
        .await
        .expect("turn survives a tool error");

    // The tool error was fed back into the transcript...
    assert!(outcome
        .messages
        .iter()
        .any(|m| m.as_text().contains("boom")));
    // ...and the model still produced its final answer.
    assert_eq!(outcome.text, "done: tool result received");
}

#[tokio::test]
async fn deadline_times_out_a_slow_provider() {
    let mut reg = Registry::new();
    reg.add_provider(Arc::new(SlowProvider {
        delay: Duration::from_secs(10),
    }));
    let kernel = Kernel::new(reg, "slow");

    let ctx = Ctx::default().with_deadline(Duration::from_millis(50));
    let err = kernel.run_turn("hi", &ctx, &NullSink).await.unwrap_err();
    assert!(matches!(err, OxioError::Timeout));
}
