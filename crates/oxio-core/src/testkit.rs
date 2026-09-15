//! Deterministic, network-free implementations for tests and proof-of-life.
//! These are NOT real modules - the real provider/tools land in later crates.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::ctx::Ctx;
use crate::traits::{Provider, Tool, ToolKind};
use crate::types::{
    Content, Message, OxioError, Request, Response, Result, Role, StopReason, ToolCall, ToolOutput,
    ToolSpec, Usage,
};

/// A provider scripted to first emit one tool call, then (once it sees the tool
/// result) emit final text - so a single `run_turn` exercises the full
/// Start -> Model -> Tool -> Model -> End path deterministically.
pub struct StubProvider {
    tool_name: String,
}

impl StubProvider {
    pub fn new(tool_name: impl Into<String>) -> Self {
        StubProvider {
            tool_name: tool_name.into(),
        }
    }
}

#[async_trait]
impl Provider for StubProvider {
    fn name(&self) -> &str {
        "stub"
    }

    async fn complete(&self, req: Request, _ctx: &Ctx) -> Result<Response> {
        let saw_tool_result = req.messages.iter().any(|m| m.role == Role::Tool);
        if saw_tool_result {
            Ok(Response {
                message: Message::text(Role::Assistant, "done: tool result received"),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
                stop_reason: StopReason::EndTurn,
            })
        } else {
            let msg = Message {
                role: Role::Assistant,
                content: vec![Content::Text {
                    text: "calling tool".into(),
                }],
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: self.tool_name.clone(),
                    arguments: json!({ "value": "ping" }),
                }],
                tool_call_id: None,
            };
            Ok(Response {
                message: msg,
                usage: Usage {
                    input_tokens: 8,
                    output_tokens: 4,
                },
                stop_reason: StopReason::ToolUse,
            })
        }
    }
}

/// Echoes its `value` argument back.
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".into(),
            description: "Echo the input value.".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        }
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let v = input.get("value").and_then(|v| v.as_str()).unwrap_or("");
        Ok(ToolOutput::ok(format!("echo: {v}")))
    }
}

/// A tool that always errors - for testing that tool failures are fed back to
/// the model instead of aborting the turn. Registered under the name "echo" so
/// the stub provider's tool call reaches it.
pub struct FailTool;

#[async_trait]
impl Tool for FailTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".into(),
            description: "Always fails.".into(),
            input_schema: json!({ "type": "object" }),
        }
    }

    async fn call(&self, _input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        Err(OxioError::Tool("boom".into()))
    }
}

/// A provider that sleeps before responding - for testing deadline enforcement.
pub struct SlowProvider {
    pub delay: Duration,
}

#[async_trait]
impl Provider for SlowProvider {
    fn name(&self) -> &str {
        "slow"
    }

    async fn complete(&self, _req: Request, _ctx: &Ctx) -> Result<Response> {
        tokio::time::sleep(self.delay).await;
        Ok(Response {
            message: Message::text(Role::Assistant, "late"),
            usage: Usage::default(),
            stop_reason: StopReason::EndTurn,
        })
    }
}
