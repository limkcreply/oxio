//! OpenAI **Responses API** wire adapter (`/responses`). Second wire format after
//! Chat Completions; used by GPT and reasoning models. It plugs into the frozen
//! `Provider` seam; the field and event shapes follow the Responses API spec.
//!
//! Shape differences from Chat Completions this adapter handles:
//! - the system prompt is a top-level `instructions` string, NOT a message;
//! - conversation is an `input` array of typed items (`message` with
//!   `input_text`/`output_text` parts, `function_call`, `function_call_output`);
//! - tools are flat (`{type:"function", name, description, parameters}`), not
//!   nested under a `function` key;
//! - reasoning is streamed as explicit `response.reasoning_*` events, so we route
//!   them straight to `ThinkingDelta` (no marker-splitter needed);
//! - streaming is typed SSE: each `data:` line carries a `type` we switch on.

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};

use oxio_core::{
    Content, Ctx, Message, OxioError, Provider, Request, Response, Result, Role, StopReason,
    StreamEvent, ToolCall, ToolSpec, Usage,
};

use crate::chat_completions::Sampling;

pub struct ResponsesAdapter {
    name: String,
    base_url: String,
    model: String,
    api_key: Option<String>,
    sampling: Sampling,
    http: reqwest::Client,
}

impl ResponsesAdapter {
    /// `base_url` is the API root (e.g. `https://api.openai.com/v1`); the adapter
    /// appends `/responses`.
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        let base = base_url.into();
        Self {
            name: name.into(),
            base_url: base.trim_end_matches('/').to_string(),
            model: model.into(),
            api_key,
            sampling: Sampling::default(),
            http: reqwest::Client::new(),
        }
    }

    /// Attach per-provider sampling defaults (model-class-aware).
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/responses", self.base_url)
    }

    fn build_body(&self, req: &Request, stream: bool) -> Value {
        let (instructions, input) = build_input(&req.messages);
        let mut body = json!({
            "model": self.model,
            "input": input,
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "stream": stream,
            // We manage history ourselves (session module); never let the server persist it.
            "store": false,
        });
        if !instructions.is_empty() {
            body["instructions"] = instructions.into();
        }
        if !req.tools.is_empty() {
            body["tools"] = req
                .tools
                .iter()
                .map(tool_to_responses)
                .collect::<Vec<_>>()
                .into();
        }
        // Model-class-aware sampling (per-turn Params override provider defaults).
        // A reasoning model takes `reasoning.effort` and ignores temperature/top_p;
        // a standard model takes temperature/top_p. `max_output_tokens` applies to both.
        let s = &self.sampling;
        if let Some(effort) = &s.reasoning_effort {
            body["reasoning"] = json!({ "effort": effort });
        } else {
            if let Some(t) = req.params.temperature.or(s.temperature) {
                body["temperature"] = t.into();
            }
            if let Some(p) = req.params.top_p.or(s.top_p) {
                body["top_p"] = p.into();
            }
        }
        if let Some(m) = req.params.max_tokens.or(s.max_tokens) {
            body["max_output_tokens"] = m.into();
        }
        body
    }
}

#[async_trait]
impl Provider for ResponsesAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, req: Request, _ctx: &Ctx) -> Result<Response> {
        let body = self.build_body(&req, false);
        let mut rb = self.http.post(self.endpoint()).json(&body);
        if let Some(k) = &self.api_key {
            rb = rb.bearer_auth(k);
        }
        let resp = rb
            .send()
            .await
            .map_err(|e| OxioError::Provider(e.to_string()))?;
        if !resp.status().is_success() {
            let st = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(OxioError::Provider(format!("http {st}: {t}")));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| OxioError::Provider(e.to_string()))?;
        parse_response(&v)
    }

    async fn stream(
        &self,
        req: Request,
        _ctx: &Ctx,
    ) -> Result<BoxStream<'static, Result<StreamEvent>>> {
        let client = self.http.clone();
        let endpoint = self.endpoint();
        let api_key = self.api_key.clone();
        let body = self.build_body(&req, true);

        let s = async_stream::try_stream! {
            let mut rb = client.post(&endpoint).json(&body);
            if let Some(k) = &api_key {
                rb = rb.bearer_auth(k);
            }
            let resp = rb.send().await.map_err(|e| OxioError::Provider(e.to_string()))?;
            let status = resp.status();
            if !status.is_success() {
                Err(OxioError::Provider(format!("http {status}")))?;
            }

            let mut bytes = resp.bytes_stream();
            let mut buf = String::new();
            let mut text = String::new();
            let mut think = String::new();
            let mut calls: Vec<ToolCall> = Vec::new();
            let mut usage = Usage::default();
            let mut stop = StopReason::EndTurn;

            'outer: while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|e| OxioError::Provider(e.to_string()))?;
                buf.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim().to_string();
                    buf.drain(..=pos);
                    // Responses SSE carries both `event:` and `data:` lines; the
                    // `type` we switch on is inside the data JSON, so `event:` is ignored.
                    let data = match line.strip_prefix("data:") {
                        Some(d) => d.trim(),
                        None => continue,
                    };
                    if data == "[DONE]" {
                        break 'outer;
                    }
                    let v: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    match v.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                        "response.output_text.delta" => {
                            if let Some(d) = v.get("delta").and_then(|x| x.as_str()) {
                                if !d.is_empty() {
                                    text.push_str(d);
                                    yield StreamEvent::TextDelta(d.to_string());
                                }
                            }
                        }
                        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                            if let Some(d) = v.get("delta").and_then(|x| x.as_str()) {
                                if !d.is_empty() {
                                    think.push_str(d);
                                    yield StreamEvent::ThinkingDelta(d.to_string());
                                }
                            }
                        }
                        // A complete output item. Function calls arrive whole here.
                        "response.output_item.done" => {
                            if let Some(item) = v.get("item") {
                                if item.get("type").and_then(|x| x.as_str()) == Some("function_call") {
                                    let call = function_call_from_item(item);
                                    if let Some(c) = call {
                                        yield StreamEvent::ToolCallDelta {
                                            id: c.id.clone(),
                                            name: c.name.clone(),
                                            arguments_delta: serde_json::to_string(&c.arguments).unwrap_or_default(),
                                        };
                                        calls.push(c);
                                    }
                                }
                            }
                        }
                        "response.completed" => {
                            if let Some(u) = v.get("response").and_then(|r| r.get("usage")) {
                                usage = usage_from(u);
                            }
                            break 'outer;
                        }
                        "response.failed" | "response.incomplete" => {
                            let msg = v.get("response")
                                .and_then(|r| r.get("error"))
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str())
                                .unwrap_or("response failed");
                            Err(OxioError::Provider(msg.to_string()))?;
                        }
                        _ => {}
                    }
                }
            }

            if !calls.is_empty() && stop == StopReason::EndTurn {
                stop = StopReason::ToolUse;
            }
            let mut content = Vec::new();
            if !think.is_empty() {
                content.push(Content::Thinking { text: think });
            }
            if !text.is_empty() {
                content.push(Content::Text { text });
            }
            let message = Message { role: Role::Assistant, content, tool_calls: calls, tool_call_id: None };
            yield StreamEvent::Done { stop_reason: stop, usage, message };
        };

        Ok(Box::pin(s))
    }
}

/// Split oxio messages into the Responses `(instructions, input[])` pair.
/// System messages become the top-level instructions string; everything else
/// becomes typed input items.
fn build_input(messages: &[Message]) -> (String, Vec<Value>) {
    let mut instructions = String::new();
    let mut input = Vec::new();
    for m in messages {
        match m.role {
            Role::System => {
                let t = m.as_text();
                if !t.is_empty() {
                    if !instructions.is_empty() {
                        instructions.push_str("\n\n");
                    }
                    instructions.push_str(&t);
                }
            }
            Role::User => input.push(message_item("user", m)),
            Role::Assistant => {
                let t = m.as_text();
                if !t.is_empty() || m.content.iter().any(|c| matches!(c, Content::Image { .. })) {
                    input.push(message_item("assistant", m));
                }
                for tc in &m.tool_calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".into()),
                    }));
                }
            }
            Role::Tool => input.push(json!({
                "type": "function_call_output",
                "call_id": m.tool_call_id.clone().unwrap_or_default(),
                "output": m.as_text(),
            })),
        }
    }
    (instructions, input)
}

/// A `message` input item with typed content parts. `input_text`/`input_image`
/// for the user side, `output_text` for assistant history.
fn message_item(role: &str, m: &Message) -> Value {
    let part_text = if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    };
    let text = m.as_text();
    let mut parts = Vec::new();
    if !text.is_empty() {
        parts.push(json!({ "type": part_text, "text": text }));
    }
    for c in &m.content {
        if let Content::Image { media_type, data } = c {
            parts.push(json!({
                "type": "input_image",
                "image_url": format!("data:{media_type};base64,{data}"),
            }));
        }
    }
    json!({ "type": "message", "role": role, "content": parts })
}

/// Responses tools are flat: `{type:"function", name, description, parameters}`,
/// unlike Chat Completions' nested `function` object.
fn tool_to_responses(t: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.input_schema,
    })
}

fn function_call_from_item(item: &Value) -> Option<ToolCall> {
    let name = item.get("name").and_then(|x| x.as_str())?.to_string();
    // call_id is the model-facing id; fall back to the item id if absent.
    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    let arguments = match item.get("arguments") {
        Some(Value::String(s)) => crate::toolcalls::parse_tool_args(s),
        Some(v) => v.clone(),
        None => json!({}),
    };
    Some(ToolCall {
        id,
        name,
        arguments,
    })
}

fn usage_from(u: &Value) -> Usage {
    Usage {
        input_tokens: u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0),
        output_tokens: u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0),
    }
}

/// Parse a non-streaming `/responses` result: the `output` array holds message
/// and function_call items; `usage` holds token counts.
fn parse_response(v: &Value) -> Result<Response> {
    let output = v
        .get("output")
        .and_then(|o| o.as_array())
        .ok_or_else(|| OxioError::Provider("response had no output".into()))?;

    let mut text = String::new();
    let mut think = String::new();
    let mut tool_calls = Vec::new();
    for item in output {
        match item.get("type").and_then(|x| x.as_str()).unwrap_or("") {
            "message" => {
                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    for p in parts {
                        if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                            text.push_str(t);
                        }
                    }
                }
            }
            "reasoning" => {
                // Reasoning items carry summary parts; concatenate any text.
                if let Some(parts) = item.get("summary").and_then(|c| c.as_array()) {
                    for p in parts {
                        if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                            think.push_str(t);
                        }
                    }
                }
            }
            "function_call" => {
                if let Some(c) = function_call_from_item(item) {
                    tool_calls.push(c);
                }
            }
            _ => {}
        }
    }

    let stop = if !tool_calls.is_empty() {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    };
    let usage = v.get("usage").map(usage_from).unwrap_or_default();
    let mut content = Vec::new();
    if !think.is_empty() {
        content.push(Content::Thinking { text: think });
    }
    if !text.is_empty() {
        content.push(Content::Text { text });
    }
    let message = Message {
        role: Role::Assistant,
        content,
        tool_calls,
        tool_call_id: None,
    };
    Ok(Response {
        message,
        stop_reason: stop,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxio_core::Params;

    fn req(messages: Vec<Message>, tools: Vec<ToolSpec>) -> Request {
        Request {
            model: "gpt".into(),
            messages,
            tools,
            params: Params::default(),
        }
    }

    #[test]
    fn system_becomes_instructions_not_an_input_item() {
        let a = ResponsesAdapter::new("p", "http://x/v1", "gpt", None);
        let r = req(
            vec![
                Message::text(Role::System, "be terse"),
                Message::text(Role::User, "hi"),
            ],
            vec![],
        );
        let b = a.build_body(&r, false);
        assert_eq!(
            b["instructions"], "be terse",
            "system prompt is top-level instructions"
        );
        let input = b["input"].as_array().unwrap();
        assert_eq!(input.len(), 1, "only the user message is an input item");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
    }

    #[test]
    fn tool_call_and_result_roundtrip_to_typed_items() {
        let a = ResponsesAdapter::new("p", "http://x/v1", "gpt", None);
        let mut assistant = Message {
            role: Role::Assistant,
            content: vec![],
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: json!({"path":"a.rs"}),
            }],
            tool_call_id: None,
        };
        assistant.content.clear();
        let mut tool_result = Message::text(Role::Tool, "file contents");
        tool_result.tool_call_id = Some("call_1".into());
        let b = a.build_body(&req(vec![assistant, tool_result], vec![]), false);
        let input = b["input"].as_array().unwrap();
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["name"], "read_file");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["output"], "file contents");
    }

    #[test]
    fn tools_are_flat_not_nested() {
        let a = ResponsesAdapter::new("p", "http://x/v1", "gpt", None);
        let tool = ToolSpec {
            name: "grep".into(),
            description: "search".into(),
            input_schema: json!({"type":"object"}),
        };
        let b = a.build_body(
            &req(vec![Message::text(Role::User, "x")], vec![tool]),
            false,
        );
        let t = &b["tools"][0];
        assert_eq!(t["type"], "function");
        assert_eq!(t["name"], "grep", "name is flat, not under a function key");
        assert!(t.get("function").is_none(), "no nested function wrapper");
    }

    #[test]
    fn parses_output_message_and_function_call() {
        let v = json!({
            "output": [
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "hello" }] },
                { "type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{\"path\":\".\"}" }
            ],
            "usage": { "input_tokens": 10, "output_tokens": 3 }
        });
        let r = parse_response(&v).unwrap();
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.message.as_text(), "hello");
        assert_eq!(r.message.tool_calls.len(), 1);
        assert_eq!(r.message.tool_calls[0].name, "ls");
        assert_eq!(r.message.tool_calls[0].arguments["path"], ".");
        assert_eq!(r.usage.input_tokens, 10);
        assert_eq!(r.usage.output_tokens, 3);
    }

    #[test]
    fn reasoning_effort_omits_temperature() {
        let a = ResponsesAdapter::new("p", "http://x/v1", "o1", None).with_sampling(Sampling {
            reasoning_effort: Some("high".into()),
            temperature: Some(0.7),
            ..Default::default()
        });
        let b = a.build_body(&req(vec![Message::text(Role::User, "x")], vec![]), false);
        assert_eq!(b["reasoning"]["effort"], "high");
        assert!(
            b.get("temperature").is_none(),
            "reasoning model omits temperature"
        );
    }
}
