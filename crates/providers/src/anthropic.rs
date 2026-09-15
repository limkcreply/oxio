//! **Anthropic Messages API** wire adapter (`/v1/messages`). Third and final wire
//! format. It plugs into the frozen `Provider` seam; the request assembly, SSE event
//! names, and content-block translation follow the Anthropic Messages spec.
//!
//! Shape differences this adapter handles vs OpenAI wires:
//! - auth is the `x-api-key` header plus a required `anthropic-version` header
//!   (not bearer);
//! - the system prompt is a top-level `system` string, NOT a message;
//! - content is typed blocks: `text`, `image` (base64 `source`), `tool_use`
//!   (assistant), `tool_result` (user);
//! - tools use `input_schema` (not `parameters`) and `tool_choice` is an object;
//! - `max_tokens` is REQUIRED;
//! - Anthropic requires alternating user/assistant turns, so adjacent same-role
//!   messages (notably a run of tool results) are coalesced into one turn;
//! - streaming SSE is block-structured: `content_block_start` (text/tool_use),
//!   `content_block_delta` (`text_delta` / `input_json_delta` / `thinking_delta`),
//!   `message_delta` (stop_reason + output usage).

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};

use oxio_core::{
    Content, Ctx, Message, OxioError, Provider, Request, Response, Result, Role, StopReason,
    StreamEvent, ToolCall, ToolSpec, Usage,
};

use crate::chat_completions::Sampling;

/// Anthropic requires `max_tokens`; used when neither per-turn nor provider
/// sampling supplies one.
const DEFAULT_MAX_TOKENS: u32 = 4096;
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicAdapter {
    name: String,
    base_url: String,
    model: String,
    api_key: Option<String>,
    sampling: Sampling,
    http: reqwest::Client,
}

impl AnthropicAdapter {
    /// `base_url` is the API root (e.g. `https://api.anthropic.com/v1`); the
    /// adapter appends `/messages`.
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

    /// Attach per-provider sampling defaults.
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/messages", self.base_url)
    }

    fn build_body(&self, req: &Request, stream: bool) -> Value {
        let (system, messages) = build_messages(&req.messages);
        let s = &self.sampling;
        let max_tokens = req
            .params
            .max_tokens
            .or(s.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS);
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        });
        if !system.is_empty() {
            body["system"] = system.into();
        }
        if !req.tools.is_empty() {
            body["tools"] = req
                .tools
                .iter()
                .map(tool_to_anthropic)
                .collect::<Vec<_>>()
                .into();
            body["tool_choice"] = json!({ "type": "auto" });
        }
        // Anthropic has no reasoning_effort knob; temperature applies to all models.
        if let Some(t) = req.params.temperature.or(s.temperature) {
            body["temperature"] = t.into();
        }
        if let Some(p) = req.params.top_p.or(s.top_p) {
            body["top_p"] = p.into();
        }
        body
    }

    fn send_headers(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut rb = rb.header("anthropic-version", ANTHROPIC_VERSION);
        if let Some(k) = &self.api_key {
            rb = rb.header("x-api-key", k);
        }
        rb
    }
}

#[async_trait]
impl Provider for AnthropicAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, req: Request, _ctx: &Ctx) -> Result<Response> {
        let body = self.build_body(&req, false);
        let rb = self.send_headers(self.http.post(self.endpoint()).json(&body));
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
            let mut rb = client.post(&endpoint).json(&body).header("anthropic-version", ANTHROPIC_VERSION);
            if let Some(k) = &api_key {
                rb = rb.header("x-api-key", k);
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
            // Per content-block index: (id, name, json-args-accumulator). Only
            // tool_use blocks populate id/name; text/thinking blocks stay empty.
            let mut blocks: Vec<(String, String, String)> = Vec::new();
            let mut usage = Usage::default();
            let mut stop = StopReason::EndTurn;

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|e| OxioError::Provider(e.to_string()))?;
                buf.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim().to_string();
                    buf.drain(..=pos);
                    // Anthropic SSE carries `event:` and `data:` lines; the event
                    // kind is duplicated in the data JSON `type`, so we key on that.
                    let data = match line.strip_prefix("data:") {
                        Some(d) => d.trim(),
                        None => continue,
                    };
                    let v: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    match v.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                        "message_start" => {
                            if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
                                if let Some(i) = u.get("input_tokens").and_then(|x| x.as_u64()) {
                                    usage.input_tokens = i;
                                }
                            }
                        }
                        "content_block_start" => {
                            let idx = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                            while blocks.len() <= idx {
                                blocks.push((String::new(), String::new(), String::new()));
                            }
                            if let Some(cb) = v.get("content_block") {
                                if cb.get("type").and_then(|x| x.as_str()) == Some("tool_use") {
                                    blocks[idx].0 = cb.get("id").and_then(|x| x.as_str()).unwrap_or_default().to_string();
                                    blocks[idx].1 = cb.get("name").and_then(|x| x.as_str()).unwrap_or_default().to_string();
                                }
                            }
                        }
                        "content_block_delta" => {
                            let idx = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                            while blocks.len() <= idx {
                                blocks.push((String::new(), String::new(), String::new()));
                            }
                            let delta = match v.get("delta") { Some(d) => d, None => continue };
                            match delta.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                                "text_delta" => {
                                    if let Some(t) = delta.get("text").and_then(|x| x.as_str()) {
                                        if !t.is_empty() { text.push_str(t); yield StreamEvent::TextDelta(t.to_string()); }
                                    }
                                }
                                "thinking_delta" => {
                                    if let Some(t) = delta.get("thinking").and_then(|x| x.as_str()) {
                                        if !t.is_empty() { think.push_str(t); yield StreamEvent::ThinkingDelta(t.to_string()); }
                                    }
                                }
                                "input_json_delta" => {
                                    if let Some(pj) = delta.get("partial_json").and_then(|x| x.as_str()) {
                                        blocks[idx].2.push_str(pj);
                                        yield StreamEvent::ToolCallDelta {
                                            id: blocks[idx].0.clone(),
                                            name: blocks[idx].1.clone(),
                                            arguments_delta: pj.to_string(),
                                        };
                                    }
                                }
                                _ => {}
                            }
                        }
                        "message_delta" => {
                            if let Some(sr) = v.get("delta").and_then(|d| d.get("stop_reason")).and_then(|x| x.as_str()) {
                                stop = map_stop(sr);
                            }
                            if let Some(o) = v.get("usage").and_then(|u| u.get("output_tokens")).and_then(|x| x.as_u64()) {
                                usage.output_tokens = o;
                            }
                        }
                        "message_stop" => break,
                        "error" => {
                            let msg = v.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()).unwrap_or("anthropic error");
                            Err(OxioError::Provider(msg.to_string()))?;
                        }
                        _ => {}
                    }
                }
            }

            let tool_calls: Vec<ToolCall> = blocks
                .into_iter()
                .filter(|(id, name, _)| !id.is_empty() || !name.is_empty())
                .map(|(id, name, args)| ToolCall {
                    id,
                    name,
                    arguments: crate::toolcalls::parse_tool_args(&args),
                })
                .collect();
            let mut content = Vec::new();
            if !think.is_empty() { content.push(Content::Thinking { text: think }); }
            if !text.is_empty() { content.push(Content::Text { text }); }
            let message = Message { role: Role::Assistant, content, tool_calls, tool_call_id: None };
            yield StreamEvent::Done { stop_reason: stop, usage, message };
        };

        Ok(Box::pin(s))
    }
}

fn map_stop(reason: &str) -> StopReason {
    match reason {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

/// Split oxio messages into Anthropic's `(system, messages[])`. System text
/// becomes the top-level `system` string; the rest become typed-block turns.
/// Adjacent same-role turns are coalesced (Anthropic requires alternation, and a
/// run of tool results must land in one user turn).
fn build_messages(messages: &[Message]) -> (String, Vec<Value>) {
    let mut system = String::new();
    let mut turns: Vec<(String, Vec<Value>)> = Vec::new();

    let mut push = |role: &str, blocks: Vec<Value>| {
        if blocks.is_empty() {
            return;
        }
        match turns.last_mut() {
            Some((r, existing)) if r == role => existing.extend(blocks),
            _ => turns.push((role.to_string(), blocks)),
        }
    };

    for m in messages {
        match m.role {
            Role::System => {
                let t = m.as_text();
                if !t.is_empty() {
                    if !system.is_empty() {
                        system.push_str("\n\n");
                    }
                    system.push_str(&t);
                }
            }
            Role::User => push("user", user_blocks(m)),
            Role::Assistant => push("assistant", assistant_blocks(m)),
            Role::Tool => push(
                "user",
                vec![json!({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                    "content": m.as_text(),
                })],
            ),
        }
    }

    let messages = turns
        .into_iter()
        .map(|(role, content)| json!({ "role": role, "content": content }))
        .collect();
    (system, messages)
}

/// User-turn blocks: text plus any base64 image blocks.
fn user_blocks(m: &Message) -> Vec<Value> {
    let mut blocks = Vec::new();
    let text = m.as_text();
    if !text.is_empty() {
        blocks.push(json!({ "type": "text", "text": text }));
    }
    for c in &m.content {
        if let Content::Image { media_type, data } = c {
            blocks.push(json!({
                "type": "image",
                "source": { "type": "base64", "media_type": media_type, "data": data },
            }));
        }
    }
    blocks
}

/// Assistant-turn blocks: text followed by `tool_use` blocks.
fn assistant_blocks(m: &Message) -> Vec<Value> {
    let mut blocks = Vec::new();
    let text = m.as_text();
    if !text.is_empty() {
        blocks.push(json!({ "type": "text", "text": text }));
    }
    for tc in &m.tool_calls {
        blocks.push(json!({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": tc.arguments,
        }));
    }
    blocks
}

/// Anthropic tools use `input_schema` (not `parameters`).
fn tool_to_anthropic(t: &ToolSpec) -> Value {
    json!({
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema,
    })
}

/// Parse a non-streaming `/messages` result: `content` is a block array of
/// `text` / `tool_use`; `usage` holds token counts; `stop_reason` the finish.
fn parse_response(v: &Value) -> Result<Response> {
    let content_blocks = v
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| OxioError::Provider("response had no content".into()))?;

    let mut text = String::new();
    let mut think = String::new();
    let mut tool_calls = Vec::new();
    for b in content_blocks {
        match b.get("type").and_then(|x| x.as_str()).unwrap_or("") {
            "text" => {
                if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                    text.push_str(t);
                }
            }
            "thinking" => {
                if let Some(t) = b.get("thinking").and_then(|x| x.as_str()) {
                    think.push_str(t);
                }
            }
            "tool_use" => {
                tool_calls.push(ToolCall {
                    id: b
                        .get("id")
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    name: b
                        .get("name")
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    arguments: b.get("input").cloned().unwrap_or_else(|| json!({})),
                });
            }
            _ => {}
        }
    }

    let stop = v
        .get("stop_reason")
        .and_then(|x| x.as_str())
        .map(map_stop)
        .unwrap_or(StopReason::EndTurn);
    let usage = v
        .get("usage")
        .map(|u| Usage {
            input_tokens: u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0),
            output_tokens: u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0),
        })
        .unwrap_or_default();
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
            model: "claude".into(),
            messages,
            tools,
            params: Params::default(),
        }
    }

    #[test]
    fn system_is_top_level_and_max_tokens_present() {
        let a = AnthropicAdapter::new("p", "http://x/v1", "claude", None);
        let b = a.build_body(
            &req(
                vec![
                    Message::text(Role::System, "be terse"),
                    Message::text(Role::User, "hi"),
                ],
                vec![],
            ),
            false,
        );
        assert_eq!(b["system"], "be terse");
        assert!(
            b["max_tokens"].as_u64().unwrap() > 0,
            "max_tokens is required by Anthropic"
        );
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["type"], "text");
    }

    #[test]
    fn tool_use_and_result_become_typed_blocks() {
        let a = AnthropicAdapter::new("p", "http://x/v1", "claude", None);
        let mut assistant = Message {
            role: Role::Assistant,
            content: vec![],
            tool_calls: vec![ToolCall {
                id: "tu_1".into(),
                name: "read_file".into(),
                arguments: json!({"path":"a.rs"}),
            }],
            tool_call_id: None,
        };
        assistant.content.clear();
        let mut tr = Message::text(Role::Tool, "contents");
        tr.tool_call_id = Some("tu_1".into());
        let b = a.build_body(&req(vec![assistant, tr], vec![]), false);
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"][0]["type"], "tool_use");
        assert_eq!(msgs[0]["content"][0]["id"], "tu_1");
        assert_eq!(msgs[0]["content"][0]["input"]["path"], "a.rs");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[1]["content"][0]["tool_use_id"], "tu_1");
    }

    #[test]
    fn adjacent_tool_results_coalesce_into_one_user_turn() {
        let a = AnthropicAdapter::new("p", "http://x/v1", "claude", None);
        let mut t1 = Message::text(Role::Tool, "r1");
        t1.tool_call_id = Some("a".into());
        let mut t2 = Message::text(Role::Tool, "r2");
        t2.tool_call_id = Some("b".into());
        let b = a.build_body(&req(vec![t1, t2], vec![]), false);
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(
            msgs.len(),
            1,
            "two tool results merge into one user turn (alternation)"
        );
        assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn tools_use_input_schema() {
        let a = AnthropicAdapter::new("p", "http://x/v1", "claude", None);
        let tool = ToolSpec {
            name: "grep".into(),
            description: "search".into(),
            input_schema: json!({"type":"object"}),
        };
        let b = a.build_body(
            &req(vec![Message::text(Role::User, "x")], vec![tool]),
            false,
        );
        assert_eq!(b["tools"][0]["name"], "grep");
        assert_eq!(b["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(b["tool_choice"]["type"], "auto");
    }

    #[test]
    fn parses_content_blocks_and_tool_use() {
        let v = json!({
            "content": [
                { "type": "text", "text": "hello" },
                { "type": "tool_use", "id": "tu_9", "name": "ls", "input": { "path": "." } }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 12, "output_tokens": 4 }
        });
        let r = parse_response(&v).unwrap();
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.message.as_text(), "hello");
        assert_eq!(r.message.tool_calls[0].name, "ls");
        assert_eq!(r.message.tool_calls[0].arguments["path"], ".");
        assert_eq!(r.usage.input_tokens, 12);
    }
}
