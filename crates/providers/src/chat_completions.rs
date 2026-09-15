//! OpenAI-compatible Chat Completions wire adapter. Talks to any server speaking
//! `/v1/chat/completions` - local (Ollama, LM Studio, llama.cpp, vLLM) and most
//! vendors. Implements `Provider::complete` (non-stream) and `Provider::stream`
//! (real SSE with text + tool-call deltas).

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};

use oxio_core::{
    Content, Ctx, Message, OxioError, Provider, Request, Response, Result, Role, StopReason,
    StreamEvent, ToolCall, ToolSpec, Usage,
};

use crate::thinking::{Seg, ThinkingSplitter};

/// Per-provider sampling defaults. For a reasoning model set `reasoning_effort`
/// and temperature/top_p are omitted (they are ignored/rejected there); for a
/// standard model set temperature/top_p/max_tokens. Per-turn `Params` override.
#[derive(Clone, Default)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub reasoning_effort: Option<String>,
}

pub struct ChatCompletionsAdapter {
    name: String,
    base_url: String,
    model: String,
    api_key: Option<String>,
    sampling: Sampling,
    http: reqwest::Client,
}

impl ChatCompletionsAdapter {
    /// `base_url` is the API root (e.g. `http://localhost:11434/v1`); the
    /// adapter appends `/chat/completions`.
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
        format!("{}/chat/completions", self.base_url)
    }

    fn build_body(&self, req: &Request, stream: bool) -> Value {
        let messages: Vec<Value> = coalesce_system(&req.messages)
            .iter()
            .map(msg_to_oai)
            .collect();
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
        });
        if !req.tools.is_empty() {
            body["tools"] = req.tools.iter().map(tool_to_oai).collect::<Vec<_>>().into();
        }
        // Model-class-aware sampling (per-turn Params override provider defaults):
        // reasoning models take `reasoning_effort` and IGNORE temperature/top_p;
        // standard models take temperature/top_p. max_tokens applies to both.
        let s = &self.sampling;
        if let Some(effort) = &s.reasoning_effort {
            body["reasoning_effort"] = effort.clone().into();
        } else {
            if let Some(t) = req.params.temperature.or(s.temperature) {
                body["temperature"] = t.into();
            }
            if let Some(p) = req.params.top_p.or(s.top_p) {
                body["top_p"] = p.into();
            }
        }
        if let Some(m) = req.params.max_tokens.or(s.max_tokens) {
            body["max_tokens"] = m.into();
        }
        if !req.params.stop.is_empty() {
            body["stop"] = req.params.stop.clone().into();
        }
        // Thinking/reasoning toggle → `chat_template_kwargs.enable_thinking`, the
        // vLLM / llama.cpp-server / Qwen convention. Sent explicitly for BOTH on and off
        // (the server honours `false` = answer-only); `None` (auto) omits it so the
        // model keeps its own default. Thinking then returns in `reasoning_content`.
        if let Some(on) = req.params.thinking {
            body["chat_template_kwargs"] = json!({ "enable_thinking": on });
        }
        if stream {
            body["stream_options"] = json!({ "include_usage": true });
        }
        body
    }
}

#[async_trait]
impl Provider for ChatCompletionsAdapter {
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
            // On error, consume the body so the reason surfaces - a single-flight local
            // server (one model loaded at a time) returns its "busy / reloading" or
            // request-rejection detail here. Matches the non-streaming
            // `complete` path, which already includes the body; the stream path dropped it.
            // Rebind `resp` so the success path still owns it (the error arm diverges via `?`).
            let resp = if status.is_success() {
                resp
            } else {
                let body = resp.text().await.unwrap_or_default();
                let detail = body.trim();
                let msg = if detail.is_empty() {
                    format!("http {status}")
                } else {
                    format!("http {status}: {detail}")
                };
                Err(OxioError::Provider(msg))?;
                unreachable!("Err(..)? above diverges");
            };

            let mut bytes = resp.bytes_stream();
            let mut buf = String::new();
            let mut text = String::new();
            let mut think = String::new();
            let mut splitter = ThinkingSplitter::new();
            let mut calls: Vec<(String, String, String)> = Vec::new(); // (id, name, args)
            let mut usage = Usage::default();
            let mut stop = StopReason::EndTurn;

            'outer: while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|e| OxioError::Provider(e.to_string()))?;
                buf.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim().to_string();
                    buf.drain(..=pos);
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
                    if let Some(u) = v.get("usage") {
                        if let Some(p) = u.get("prompt_tokens").and_then(|x| x.as_u64()) {
                            usage.input_tokens = p;
                        }
                        if let Some(c) = u.get("completion_tokens").and_then(|x| x.as_u64()) {
                            usage.output_tokens = c;
                        }
                    }
                    let choice = match v.get("choices").and_then(|c| c.get(0)) {
                        Some(c) => c,
                        None => continue,
                    };
                    if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
                        stop = map_finish(Some(fr));
                    }
                    let delta = match choice.get("delta") {
                        Some(d) => d,
                        None => continue,
                    };
                    if let Some(r) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
                        if !r.is_empty() {
                            think.push_str(r);
                            yield StreamEvent::ThinkingDelta(r.to_string());
                        }
                    }
                    if let Some(c) = delta.get("content").and_then(|x| x.as_str()) {
                        if !c.is_empty() {
                            for seg in splitter.push(c) {
                                match seg {
                                    Seg::Thinking(t) => { think.push_str(&t); yield StreamEvent::ThinkingDelta(t); }
                                    Seg::Answer(a) => { text.push_str(&a); yield StreamEvent::TextDelta(a); }
                                }
                            }
                        }
                    }
                    if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
                        for tc in tcs {
                            let idx = tc.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                            while calls.len() <= idx {
                                calls.push((String::new(), String::new(), String::new()));
                            }
                            if let Some(id) = tc.get("id").and_then(|x| x.as_str()) {
                                if !id.is_empty() {
                                    calls[idx].0 = id.to_string();
                                }
                            }
                            if let Some(f) = tc.get("function") {
                                if let Some(n) = f.get("name").and_then(|x| x.as_str()) {
                                    if !n.is_empty() {
                                        calls[idx].1 = n.to_string();
                                    }
                                }
                                if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                                    calls[idx].2.push_str(a);
                                    yield StreamEvent::ToolCallDelta {
                                        id: calls[idx].0.clone(),
                                        name: calls[idx].1.clone(),
                                        arguments_delta: a.to_string(),
                                    };
                                }
                            }
                        }
                    }
                }
            }

            for seg in splitter.finish() {
                match seg {
                    Seg::Thinking(t) => { think.push_str(&t); yield StreamEvent::ThinkingDelta(t); }
                    Seg::Answer(a) => { text.push_str(&a); yield StreamEvent::TextDelta(a); }
                }
            }

            let mut tool_calls: Vec<ToolCall> = calls
                .into_iter()
                .filter(|(id, name, _)| !id.is_empty() || !name.is_empty())
                .map(|(id, name, args)| ToolCall {
                    id,
                    name,
                    arguments: crate::toolcalls::parse_tool_args(&args),
                })
                .collect();
            // Fallback: some local servers emit tool calls as TEXT, not native
            // `tool_calls`. The toolcalls layer parses the common formats.
            if tool_calls.is_empty() {
                let (tc, cleaned) = crate::toolcalls::parse(&text);
                if !tc.is_empty() {
                    tool_calls = tc;
                    text = cleaned;
                }
            }
            if !tool_calls.is_empty() && stop == StopReason::EndTurn {
                stop = StopReason::ToolUse;
            }
            let mut content = Vec::new();
            if !think.is_empty() {
                content.push(Content::Thinking { text: think });
            }
            if !text.is_empty() {
                content.push(Content::Text { text });
            }
            let message = Message { role: Role::Assistant, content, tool_calls, tool_call_id: None };
            yield StreamEvent::Done { stop_reason: stop, usage, message };
        };

        Ok(Box::pin(s))
    }
}

fn map_finish(reason: Option<&str>) -> StopReason {
    match reason {
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("stop") | None => StopReason::EndTurn,
        _ => StopReason::EndTurn,
    }
}

/// Message content for the wire: a plain string, or (when image blocks are
/// present) an OpenAI vision content array `[{type:text},{type:image_url}]`.
fn content_value(m: &Message, text: &str) -> Value {
    if !m.content.iter().any(|c| matches!(c, Content::Image { .. })) {
        return if text.is_empty() {
            Value::Null
        } else {
            json!(text)
        };
    }
    let mut parts = Vec::new();
    if !text.is_empty() {
        parts.push(json!({ "type": "text", "text": text }));
    }
    for c in &m.content {
        if let Content::Image { media_type, data } = c {
            parts.push(json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{media_type};base64,{data}") }
            }));
        }
    }
    json!(parts)
}

/// Merge consecutive `System` messages into one. oxio assembles the system
/// prompt and other system content (memory, env) as SEPARATE system messages via
/// PreTurn transformers; but strict chat templates (Qwen3, gemma) raise
/// "System message must be at the beginning" on any system message past index 0.
/// Coalescing to a single leading system message is model-agnostic - lenient
/// templates are unaffected - and is a widely-accepted shape. Text is joined with a
/// blank line; non-system messages
/// pass through untouched.
fn coalesce_system(msgs: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(msgs.len());
    for m in msgs {
        if m.role == Role::System {
            if let Some(last) = out.last_mut() {
                if last.role == Role::System {
                    let (a, b) = (last.as_text(), m.as_text());
                    let joined = match (a.is_empty(), b.is_empty()) {
                        (false, false) => format!("{a}\n\n{b}"),
                        (true, _) => b.to_string(),
                        (_, true) => a.to_string(),
                    };
                    *last = Message::text(Role::System, joined);
                    continue;
                }
            }
        }
        out.push(m.clone());
    }
    out
}

fn msg_to_oai(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let text = m.as_text();
    let mut o = json!({ "role": role });
    if m.role == Role::Tool {
        o["tool_call_id"] = json!(m.tool_call_id.clone().unwrap_or_default());
        o["content"] = json!(text);
    } else if !m.tool_calls.is_empty() {
        o["content"] = content_value(m, &text);
        o["tool_calls"] = m
            .tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".into()),
                    }
                })
            })
            .collect::<Vec<_>>()
            .into();
    } else {
        o["content"] = content_value(m, &text);
    }
    o
}

fn tool_to_oai(t: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
    })
}

fn parse_response(v: &Value) -> Result<Response> {
    let choice = v
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or_else(|| OxioError::Provider("response had no choices".into()))?;
    let m = choice
        .get("message")
        .ok_or_else(|| OxioError::Provider("choice had no message".into()))?;
    let raw = m.get("content").and_then(|x| x.as_str()).unwrap_or("");
    let mut think = String::new();
    if let Some(r) = m.get("reasoning_content").and_then(|x| x.as_str()) {
        think.push_str(r);
    }
    let mut answer = String::new();
    let mut sp = ThinkingSplitter::new();
    let mut segs = sp.push(raw);
    segs.extend(sp.finish());
    for s in segs {
        match s {
            Seg::Thinking(t) => think.push_str(&t),
            Seg::Answer(a) => answer.push_str(&a),
        }
    }
    let mut tool_calls: Vec<ToolCall> = m
        .get("tool_calls")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let id = tc.get("id")?.as_str()?.to_string();
                    let f = tc.get("function")?;
                    let name = f.get("name")?.as_str()?.to_string();
                    let args = f.get("arguments").and_then(|x| x.as_str()).unwrap_or("{}");
                    Some(ToolCall {
                        id,
                        name,
                        arguments: serde_json::from_str(args).unwrap_or_else(|_| json!({})),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    // Fallback: parse text-format tool calls when native ones are absent.
    if tool_calls.is_empty() {
        let (tc, cleaned) = crate::toolcalls::parse(&answer);
        if !tc.is_empty() {
            tool_calls = tc;
            answer = cleaned;
        }
    }
    let mut stop = map_finish(choice.get("finish_reason").and_then(|x| x.as_str()));
    if !tool_calls.is_empty() && stop == StopReason::EndTurn {
        stop = StopReason::ToolUse;
    }
    let u = v.get("usage");
    let usage = Usage {
        input_tokens: u
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|x| x.as_u64())
            .unwrap_or(0),
        output_tokens: u
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|x| x.as_u64())
            .unwrap_or(0),
    };
    let mut content = Vec::new();
    if !think.is_empty() {
        content.push(Content::Thinking { text: think });
    }
    if !answer.is_empty() {
        content.push(Content::Text { text: answer });
    }
    let message = Message {
        role: Role::Assistant,
        content,
        tool_calls,
        tool_call_id: None,
    };
    Ok(Response {
        message,
        usage,
        stop_reason: stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_tool_result_message() {
        let mut m = Message::text(Role::Tool, "echo: hi");
        m.tool_call_id = Some("call_1".into());
        let o = msg_to_oai(&m);
        assert_eq!(o["role"], "tool");
        assert_eq!(o["tool_call_id"], "call_1");
        assert_eq!(o["content"], "echo: hi");
    }

    #[test]
    fn maps_image_content_to_vision_array() {
        let m = Message {
            role: Role::User,
            content: vec![
                Content::Text {
                    text: "what is this?".into(),
                },
                Content::Image {
                    media_type: "image/png".into(),
                    data: "AAAA".into(),
                },
            ],
            tool_calls: vec![],
            tool_call_id: None,
        };
        let o = msg_to_oai(&m);
        assert_eq!(o["content"][0]["type"], "text");
        assert_eq!(o["content"][0]["text"], "what is this?");
        assert_eq!(o["content"][1]["type"], "image_url");
        assert_eq!(
            o["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    fn req(params: oxio_core::Params) -> Request {
        Request {
            model: "m".into(),
            messages: vec![],
            tools: vec![],
            params,
        }
    }

    #[test]
    fn sampling_standard_sends_temperature_and_top_p() {
        let a =
            ChatCompletionsAdapter::new("p", "http://x/v1", "m", None).with_sampling(Sampling {
                temperature: Some(0.7),
                top_p: Some(0.9),
                max_tokens: Some(256),
                reasoning_effort: None,
            });
        let b = a.build_body(&req(oxio_core::Params::default()), false);
        assert!((b["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-4);
        assert!((b["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-4);
        assert_eq!(b["max_tokens"], 256);
        assert!(b.get("reasoning_effort").is_none());
    }

    #[test]
    fn sampling_reasoning_sends_effort_and_omits_temperature() {
        let a =
            ChatCompletionsAdapter::new("p", "http://x/v1", "m", None).with_sampling(Sampling {
                temperature: Some(0.7),
                top_p: Some(0.9),
                max_tokens: Some(256),
                reasoning_effort: Some("high".into()),
            });
        let b = a.build_body(&req(oxio_core::Params::default()), false);
        assert_eq!(b["reasoning_effort"], "high");
        assert!(
            b.get("temperature").is_none(),
            "reasoning models ignore temperature - must be omitted"
        );
        assert!(b.get("top_p").is_none());
        assert_eq!(b["max_tokens"], 256);
    }

    #[test]
    fn consecutive_system_messages_coalesce_to_one() {
        use oxio_core::Role;
        let msgs = vec![
            Message::text(Role::System, "prompt"),
            Message::text(Role::System, "memory"),
            Message::text(Role::User, "hi"),
        ];
        let out = coalesce_system(&msgs);
        assert_eq!(
            out.len(),
            2,
            "two system messages merge into one, user preserved"
        );
        assert_eq!(out[0].role, Role::System);
        assert_eq!(out[0].as_text(), "prompt\n\nmemory");
        assert_eq!(out[1].role, Role::User);
    }

    #[test]
    fn thinking_toggle_maps_to_chat_template_kwargs() {
        let a = ChatCompletionsAdapter::new("p", "http://x/v1", "m", None);
        // auto (None) → no flag sent (model keeps its default)
        let auto = a.build_body(&req(oxio_core::Params::default()), false);
        assert!(
            auto.get("chat_template_kwargs").is_none(),
            "auto sends no thinking flag"
        );
        // on → enable_thinking: true
        let on = a.build_body(
            &req(oxio_core::Params {
                thinking: Some(true),
                ..Default::default()
            }),
            false,
        );
        assert_eq!(on["chat_template_kwargs"]["enable_thinking"], true);
        // off → enable_thinking: false (server honours answer-only)
        let off = a.build_body(
            &req(oxio_core::Params {
                thinking: Some(false),
                ..Default::default()
            }),
            false,
        );
        assert_eq!(off["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn sampling_per_turn_params_override_provider_default() {
        let a =
            ChatCompletionsAdapter::new("p", "http://x/v1", "m", None).with_sampling(Sampling {
                temperature: Some(0.7),
                ..Default::default()
            });
        let b = a.build_body(
            &req(oxio_core::Params {
                temperature: Some(0.1),
                ..Default::default()
            }),
            false,
        );
        assert!(
            (b["temperature"].as_f64().unwrap() - 0.1).abs() < 1e-4,
            "per-turn Params override the provider default"
        );
    }

    #[test]
    fn maps_assistant_tool_call() {
        let m = Message {
            role: Role::Assistant,
            content: vec![],
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                arguments: json!({"value":"x"}),
            }],
            tool_call_id: None,
        };
        let o = msg_to_oai(&m);
        assert_eq!(o["tool_calls"][0]["function"]["name"], "echo");
        // arguments serialized as a JSON string per the OpenAI wire format
        assert_eq!(
            o["tool_calls"][0]["function"]["arguments"],
            "{\"value\":\"x\"}"
        );
    }

    #[test]
    fn parses_text_response() {
        let v = json!({
            "choices": [ { "message": { "content": "pong" }, "finish_reason": "stop" } ],
            "usage": { "prompt_tokens": 5, "completion_tokens": 1 }
        });
        let r = parse_response(&v).unwrap();
        assert_eq!(r.message.as_text(), "pong");
        assert_eq!(r.stop_reason, StopReason::EndTurn);
        assert_eq!(r.usage.output_tokens, 1);
    }

    #[test]
    fn parses_tool_call_response() {
        let v = json!({
            "choices": [ {
                "message": { "content": null, "tool_calls": [
                    { "id": "c1", "type": "function", "function": { "name": "echo", "arguments": "{\"value\":\"hi\"}" } }
                ]},
                "finish_reason": "tool_calls"
            } ]
        });
        let r = parse_response(&v).unwrap();
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.message.tool_calls.len(), 1);
        assert_eq!(r.message.tool_calls[0].name, "echo");
        assert_eq!(r.message.tool_calls[0].arguments["value"], "hi");
    }
}
