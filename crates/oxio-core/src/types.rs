//! Frozen domain vocabulary + error type. Do NOT redefine - modules build on these.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Crate-wide error.
#[derive(Debug, Error)]
pub enum OxioError {
    #[error("provider error: {0}")]
    Provider(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("no provider registered")]
    NoProvider,
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("cancelled")]
    Cancelled,
    #[error("timed out")]
    Timeout,
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, OxioError>;

/// Who authored a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A content block. Extensible later (image, etc.) without redefining `Message`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    Text {
        text: String,
    },
    /// Model reasoning/thinking, kept separate from the answer so the UI can show
    /// it collapsibly (Ctrl+O) and it never pollutes `as_text()`.
    Thinking {
        text: String,
    },
    /// An image (base64-encoded) for multimodal providers. `media_type` is a MIME
    /// like "image/png". Vision adapters serialize this to their wire format;
    /// text-only adapters ignore it. Additive - `as_text()`/`thinking()` skip it.
    Image {
        media_type: String,
        data: String,
    },
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// One conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: Vec<Content>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Build a plain-text message.
    pub fn text(role: Role, s: impl Into<String>) -> Self {
        Message {
            role,
            content: vec![Content::Text { text: s.into() }],
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// The answer text (concatenate Text blocks; excludes thinking).
    pub fn as_text(&self) -> String {
        let mut out = String::new();
        for c in &self.content {
            if let Content::Text { text } = c {
                out.push_str(text);
            }
        }
        out
    }

    /// The model's reasoning/thinking, if any (concatenate Thinking blocks).
    pub fn thinking(&self) -> String {
        let mut out = String::new();
        for c in &self.content {
            if let Content::Thinking { text } = c {
                out.push_str(text);
            }
        }
        out
    }
}

/// What a [`crate::traits::Tool`] returns; the kernel wraps it into a `Message`
/// carrying the originating tool-call id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        ToolOutput {
            content: content.into(),
            is_error: false,
        }
    }
    pub fn error(content: impl Into<String>) -> Self {
        ToolOutput {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Token accounting for a model call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
    }
}

/// Sampling parameters. Grows without redefining callers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Params {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Reasoning/thinking mode: `Some(true)`/`Some(false)` explicitly turns the
    /// model's thinking on/off for this turn; `None` leaves the model's default.
    /// Wire adapters map it to their engine's control (e.g. chat-completions →
    /// `chat_template_kwargs.enable_thinking`). Runtime-toggled from the UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
}

/// Tool description advertised to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool input.
    pub input_schema: Value,
}

/// A single model request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub params: Params,
}

/// Why the model stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Stop,
}

/// A single model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub message: Message,
    #[serde(default)]
    pub usage: Usage,
    pub stop_reason: StopReason,
}

/// Streaming vocabulary - defined now so it is never redefined, even though the
/// first providers may only implement `complete`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StreamEvent {
    /// Model reasoning delta - the UI shows this collapsibly (Ctrl+O), separate
    /// from the answer.
    ThinkingDelta(String),
    TextDelta(String),
    ToolCallDelta {
        id: String,
        name: String,
        arguments_delta: String,
    },
    /// Terminal event carrying the assembled message the kernel uses for turn
    /// logic (tool-call dispatch), plus stop reason and usage.
    Done {
        stop_reason: StopReason,
        usage: Usage,
        message: Message,
    },
    /// Out-of-band UI notice that is NOT model output. Emitted by modules at hook
    /// points (tool dispatch, compaction, provider fallback, memory ops) so runtime
    /// work surfaces to the user instead of dying silently. Additive extension in
    /// the same class as `Content::Image`; sinks that don't care ignore it.
    Notice {
        level: NoticeLevel,
        text: String,
    },
}

/// Severity/kind of a [`StreamEvent::Notice`], so the UI can style it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoticeLevel {
    /// Transient progress: "compacting…", "→ read_file …". Rendered dimmed.
    Progress,
    /// Informational: fallback switch, memory saved.
    Info,
    /// Something the user should notice: degraded mode, retry.
    Warn,
}
