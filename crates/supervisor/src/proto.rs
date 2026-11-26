//! The minimal private vocabulary the supervisor uses at the engine boundary.
//!
//! Trimmed copy of the oxa-proto / oxa-engine request types this crate needs: the typed chat
//! request, the reply it assembles, and one error type. There is no agent, session, hook or
//! provider vocabulary here.
pub use serde_json::Value;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Poll, Waker};

/// The one error type of the supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OxaError {
    pub code: &'static str,
    pub message: String,
    pub hint: Option<String>,
}

impl OxaError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        OxaError {
            code,
            message: message.into(),
            hint: None,
        }
    }
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
    pub fn unsupported(what: &str) -> Self {
        OxaError::new(codes::UNSUPPORTED, format!("{what} is not supported"))
    }
    pub fn io(context: &str, e: &std::io::Error) -> Self {
        OxaError::new(codes::IO, format!("{context}: {e}"))
    }
}

impl fmt::Display for OxaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(h) = &self.hint {
            write!(f, "\n{h}")?;
        }
        Ok(())
    }
}

impl std::error::Error for OxaError {}

/// Message prefix of a `stream_broken` that wraps a refusal arriving after reply data.
pub const CUT_BY_REFUSAL: &str = "the reply was cut: ";

/// The stable error codes this crate uses.
pub mod codes {
    pub const ENGINE_DOWN: &str = "engine_down";
    pub const ENGINE_HTTP: &str = "engine_http";
    pub const CONTEXT_OVERFLOW: &str = "context_overflow";
    pub const STREAM_BROKEN: &str = "stream_broken";
    pub const ABORTED: &str = "aborted";
    pub const UNSUPPORTED: &str = "unsupported";
    pub const IO: &str = "io";
    pub const NOT_FOUND: &str = "not_found";
    pub const INVALID: &str = "invalid";
    pub const BUSY: &str = "busy";
    pub const ENGINE_BUSY: &str = "engine_busy";
    pub const TIMEOUT: &str = "timeout";
    pub const INTERNAL: &str = "internal";
}

/// A cooperative cancellation flag shared between the caller and a worker.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<CancelState>);

#[derive(Debug, Default)]
struct CancelState {
    flag: AtomicBool,
    waiters: Mutex<Vec<Weak<Mutex<Option<Waker>>>>>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.flag.store(true, Ordering::SeqCst);
        let waiters =
            std::mem::take(&mut *self.0.waiters.lock().unwrap_or_else(|e| e.into_inner()));
        for waiter in waiters.into_iter().filter_map(|waiter| waiter.upgrade()) {
            let waker = waiter.lock().unwrap_or_else(|e| e.into_inner()).take();
            if let Some(waker) = waker {
                waker.wake();
            }
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.flag.load(Ordering::SeqCst)
    }
    /// Wake an async operation once cancelled, without a timer or worker thread.
    pub fn cancelled(&self) -> impl std::future::Future<Output = ()> + Send + 'static {
        let token = self.clone();
        async move {
            if token.is_cancelled() {
                return;
            }
            let waiter = Arc::new(Mutex::new(None::<Waker>));
            {
                let mut waiters = token.0.waiters.lock().unwrap_or_else(|e| e.into_inner());
                waiters.retain(|waiter| waiter.strong_count() > 0);
                waiters.push(Arc::downgrade(&waiter));
            }
            std::future::poll_fn(move |context| {
                let mut registered = waiter.lock().unwrap_or_else(|e| e.into_inner());
                if !registered
                    .as_ref()
                    .is_some_and(|waker| waker.will_wake(context.waker()))
                {
                    *registered = Some(context.waker().clone());
                }
                if token.is_cancelled() {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            })
            .await;
        }
    }
}

/// Chat role on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
    pub fn parse(s: &str) -> Option<Role> {
        Some(match s {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => return None,
        })
    }
}

/// One tool call of an assistant message. `args_raw` is the streamed argument string exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args_raw: String,
}

/// One message of the wire (`messages` of `/v1/chat/completions`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireMsg {
    pub role: Role,
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

impl WireMsg {
    pub fn user(text: impl Into<String>) -> Self {
        Self::plain(Role::User, text.into())
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::plain(Role::Assistant, text.into())
    }
    pub fn tool_result(tool_call_id: impl Into<String>, text: impl Into<String>) -> Self {
        WireMsg {
            role: Role::Tool,
            content: Some(text.into()),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
    fn plain(role: Role, text: String) -> Self {
        WireMsg {
            role,
            content: Some(text),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// Append the canonical JSON bytes of this message to `out` (fixed key order, no
    /// whitespace). This is the only serializer of a wire message.
    pub fn write_json(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(b"{\"role\":\"");
        out.extend_from_slice(self.role.as_str().as_bytes());
        out.push(b'"');
        if self.role == Role::Tool {
            out.extend_from_slice(b",\"tool_call_id\":");
            write_json_str(self.tool_call_id.as_deref().unwrap_or(""), out);
        }
        out.extend_from_slice(b",\"content\":");
        match &self.content {
            Some(c) => write_json_str(c, out),
            None => out.extend_from_slice(b"null"),
        }
        if self.role == Role::Assistant {
            if let Some(r) = &self.reasoning {
                out.extend_from_slice(b",\"reasoning_content\":");
                write_json_str(r, out);
            }
            if !self.tool_calls.is_empty() {
                out.extend_from_slice(b",\"tool_calls\":[");
                for (i, t) in self.tool_calls.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(b"{\"id\":");
                    write_json_str(&t.id, out);
                    out.extend_from_slice(b",\"type\":\"function\",\"function\":{\"name\":");
                    write_json_str(&t.name, out);
                    out.extend_from_slice(b",\"arguments\":");
                    write_json_str(&t.args_raw, out);
                    out.extend_from_slice(b"}}");
                }
                out.push(b']');
            }
        }
        out.push(b'}');
    }

    pub fn to_json(&self) -> Vec<u8> {
        let mut v = Vec::new();
        self.write_json(&mut v);
        v
    }
}

/// Append `s` as a JSON string literal.
pub fn write_json_str(s: &str, out: &mut Vec<u8>) {
    serde_json::to_writer(&mut *out, s).expect("writing a string to a Vec cannot fail");
}

/// H5 `X-Oxe-Turn-Kind`: a free ledger label; it never changes generated bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TurnKind {
    Chat,
    Compaction,
    BranchSummary,
    Prefill,
    Subagent,
}

impl TurnKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TurnKind::Chat => "chat",
            TurnKind::Compaction => "compaction",
            TurnKind::BranchSummary => "branch_summary",
            TurnKind::Prefill => "prefill",
            TurnKind::Subagent => "subagent",
        }
    }
}

/// Token usage of one request as reported by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt: u32,
    pub completion: u32,
    pub cached: u32,
    pub cached_missing: bool,
    pub unreported: bool,
}

impl Usage {
    pub fn unreported() -> Self {
        Self {
            cached_missing: true,
            unreported: true,
            ..Self::default()
        }
    }
}

/// Why the engine stopped. An aborted or failed stream has no `Finish`: it is an `Err`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
}

/// `timings.*` of the engine's final chunk.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Timings {
    pub cache_n: u32,
    pub prompt_n: u32,
    pub prompt_ms: f64,
    pub predicted_n: u32,
    pub predicted_ms: f64,
    pub snap_pos: Option<u32>,
    pub snap_asked: bool,
}

/// One increment of a streaming reply, in arrival order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    Warning(&'static str),
    Prefill {
        done: u32,
        total: u32,
        secs: u32,
    },
    Queued {
        pos: Option<u32>,
        wait_s: u32,
    },
    Reasoning(String),
    Text(String),
    ToolArgs {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        delta: String,
    },
}

/// A complete, successful reply.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantTurn {
    pub msg: WireMsg,
    pub finish: Finish,
    pub usage: Usage,
    pub timings: Option<Timings>,
    pub model: String,
}

/// Sampling parameters sent with a request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
}

/// The optional, client-neutral hints of the engine protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hints<'a> {
    pub session: &'a str,
    pub idem_key: &'a str,
    pub kind: TurnKind,
    pub background: bool,
    pub snapshot_keep: bool,
}

impl Default for Hints<'_> {
    fn default() -> Self {
        Hints {
            session: "",
            idem_key: "",
            kind: TurnKind::Chat,
            background: false,
            snapshot_keep: false,
        }
    }
}

/// One chat request. The body is a pure function of this value.
#[derive(Debug, Clone, Copy)]
pub struct ChatRequest<'a> {
    pub prefix: &'a FrozenPrefix,
    pub messages: &'a [WireMsg],
    pub model: &'a str,
    pub max_tokens: u32,
    pub thinking: bool,
    pub thinking_budget: Option<u64>,
    pub preserve_thinking: bool,
    pub sampling: Option<Sampling>,
    pub hints: Hints<'a>,
}

impl<'a> ChatRequest<'a> {
    pub fn new(prefix: &'a FrozenPrefix, messages: &'a [WireMsg], model: &'a str) -> Self {
        ChatRequest {
            prefix,
            messages,
            model,
            max_tokens: 1024,
            thinking: false,
            thinking_budget: None,
            preserve_thinking: false,
            sampling: None,
            hints: Hints::default(),
        }
    }
}

/// The frozen prefix of one epoch: the system prompt sections and the tool list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenPrefix {
    pub sections: Vec<(String, String)>,
    pub tools_json: String,
    pub sha256: [u8; 32],
}

impl FrozenPrefix {
    /// The system message text: the section texts joined by one blank line.
    pub fn system_text(&self) -> String {
        let mut s = String::new();
        for (i, (_, text)) in self.sections.iter().enumerate() {
            if i > 0 {
                s.push_str("\n\n");
            }
            s.push_str(text);
        }
        s
    }
}

/// What the engine reports about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    pub base_url: String,
    pub models: Vec<String>,
    pub model: String,
    pub limits_for: String,
    pub n_ctx: u32,
    pub max_output: u32,
    pub queue: u32,
    pub compact_at: Option<u32>,
    pub prefill_only: bool,
    pub caps: EngineCaps,
    pub thinking: Option<bool>,
    pub images: Option<bool>,
}

/// What the engine lets a client use with a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngineCaps {
    pub hint_clear: bool,
    pub remote: bool,
    pub cancel: bool,
    pub context_window: Option<u32>,
}

/// The engine as the supervisor sees it.
pub trait Engine: Send + Sync {
    fn info(&self, target: Option<&str>) -> Result<EngineInfo, OxaError>;
    fn local_memory_status(&self, _model: &str) -> Option<(u32, &'static str, u64)> {
        None
    }
    fn count_tokens(&self, text: &str) -> Result<u32, OxaError>;
    fn chat(
        &self,
        r: &ChatRequest,
        cancel: &CancelToken,
        on: &mut dyn FnMut(StreamEvent),
    ) -> Result<AssistantTurn, OxaError>;
    fn clear_hint(&self, hint: &str) -> Result<(), OxaError> {
        let _ = hint;
        Ok(())
    }
}

/// JSON value type used at the engine boundary.
pub type JsonValue = Value;
