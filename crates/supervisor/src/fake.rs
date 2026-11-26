//! A test-only fake engine over HTTP + SSE, and an isolated test directory.
//!
//! Trimmed port of oxa-fake's `FakeServer`/`TestEnv`: only what this crate's tests exercise —
//! the cache model, prefill (`max_tokens: 0`), tool-call and reasoning deltas, usage/timings,
//! and the faults a transport test needs. Never links weights and never generates anything.
#![doc(hidden)]
use crate::proto::{codes, Finish, OxaError, Role, ToolCall, Usage, Value, WireMsg};
use serde_json::json;
use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

/// Server-wide options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeOpts {
    /// `0` = pick a free port.
    pub port: u16,
    pub n_ctx: u32,
    pub model: String,
    /// Owned native-engine build identity; `None` models an older/unidentified peer.
    pub build_info: Option<String>,
    /// Accepted for source compatibility with the controller-endpoint fake. The trimmed fake
    /// serves only the plain engine routes.
    pub controller: bool,
}

impl Default for FakeOpts {
    fn default() -> Self {
        FakeOpts {
            port: 0,
            n_ctx: 1 << 20,
            model: "fake-model".to_string(),
            build_info: Some(format!("oxe-{}", env!("CARGO_PKG_VERSION"))),
            controller: false,
        }
    }
}

/// A fault injected instead of, or during, a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// Answer with this HTTP status and body instead of a stream.
    Http { status: u16, body: String },
    /// Answer 400 `exceed_context_size_error`.
    Overflow,
    /// Close the connection after this many bytes of the SSE body.
    DropAt { bytes: usize },
    /// Send only `: queued` comments every second for this long before the reply.
    CommentsOnly { ms: u64 },
    /// `count` queued comments in the unified format before the reply, one per 10 ms.
    Queued { count: u32, pos: bool },
    /// An error after early headers, without `[DONE]`.
    StreamError {
        queued: u32,
        status: u16,
        error_type: String,
        event: bool,
    },
    /// The controller's queue-full reply: 503 with `Retry-After` when `Some`.
    Busy { retry_after: Option<u32> },
}

/// One scripted reply. A request with `max_tokens > 0` consumes the next turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptTurn {
    pub reasoning: Option<String>,
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish: Finish,
    /// `None` = canonical reply bytes + 1 (the fake's cache model).
    pub completion_tokens: Option<u32>,
    /// Delta size in bytes for streaming (0 = one delta per field).
    pub chunk: usize,
    pub fault: Option<Fault>,
}

impl ScriptTurn {
    pub fn text(t: &str) -> ScriptTurn {
        ScriptTurn {
            reasoning: None,
            text: Some(t.to_string()),
            tool_calls: Vec::new(),
            finish: Finish::Stop,
            completion_tokens: None,
            chunk: 0,
            fault: None,
        }
    }
    pub fn tool(id: &str, name: &str, args_raw: &str) -> ScriptTurn {
        ScriptTurn {
            reasoning: None,
            text: None,
            tool_calls: vec![ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                args_raw: args_raw.to_string(),
            }],
            finish: Finish::ToolCalls,
            completion_tokens: None,
            chunk: 0,
            fault: None,
        }
    }
    pub fn fault(f: Fault) -> ScriptTurn {
        ScriptTurn {
            fault: Some(f),
            ..ScriptTurn::text("")
        }
    }
    pub fn message(&self) -> WireMsg {
        WireMsg {
            role: Role::Assistant,
            content: self.text.clone(),
            reasoning: self.reasoning.clone(),
            tool_calls: self.tool_calls.clone(),
            tool_call_id: None,
        }
    }
}

/// One request as seen by the fake engine, plus what it answered.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorded {
    pub seq: usize,
    pub system: String,
    pub tools_json: String,
    pub messages: Vec<WireMsg>,
    pub model: String,
    pub max_tokens: u32,
    pub thinking: bool,
    pub session: String,
    pub idem_key: String,
    pub kind: String,
    pub background: bool,
    pub snapshot_keep: bool,
    pub preserve_thinking: Option<bool>,
    pub temperature: f64,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub reply: Option<WireMsg>,
    pub usage: Option<Usage>,
}

/// The bytes the fake's cache model counts.
pub fn cache_bytes(system: &str, tools_json: &str, msgs: &[WireMsg]) -> Vec<u8> {
    let mut v = Vec::with_capacity(system.len() + tools_json.len() + 2);
    v.extend_from_slice(system.as_bytes());
    v.push(0);
    v.extend_from_slice(tools_json.as_bytes());
    v.push(0);
    for m in msgs {
        m.write_json(&mut v);
        v.push(b'\n');
    }
    v
}

struct Shared {
    opts: FakeOpts,
    core: Mutex<Core>,
    state: Arc<Mutex<Vec<Recorded>>>,
    hits: Arc<Mutex<Vec<(String, String)>>>,
}

struct Core {
    turns: VecDeque<ScriptTurn>,
    /// Per session: the bytes the engine holds (last prompt + reply).
    cache: Vec<(String, Vec<u8>)>,
}

/// A running fake engine on 127.0.0.1.
pub struct FakeServer {
    port: u16,
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<Vec<Recorded>>>,
    hits: Arc<Mutex<Vec<(String, String)>>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl FakeServer {
    pub fn start(script: Vec<ScriptTurn>, opts: FakeOpts) -> Result<FakeServer, OxaError> {
        let listener = TcpListener::bind(("127.0.0.1", opts.port))
            .map_err(|e| OxaError::io("bind fake engine port", &e))?;
        let port = listener
            .local_addr()
            .map_err(|e| OxaError::io("fake engine address", &e))?
            .port();
        listener
            .set_nonblocking(true)
            .map_err(|e| OxaError::io("fake engine listener", &e))?;
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(Vec::new()));
        let hits = Arc::new(Mutex::new(Vec::new()));
        let shared = Arc::new(Shared {
            opts,
            core: Mutex::new(Core {
                turns: script.into(),
                cache: Vec::new(),
            }),
            state: state.clone(),
            hits: hits.clone(),
        });
        let (stop2, shared2) = (stop.clone(), shared.clone());
        let thread = std::thread::spawn(move || accept_loop(listener, stop2, shared2));
        Ok(FakeServer {
            port,
            stop,
            state,
            hits,
            thread: Some(thread),
        })
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn requests(&self) -> Vec<Recorded> {
        lock(&self.state).clone()
    }

    /// How many requests with this method and path the fake has seen.
    pub fn hits(&self, method: &str, path: &str) -> usize {
        lock(&self.hits)
            .iter()
            .filter(|(m, p)| m == method && p == path)
            .count()
    }
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn accept_loop(listener: TcpListener, stop: Arc<AtomicBool>, shared: Arc<Shared>) {
    while !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let sh = shared.clone();
                std::thread::spawn(move || {
                    let _ = handle(stream, &sh);
                });
            }
            Err(_) => std::thread::sleep(Duration::from_millis(5)),
        }
    }
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn handle(mut stream: TcpStream, sh: &Shared) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let req = match read_request(&mut stream) {
        Ok(r) => r,
        Err(_) => return Ok(()),
    };
    lock(&sh.hits).push((req.method.clone(), req.path.clone()));
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => respond_json(&mut stream, 200, &json!({"status":"ok"})),
        ("GET", "/props") => respond_json(
            &mut stream,
            200,
            &json!({"build_info": sh.opts.build_info, "model": sh.opts.model}),
        ),
        ("POST", "/state-cache/clear") => respond_json(&mut stream, 200, &json!({"status":"ok"})),
        ("POST", "/v1/chat/completions") => chat(&mut stream, sh, &req),
        _ => respond_json(
            &mut stream,
            404,
            &json!({"error": {"code": "not_found", "message": "not found"}}),
        ),
    }
}

fn read_request(s: &mut TcpStream) -> Result<Request, String> {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match s.read(&mut byte) {
            Ok(0) => return Err("connection closed before headers".into()),
            Ok(_) => head.push(byte[0]),
            Err(e) => return Err(e.to_string()),
        }
        if head.ends_with(b"\r\n\r\n") {
            break;
        }
        if head.len() > 1 << 20 {
            return Err("headers too large".into());
        }
    }
    let text = String::from_utf8(head).map_err(|_| "headers are not UTF-8".to_string())?;
    let text = text.strip_suffix("\r\n\r\n").unwrap_or(&text);
    let (start, block) = text.split_once("\r\n").unwrap_or((text, ""));
    let mut parts = start.split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let path = parts
        .next()
        .unwrap_or("")
        .split('?')
        .next()
        .unwrap_or("")
        .to_string();
    let mut headers = Vec::new();
    for l in block.split("\r\n") {
        if let Some((k, v)) = l.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let len: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    if len > 64 << 20 {
        return Err("body too large".into());
    }
    let mut tmp = [0u8; 4096];
    let mut body = Vec::new();
    while body.len() < len {
        let n = s.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("connection closed inside the body".into());
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(len);
    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

fn respond(s: &mut TcpStream, status: u16, ctype: &str, body: &[u8]) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason(status),
        body.len()
    );
    s.write_all(head.as_bytes())?;
    s.write_all(body)?;
    s.flush()
}

fn respond_json(s: &mut TcpStream, status: u16, v: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(v).expect("json serialises");
    respond(s, status, "application/json", &body)
}

fn overflow_value(prompt: usize, n_ctx: u32) -> Value {
    json!({"error": {
        "code": 400,
        "type": "exceed_context_size_error",
        "message": "the request exceeds the available context size",
        "n_prompt_tokens": prompt,
        "n_ctx": n_ctx,
    }})
}

fn chat(s: &mut TcpStream, sh: &Shared, req: &Request) -> io::Result<()> {
    let mut rec = match parse_recorded(0, &req.headers, &req.body) {
        Ok(r) => r,
        Err(e) => {
            let body =
                serde_json::to_vec(&json!({"error": {"code": e.code, "message": e.message}}))
                    .expect("json serialises");
            return respond(s, 400, "application/json", &body);
        }
    };
    let mut core = lock(&sh.core);
    let seen_len = lock(&sh.state).len();
    rec.seq = seen_len;
    let n_ctx = sh.opts.n_ctx;
    let prompt = cache_bytes(&rec.system, &rec.tools_json, &rec.messages);
    let plen = prompt.len();
    let cached = cached_of(&core.cache, &rec.session, &prompt);

    if plen as u64 >= n_ctx as u64 {
        lock(&sh.state).push(rec);
        return respond_json(s, 400, &overflow_value(plen, n_ctx));
    }

    if rec.max_tokens == 0 {
        let usage = Usage {
            prompt: plen as u32,
            completion: 0,
            cached: cached as u32,
            cached_missing: false,
            unreported: false,
        };
        rec.usage = Some(usage);
        core.cache.retain(|(s, _)| *s != rec.session);
        core.cache.push((rec.session.clone(), prompt));
        let mut body = format!(": prefill {plen}/{plen} 0s\n\n").into_bytes();
        sse_data(
            &mut body,
            &delta_value(json!({"role": "assistant", "content": null})),
        );
        sse_data(
            &mut body,
            &json!({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": rec.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}),
        );
        sse_data(
            &mut body,
            &json!({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": rec.model,
                    "choices": [], "usage": usage_value(usage, 0),
                    "timings": timings_value(plen, cached, 0)}),
        );
        body.extend_from_slice(b"data: [DONE]\n\n");
        lock(&sh.state).push(rec);
        return write_stream(s, StreamPlan::new(body));
    }

    let turn = match core.turns.pop_front() {
        Some(t) => t,
        None => {
            lock(&sh.state).push(rec);
            return respond_json(
                s,
                500,
                &json!({"error": {"code": codes::INTERNAL, "message": "script exhausted"}}),
            );
        }
    };
    let mut pre = Vec::new();
    match &turn.fault {
        Some(Fault::Overflow) => {
            lock(&sh.state).push(rec);
            return respond_json(s, 400, &overflow_value(plen, n_ctx));
        }
        Some(Fault::Http { status, body }) => {
            lock(&sh.state).push(rec);
            return respond(s, *status, "application/json", body.as_bytes());
        }
        Some(Fault::Busy { retry_after }) => {
            lock(&sh.state).push(rec);
            let secs = retry_after.map_or(String::new(), |n| format!(" in {n} s"));
            let body = json!({"error": {"message": format!("Too many requests are waiting for the local model; retry{secs}."),
                                        "type": "local_backend_busy", "param": null, "code": "queue_full"}});
            let bytes = serde_json::to_vec(&body).expect("json serialises");
            let mut head = format!(
                "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
                bytes.len()
            );
            if let Some(n) = retry_after {
                head.push_str(&format!("Retry-After: {n}\r\n"));
            }
            head.push_str("Connection: close\r\n\r\n");
            s.write_all(head.as_bytes())?;
            s.write_all(&bytes)?;
            return s.flush();
        }
        Some(Fault::StreamError {
            queued,
            status,
            error_type,
            event,
        }) => {
            let pre = (0..*queued)
                .map(|i| queued_comment(Some(queued - i), i))
                .collect();
            let mut body = Vec::new();
            if *event {
                body.extend_from_slice(b"event: error\n");
            }
            let value = if error_type == "exceed_context_size_error" {
                overflow_value(plen, n_ctx)
            } else {
                json!({"error": {"code": status, "message": format!("the engine failed with HTTP {status}"), "type": error_type}})
            };
            sse_data(&mut body, &value);
            lock(&sh.state).push(rec);
            return write_stream(
                s,
                StreamPlan {
                    pre,
                    ..StreamPlan::new(body)
                },
            );
        }
        Some(Fault::Queued { count, pos }) => {
            pre = (0..*count)
                .map(|i| queued_comment(pos.then_some(count - i), i))
                .collect();
        }
        _ => {}
    }

    let comments_ms = match turn.fault {
        Some(Fault::CommentsOnly { ms }) => ms,
        _ => 0,
    };
    let cut = match turn.fault {
        Some(Fault::DropAt { bytes }) => Some(bytes),
        _ => None,
    };

    let room = n_ctx - plen as u32;
    let (msg, finish, completion) = generate(&turn, rec.max_tokens.min(room));

    let mut body = Vec::new();
    if let Some(r) = &msg.reasoning {
        for c in split_chunks(r, turn.chunk) {
            sse_data(&mut body, &delta_value(json!({"reasoning_content": c})));
        }
    }
    if let Some(t) = &msg.content {
        for c in split_chunks(t, turn.chunk) {
            sse_data(&mut body, &delta_value(json!({"content": c})));
        }
    }
    for (index, call) in msg.tool_calls.iter().enumerate() {
        for (j, part) in split_chunks(&call.args_raw, turn.chunk)
            .into_iter()
            .enumerate()
        {
            let mut tc = json!({"index": index, "function": {"arguments": part}});
            if j == 0 {
                tc["id"] = json!(call.id);
                tc["type"] = json!("function");
                tc["function"]["name"] = json!(call.name);
            }
            sse_data(&mut body, &delta_value(json!({"tool_calls": [tc]})));
        }
    }
    let usage = Usage {
        prompt: plen as u32,
        completion,
        cached: cached as u32,
        cached_missing: false,
        unreported: false,
    };
    let fin = json!({
        "id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": rec.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_value(finish)}],
        "usage": usage_value(usage, msg.reasoning.as_deref().map_or(0, str::len)),
        "timings": timings_value(plen, cached, completion),
    });
    sse_data(&mut body, &fin);
    body.extend_from_slice(b"data: [DONE]\n\n");

    if cut.is_none() {
        let mut held = prompt;
        held.extend_from_slice(&msg.to_json());
        rec.reply = Some(msg);
        rec.usage = Some(usage);
        core.cache.retain(|(s, _)| *s != rec.session);
        core.cache.push((rec.session.clone(), held));
    }
    lock(&sh.state).push(rec);
    write_stream(
        s,
        StreamPlan {
            pre,
            comments_ms,
            cut,
            ..StreamPlan::new(body)
        },
    )
}

fn sse_data(out: &mut Vec<u8>, v: &Value) {
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(&serde_json::to_vec(v).expect("json serialises"));
    out.extend_from_slice(b"\n\n");
}

fn delta_value(delta: Value) -> Value {
    json!({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
           "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}]})
}

fn usage_value(u: Usage, reasoning_tokens: usize) -> Value {
    let mut v = json!({
        "prompt_tokens": u.prompt,
        "completion_tokens": u.completion,
        "total_tokens": u.prompt + u.completion,
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    });
    if !u.cached_missing {
        v["prompt_tokens_details"] = json!({"cached_tokens": u.cached});
    }
    v
}

fn timings_value(prompt: usize, cached: usize, predicted: u32) -> Value {
    json!({
        "cache_n": cached,
        "prompt_n": prompt - cached,
        "prompt_ms": 0.0,
        "predicted_n": predicted,
        "predicted_ms": 0.0,
    })
}

fn finish_value(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "stop",
        Finish::ToolCalls => "tool_calls",
        Finish::Length => "length",
        Finish::ContentFilter => "content_filter",
    }
}

fn split_chunks(s: &str, chunk: usize) -> Vec<&str> {
    if chunk == 0 || s.len() <= chunk {
        return vec![s];
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < s.len() {
        let mut end = (start + chunk).min(s.len());
        while end < s.len() && !s.is_char_boundary(end) {
            end += 1;
        }
        out.push(&s[start..end]);
        start = end;
    }
    out
}

fn generate(turn: &ScriptTurn, limit: u32) -> (WireMsg, Finish, u32) {
    let msg = turn.message();
    let natural = turn
        .completion_tokens
        .unwrap_or(msg.to_json().len() as u32 + 1);
    if natural <= limit {
        return (msg, turn.finish, natural);
    }
    let mut budget = limit as usize;
    let mut cut = |s: &Option<String>| -> Option<String> {
        let t = s.as_ref()?;
        let mut end = budget.min(t.len());
        while !t.is_char_boundary(end) {
            end -= 1;
        }
        budget -= end;
        Some(t[..end].to_string())
    };
    let reasoning = cut(&msg.reasoning);
    let text = cut(&msg.content);
    let out = WireMsg {
        role: Role::Assistant,
        content: text,
        reasoning,
        tool_calls: Vec::new(),
        tool_call_id: None,
    };
    (out, Finish::Length, limit)
}

fn queued_comment(pos: Option<u32>, wait_s: u32) -> Vec<u8> {
    match pos {
        Some(p) => format!(": queued pos={p} wait_s={wait_s}\n\n").into_bytes(),
        None => format!(": queued wait_s={wait_s}\n\n").into_bytes(),
    }
}

struct StreamPlan {
    comments_ms: u64,
    pre: Vec<Vec<u8>>,
    body: Vec<u8>,
    cut: Option<usize>,
}

impl StreamPlan {
    fn new(body: Vec<u8>) -> StreamPlan {
        StreamPlan {
            comments_ms: 0,
            pre: Vec::new(),
            body,
            cut: None,
        }
    }
}

fn write_stream(s: &mut TcpStream, p: StreamPlan) -> io::Result<()> {
    s.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
          Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )?;
    let mut waited = 0u64;
    while waited < p.comments_ms {
        chunk(s, &queued_comment(None, (waited / 1000) as u32))?;
        let step = (p.comments_ms - waited).min(1000);
        std::thread::sleep(Duration::from_millis(step));
        waited += step;
    }
    for c in &p.pre {
        chunk(s, c)?;
        std::thread::sleep(Duration::from_millis(10));
    }
    if let Some(n) = p.cut {
        let n = n.min(p.body.len());
        chunk(s, &p.body[..n])?;
        let _ = s.shutdown(Shutdown::Both);
        return Ok(());
    }
    chunk(s, &p.body)?;
    s.write_all(b"0\r\n\r\n")?;
    s.flush()
}

fn chunk(s: &mut TcpStream, data: &[u8]) -> io::Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    s.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
    s.write_all(data)?;
    s.write_all(b"\r\n")?;
    s.flush()
}

/// The anchor of a prompt in the cache model: system text, `\0`, tools JSON, `\0`.
fn anchor_len(prompt: &[u8]) -> Option<usize> {
    let first = prompt.iter().position(|&b| b == 0)?;
    let second = prompt[first + 1..].iter().position(|&b| b == 0)?;
    Some(first + 1 + second + 1)
}

fn cached_of(cache: &[(String, Vec<u8>)], session: &str, prompt: &[u8]) -> usize {
    let own = cache
        .iter()
        .find(|(s, _)| s == session)
        .map(|(_, held)| held.iter().zip(prompt).take_while(|(a, b)| a == b).count())
        .unwrap_or(0);
    let shared = anchor_len(prompt)
        .filter(|&n| {
            cache
                .iter()
                .any(|(_, held)| held.len() >= n && held[..n] == prompt[..n])
        })
        .unwrap_or(0);
    own.max(shared)
}

fn parse_recorded(
    seq: usize,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<Recorded, OxaError> {
    let invalid = |msg: String| OxaError::new(codes::INVALID, msg);
    let text =
        std::str::from_utf8(body).map_err(|_| invalid("request body is not UTF-8".into()))?;
    let v: Value = serde_json::from_str(text)
        .map_err(|e| invalid(format!("request body is not JSON: {e}")))?;
    let obj = v
        .as_object()
        .ok_or_else(|| invalid("request body is not a JSON object".into()))?;
    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let max_tokens = obj
        .get("max_tokens")
        .and_then(Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
        .ok_or_else(|| invalid("`max_tokens` must be a non-negative integer".into()))?;
    let thinking = obj
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let preserve_thinking = obj
        .get("chat_template_kwargs")
        .and_then(|k| k.get("preserve_thinking"))
        .and_then(Value::as_bool);
    let temperature = obj
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let msgs = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("`messages` must be an array".into()))?;
    let mut system = String::new();
    let mut messages = Vec::new();
    for (i, m) in msgs.iter().enumerate() {
        let w = wire_from_value(m)?;
        if i == 0 && w.role == Role::System {
            system = w.content.unwrap_or_default();
        } else {
            messages.push(w);
        }
    }
    let tools_json = obj
        .get("tools")
        .map(|v| v.to_string())
        .unwrap_or_else(|| "[]".to_string());
    let h = |name: &str| -> Option<String> {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
    };
    Ok(Recorded {
        seq,
        system,
        tools_json,
        messages,
        model,
        max_tokens,
        thinking,
        session: h("x-oxe-session")
            .or_else(|| h("x-session-id"))
            .unwrap_or_default(),
        idem_key: h("idempotency-key").unwrap_or_default(),
        kind: h("x-oxe-turn-kind").unwrap_or_default(),
        background: h("x-oxe-priority").as_deref() == Some("background"),
        snapshot_keep: h("x-oxe-snapshot").as_deref() == Some("keep"),
        preserve_thinking,
        temperature,
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
            .collect(),
        body: body.to_vec(),
        reply: None,
        usage: None,
    })
}

fn wire_from_value(m: &Value) -> Result<WireMsg, OxaError> {
    let invalid = |msg: String| OxaError::new(codes::INVALID, msg);
    let role_s = m
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("message without a role".into()))?;
    let role = Role::parse(role_s).ok_or_else(|| invalid(format!("unknown role `{role_s}`")))?;
    let content = match m.get("content") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(invalid("message content must be a string or null".into())),
    };
    let mut out = WireMsg {
        role,
        content,
        reasoning: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
    };
    match role {
        Role::Tool => {
            out.tool_call_id = m
                .get("tool_call_id")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        Role::Assistant => {
            out.reasoning = m
                .get("reasoning_content")
                .and_then(Value::as_str)
                .map(str::to_string);
            if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                for c in calls {
                    let f = c
                        .get("function")
                        .ok_or_else(|| invalid("tool call without a function".into()))?;
                    let id = c
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let name = f
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let args_raw = match f.get("arguments") {
                        Some(Value::String(s)) => s.clone(),
                        Some(other) => other.to_string(),
                        None => String::new(),
                    };
                    out.tool_calls.push(ToolCall { id, name, args_raw });
                }
            }
        }
        Role::System | Role::User => {}
    }
    Ok(out)
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// A private directory tree for one test, removed on drop.
#[derive(Debug)]
pub struct TestEnv {
    root: PathBuf,
}

impl TestEnv {
    pub fn new() -> TestEnv {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!(
            "oxe-supervisor-test-{}_{}_{}",
            std::process::id(),
            n,
            nanos
        ));
        for d in ["home", "tmp", "work"] {
            std::fs::create_dir_all(root.join(d)).expect("create test dir");
        }
        TestEnv { root }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn work(&self) -> PathBuf {
        self.root.join("work")
    }
    pub fn write(&self, rel: &str, content: &str) -> PathBuf {
        let p = self.root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        std::fs::write(&p, content).expect("write test file");
        p
    }
}

impl Default for TestEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
