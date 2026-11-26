//! The minimal HTTP/1.1 + SSE client used at the owned loopback engine boundary.
//!
//! Trimmed copy of the pieces of oxa-engine / oxa-http this crate uses: the pure request
//! builder, a bounded SSE reader, the reply assembler and the shared error classification.
//! Plain `std::net::TcpStream`; no async runtime, proxy, redirect or TLS.
pub use crate::proto::Value;

use crate::proto::{
    codes, write_json_str, AssistantTurn, CancelToken, ChatRequest, Finish, Hints, OxaError,
    StreamEvent, Timings, ToolCall, Usage, WireMsg,
};
use std::fmt::Write as _;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// How often a blocked socket call wakes up to poll cancellation and the silence limit.
const POLL: Duration = Duration::from_millis(100);
/// Largest response head (status line + headers) or chunk-size line we accept.
const MAX_HEAD: usize = 64 * 1024;
const ENGINE_HINT: &str = "start oxe serve and check the configured engine binary";

/// A parsed `http://host[:port][/base]` URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    pub host: String,
    pub port: u16,
    pub base_path: String,
}

impl Url {
    pub fn parse(s: &str) -> Result<Url, OxaError> {
        let s = s.trim();
        let bad = |msg: String| OxaError::new(codes::INVALID, msg);
        if s.get(..8)
            .is_some_and(|p| p.eq_ignore_ascii_case("https://"))
        {
            return Err(bad(format!(
                "https is not supported for the engine URL: {s}"
            )));
        }
        let rest = match s.get(..7) {
            Some(p) if p.eq_ignore_ascii_case("http://") => &s[7..],
            _ => return Err(bad(format!("engine URL must start with http://: {s}"))),
        };
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        if path
            .chars()
            .any(|c| c == '?' || c == '#' || c.is_whitespace())
        {
            return Err(bad(format!(
                "engine URL has a query, fragment or whitespace: {s}"
            )));
        }
        if authority.contains('@') {
            return Err(bad(format!(
                "credentials in the engine URL are not supported: {s}"
            )));
        }
        let (host, port_text): (&str, Option<&str>) = match authority.strip_prefix('[') {
            Some(inner) => {
                let end = inner
                    .find(']')
                    .ok_or_else(|| bad(format!("unclosed [ in engine URL: {s}")))?;
                let after = &inner[end + 1..];
                match after.strip_prefix(':') {
                    Some(p) => (&inner[..end], Some(p)),
                    None if after.is_empty() => (&inner[..end], None),
                    None => return Err(bad(format!("bad engine URL authority: {s}"))),
                }
            }
            None => match authority.rsplit_once(':') {
                Some((h, p)) => (h, Some(p)),
                None => (authority, None),
            },
        };
        if host.is_empty()
            || host
                .chars()
                .any(|c| c.is_whitespace() || c == '[' || c == ']')
        {
            return Err(bad(format!("engine URL has no host: {s}")));
        }
        let port = match port_text {
            None => 80,
            Some(p) => p
                .parse::<u16>()
                .ok()
                .filter(|&p| p != 0)
                .ok_or_else(|| bad(format!("invalid port in engine URL: {p:?}")))?,
        };
        Ok(Url {
            host: host.to_string(),
            port,
            base_path: path.trim_end_matches('/').to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest<'a> {
    pub method: &'a str,
    pub url: &'a Url,
    pub path: &'a str,
    pub headers: &'a [(&'a str, String)],
    pub body: &'a [u8],
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// One non-streaming request (`Connection: close`; handles `Content-Length` and chunked bodies).
pub fn http_request(r: &HttpRequest) -> Result<HttpResponse, OxaError> {
    let never = CancelToken::new();
    let stream = connect(r.url, r.timeout_ms, &never)?;
    let mut conn = Conn::new(
        stream,
        never.clone(),
        r.timeout_ms,
        codes::TIMEOUT,
        codes::ENGINE_HTTP,
    )?;
    let req = build_request(r.method, r.url, r.path, r.headers, r.body)?;
    conn.write_all(&req)?;
    let (head, rest) = conn.read_head()?;
    let parsed = parse_head(&head).map_err(|m| OxaError::new(codes::ENGINE_HTTP, m))?;
    let framing = framing_of(&parsed.headers).map_err(|m| OxaError::new(codes::ENGINE_HTTP, m))?;
    let mut dec = BodyDecoder::new(framing);
    let mut body = Vec::new();
    dec.feed(&rest, &mut body)?;
    let mut buf = vec![0u8; 16 * 1024];
    while !dec.done() {
        let n = conn.read(&mut buf)?;
        if n == 0 {
            break;
        }
        dec.feed(&buf[..n], &mut body)?;
    }
    if !dec.complete() {
        return Err(OxaError::new(
            codes::ENGINE_HTTP,
            "the engine closed the connection before the response body was complete",
        ));
    }
    Ok(HttpResponse {
        status: parsed.status,
        headers: parsed.headers,
        body,
    })
}

/// Open a TCP connection to `url`, polling `cancel` while the connect runs.
fn connect(url: &Url, timeout_ms: u64, cancel: &CancelToken) -> Result<TcpStream, OxaError> {
    if cancel.is_cancelled() {
        return Err(aborted());
    }
    let timeout = Duration::from_millis(timeout_ms.max(1));
    let (tx, rx) = mpsc::channel();
    let (host, port) = (url.host.clone(), url.port);
    thread::Builder::new()
        .name("supervisor-connect".into())
        .spawn(move || {
            let res = connect_now(&host, port, timeout);
            let _ = tx.send(res);
        })
        .map_err(|e| OxaError::io("spawn connect thread", &e))?;
    loop {
        match rx.recv_timeout(POLL) {
            Ok(Ok(stream)) => return Ok(stream),
            Ok(Err(e)) => return Err(map_connect_err(url, &e)),
            Err(RecvTimeoutError::Timeout) => {
                if cancel.is_cancelled() {
                    return Err(aborted());
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(OxaError::new(
                    codes::INTERNAL,
                    "connect thread ended without a result",
                ))
            }
        }
    }
}

fn connect_now(host: &str, port: u16, timeout: Duration) -> io::Result<TcpStream> {
    let mut last = None;
    for addr in (host, port).to_socket_addrs()? {
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(stream) => return Ok(stream),
            Err(e) => last = Some(e),
        }
    }
    Err(last
        .unwrap_or_else(|| io::Error::new(ErrorKind::NotFound, "no address for the engine host")))
}

fn map_connect_err(url: &Url, e: &io::Error) -> OxaError {
    let at = format!("{}:{}", url.host, url.port);
    if e.kind() == ErrorKind::TimedOut {
        OxaError::new(
            codes::TIMEOUT,
            format!("connecting to the engine at {at} timed out"),
        )
    } else {
        OxaError::new(
            codes::ENGINE_DOWN,
            format!("cannot reach the engine at {at}: {e}"),
        )
        .with_hint(ENGINE_HINT)
    }
}

fn aborted() -> OxaError {
    OxaError::new(codes::ABORTED, "request cancelled")
}

/// Serialize one request: request line, `Host`, `Connection: close`, `Content-Length`.
fn build_request(
    method: &str,
    url: &Url,
    path: &str,
    headers: &[(&str, String)],
    body: &[u8],
) -> Result<Vec<u8>, OxaError> {
    let bad = |what: &str| OxaError::new(codes::INVALID, format!("invalid HTTP request: {what}"));
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(bad("method"));
    }
    if !path.starts_with('/') || path.bytes().any(|b| b <= b' ') {
        return Err(bad("path must start with / and contain no spaces"));
    }
    let host = if url.host.contains(':') {
        format!("[{}]:{}", url.host, url.port)
    } else {
        format!("{}:{}", url.host, url.port)
    };
    let mut head = String::with_capacity(256);
    let _ = write!(
        head,
        "{method} {}{path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Length: {}\r\n",
        url.base_path,
        body.len()
    );
    for (name, value) in headers {
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(bad("header name"));
        }
        if value.contains(['\r', '\n']) {
            return Err(bad("header value contains a line break"));
        }
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "host" | "content-length" | "connection" | "transfer-encoding"
        ) {
            continue;
        }
        let _ = write!(head, "{name}: {value}\r\n");
    }
    head.push_str("\r\n");
    let mut out = head.into_bytes();
    out.extend_from_slice(body);
    Ok(out)
}

/// A connection with the liveness rule: any byte resets the silence clock.
struct Conn {
    stream: TcpStream,
    cancel: CancelToken,
    silence: Duration,
    last: Instant,
    silent_code: &'static str,
    broken_code: &'static str,
}

impl Conn {
    fn new(
        stream: TcpStream,
        cancel: CancelToken,
        silence_ms: u64,
        silent_code: &'static str,
        broken_code: &'static str,
    ) -> Result<Self, OxaError> {
        let setup = |e: io::Error| OxaError::io("socket setup", &e);
        stream.set_nodelay(true).map_err(setup)?;
        let silence = Duration::from_millis(silence_ms.max(1));
        stream.set_read_timeout(Some(POLL)).map_err(setup)?;
        stream.set_write_timeout(Some(POLL)).map_err(setup)?;
        Ok(Conn {
            stream,
            cancel,
            silence,
            last: Instant::now(),
            silent_code,
            broken_code,
        })
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), OxaError> {
        let mut off = 0;
        let mut progress = Instant::now();
        while off < bytes.len() {
            if self.cancel.is_cancelled() {
                return Err(aborted());
            }
            match self.stream.write(&bytes[off..]) {
                Ok(0) => {
                    return Err(OxaError::new(
                        self.broken_code,
                        "request write failed: the engine closed the connection",
                    ))
                }
                Ok(n) => {
                    off += n;
                    progress = Instant::now();
                }
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    if progress.elapsed() >= self.silence {
                        return Err(self.silent_error());
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => {
                    return Err(OxaError::new(
                        self.broken_code,
                        format!("request write failed: {e}"),
                    ))
                }
            }
        }
        self.last = Instant::now();
        Ok(())
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<usize, OxaError> {
        loop {
            if self.cancel.is_cancelled() {
                return Err(aborted());
            }
            match self.stream.read(buf) {
                Ok(0) => return Ok(0),
                Ok(n) => {
                    self.last = Instant::now();
                    return Ok(n);
                }
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    if self.last.elapsed() >= self.silence {
                        return Err(self.silent_error());
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => {
                    return Err(OxaError::new(
                        self.broken_code,
                        format!("connection read failed: {e}"),
                    ))
                }
            }
        }
    }

    fn read_head(&mut self) -> Result<(Vec<u8>, Vec<u8>), OxaError> {
        let mut acc = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            if let Some(pos) = find(&acc, b"\r\n\r\n") {
                let rest = acc.split_off(pos + 4);
                acc.truncate(pos);
                return Ok((acc, rest));
            }
            if acc.len() > MAX_HEAD {
                return Err(OxaError::new(
                    self.broken_code,
                    "engine response head is too large",
                ));
            }
            let n = self.read(&mut chunk)?;
            if n == 0 {
                return Err(OxaError::new(
                    self.broken_code,
                    "the engine closed the connection before the response headers",
                ));
            }
            acc.extend_from_slice(&chunk[..n]);
        }
    }

    fn silent_error(&self) -> OxaError {
        OxaError::new(
            self.silent_code,
            format!(
                "no bytes from the engine for {} ms",
                self.silence.as_millis()
            ),
        )
    }
}

struct Head {
    status: u16,
    /// Names lowercased, values trimmed.
    headers: Vec<(String, String)>,
}

fn parse_head(head: &[u8]) -> Result<Head, String> {
    let text = String::from_utf8_lossy(head);
    let mut lines = text.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let mut parts = status_line.split_whitespace();
    let version = parts.next().unwrap_or("");
    let status = parts.next().and_then(|s| s.parse::<u16>().ok());
    if !version.starts_with("HTTP/") {
        return Err(format!("malformed status line: {status_line:?}"));
    }
    let status = status.ok_or_else(|| format!("malformed status line: {status_line:?}"))?;
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        match line.split_once(':') {
            Some((name, value)) => {
                headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()))
            }
            None => return Err(format!("malformed header line: {line:?}")),
        }
    }
    Ok(Head { status, headers })
}

fn header<'h>(headers: &'h [(String, String)], name: &str) -> Option<&'h str> {
    headers
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
}

/// How the body of a response ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Framing {
    Eof,
    Length(usize),
    Chunked,
}

fn framing_of(headers: &[(String, String)]) -> Result<Framing, String> {
    if let Some(te) = header(headers, "transfer-encoding") {
        if te.to_ascii_lowercase().contains("chunked") {
            return Ok(Framing::Chunked);
        }
    }
    match header(headers, "content-length") {
        Some(cl) => cl
            .parse::<usize>()
            .map(Framing::Length)
            .map_err(|_| format!("bad content-length: {cl:?}")),
        None => Ok(Framing::Eof),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkState {
    Size,
    Data(usize),
    DataEnd,
    Trailer,
}

/// Incremental body decoder for all three framings.
struct BodyDecoder {
    framing: Framing,
    chunk: ChunkState,
    buf: Vec<u8>,
    remaining: usize,
    done: bool,
}

impl BodyDecoder {
    fn new(framing: Framing) -> Self {
        BodyDecoder {
            framing,
            chunk: ChunkState::Size,
            buf: Vec::new(),
            remaining: match framing {
                Framing::Length(n) => n,
                _ => 0,
            },
            done: framing == Framing::Length(0),
        }
    }

    fn done(&self) -> bool {
        self.done
    }

    fn complete(&self) -> bool {
        self.done || self.framing == Framing::Eof
    }

    fn feed(&mut self, raw: &[u8], out: &mut Vec<u8>) -> Result<(), OxaError> {
        if self.done {
            return Ok(());
        }
        match self.framing {
            Framing::Eof => out.extend_from_slice(raw),
            Framing::Length(_) => {
                let n = raw.len().min(self.remaining);
                out.extend_from_slice(&raw[..n]);
                self.remaining -= n;
                self.done = self.remaining == 0;
            }
            Framing::Chunked => {
                self.buf.extend_from_slice(raw);
                self.decode_chunked(out)?;
            }
        }
        Ok(())
    }

    fn decode_chunked(&mut self, out: &mut Vec<u8>) -> Result<(), OxaError> {
        loop {
            match self.chunk {
                ChunkState::Size => {
                    let pos = match find(&self.buf, b"\r\n") {
                        Some(p) => p,
                        None if self.buf.len() > MAX_HEAD => {
                            return Err(bad_chunk("size line too long"))
                        }
                        None => return Ok(()),
                    };
                    let line = String::from_utf8_lossy(&self.buf[..pos]).into_owned();
                    let size_text = line.split(';').next().unwrap_or("").trim();
                    let size = usize::from_str_radix(size_text, 16)
                        .map_err(|_| bad_chunk(&format!("bad chunk size {size_text:?}")))?;
                    self.buf.drain(..pos + 2);
                    self.chunk = if size == 0 {
                        ChunkState::Trailer
                    } else {
                        ChunkState::Data(size)
                    };
                }
                ChunkState::Data(rem) => {
                    if self.buf.is_empty() {
                        return Ok(());
                    }
                    let n = rem.min(self.buf.len());
                    out.extend_from_slice(&self.buf[..n]);
                    self.buf.drain(..n);
                    self.chunk = if n == rem {
                        ChunkState::DataEnd
                    } else {
                        ChunkState::Data(rem - n)
                    };
                }
                ChunkState::DataEnd => {
                    if self.buf.len() < 2 {
                        return Ok(());
                    }
                    if &self.buf[..2] != b"\r\n" {
                        return Err(bad_chunk("missing CRLF after chunk data"));
                    }
                    self.buf.drain(..2);
                    self.chunk = ChunkState::Size;
                }
                ChunkState::Trailer => {
                    if self.buf.len() < 2 {
                        return Ok(());
                    }
                    if self.buf.starts_with(b"\r\n") {
                        self.buf.drain(..2);
                        self.done = true;
                        return Ok(());
                    }
                    match find(&self.buf, b"\r\n\r\n") {
                        Some(p) => {
                            self.buf.drain(..p + 4);
                            self.done = true;
                            return Ok(());
                        }
                        None if self.buf.len() > MAX_HEAD => {
                            return Err(bad_chunk("trailer too long"))
                        }
                        None => return Ok(()),
                    }
                }
            }
        }
    }
}

fn bad_chunk(what: &str) -> OxaError {
    OxaError::new(
        codes::STREAM_BROKEN,
        format!("malformed chunked body: {what}"),
    )
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

// ---- SSE framing and streaming ----

/// One framed SSE item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    Data(String),
    Comment(String),
    Done,
}

/// Byte-wise SSE framing. Lines end at `\n`, `\r\n` or a lone `\r`.
#[derive(Default)]
struct Framer {
    line: Vec<u8>,
    data: Vec<u8>,
    after_cr: bool,
}

enum Frame {
    Data(Vec<u8>),
    Comment(Vec<u8>),
}

impl Framer {
    fn push(&mut self, byte: u8) -> Option<Frame> {
        if byte == b'\n' && self.after_cr {
            self.after_cr = false;
            return None;
        }
        self.after_cr = byte == b'\r';
        if !matches!(byte, b'\r' | b'\n') {
            self.line.push(byte);
            return None;
        }
        let line = std::mem::take(&mut self.line);
        if line.is_empty() {
            if self.data.is_empty() {
                return None;
            }
            let mut data = std::mem::take(&mut self.data);
            data.pop();
            return Some(Frame::Data(data));
        }
        if let Some(comment) = line.strip_prefix(b":") {
            return Some(Frame::Comment(comment.to_vec()));
        }
        let data = if line == b"data" {
            Some(&b""[..])
        } else {
            line.strip_prefix(b"data:")
        };
        if let Some(data) = data {
            self.data
                .extend_from_slice(data.strip_prefix(b" ").unwrap_or(data));
            self.data.push(b'\n');
        }
        None
    }

    fn pending_line(&self) -> usize {
        self.line.len()
    }

    fn pending_data(&self) -> usize {
        self.data.len()
    }
}

/// Incremental SSE parser, correct for any split of the byte stream.
#[derive(Default)]
pub struct SseParser {
    framer: Framer,
    after_cr: bool,
    data_bytes: usize,
}

impl std::fmt::Debug for SseParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SseParser").finish_non_exhaustive()
    }
}

impl SseParser {
    pub fn new() -> SseParser {
        SseParser::default()
    }

    /// Bound incomplete lines and multiline event data before buffering another byte.
    pub fn feed_bounded(
        &mut self,
        bytes: &[u8],
        maximum: usize,
    ) -> Result<Vec<SseEvent>, OxaError> {
        let mut events = Vec::new();
        for &byte in bytes {
            if self.framer.pending_line().saturating_add(self.data_bytes) >= maximum {
                return Err(broken("SSE event exceeds its buffer limit"));
            }
            events.extend(self.push(byte));
        }
        Ok(events)
    }

    fn push(&mut self, byte: u8) -> Option<SseEvent> {
        let line = self.framer.pending_line();
        let data = self.framer.pending_data();
        let continues_crlf = byte == b'\n' && self.after_cr;
        self.after_cr = byte == b'\r';
        let frame = self.framer.push(byte);
        if matches!(byte, b'\r' | b'\n') && !continues_crlf {
            if line == 0 {
                self.data_bytes = 0;
            } else if self.framer.pending_data() > data {
                self.data_bytes = self.data_bytes.saturating_add(line);
            }
        }
        frame.map(|frame| match frame {
            Frame::Data(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                if text == "[DONE]" {
                    SseEvent::Done
                } else {
                    SseEvent::Data(text.into_owned())
                }
            }
            Frame::Comment(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                SseEvent::Comment(text.strip_prefix(' ').unwrap_or(&text).to_string())
            }
        })
    }
}

/// Whether the caller wants more events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamOpts {
    pub connect_timeout_ms: u64,
    pub silence_limit_ms: u64,
}

impl Default for StreamOpts {
    fn default() -> Self {
        StreamOpts {
            connect_timeout_ms: 5_000,
            silence_limit_ms: 300_000,
        }
    }
}

/// A decoded gateway stream item. The supervisor accepts only chunks and completion; other
/// native-gateway events carry no local replay authority.
pub enum GatewayEvent {
    Chunk(Value),
    Done,
    DoneWithHistory(std::sync::Arc<dyn Fn(&str) + Send + Sync>),
}

/// How a stream ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEnd {
    pub status: u16,
    pub got_data: bool,
    pub done: bool,
    pub error_body: Option<String>,
    pub retry_after: Option<u32>,
}

/// The SSE transport with a cap on buffered events and HTTP error bodies.
pub struct BoundedStreamOpts {
    pub stream: StreamOpts,
    pub max_event_bytes: usize,
}

pub fn post_sse_bounded(
    url: &Url,
    path: &str,
    headers: &[(&str, String)],
    body: &[u8],
    opts: &BoundedStreamOpts,
    cancel: &CancelToken,
    on: &mut dyn FnMut(SseEvent) -> Flow,
) -> Result<SseEnd, OxaError> {
    if opts.max_event_bytes == 0 {
        return Err(broken("SSE buffer limit must be positive"));
    }
    let maximum = opts.max_event_bytes;
    let stream_opts = &opts.stream;
    let mut all: Vec<(&str, String)> = headers.to_vec();
    if !has_header(headers, "content-type") {
        all.push(("Content-Type", "application/json".to_string()));
    }
    if !has_header(headers, "accept") {
        all.push(("Accept", "text/event-stream".to_string()));
    }
    let req = build_request("POST", url, path, &all, body)?;

    let stream = connect(url, stream_opts.connect_timeout_ms, cancel)?;
    let mut conn = Conn::new(
        stream,
        cancel.clone(),
        stream_opts.silence_limit_ms,
        codes::STREAM_BROKEN,
        codes::STREAM_BROKEN,
    )?;
    conn.write_all(&req)?;

    let (head, rest) = conn.read_head()?;
    let parsed = parse_head(&head).map_err(|m| broken(&m))?;
    let status = parsed.status;
    let framing = framing_of(&parsed.headers).map_err(|m| broken(&m))?;
    let mut dec = BodyDecoder::new(framing);

    if !(200..300).contains(&status) {
        let mut body = Vec::new();
        let mut failure = dec.feed(&rest, &mut body).err();
        if body.len() > maximum {
            return Err(broken("HTTP error body exceeds its buffer limit"));
        }
        let mut buf = [0u8; 4096];
        while failure.is_none() && !dec.done() {
            match conn.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = dec.feed(&buf[..n], &mut body) {
                        failure = Some(e);
                    }
                    if body.len() > maximum {
                        return Err(broken("HTTP error body exceeds its buffer limit"));
                    }
                }
                Err(e) => failure = Some(e),
            }
        }
        if let Some(e) = failure {
            if e.code == codes::ABORTED {
                return Err(e);
            }
        }
        let retry_after =
            header(&parsed.headers, "retry-after").and_then(|v| v.trim().parse::<u32>().ok());
        return Ok(SseEnd {
            status,
            got_data: false,
            done: false,
            error_body: Some(String::from_utf8_lossy(&body).into_owned()),
            retry_after,
        });
    }

    let mut parser = SseParser::new();
    let mut end = SseEnd {
        status,
        got_data: false,
        done: false,
        error_body: None,
        retry_after: None,
    };
    if deliver(&rest, &mut dec, &mut parser, &mut end, maximum, on)? {
        return Ok(end);
    }
    let mut chunk = vec![0u8; 16 * 1024];
    while !dec.done() {
        let n = conn.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        if deliver(&chunk[..n], &mut dec, &mut parser, &mut end, maximum, on)? {
            return Ok(end);
        }
    }
    if end.done {
        return Ok(end);
    }
    Err(broken("the engine closed the stream before [DONE]"))
}

/// An open raw response: status, headers and the body bytes as they arrive.
///
/// This is the passthrough form used by `oxe serve`: payload bytes are returned exactly as the
/// engine wrote them (SSE comments included) and the caller decides how to frame them.
pub struct RawResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    conn: Conn,
    dec: BodyDecoder,
    pending: Vec<u8>,
    done: bool,
}

/// POST `body` and return the response head plus an open body reader. Non-2xx statuses are
/// returned the same way; their bodies are read through [`RawResponse::read_chunk`].
pub fn open_raw(
    url: &Url,
    path: &str,
    headers: &[(&str, String)],
    body: &[u8],
    opts: &StreamOpts,
    cancel: &CancelToken,
) -> Result<RawResponse, OxaError> {
    let mut all: Vec<(&str, String)> = headers.to_vec();
    if !has_header(headers, "content-type") {
        all.push(("Content-Type", "application/json".to_string()));
    }
    if !has_header(headers, "accept") {
        all.push(("Accept", "text/event-stream".to_string()));
    }
    let req = build_request("POST", url, path, &all, body)?;
    let stream = connect(url, opts.connect_timeout_ms, cancel)?;
    let mut conn = Conn::new(
        stream,
        cancel.clone(),
        opts.silence_limit_ms,
        codes::STREAM_BROKEN,
        codes::STREAM_BROKEN,
    )?;
    conn.write_all(&req)?;
    let (head, rest) = conn.read_head()?;
    let parsed = parse_head(&head).map_err(|m| broken(&m))?;
    let framing = framing_of(&parsed.headers).map_err(|m| broken(&m))?;
    Ok(RawResponse {
        status: parsed.status,
        headers: parsed.headers,
        conn,
        dec: BodyDecoder::new(framing),
        pending: rest,
        done: false,
    })
}

impl RawResponse {
    /// The next payload bytes, or `None` at the end of the response body.
    pub fn read_chunk(&mut self) -> Result<Option<Vec<u8>>, OxaError> {
        if self.done {
            return Ok(None);
        }
        let mut out = Vec::new();
        loop {
            if !self.pending.is_empty() {
                let pending = std::mem::take(&mut self.pending);
                self.dec.feed(&pending, &mut out)?;
                if self.dec.done() {
                    self.done = true;
                }
                if !out.is_empty() {
                    return Ok(Some(out));
                }
                if self.done {
                    return Ok(None);
                }
            }
            if self.dec.done() {
                self.done = true;
                return Ok(None);
            }
            let mut buf = [0u8; 16 * 1024];
            let n = self.conn.read(&mut buf)?;
            if n == 0 {
                self.done = true;
                return Ok(None);
            }
            self.dec.feed(&buf[..n], &mut out)?;
            if self.dec.done() {
                self.done = true;
            }
            if !out.is_empty() {
                return Ok(Some(out));
            }
            if self.done {
                return Ok(None);
            }
        }
    }
}

fn deliver(
    bytes: &[u8],
    dec: &mut BodyDecoder,
    parser: &mut SseParser,
    end: &mut SseEnd,
    maximum: usize,
    on: &mut dyn FnMut(SseEvent) -> Flow,
) -> Result<bool, OxaError> {
    let mut payload = Vec::new();
    dec.feed(bytes, &mut payload)?;
    let events = parser.feed_bounded(&payload, maximum)?;
    for ev in events {
        let is_done = ev == SseEvent::Done;
        match ev {
            SseEvent::Data(_) => end.got_data = true,
            SseEvent::Done => end.done = true,
            SseEvent::Comment(_) => {}
        }
        let flow = on(ev);
        if is_done || flow == Flow::Stop {
            return Ok(true);
        }
    }
    Ok(false)
}

fn has_header(headers: &[(&str, String)], name: &str) -> bool {
    headers.iter().any(|(n, _)| n.eq_ignore_ascii_case(name))
}

fn broken(what: &str) -> OxaError {
    OxaError::new(codes::STREAM_BROKEN, what.to_string())
}

// ---- request body, headers and reply assembly ----

/// Append the request body for `r` to `out`. Pure; the only body builder.
pub fn build_body(r: &ChatRequest, out: &mut Vec<u8>) {
    out.extend_from_slice(b"{\"model\":");
    write_json_str(r.model, out);
    out.extend_from_slice(b",\"messages\":[{\"role\":\"system\",\"content\":");
    write_json_str(&r.prefix.system_text(), out);
    out.push(b'}');
    for m in r.messages {
        out.push(b',');
        m.write_json(out);
    }
    out.push(b']');
    let tools = r.prefix.tools_json.trim();
    if !tools.is_empty() && tools != "[]" {
        out.extend_from_slice(b",\"tools\":");
        out.extend_from_slice(r.prefix.tools_json.as_bytes());
    }
    out.extend_from_slice(
        b",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"temperature\":",
    );
    match &r.sampling {
        None => out.push(b'0'),
        Some(s) => {
            let mut keys = format!(
                "{},\"top_p\":{},\"top_k\":{},\"min_p\":{},\"presence_penalty\":{}",
                s.temperature, s.top_p, s.top_k, s.min_p, s.presence_penalty
            );
            if let Some(seed) = s.seed {
                keys.push_str(&format!(",\"seed\":{seed}"));
            }
            out.extend_from_slice(keys.as_bytes());
        }
    }
    if r.thinking && r.max_tokens > 0 {
        if let Some(budget) = r.thinking_budget {
            out.extend_from_slice(b",\"thinking_budget\":");
            out.extend_from_slice(budget.to_string().as_bytes());
        }
    }
    out.extend_from_slice(b",\"cache_prompt\":true,\"max_tokens\":");
    out.extend_from_slice(r.max_tokens.to_string().as_bytes());
    out.extend_from_slice(b",\"chat_template_kwargs\":{");
    if r.preserve_thinking {
        out.extend_from_slice(b"\"preserve_thinking\":true,");
    }
    out.extend_from_slice(b"\"enable_thinking\":");
    out.extend_from_slice(if r.thinking { b"true" } else { b"false" });
    out.extend_from_slice(b"}}");
}

/// The hint headers for one request, in this fixed order, each only when applicable.
pub fn build_headers(h: &Hints) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::with_capacity(6);
    if !h.session.is_empty() {
        out.push(("X-Oxe-Session", h.session.to_string()));
        out.push(("X-Session-Id", h.session.to_string()));
    }
    if !h.idem_key.is_empty() {
        out.push(("Idempotency-Key", h.idem_key.to_string()));
    }
    if h.background {
        out.push(("X-Oxe-Priority", "background".to_string()));
    }
    if h.snapshot_keep {
        out.push(("X-Oxe-Snapshot", "keep".to_string()));
    }
    out.push(("X-Oxe-Turn-Kind", h.kind.as_str().to_string()));
    out
}

/// Builds an [`AssistantTurn`] from the `data:` chunks of one stream.
#[derive(Debug, Default)]
pub struct ReplyAssembler {
    content: Option<String>,
    reasoning: Option<String>,
    calls: Vec<ToolCall>,
    finish: Option<Finish>,
    usage: Option<Usage>,
    timings: Option<Timings>,
    model: String,
    indices: Vec<usize>,
    in_think: bool,
    think_done: bool,
    held: String,
}

impl ReplyAssembler {
    pub fn new() -> ReplyAssembler {
        ReplyAssembler::default()
    }

    pub fn chunk(&mut self, data: &str) -> Result<Vec<StreamEvent>, OxaError> {
        let v: Value = serde_json::from_str(data)
            .map_err(|e| broken(&format!("malformed stream chunk: {e}")))?;
        self.chunk_value(&v)
    }

    pub fn chunk_value(&mut self, v: &Value) -> Result<Vec<StreamEvent>, OxaError> {
        let obj = v
            .as_object()
            .ok_or_else(|| broken("a stream chunk is not a JSON object"))?;
        if let Some(err) = obj.get("error") {
            return Err(stream_error(err));
        }
        if let Some(m) = obj.get("model").and_then(Value::as_str) {
            self.model = m.to_string();
        }
        if let Some(u) = obj.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(parse_usage(u)?);
        }
        if let Some(t) = obj.get("timings").filter(|t| !t.is_null()) {
            self.timings = Some(parse_timings(t));
        }

        let mut events = Vec::new();
        let choices = match obj.get("choices").and_then(Value::as_array) {
            Some(c) => c,
            None => return Ok(events),
        };
        for choice in choices {
            if let Some(reason) = choice.get("finish_reason").filter(|f| !f.is_null()) {
                self.finish = Some(parse_finish(reason)?);
            }
            let delta = match choice.get("delta") {
                Some(d) => d,
                None => continue,
            };
            if let Some(r) = delta.get("reasoning_content").and_then(Value::as_str) {
                push_text(&mut self.reasoning, r);
                if !r.is_empty() {
                    events.push(StreamEvent::Reasoning(r.to_string()));
                }
            }
            if let Some(c) = delta.get("content").and_then(Value::as_str) {
                self.content_delta(c, &mut events);
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for (pos, tc) in calls.iter().enumerate() {
                    let index = tc
                        .get("index")
                        .and_then(Value::as_u64)
                        .and_then(|i| usize::try_from(i).ok())
                        .unwrap_or(pos);
                    let slot = self.slot(index);
                    let id = tc
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty());
                    let name = tc
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty());
                    let args = tc
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if let Some(id) = id {
                        self.calls[slot].id = id.to_string();
                    }
                    if let Some(name) = name {
                        let cur = &mut self.calls[slot].name;
                        if cur.is_empty() || cur == name {
                            *cur = name.to_string();
                        } else {
                            cur.push_str(name);
                        }
                    }
                    self.calls[slot].args_raw.push_str(args);
                    if id.is_some() || name.is_some() || !args.is_empty() {
                        events.push(StreamEvent::ToolArgs {
                            index,
                            id: id.map(str::to_string),
                            name: name.map(|_| self.calls[slot].name.clone()),
                            delta: args.to_string(),
                        });
                    }
                }
            }
        }
        Ok(events)
    }

    fn content_delta(&mut self, c: &str, events: &mut Vec<StreamEvent>) {
        const OPEN: &str = " thinking";
        const CLOSE: &str = "";
        let mut buf = std::mem::take(&mut self.held);
        buf.push_str(c);
        loop {
            if self.in_think {
                if let Some(i) = buf.find(CLOSE) {
                    self.emit_reasoning(&buf[..i], events);
                    buf = buf[i + CLOSE.len()..].to_string();
                    self.in_think = false;
                    self.think_done = true;
                    continue;
                }
                let keep = tag_prefix_len(&buf, CLOSE);
                let cut = buf.len() - keep;
                self.emit_reasoning(&buf[..cut], events);
                self.held = buf[cut..].to_string();
                return;
            }
            let at_start =
                !self.think_done && self.content.as_deref().is_none_or(|t| t.trim().is_empty());
            if at_start {
                let lead = buf.len() - buf.trim_start().len();
                let rest = &buf[lead..];
                if let Some(after) = rest.strip_prefix(OPEN) {
                    buf = after.to_string();
                    self.in_think = true;
                    continue;
                }
                if rest.is_empty() || OPEN.starts_with(rest) {
                    self.held = buf;
                    return;
                }
            }
            self.emit_text(&buf, events);
            return;
        }
    }

    fn emit_reasoning(&mut self, r: &str, events: &mut Vec<StreamEvent>) {
        if !r.is_empty() {
            push_text(&mut self.reasoning, r);
            events.push(StreamEvent::Reasoning(r.to_string()));
        }
    }

    fn emit_text(&mut self, t: &str, events: &mut Vec<StreamEvent>) {
        if !t.is_empty() {
            push_text(&mut self.content, t);
            events.push(StreamEvent::Text(t.to_string()));
        }
    }

    /// Parse a comment line (`prefill <done>/<total>` or `queued ...`); other comments -> `None`.
    pub fn comment(&self, comment: &str, secs: u32) -> Option<StreamEvent> {
        let text = comment.trim();
        if let Some(rest) = text.strip_prefix("prefill ") {
            let (done, tail) = rest.split_once('/')?;
            let mut fields = tail.split_whitespace();
            let total = fields.next()?.parse().ok()?;
            let own = fields.find_map(|f| f.strip_suffix('s').and_then(|n| n.parse::<u32>().ok()));
            return Some(StreamEvent::Prefill {
                done: done.trim().parse().ok()?,
                total,
                secs: own.unwrap_or(secs),
            });
        }
        let rest = match text.strip_prefix("queued") {
            Some(r) if r.is_empty() || r.starts_with(char::is_whitespace) => r,
            _ => return None,
        };
        let mut pos = None;
        let mut wait_s = None;
        for field in rest.split_whitespace() {
            match field.split_once('=') {
                Some(("pos", v)) => pos = v.parse().ok(),
                Some(("wait_s", v)) => wait_s = v.parse().ok(),
                Some(_) => {}
                None => {
                    if let Some(n) = field.strip_suffix('s').and_then(|n| n.parse().ok()) {
                        wait_s = Some(n);
                    } else if let Ok(n) = field.parse() {
                        pos = Some(n);
                    }
                }
            }
        }
        Some(StreamEvent::Queued {
            pos,
            wait_s: wait_s.unwrap_or(secs),
        })
    }

    /// Deliver the held tail once the stream has ended.
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        let held = std::mem::take(&mut self.held);
        let mut events = Vec::new();
        if self.in_think {
            self.emit_reasoning(&held, &mut events);
        } else {
            self.emit_text(&held, &mut events);
        }
        events
    }

    /// Finish after `[DONE]`. No `finish_reason` or no usage = `stream_broken`.
    pub fn finish(self) -> Result<AssistantTurn, OxaError> {
        let finish = self
            .finish
            .ok_or_else(|| broken("the stream ended without a finish_reason"))?;
        let usage = self
            .usage
            .ok_or_else(|| broken("the stream ended without usage"))?;
        if let Some(call) = self
            .calls
            .iter()
            .find(|c| c.id.is_empty() || c.name.is_empty())
        {
            return Err(broken(&format!(
                "a tool call arrived without an id or a name (id {:?}, name {:?})",
                call.id, call.name
            )));
        }
        Ok(AssistantTurn {
            msg: WireMsg {
                role: crate::proto::Role::Assistant,
                content: self.content,
                reasoning: self.reasoning,
                tool_calls: self.calls,
                tool_call_id: None,
            },
            finish,
            usage,
            timings: self.timings,
            model: self.model,
        })
    }

    fn slot(&mut self, index: usize) -> usize {
        if let Some(p) = self.indices.iter().position(|&i| i == index) {
            return p;
        }
        self.indices.push(index);
        self.calls.push(ToolCall {
            id: String::new(),
            name: String::new(),
            args_raw: String::new(),
        });
        self.calls.len() - 1
    }
}

fn tag_prefix_len(s: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&n| s.ends_with(&tag[..n]))
        .unwrap_or(0)
}

fn push_text(field: &mut Option<String>, delta: &str) {
    field.get_or_insert_with(String::new).push_str(delta);
}

fn parse_usage(u: &Value) -> Result<Usage, OxaError> {
    let count = |v: Option<&Value>| {
        v.and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
    };
    let prompt =
        count(u.get("prompt_tokens")).ok_or_else(|| broken("usage has no prompt_tokens"))?;
    let completion = count(u.get("completion_tokens"))
        .ok_or_else(|| broken("usage has no completion_tokens"))?;
    let reported = count(u.pointer("/prompt_tokens_details/cached_tokens"));
    Ok(Usage {
        prompt,
        completion,
        cached: reported.unwrap_or(0),
        cached_missing: reported.is_none(),
        unreported: false,
    })
}

fn parse_timings(t: &Value) -> Timings {
    let count = |key: &str| {
        t.get(key)
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0)
    };
    let ms = |key: &str| t.get(key).and_then(Value::as_f64).unwrap_or(0.0);
    Timings {
        cache_n: count("cache_n"),
        prompt_n: count("prompt_n"),
        prompt_ms: ms("prompt_ms"),
        predicted_n: count("predicted_n"),
        predicted_ms: ms("predicted_ms"),
        snap_pos: t
            .get("snap_pos")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok()),
        snap_asked: t.get("snap_pos").is_some(),
    }
}

fn parse_finish(v: &Value) -> Result<Finish, OxaError> {
    match v.as_str() {
        Some("stop") => Ok(Finish::Stop),
        Some("tool_calls") => Ok(Finish::ToolCalls),
        Some("length") => Ok(Finish::Length),
        Some("content_filter") => Ok(Finish::ContentFilter),
        _ => Err(broken(&format!("unknown finish_reason {v}"))),
    }
}

// ---- HTTP error classification ----

/// Classify an OpenAI-compatible status reply: overflow, busy (503 + Retry-After), or HTTP error.
pub fn normalize_http_error(status: u16, body: &[u8], retry_after: Option<u32>) -> OxaError {
    let text = String::from_utf8_lossy(body);
    let parsed: Option<Value> = serde_json::from_str(&text).ok();
    let err = parsed.as_ref().and_then(|v| v.get("error"));
    if err.is_some_and(is_overflow) {
        return overflow();
    }
    let detail = err
        .and_then(|e| e.get("message").and_then(Value::as_str))
        .map(str::to_string)
        .unwrap_or_else(|| text.trim().chars().take(300).collect());
    status_error(status, &detail, retry_after)
}

/// Classify the error object of an in-stream error.
pub fn stream_error(err: &Value) -> OxaError {
    if is_overflow(err) {
        return overflow();
    }
    let detail = err
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| err.as_str().map(str::to_string))
        .unwrap_or_else(|| err.to_string());
    let as_status = |v: &Value| -> Option<u16> {
        let n = match v {
            Value::Number(n) => n.as_u64()?,
            Value::String(s) if s.len() == 3 => s.parse::<u64>().ok()?,
            _ => return None,
        };
        u16::try_from(n).ok().filter(|s| (100..600).contains(s))
    };
    let status = err
        .get("code")
        .and_then(as_status)
        .or_else(|| err.get("status").and_then(as_status))
        .or_else(|| {
            (err.get("type").and_then(Value::as_str) == Some("local_backend_busy")).then_some(503)
        });
    match status {
        Some(s) => status_error(s, &detail, None),
        None => OxaError::new(
            codes::STREAM_BROKEN,
            format!("the engine reported an error mid-stream: {detail}"),
        ),
    }
}

fn is_overflow(err: &Value) -> bool {
    ["type", "code"]
        .iter()
        .any(|k| err.get(*k).and_then(Value::as_str) == Some("exceed_context_size_error"))
}

fn overflow() -> OxaError {
    OxaError::new(
        codes::CONTEXT_OVERFLOW,
        "the request does not fit the engine's context window",
    )
    .with_hint("the session is too long for this context; compaction shrinks it")
}

fn status_error(status: u16, detail: &str, retry_after: Option<u32>) -> OxaError {
    let msg = format!("the engine answered HTTP {status}: {detail}");
    if status == 503 {
        let msg = match retry_after {
            Some(n) => format!("{msg} (retry after {n} s)"),
            None => msg,
        };
        return OxaError::new(codes::ENGINE_BUSY, msg).with_hint(
            "the engine is busy (its queue is full or no model is loaded); the request is retried",
        );
    }
    OxaError::new(codes::ENGINE_HTTP, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(text: &str) -> SseEvent {
        SseEvent::Data(text.to_string())
    }

    #[test]
    fn events_comments_and_done() {
        let mut p = SseParser::new();
        let got = p.feed_bounded(
            b": prefill 10/200\r\ndata: a\ndata: b\n\ndata: [DONE]\n\n",
            1 << 20,
        );
        assert_eq!(
            got.unwrap(),
            vec![
                SseEvent::Comment("prefill 10/200".into()),
                data("a\nb"),
                SseEvent::Done
            ]
        );
    }

    #[test]
    fn overflow_and_busy_status_replies_keep_their_class() {
        let busy = normalize_http_error(503, br#"{"error":{"message":"queue full"}}"#, Some(3));
        assert_eq!(busy.code, codes::ENGINE_BUSY);
        assert!(busy.message.contains("retry after 3 s"));
        let overflow = normalize_http_error(
            400,
            br#"{"error":{"type":"exceed_context_size_error","message":"too long"}}"#,
            None,
        );
        assert_eq!(overflow.code, codes::CONTEXT_OVERFLOW);
    }
}
