//! Typed local chat over the owned loopback engine. No controller routes or retries.
use crate::lifecycle::{Lifecycle, MemorySample, RequestLease};
use crate::State;
use oxa_engine::{
    build_body, build_headers, post_sse_bounded, BoundedStreamOpts, Flow, GatewayEvent, ReplyAssembler,
    SseEnd, SseEvent, StreamOpts, Url,
};
use oxa_proto::{codes, AssistantTurn, CancelToken, ChatRequest, OxaError, StreamEvent};
use std::io;
use std::net::SocketAddr;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::time::{Duration, Instant};

const MAX_REQUEST: usize = 16 * 1024 * 1024;
const MAX_EVENT: usize = 1024 * 1024;

/// Fixed numeric loopback endpoint. No proxy, redirect, DNS or controller fallback.
/// The caller owns endpoint authentication and must match this address to its child.
pub struct Transport {
    url: Url,
    opts: StreamOpts,
}
/// Caller-owned cancellation and bounded process/memory probes. No hidden reader.
pub struct Monitor<'a> {
    pub cancel: &'a CancelToken,
    pub memory: &'a mut dyn FnMut() -> io::Result<MemorySample>,
    pub healthy: &'a mut dyn FnMut() -> io::Result<bool>,
}
impl Transport {
    pub fn new(address: SocketAddr, opts: StreamOpts) -> Result<Self, OxaError> {
        if !address.ip().is_loopback()
            || address.port() == 0
            || opts.connect_timeout_ms == 0
            || opts.silence_limit_ms == 0
        {
            return Err(OxaError::new(codes::INVALID, "invalid local engine transport"));
        }
        Ok(Self {
            url: Url {
                host: address.ip().to_string(),
                port: address.port(),
                base_path: String::new(),
            },
            opts,
        })
    }

    /// Borrow the gateway's shared ChatRequest, encoding only at the actual engine
    /// HTTP boundary. Delivery stays on the caller thread. One scoped I/O worker
    /// and a single-event channel exist only during the call; callbacks need not
    /// be Send. Samples/probes must be fresh and bounded, as for Lifecycle::poll.
    ///
    /// A failed or panicking request transfers its lease into lifecycle cleanup.
    /// Continue polling until Stopped: returning an error never releases a still
    /// running engine to the next request. No failed generation is replayed.
    pub fn chat(
        &self,
        owner: &mut Lifecycle,
        lease: RequestLease,
        request: &ChatRequest<'_>,
        monitor: &mut Monitor<'_>,
        on: &mut dyn FnMut(StreamEvent),
    ) -> Result<AssistantTurn, OxaError> {
        if !owner.owns(&lease)
            || owner.status().state != State::Ready
            || owner.status().model != Some(request.model)
        {
            return Err(OxaError::new(
                codes::BUSY,
                "local model is not ready for this request",
            ));
        }
        if monitor.cancel.is_cancelled() {
            return Err(aborted());
        }
        let mut body = Vec::new();
        build_body(request, &mut body);
        if body.len() > MAX_REQUEST {
            return Err(OxaError::new(
                codes::INVALID,
                "local chat request exceeds its size limit",
            ));
        }
        let headers = build_headers(&request.hints);
        let mut active = Active {
            owner,
            lease: Some(lease),
        };
        active
            .owner
            .poll(Instant::now(), (monitor.memory)(), &mut *monitor.healthy)
            .map_err(|_| down())?;
        if active.owner.status().state != State::Ready {
            return Err(down());
        }
        let result = std::thread::scope(|scope| {
            let local_cancel = CancelToken::new();
            // On callback panic, cancel before scope joins a blocked socket or sender.
            let cancel_on_drop = CancelOnDrop(local_cancel.clone());
            let (send, receive) = mpsc::sync_channel(1);
            let worker_cancel = local_cancel.clone();
            let worker = std::thread::Builder::new()
                .name("oxc-local-io".into())
                .spawn_scoped(scope, move || {
                    let mut malformed = None;
                    let result = post_sse_bounded(
                        &self.url,
                        "/v1/chat/completions",
                        &headers,
                        &body,
                        &BoundedStreamOpts {
                            stream: self.opts,
                            max_event_bytes: MAX_EVENT,
                        },
                        &worker_cancel,
                        &mut |event| {
                            let event = match event {
                                SseEvent::Data(data) => {
                                    if data.len() > MAX_EVENT {
                                        malformed = Some(broken("local stream event exceeds its size limit"));
                                        return Flow::Stop;
                                    }
                                    match serde_json::from_str(&data) {
                                        Ok(value) => Message::Event(GatewayEvent::Chunk(value)),
                                        Err(_) => {
                                            malformed = Some(broken("invalid local stream event"));
                                            return Flow::Stop;
                                        }
                                    }
                                }
                                SseEvent::Done => Message::Event(GatewayEvent::Done),
                                SseEvent::Comment(text) => Message::Comment(text),
                            };
                            if deliver(&send, event, &worker_cancel) {
                                Flow::Continue
                            } else {
                                Flow::Stop
                            }
                        },
                    );
                    let result = malformed.map_or(result, Err);
                    deliver(&send, Message::Finished(result), &worker_cancel);
                })
                .map_err(|_| down())?;
            let mut emitted = false;
            let result = drive(&receive, &mut active, request, monitor, &mut |event| {
                emitted |= matches!(
                    event,
                    StreamEvent::Text(_) | StreamEvent::Reasoning(_) | StreamEvent::ToolArgs { .. }
                );
                on(event);
            })
            .map_err(|error| {
                if emitted
                    && error.code != codes::ABORTED
                    && error.code != codes::STREAM_BROKEN
                    && !error.message.starts_with(oxa_proto::CUT_BY_REFUSAL)
                {
                    broken(&format!("{}{}", oxa_proto::CUT_BY_REFUSAL, error.message))
                } else {
                    error
                }
            });
            drop(cancel_on_drop);
            worker
                .join()
                .map_err(|_| broken("local transport worker failed"))?;
            result
        });
        if result.is_ok() {
            drop(active.lease.take());
        }
        result
    }
}

struct Active<'a> {
    owner: &'a mut Lifecycle,
    lease: Option<RequestLease>,
}
impl Drop for Active<'_> {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            // Stop failures retain the lease and are retried by the owner poll.
            let _ = self.owner.interrupt(lease, Instant::now());
        }
    }
}
struct CancelOnDrop(CancelToken);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

enum Message {
    Event(GatewayEvent),
    Comment(String),
    Finished(Result<SseEnd, OxaError>),
}
fn deliver(send: &SyncSender<Message>, mut message: Message, cancel: &CancelToken) -> bool {
    loop {
        if cancel.is_cancelled() {
            return false;
        }
        match send.try_send(message) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(value)) => message = value,
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn drive(
    receive: &Receiver<Message>,
    active: &mut Active<'_>,
    request: &ChatRequest<'_>,
    monitor: &mut Monitor<'_>,
    on: &mut dyn FnMut(StreamEvent),
) -> Result<AssistantTurn, OxaError> {
    let mut assembler = ReplyAssembler::new();
    let mut done = false;
    let started = Instant::now();
    let mut next_poll = started + Duration::from_millis(250);
    loop {
        if !done && monitor.cancel.is_cancelled() {
            return Err(aborted());
        }
        if !done && Instant::now() >= next_poll {
            active
                .owner
                .poll(Instant::now(), (monitor.memory)(), &mut *monitor.healthy)
                .map_err(|_| down())?;
            if active.owner.status().state != State::Ready {
                return Err(down());
            }
            next_poll = Instant::now() + Duration::from_millis(250);
        }
        match receive.recv_timeout(Duration::from_millis(50)) {
            Ok(Message::Event(GatewayEvent::Chunk(value))) => {
                if done {
                    return Err(broken("local stream data after completion"));
                }
                for event in assembler.chunk_value(&value)? {
                    if monitor.cancel.is_cancelled() {
                        return Err(aborted());
                    }
                    on(event);
                }
            }
            Ok(Message::Event(GatewayEvent::Done)) => done = true,
            Ok(Message::Comment(text)) => {
                if let Some(event) =
                    assembler.comment(&text, started.elapsed().as_secs().try_into().unwrap_or(u32::MAX))
                {
                    on(event);
                }
            }
            Ok(Message::Finished(result)) => {
                let end = result?;
                if !(200..300).contains(&end.status) {
                    return Err(oxa_engine::normalize_http_error(
                        end.status,
                        end.error_body.as_deref().unwrap_or("").as_bytes(),
                        end.retry_after,
                    ));
                }
                if !done || !end.done {
                    return Err(broken("local stream ended before completion"));
                }
                for event in assembler.flush() {
                    on(event);
                }
                let mut reply = assembler.finish(request.effective_level())?;
                if reply.model.is_empty() {
                    reply.model = request.model.into();
                }
                return Ok(reply);
            }
            Err(RecvTimeoutError::Timeout) => (),
            Err(RecvTimeoutError::Disconnected) => return Err(broken("local transport ended unexpectedly")),
        }
    }
}
fn aborted() -> OxaError {
    OxaError::new(codes::ABORTED, "local request cancelled")
}
fn down() -> OxaError {
    OxaError::new(codes::ENGINE_DOWN, "local engine stopped during chat")
}
fn broken(message: &str) -> OxaError {
    OxaError::new(codes::STREAM_BROKEN, message)
}
