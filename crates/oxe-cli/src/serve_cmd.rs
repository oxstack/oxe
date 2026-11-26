//! `oxe serve`: one OpenAI-compatible listener over one supervised `oxe-server` child.
//!
//! One model is resident at a time. A request is routed by its `model` field; switching waits
//! for the active lease, stops the old child and admits the new launch profile (memory first).
//! SSE bytes pass through unchanged, including `:` comments and in-stream error events.
use crate::models_cmd::{self, Config, ModelConfig};
use bytes::Bytes;
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use oxe_supervisor::admission;
use oxe_supervisor::child_auth::ChildAuth;
use oxe_supervisor::engine::{open_raw, StreamOpts, Url};
use oxe_supervisor::lifecycle::{Lifecycle, MemorySample, Model, Policy, RequestLease};
use oxe_supervisor::memory::MemorySource;
use oxe_supervisor::proto::CancelToken;
use oxe_supervisor::queue::AdmissionError;
use oxe_supervisor::Config as LaunchConfig;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::ffi::OsString;
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type ServeBody = BoxBody<Bytes, Infallible>;

const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

const USAGE: &str = "\
usage: oxe serve [--config PATH] [--listen 127.0.0.1:8090] [--allow-remote]

Serve every model in ~/.config/oxe/models.toml behind one OpenAI-compatible listener.
GET /health, GET /v1/models, POST /v1/chat/completions and POST /v1/completions.
One oxe-server child is resident at a time and is started on demand on a private loopback
port. Loopback-only by default; --allow-remote permits a non-loopback --listen address.
Any Authorization bearer token is accepted and ignored.
SIGTERM and SIGINT stop the listener, stop the engine child (TERM, grace, KILL, wait) and exit 0.";

pub fn run(args: &[String]) -> ! {
    let mut config = models_cmd::default_config_path();
    let mut listen: Option<String> = None;
    let mut allow_remote = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                config = PathBuf::from(value(args, i, "--config"));
            }
            "--listen" => {
                i += 1;
                listen = Some(value(args, i, "--listen").to_string());
            }
            "--allow-remote" => allow_remote = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0)
            }
            flag => fail(&format!("unknown argument {flag}\n\n{USAGE}")),
        }
        i += 1;
    }
    let mut cfg = Config::load(&config).unwrap_or_else(|e| {
        if config.exists() {
            fail(&e)
        } else {
            fail(&format!(
                "no config at {}; run `oxe models add <model.gguf>` (or `oxe import-oxa`) first\n{e}",
                config.display()
            ))
        }
    });
    if let Some(listen) = listen {
        cfg.listen = listen;
    }
    cfg.allow_remote |= allow_remote;
    if cfg.models.is_empty() {
        fail(&format!(
            "no models configured in {}; run `oxe models add <model.gguf>` first",
            config.display()
        ));
    }
    let address: SocketAddr = cfg
        .listen
        .parse()
        .unwrap_or_else(|_| fail(&format!("--listen {}: expected IP:PORT", cfg.listen)));
    if !address.ip().is_loopback() && !cfg.allow_remote {
        fail(&format!(
            "refusing non-loopback listen address {}; pass --allow-remote (and an API key on the client side) if you really want that",
            address
        ));
    }
    let host = Arc::new(Host::new(cfg).unwrap_or_else(|e| fail(&e)));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| fail(&format!("tokio runtime: {e}")));
    if let Err(error) = runtime.block_on(serve(host, address)) {
        fail(&format!("serve: {error}"));
    }
    std::process::exit(0)
}

async fn serve(host: Arc<Host>, address: SocketAddr) -> io::Result<()> {
    let listener = tokio::net::TcpListener::bind(address).await?;
    // SIGTERM (systemd stop, `kill`) and SIGINT (Ctrl-C) both stop accepting, then take the
    // engine down through the supervisor's own stop path instead of leaving it orphaned.
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    eprintln!(
        "oxe serve: listening on http://{address}/v1 ({} model(s), engine {})",
        host.models.len(),
        host.binary.display()
    );
    loop {
        let (stream, _) = tokio::select! {
            accepted = listener.accept() => accepted?,
            _ = terminate.recv() => break,
            _ = interrupt.recv() => break,
        };
        let io = TokioIo::new(stream);
        let host = host.clone();
        tokio::spawn(async move {
            let service = service_fn(move |request| {
                let host = host.clone();
                async move { Ok::<_, Infallible>(handle(request, host).await) }
            });
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await;
        });
    }
    drop(listener);
    eprintln!("oxe serve: stopping the engine");
    let stopping = host.clone();
    tokio::task::spawn_blocking(move || stopping.shutdown())
        .await
        .unwrap_or_else(|error| Err(io::Error::other(format!("shutdown task: {error}"))))
}

async fn handle(request: Request<Incoming>, host: Arc<Host>) -> Response<ServeBody> {
    match (request.method(), request.uri().path()) {
        (&Method::GET, "/health") | (&Method::GET, "/v1/health") => {
            json_response(StatusCode::OK, host.health_body())
        }
        (&Method::GET, "/v1/models") | (&Method::GET, "/models") => {
            json_response(StatusCode::OK, host.models_body())
        }
        (&Method::POST, "/v1/chat/completions") | (&Method::POST, "/v1/completions") => {
            let path = request.uri().path().to_string();
            chat(request, host, path).await
        }
        (&Method::GET, "/") => json_response(
            StatusCode::OK,
            json!({"service": "oxe", "models": host.models.len()}),
        ),
        (method, path) => openai_error(
            StatusCode::NOT_FOUND,
            format!("Unknown route {method} {path}"),
            "invalid_request_error",
            "not_found",
        ),
    }
}

async fn chat(request: Request<Incoming>, host: Arc<Host>, path: String) -> Response<ServeBody> {
    let body = match read_body(request.into_body()).await {
        Ok(body) => body,
        Err(error) => return error_response(error),
    };
    let model = match serde_json::from_slice::<Value>(&body) {
        Ok(value) => match value.get("model").and_then(Value::as_str) {
            Some(model) => model.to_string(),
            None => {
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    "the request has no model field".into(),
                    "invalid_request_error",
                    "missing_model",
                )
            }
        },
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("the request body is not JSON: {error}"),
                "invalid_request_error",
                "invalid_json",
            )
        }
    };
    if !host.models.contains_key(&model) {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("model {model:?} is not configured; GET /v1/models lists the available ids"),
            "invalid_request_error",
            "model_not_found",
        );
    }
    let active = match host.acquire(&model).await {
        Ok(active) => active,
        Err(ServeError::Busy) => return busy_response(),
        Err(ServeError::Unknown) => {
            return openai_error(
                StatusCode::NOT_FOUND,
                format!("model {model:?} is not configured"),
                "invalid_request_error",
                "model_not_found",
            )
        }
        Err(ServeError::Engine(message)) => {
            return openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("model {model:?} could not start: {message}"),
                "unavailable_error",
                "engine_unavailable",
            )
        }
    };
    let (send, mut receive) = tokio::sync::mpsc::channel::<Upstream>(16);
    let cancel = CancelToken::new();
    let worker_cancel = cancel.clone();
    let endpoint = host.endpoints[&model].clone();
    tokio::task::spawn_blocking(move || {
        upstream(active, endpoint, path, body, send, worker_cancel);
    });
    let first = match receive.recv().await {
        Some(first) => first,
        None => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                "the engine closed the response before its headers".into(),
                "server_error",
                "engine_stream_failed",
            )
        }
    };
    let (status, content_type) = match first {
        Upstream::Head {
            status,
            content_type,
        } => (status, content_type),
        Upstream::Failed(message) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                message,
                "server_error",
                "engine_unavailable",
            )
        }
        Upstream::Chunk(_) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                "the engine sent body bytes before its headers".into(),
                "server_error",
                "engine_stream_failed",
            )
        }
    };
    let stream = futures_util::stream::unfold(receive, |mut receive| async move {
        loop {
            match receive.recv().await {
                Some(Upstream::Chunk(bytes)) => {
                    return Some((Ok::<_, Infallible>(Frame::data(bytes)), receive))
                }
                Some(Upstream::Head { .. }) | Some(Upstream::Failed(_)) => continue,
                None => return None,
            }
        }
    });
    let mut builder = Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY))
        .header("Cache-Control", "no-cache");
    if let Some(content_type) = content_type {
        builder = builder.header("Content-Type", content_type);
    }
    builder
        .body(StreamBody::new(stream).boxed())
        .unwrap_or_else(|_| error_body(StatusCode::INTERNAL_SERVER_ERROR, "invalid response"))
}

enum Upstream {
    Head {
        status: u16,
        content_type: Option<String>,
    },
    Chunk(Bytes),
    Failed(String),
}

fn upstream(
    active: Active,
    endpoint: Endpoint,
    path: String,
    body: Vec<u8>,
    send: tokio::sync::mpsc::Sender<Upstream>,
    cancel: CancelToken,
) {
    let url = Url {
        host: "127.0.0.1".into(),
        port: endpoint.port,
        base_path: String::new(),
    };
    let headers = [("Authorization", format!("Bearer {}", endpoint.token))];
    let opts = StreamOpts {
        connect_timeout_ms: 5_000,
        silence_limit_ms: 600_000,
    };
    let mut response = match open_raw(&url, &path, &headers, &body, &opts, &cancel) {
        Ok(response) => response,
        Err(error) => {
            let _ = send.blocking_send(Upstream::Failed(error.message));
            return;
        }
    };
    let content_type = response
        .headers
        .iter()
        .find(|(name, _)| name == "content-type")
        .map(|(_, value)| value.clone());
    if send
        .blocking_send(Upstream::Head {
            status: response.status,
            content_type,
        })
        .is_err()
    {
        cancel.cancel();
        return;
    }
    loop {
        match response.read_chunk() {
            Ok(Some(bytes)) => {
                if send
                    .blocking_send(Upstream::Chunk(Bytes::from(bytes)))
                    .is_err()
                {
                    // The client went away: close the upstream socket so the engine cancels
                    // its generation instead of writing into a dead stream.
                    cancel.cancel();
                    return;
                }
            }
            Ok(None) => break,
            Err(error) => {
                if !cancel.is_cancelled() {
                    let _ = send.blocking_send(Upstream::Failed(error.message));
                }
                break;
            }
        }
    }
    drop(active);
}

async fn read_body(mut body: Incoming) -> Result<Vec<u8>, ServeError> {
    let mut out = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|error| ServeError::Engine(format!("request body: {error}")))?;
        if let Some(data) = frame.data_ref() {
            if out.len() + data.len() > MAX_REQUEST_BYTES {
                return Err(ServeError::Engine(format!(
                    "the request body exceeds {MAX_REQUEST_BYTES} bytes"
                )));
            }
            out.extend_from_slice(data);
        }
    }
    Ok(out)
}

#[derive(Debug)]
enum ServeError {
    Busy,
    Unknown,
    Engine(String),
}

#[derive(Clone)]
struct Endpoint {
    port: u16,
    token: String,
}

struct Active {
    _lease: RequestLease,
}

trait Probe: Send + Sync {
    fn host(&self) -> io::Result<MemorySample>;
    fn child(&self, pid: u32) -> io::Result<MemorySample>;
    fn host_memory_mode(&self) -> bool;
}

struct RealProbe(MemorySource);

impl Probe for RealProbe {
    fn host(&self) -> io::Result<MemorySample> {
        self.0.sample()
    }
    fn child(&self, pid: u32) -> io::Result<MemorySample> {
        self.0.sample_child(pid)
    }
    fn host_memory_mode(&self) -> bool {
        true
    }
}

/// Model-free test samples: the serve tests run a fake child and must not depend on the host's
/// cgroup layout or swap state. Enabled only by the internal test hook.
struct FakeProbe;

impl Probe for FakeProbe {
    fn host(&self) -> io::Result<MemorySample> {
        Ok(sample())
    }
    fn child(&self, _pid: u32) -> io::Result<MemorySample> {
        Ok(sample())
    }
    fn host_memory_mode(&self) -> bool {
        false
    }
}

fn sample() -> MemorySample {
    MemorySample {
        available_bytes: 32 * 1024 * 1024 * 1024,
        swap_free_bytes: 32 * 1024 * 1024 * 1024,
        psi_full_avg10: 0.0,
        swap_current_bytes: 0,
        swap_limit_zero: true,
        oom_events: 0,
        host_oom_kills: 0,
        pids_max_events: 0,
    }
}

struct Host {
    cfg: Config,
    binary: PathBuf,
    models: BTreeMap<String, Model>,
    endpoints: BTreeMap<String, Endpoint>,
    _auths: Vec<ChildAuth>,
    lifecycle: Mutex<Lifecycle>,
    gate: oxe_supervisor::lifecycle::RequestGate,
    probe: Box<dyn Probe>,
    agent: ureq::Agent,
    ticker: AtomicBool,
}

impl Host {
    fn new(cfg: Config) -> Result<Host, String> {
        let binary = match &cfg.binary {
            Some(binary) => binary.clone(),
            None => std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(|p| p.join("oxe-server")))
                .ok_or("cannot locate the oxe executable")?,
        };
        if !binary.is_absolute() {
            return Err(format!(
                "engine binary {} must be an absolute path",
                binary.display()
            ));
        }
        if !binary.is_file() {
            return Err(format!(
                "engine binary {} is missing; install oxe-server beside oxe or set `binary` in the config",
                binary.display()
            ));
        }
        let fake_memory = std::env::var_os("OXE_SERVE_FAKE_MEMORY").is_some_and(|v| v == "1");
        let probe: Box<dyn Probe> = if fake_memory {
            Box::new(FakeProbe)
        } else {
            Box::new(RealProbe(MemorySource::default()))
        };
        let mut lifecycle = Lifecycle::new(
            cfg.max_waiters,
            Policy {
                idle_timeout: Some(Duration::from_secs(cfg.idle_unload_seconds)),
            },
        );
        if probe.host_memory_mode() {
            lifecycle.use_host_memory();
        }
        let gate = lifecycle.requests();
        let run_dir = run_dir();
        let agent = ureq::Agent::config_builder()
            .proxy(None)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_millis(500)))
            .build()
            .new_agent();
        let mut models = BTreeMap::new();
        let mut endpoints = BTreeMap::new();
        let mut auths = Vec::new();
        // The child runs with a cleared environment. Forward only the harmless base it needs
        // to resolve its own caches; credentials and arbitrary ambient variables stay out.
        let mut base_environment = BTreeMap::new();
        for key in [
            "HOME",
            "PATH",
            "TMPDIR",
            "XDG_RUNTIME_DIR",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "LANG",
            "LC_ALL",
        ] {
            if let Some(value) = std::env::var_os(key).and_then(|v| v.into_string().ok()) {
                base_environment.insert(key.to_string(), value);
            }
        }
        base_environment.extend(cfg.environment.clone());
        for (id, model) in &cfg.models {
            let (model, endpoint, auth) = prepare_model(
                id,
                model,
                &binary,
                &base_environment,
                &run_dir,
                cfg.startup_timeout_seconds,
            )?;
            models.insert(id.clone(), model);
            endpoints.insert(id.clone(), endpoint);
            auths.push(auth);
        }
        Ok(Host {
            cfg,
            binary,
            models,
            endpoints,
            _auths: auths,
            lifecycle: Mutex::new(lifecycle),
            gate,
            probe,
            agent,
            ticker: AtomicBool::new(false),
        })
    }

    fn health_body(&self) -> Value {
        let lifecycle = self.lifecycle.lock().unwrap_or_else(|e| e.into_inner());
        let status = lifecycle.status();
        json!({
            "status": "ok",
            "models": self.cfg.models.len(),
            "loaded": status.model,
            "state": format!("{:?}", status.state).to_lowercase(),
        })
    }

    fn models_body(&self) -> Value {
        let data: Vec<Value> = self
            .cfg
            .models
            .iter()
            .map(|(id, model)| {
                json!({
                    "id": id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "context_window": model.context_window,
                    // OpenRouter-style spelling: clients that only know this one (ox, Claude Code)
                    // otherwise fall back to a 1M context instead of the configured window.
                    "context_length": model.context_window,
                    "max_output": model.max_output,
                })
            })
            .collect();
        json!({"object": "list", "data": data})
    }

    async fn acquire(self: &Arc<Self>, id: &str) -> Result<Active, ServeError> {
        if !self.models.contains_key(id) {
            return Err(ServeError::Unknown);
        }
        let lease = self
            .gate
            .acquire_operation()
            .await
            .map_err(|error| match error {
                AdmissionError::Full => ServeError::Busy,
                AdmissionError::Closed => ServeError::Engine("engine queue is closed".into()),
            })?;
        let host = self.clone();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || host.install(&id, lease))
            .await
            .map_err(|error| ServeError::Engine(format!("supervisor task: {error}")))?
    }

    fn install(self: &Arc<Self>, id: &str, lease: RequestLease) -> Result<Active, ServeError> {
        let model = self.models.get(id).cloned().ok_or(ServeError::Unknown)?;
        let mut lifecycle = self.lifecycle.lock().unwrap_or_else(|e| e.into_inner());
        lifecycle
            .load_reserved(lease, model, Instant::now(), self.probe.host())
            .map_err(|error| ServeError::Engine(error.to_string()))?;
        loop {
            let pid = lifecycle.status().pid;
            let current = lifecycle.status().model.map(str::to_owned);
            let healthy = || self.health(current.as_deref());
            let sample = match pid {
                Some(pid) => self.probe.child(pid),
                None => self.probe.host(),
            };
            // A transient memory or health read is not an engine death: keep polling and let
            // Stopped/Failed below decide. The poll still records the latest state.
            let _ = lifecycle.poll(Instant::now(), sample, healthy);
            let status = lifecycle.status();
            if status.state == oxe_supervisor::State::Ready && status.model == Some(id) {
                break;
            }
            if matches!(
                status.state,
                oxe_supervisor::State::Stopped | oxe_supervisor::State::Failed
            ) {
                return Err(ServeError::Engine(describe(&status)));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let lease = lifecycle
            .take_ready_request()
            .ok_or_else(|| ServeError::Engine("engine ready without its load lease".into()))?;
        self.start_ticker();
        Ok(Active { _lease: lease })
    }

    fn start_ticker(self: &Arc<Self>) {
        if self.ticker.swap(true, Ordering::SeqCst) {
            return;
        }
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(250)).await;
                let Some(host) = weak.upgrade() else { break };
                if !host.tick() {
                    host.ticker.store(false, Ordering::SeqCst);
                    break;
                }
            }
        });
    }

    /// One memory/health poll while a child exists. Returns false once nothing is owned.
    fn tick(&self) -> bool {
        let mut lifecycle = self.lifecycle.lock().unwrap_or_else(|e| e.into_inner());
        let status = lifecycle.status();
        if status.pid.is_none()
            && matches!(
                status.state,
                oxe_supervisor::State::Stopped | oxe_supervisor::State::Failed
            )
        {
            return false;
        }
        let current = status.model.map(str::to_owned);
        let pid = status.pid;
        let healthy = || self.health(current.as_deref());
        let sample = match pid {
            Some(pid) => self.probe.child(pid),
            None => self.probe.host(),
        };
        let _ = lifecycle.poll(Instant::now(), sample, healthy);
        true
    }

    fn health(&self, id: Option<&str>) -> io::Result<bool> {
        let Some(id) = id else {
            return Ok(false);
        };
        let Some(endpoint) = self.endpoints.get(id) else {
            return Ok(false);
        };
        let mut response = self
            .agent
            .get(format!("http://127.0.0.1:{}/health", endpoint.port))
            .header("Authorization", format!("Bearer {}", endpoint.token))
            .call()
            .map_err(|_| io::Error::other("local health probe failed"))?;
        let text = response
            .body_mut()
            .with_config()
            .limit(4096)
            .read_to_string()
            .map_err(|_| io::Error::other("invalid local health reply"))?;
        Ok(serde_json::from_str::<Value>(&text).is_ok_and(|v| v["status"] == "ok"))
    }

    /// SIGTERM/SIGINT teardown: stop admitting engine work, then drive the supervisor's normal
    /// stop path (TERM, grace, KILL, wait) to completion. The deadline bounds a child that
    /// refuses to die, so the parent-death signal is the last resort, never the first.
    fn shutdown(&self) -> io::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut lifecycle = self.lifecycle.lock().unwrap_or_else(|e| e.into_inner());
        lifecycle.shutdown(Instant::now())?;
        loop {
            let status = lifecycle.status();
            if status.pid.is_none()
                && matches!(
                    status.state,
                    oxe_supervisor::State::Stopped | oxe_supervisor::State::Failed
                )
            {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(io::Error::other(
                    "the engine child did not stop before the shutdown deadline",
                ));
            }
            let current = status.model.map(str::to_owned);
            let healthy = || self.health(current.as_deref());
            let sample = match status.pid {
                Some(pid) => self.probe.child(pid),
                None => self.probe.host(),
            };
            let _ = lifecycle.poll(Instant::now(), sample, healthy);
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

fn describe(status: &oxe_supervisor::lifecycle::Status<'_>) -> String {
    match status.stop_reason {
        Some(reason) => format!("engine stopped ({reason:?})"),
        None => format!("engine ended in state {:?}", status.state),
    }
}

fn prepare_model(
    id: &str,
    model: &ModelConfig,
    binary: &Path,
    environment: &BTreeMap<String, String>,
    run_dir: &Path,
    startup_timeout_seconds: u64,
) -> Result<(Model, Endpoint, ChildAuth), String> {
    if model.arguments.iter().any(|word| {
        matches!(
            word.split('=').next().unwrap_or(""),
            "--api-key" | "--api-key-file"
        )
    }) || environment.keys().any(|key| key.starts_with("OXE_API_KEY"))
    {
        return Err(format!(
            "models.{id}: never put engine credentials in the config; oxe serve creates a private child key"
        ));
    }
    let weights = std::fs::metadata(&model.model)
        .map_err(|e| format!("models.{id}: {}: {e}", model.model.display()))?;
    if !weights.is_file() {
        return Err(format!(
            "models.{id}: {} is not a file",
            model.model.display()
        ));
    }
    let required = admission::required(
        &model.allocation_budget.to_json(),
        model.context_window,
        weights.len(),
    )
    .map_err(|error| format!("models.{id}: {error}"))?;
    let mut arguments: Vec<OsString> = model.arguments.iter().map(OsString::from).collect();
    admission::bind_context(&mut arguments, model.context_window)
        .map_err(|error| format!("models.{id}: {error}"))?;
    let port = free_port().map_err(|e| format!("models.{id}: {e}"))?;
    let auth = ChildAuth::create(run_dir)
        .map_err(|e| format!("models.{id}: child authentication: {e}"))?;
    if !arguments.iter().any(|word| {
        word == "--default-max-tokens"
            || word.to_string_lossy().starts_with("--default-max-tokens=")
    }) {
        arguments.extend([
            "--default-max-tokens".into(),
            model.max_output.to_string().into(),
        ]);
    }
    arguments.extend([
        "--model".into(),
        model.model.clone().into_os_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string().into(),
        "--alias".into(),
        id.into(),
    ]);
    let mut child_environment: BTreeMap<OsString, OsString> = environment
        .iter()
        .map(|(key, value)| (OsString::from(key), OsString::from(value)))
        .collect();
    child_environment.insert("OXE_API_KEY_FILE".into(), auth.path().into_os_string());
    let config = LaunchConfig {
        executable: binary.to_path_buf(),
        arguments,
        environment: child_environment,
        startup_timeout: Duration::from_secs(startup_timeout_seconds.max(1)),
        health_interval: Duration::from_millis(250),
        unhealthy_limit: 2,
        term_grace: Duration::from_secs(3),
        restart_backoff: Duration::from_secs(1),
        max_restarts: 0,
    };
    Ok((
        Model {
            name: id.to_string(),
            required_available_bytes: required,
            config,
        },
        Endpoint {
            port,
            token: auth.token().to_string(),
        },
        auth,
    ))
}

fn free_port() -> io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    listener.local_addr().map(|address| address.port())
}

fn run_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .map(|dir| dir.join("oxe"))
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/oxe/run")))
        .unwrap_or_else(|| std::env::temp_dir().join("oxe"))
}

fn json_response(status: StatusCode, value: Value) -> Response<ServeBody> {
    let body = serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)).map_err(|e| match e {}).boxed())
        .unwrap_or_else(|_| error_body(StatusCode::INTERNAL_SERVER_ERROR, "invalid response"))
}

fn openai_error(
    status: StatusCode,
    message: String,
    kind: &str,
    code: &str,
) -> Response<ServeBody> {
    json_response(
        status,
        json!({"error": {"message": message, "type": kind, "param": Value::Null, "code": code}}),
    )
}

fn error_response(error: ServeError) -> Response<ServeBody> {
    match error {
        ServeError::Busy => busy_response(),
        ServeError::Unknown => openai_error(
            StatusCode::NOT_FOUND,
            "unknown model".into(),
            "invalid_request_error",
            "model_not_found",
        ),
        ServeError::Engine(message) => openai_error(
            StatusCode::BAD_GATEWAY,
            message,
            "server_error",
            "engine_unavailable",
        ),
    }
}

fn busy_response() -> Response<ServeBody> {
    let mut response = openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "the local model queue is full; retry shortly".into(),
        "unavailable_error",
        "queue_full",
    );
    response
        .headers_mut()
        .insert("Retry-After", "1".parse().expect("static header"));
    response
}

fn error_body(status: StatusCode, message: &str) -> Response<ServeBody> {
    Response::builder()
        .status(status)
        .header("Content-Type", "text/plain")
        .body(
            Full::new(Bytes::from(message.to_string()))
                .map_err(|e| match e {})
                .boxed(),
        )
        .expect("static response")
}

fn value<'a>(args: &'a [String], i: usize, flag: &str) -> &'a str {
    args.get(i)
        .map(String::as_str)
        .unwrap_or_else(|| fail(&format!("missing value for {flag}")))
}

fn fail(message: &str) -> ! {
    eprintln!("oxe serve: {message}");
    std::process::exit(2)
}
