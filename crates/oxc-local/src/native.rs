//! In-process Engine over one owned local child. Construction/discovery do not launch.
use crate::child_auth::ChildAuth;
use crate::lifecycle::{Lifecycle, MemorySample, Model, Policy, RequestGate, RequestLease, StopReason};
use crate::memory::MemorySource;
use crate::queue::AdmissionError;
use crate::transport::{Monitor, Transport};
use crate::State;
use oxa_engine::{http_request, HttpRequest, StreamOpts, Url};
use oxa_proto::{
    codes, AssistantTurn, CancelToken, ChatRequest, Engine, EngineCaps, EngineInfo, OxaError, StreamEvent,
};
use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex, RwLock,
};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub type Launch = Arc<dyn Fn(u16) -> Result<Model, OxaError> + Send + Sync>;
pub struct LocalModel {
    pub alias: String,
    pub context_window: u32,
    pub max_output: u32,
    pub thinking: Option<bool>,
    pub images: Option<bool>,
    /// Trusted cold-launch resolution, including the current full-resident weights size.
    pub launch: Launch,
}
#[derive(Clone, Debug)]
pub struct LocalStatus {
    pub state: State,
    pub model: Option<String>,
    pub pid: Option<u32>,
    pub waiting: usize,
    pub stop_reason: Option<StopReason>,
    pub memory_policy: Option<&'static str>,
    pub launcher_micros: Option<u64>,
}
trait Sample: Send + Sync {
    fn sample(&self) -> io::Result<MemorySample>;
    fn child(&self, _pid: Option<u32>) -> io::Result<MemorySample> {
        self.sample()
    }
    fn platform(&self) -> Option<Arc<dyn crate::platform::PlatformLauncher>> {
        None
    }
}
impl Sample for MemorySource {
    fn sample(&self) -> io::Result<MemorySample> {
        self.sample()
            .map_err(|_| crate::platform::refused("host memory/PSI and cgroup v2 metadata readable"))
    }
    fn child(&self, pid: Option<u32>) -> io::Result<MemorySample> {
        pid.map_or_else(|| self.sample(), |pid| self.sample_child(pid))
    }
    #[cfg(target_os = "linux")]
    fn platform(&self) -> Option<Arc<dyn crate::platform::PlatformLauncher>> {
        Some(Arc::new(crate::platform::LinuxLauncher(self.clone())))
    }
}

pub struct LocalEngine {
    inner: Arc<Inner>,
    models: BTreeMap<String, LocalModel>,
    gate: RequestGate,
    private_dir: PathBuf,
    opts: StreamOpts,
}
struct Inner {
    state: Mutex<RuntimeState>,
    status: RwLock<LocalStatus>,
    wake: Condvar,
    shutdown: AtomicBool,
    worker: Mutex<Option<JoinHandle<()>>>,
    source: Arc<dyn Sample>,
}
struct RuntimeState {
    // Reap the child before removing its authentication file on final drop.
    lifecycle: Lifecycle,
    endpoints: BTreeMap<String, Endpoint>,
    target: Option<String>,
    driver_running: bool,
}
struct Endpoint {
    address: SocketAddr,
    auth: ChildAuth,
    transport: Transport,
    agent: ureq::Agent,
    version_mismatch: AtomicBool,
}
impl Endpoint {
    fn health(&self, pid: Option<u32>, verify_version: bool) -> io::Result<bool> {
        if !pid.is_some_and(|pid| owns_port(pid, self.address)) {
            return Ok(false);
        }
        let mut response = self
            .agent
            .get(format!("http://{}/health", self.address))
            .header("Authorization", format!("Bearer {}", self.auth.token()))
            .call()
            .map_err(|_| io::Error::other("local health probe failed"))?;
        let text = response
            .body_mut()
            .with_config()
            .limit(4096)
            .read_to_string()
            .map_err(|_| io::Error::other("invalid local health reply"))?;
        if !serde_json::from_str::<serde_json::Value>(&text).is_ok_and(|v| v["status"] == "ok") {
            return Ok(false);
        }
        if verify_version {
            let mut response = self
                .agent
                .get(format!("http://{}/props", self.address))
                .header("Authorization", format!("Bearer {}", self.auth.token()))
                .call()
                .map_err(|_| io::Error::other("local version probe failed"))?;
            let text = response
                .body_mut()
                .with_config()
                .limit(1024 * 1024)
                .read_to_string()
                .map_err(|_| io::Error::other("invalid local version reply"))?;
            let matches = serde_json::from_str::<serde_json::Value>(&text)
                .is_ok_and(|v| v["build_info"] == concat!("oxe-", env!("CARGO_PKG_VERSION")));
            self.version_mismatch.store(!matches, Ordering::Release);
            return Ok(matches);
        }
        Ok(true)
    }
}
impl LocalEngine {
    pub fn new(
        models: Vec<LocalModel>,
        private_dir: PathBuf,
        opts: StreamOpts,
        policy: Policy,
    ) -> Result<Self, OxaError> {
        Self::with_source(
            models,
            private_dir,
            opts,
            policy,
            Arc::new(MemorySource::default()),
        )
    }
    fn with_source(
        models: Vec<LocalModel>,
        private_dir: PathBuf,
        opts: StreamOpts,
        policy: Policy,
        source: Arc<dyn Sample>,
    ) -> Result<Self, OxaError> {
        if models.is_empty()
            || !private_dir.is_absolute()
            || opts.connect_timeout_ms == 0
            || opts.silence_limit_ms == 0
        {
            return Err(invalid());
        }
        let mut catalog = BTreeMap::new();
        for model in models {
            if model.alias.is_empty()
                || model.alias.len() > 128
                || model.alias.chars().any(char::is_control)
                || model.context_window == 0
                || model.max_output == 0
                || model.max_output > model.context_window
            {
                return Err(invalid());
            }
            if catalog.insert(model.alias.clone(), model).is_some() {
                return Err(invalid());
            }
        }
        let mut lifecycle = Lifecycle::new(32, policy);
        lifecycle.platform = source.platform();
        let status = snapshot(&lifecycle);
        let gate = lifecycle.requests();
        Ok(Self {
            inner: Arc::new(Inner {
                state: Mutex::new(RuntimeState {
                    lifecycle,
                    endpoints: BTreeMap::new(),
                    target: None,
                    driver_running: false,
                }),
                status: RwLock::new(status),
                wake: Condvar::new(),
                shutdown: AtomicBool::new(false),
                worker: Mutex::new(None),
                source,
            }),
            models: catalog,
            gate,
            private_dir,
            opts,
        })
    }
    pub fn status(&self) -> LocalStatus {
        let mut status = self
            .inner
            .status
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        status.waiting = self.gate.waiting();
        status
    }
    fn ensure_driver(&self, state: &mut RuntimeState) -> Result<(), OxaError> {
        if state.driver_running {
            return Ok(());
        }
        let mut slot = self.inner.worker.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = slot.take() {
            old.join().map_err(|_| down())?;
        }
        state.driver_running = true;
        let inner = self.inner.clone();
        match std::thread::Builder::new()
            .name("oxc-local-owner".into())
            .spawn(move || watch(inner))
        {
            Ok(worker) => {
                *slot = Some(worker);
                Ok(())
            }
            Err(_) => {
                state.driver_running = false;
                Err(down())
            }
        }
    }
    fn endpoint(&self, model: &LocalModel) -> Result<(Endpoint, Model), OxaError> {
        let address = TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .map_err(|_| down())?;
        let mut launch = (model.launch)(address.port())?;
        if launch.name != model.alias || launch.required_available_bytes == 0 {
            return Err(invalid());
        }
        let config = &mut launch.config;
        if config.arguments.iter().any(|a| {
            a.to_str()
                .is_some_and(|a| matches!(a.split('=').next().unwrap_or(""), "--api-key" | "--api-key-file"))
        }) || config
            .environment
            .keys()
            .any(|k| k.to_str().is_some_and(|k| k.starts_with("OXE_API_KEY")))
        {
            return Err(invalid());
        }
        let auth = ChildAuth::create(&self.private_dir).map_err(|_| {
            OxaError::new(
                codes::ENGINE_DOWN,
                "cannot create private local child authentication",
            )
        })?;
        config
            .environment
            .insert("OXE_API_KEY_FILE".into(), auth.path().into_os_string());
        let transport = Transport::new(address, self.opts)?.with_bearer(auth.token())?;
        let agent = ureq::Agent::config_builder()
            .proxy(None)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_millis(200)))
            .build()
            .new_agent();
        Ok((
            Endpoint {
                address,
                auth,
                transport,
                agent,
                version_mismatch: AtomicBool::new(false),
            },
            launch,
        ))
    }
}
impl Engine for LocalEngine {
    fn info(&self, target: Option<&str>) -> Result<EngineInfo, OxaError> {
        let status = self.status();
        let selected = target
            .or(status.model.as_deref())
            .unwrap_or_else(|| self.models.keys().next().unwrap());
        let model = self.models.get(selected).ok_or_else(invalid)?;
        Ok(EngineInfo {
            base_url: "in-process-local".into(),
            models: self.models.keys().cloned().collect(),
            model: status.model.unwrap_or_else(|| model.alias.clone()),
            limits_for: model.alias.clone(),
            n_ctx: model.context_window,
            max_output: model.max_output,
            queue: status.waiting.try_into().unwrap_or(u32::MAX),
            compact_at: None,
            prefill_only: false,
            caps: EngineCaps {
                remote: false,
                cancel: true,
                hint_clear: true,
                context_window: Some(model.context_window),
                ..EngineCaps::default()
            },
            thinking: model.thinking,
            images: model.images,
        })
    }
    fn local_memory_status(&self, model: &str) -> Option<(u32, &'static str, u64)> {
        let status = self.status();
        (model.is_empty() || status.model.as_deref() == Some(model))
            .then_some(status.pid.zip(status.memory_policy))
            .flatten()
            .map(|(pid, policy)| (pid, policy, status.launcher_micros.unwrap_or(0)))
    }
    fn count_tokens(&self, _text: &str) -> Result<u32, OxaError> {
        Err(OxaError::unsupported(
            "exact local token counts before model load",
        ))
    }
    fn clear_hint(&self, hint: &str) -> Result<(), OxaError> {
        if self.status().model.is_none() {
            return Ok(());
        }
        let _lease = self
            .gate
            .try_acquire()
            .map_err(|_| OxaError::new(codes::BUSY, "local engine is busy"))?;
        let state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(alias) = state.lifecycle.status().model else {
            return Ok(());
        };
        if state.lifecycle.status().state != State::Ready {
            return Err(OxaError::new(codes::BUSY, "local engine is busy"));
        }
        let endpoint = state.endpoints.get(alias).ok_or_else(down)?;
        let body =
            serde_json::to_vec(&serde_json::json!({"scope":"hint","hint":hint})).map_err(|_| invalid())?;
        let headers = [
            ("Authorization", format!("Bearer {}", endpoint.auth.token())),
            ("Content-Type", "application/json".into()),
        ];
        let url = Url::parse(&format!("http://{}", endpoint.address))?;
        let response = http_request(&HttpRequest {
            method: "POST",
            url: &url,
            path: "/state-cache/clear",
            headers: &headers,
            body: &body,
            timeout_ms: 1000,
        })
        .map_err(|e| redact(e, endpoint.auth.token()))?;
        if (200..300).contains(&response.status) || response.status == 404 {
            Ok(())
        } else {
            Err(redact(
                oxa_engine::normalize_http_error(response.status, &response.body, None),
                endpoint.auth.token(),
            ))
        }
    }
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        cancel: &CancelToken,
        on: &mut dyn FnMut(StreamEvent),
    ) -> Result<AssistantTurn, OxaError> {
        self.chat_with_lease(request, cancel, on, self.gate.acquire_operation())
    }
}
impl LocalEngine {
    /// Run a validated guest request through the owner's native engine lifecycle.
    pub fn chat_guest(
        &self,
        request: &ChatRequest<'_>,
        cancel: &CancelToken,
        on: &mut dyn FnMut(StreamEvent),
    ) -> Result<AssistantTurn, OxaError> {
        self.chat_with_lease(request, cancel, on, self.gate.acquire_guest_operation())
    }

    fn chat_with_lease(
        &self,
        request: &ChatRequest<'_>,
        cancel: &CancelToken,
        on: &mut dyn FnMut(StreamEvent),
        acquire: impl Future<Output = Result<RequestLease, AdmissionError>>,
    ) -> Result<AssistantTurn, OxaError> {
        let model = self.models.get(request.model).ok_or_else(invalid)?;
        let operation = wait_operation(acquire, cancel)?;
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure_driver(&mut state)?;
        if cancel.is_cancelled() {
            return Err(aborted());
        }
        let memory = self.inner.source.sample();
        let launch = if state.lifecycle.status().model == Some(model.alias.as_str())
            && state.lifecycle.status().state == State::Ready
        {
            // Keep the admitted launch snapshot even if files change while resident.
            state.lifecycle.loaded_model().ok_or_else(down)?
        } else {
            let (endpoint, launch) = self.endpoint(model)?;
            state.endpoints.insert(model.alias.clone(), endpoint);
            launch
        };
        state.target = Some(model.alias.clone());
        let loading = state
            .lifecycle
            .load_reserved(operation, launch, Instant::now(), memory)
            .map_err(|error| {
                startup_diagnostic(&error)
                    .map_or_else(down, |message| OxaError::new(codes::ENGINE_DOWN, message))
            });
        publish(&self.inner, &state.lifecycle);
        if let Err(error) = loading {
            if state.lifecycle.status().state != State::Backoff {
                state.target = None;
                if state.lifecycle.status().state != State::Ready {
                    let _ = state.lifecycle.unload(Instant::now());
                }
                let _ = advance(&self.inner, &mut state);
                return Err(error);
            }
        }
        loop {
            if cancel.is_cancelled() {
                let _ = state.lifecycle.unload(Instant::now());
                publish(&self.inner, &state.lifecycle);
                return Err(aborted());
            }
            if let Err(error) = advance(&self.inner, &mut state) {
                if state.lifecycle.status().state != State::Backoff {
                    let _ = state.lifecycle.unload(Instant::now());
                    state.target = None;
                    publish(&self.inner, &state.lifecycle);
                    return Err(if let Some(message) = startup_diagnostic(&error) {
                        OxaError::new(codes::ENGINE_DOWN, message)
                    } else if error.kind() == io::ErrorKind::Unsupported {
                        OxaError::new(
                            codes::ENGINE_DOWN,
                            concat!(
                                "Local engine version mismatch; install matching oxe-server ",
                                env!("CARGO_PKG_VERSION"),
                                " beside oxa or set oxa.local.binary."
                            ),
                        )
                    } else {
                        down()
                    });
                }
            }
            if state.lifecycle.status().state == State::Ready {
                break;
            }
            if matches!(state.lifecycle.status().state, State::Stopped | State::Failed) {
                return Err(down());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        state.target = None;
        let lease = state.lifecycle.take_ready_request().ok_or_else(down)?;
        let RuntimeState {
            lifecycle, endpoints, ..
        } = &mut *state;
        let endpoint = endpoints.get(&model.alias).ok_or_else(down)?;
        let pid = lifecycle.status().pid;
        let result = endpoint.transport.chat(
            lifecycle,
            lease,
            request,
            &mut Monitor {
                cancel,
                memory: &mut || self.inner.source.child(pid),
                healthy: &mut || endpoint.health(pid, false),
            },
            on,
        );
        publish(&self.inner, &state.lifecycle);
        self.inner.wake.notify_all();
        result
    }
}
impl Drop for LocalEngine {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.wake.notify_all();
        if let Some(worker) = self.inner.worker.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = worker.join();
        }
    }
}
fn watch(inner: Arc<Inner>) {
    let mut state = inner.state.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if inner.shutdown.load(Ordering::Acquire) {
            let _ = state.lifecycle.unload(Instant::now());
            state.driver_running = false;
            return;
        }
        let _ = advance(&inner, &mut state);
        if state.lifecycle.status().pid.is_none()
            && matches!(state.lifecycle.status().state, State::Stopped | State::Failed)
        {
            state.driver_running = false;
            return;
        }
        state = inner
            .wake
            .wait_timeout(state, Duration::from_millis(250))
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}
fn advance(inner: &Inner, state: &mut RuntimeState) -> io::Result<()> {
    let RuntimeState {
        lifecycle, endpoints, ..
    } = state;
    lifecycle.reap(Instant::now())?;
    let status = lifecycle.status();
    let endpoint = status.model.and_then(|alias| endpoints.get(alias));
    let pid = status.pid;
    let starting = status.state == State::Starting;
    let result = lifecycle
        .poll(Instant::now(), inner.source.child(pid), || {
            endpoint.map_or(Ok(false), |e| e.health(pid, starting))
        })
        .map(|_| ());
    let version_mismatch = endpoint.is_some_and(|e| e.version_mismatch.load(Ordering::Acquire));
    publish(inner, lifecycle);
    if lifecycle.status().pid.is_none() && matches!(lifecycle.status().state, State::Stopped | State::Failed)
    {
        state.target = None;
    }
    let current = lifecycle.status().model.map(str::to_owned);
    endpoints.retain(|alias, _| current.as_ref() == Some(alias) || state.target.as_ref() == Some(alias));
    if version_mismatch {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "local engine version mismatch",
        ))
    } else {
        result
    }
}
fn startup_diagnostic(error: &io::Error) -> Option<String> {
    let detail = error.get_ref()?;
    detail
        .downcast_ref::<crate::lifecycle::AdmissionRefusal>()
        .map(ToString::to_string)
        .or_else(|| {
            detail
                .downcast_ref::<crate::platform::Predicate>()
                .map(ToString::to_string)
        })
}
fn snapshot(lifecycle: &Lifecycle) -> LocalStatus {
    let status = lifecycle.status();
    LocalStatus {
        state: status.state,
        model: status.model.map(str::to_owned),
        pid: status.pid,
        waiting: status.waiting,
        stop_reason: status.stop_reason,
        memory_policy: lifecycle.memory_policy(),
        launcher_micros: lifecycle.memory_policy().map(|_| lifecycle.launcher_micros),
    }
}
fn publish(inner: &Inner, lifecycle: &Lifecycle) {
    *inner.status.write().unwrap_or_else(|e| e.into_inner()) = snapshot(lifecycle);
}
fn owns_port(pid: u32, address: SocketAddr) -> bool {
    let Ok(table) = std::fs::read_to_string("/proc/net/tcp") else {
        return false;
    };
    let wanted = format!("0100007F:{:04X}", address.port());
    let inode = table
        .lines()
        .filter_map(|line| {
            let f: Vec<_> = line.split_whitespace().collect();
            (f.get(1) == Some(&wanted.as_str()) && f.get(3) == Some(&"0A"))
                .then(|| f.get(9).copied())
                .flatten()
        })
        .next();
    let Some(inode) = inode else {
        return false;
    };
    let Ok(entries) = std::fs::read_dir(format!("/proc/{pid}/fd")) else {
        return false;
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|e| std::fs::read_link(e.path()).ok())
        .any(|p| p.as_os_str() == format!("socket:[{inode}]").as_str())
}
struct Signal {
    ready: Mutex<bool>,
    wake: Condvar,
}
impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        *self.ready.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.wake.notify_one();
    }
}
fn wait_operation(
    acquire: impl Future<Output = Result<RequestLease, AdmissionError>>,
    cancel: &CancelToken,
) -> Result<RequestLease, OxaError> {
    let signal = Arc::new(Signal {
        ready: Mutex::new(false),
        wake: Condvar::new(),
    });
    let waker = Waker::from(signal.clone());
    let mut context = Context::from_waker(&waker);
    let mut acquire = std::pin::pin!(acquire);
    let mut cancelled = std::pin::pin!(cancel.cancelled());
    loop {
        *signal.ready.lock().unwrap_or_else(|e| e.into_inner()) = false;
        if cancelled.as_mut().poll(&mut context).is_ready() {
            return Err(aborted());
        }
        if let Poll::Ready(result) = acquire.as_mut().poll(&mut context) {
            return result.map_err(|_| OxaError::new(codes::BUSY, "local request queue is unavailable"));
        }
        let mut ready = signal.ready.lock().unwrap_or_else(|e| e.into_inner());
        while !*ready {
            ready = signal.wake.wait(ready).unwrap_or_else(|e| e.into_inner());
        }
    }
}
fn invalid() -> OxaError {
    OxaError::new(codes::INVALID, "invalid user local-model configuration")
}
fn down() -> OxaError {
    OxaError::new(
        codes::ENGINE_DOWN,
        "local engine unavailable; check launch paths and inherited no-swap memory policy",
    )
}
fn aborted() -> OxaError {
    OxaError::new(codes::ABORTED, "local request cancelled")
}
fn redact(mut error: OxaError, token: &str) -> OxaError {
    error.message = error.message.replace(token, "[redacted]");
    if let Some(hint) = error.hint.as_mut() {
        *hint = hint.replace(token, "[redacted]");
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use oxa_fake::{FakeOpts, FakeServer, ScriptTurn, TestEnv};
    use oxa_proto::{Finish, FrozenPrefix, WireMsg};
    struct Memory {
        fail: AtomicBool,
    }
    impl Sample for Memory {
        fn sample(&self) -> io::Result<MemorySample> {
            if self.fail.load(Ordering::Acquire) {
                return Err(io::Error::other("synthetic missing sample"));
            }
            Ok(MemorySample {
                available_bytes: 8 * 1024 * 1024 * 1024,
                swap_free_bytes: 0,
                psi_full_avg10: 0.0,
                swap_current_bytes: 0,
                swap_limit_zero: true,
                oom_events: 0,
                host_oom_kills: 0,
                pids_max_events: 0,
            })
        }
    }
    fn engine(env: &TestEnv, policy: Policy) -> LocalEngine {
        engine_with_version(env, policy, concat!("oxe-", env!("CARGO_PKG_VERSION")))
    }
    fn engine_with_version(env: &TestEnv, policy: Policy, version: &str) -> LocalEngine {
        let version = version.to_owned();
        let models = ["fake-model", "other-model"]
            .into_iter()
            .map(|alias| {
                let alias_owned = alias.to_owned();
                let version = version.clone();
                LocalModel {
                    alias: alias.into(),
                    context_window: 1048576,
                    max_output: 1024,
                    thinking: None,
                    images: Some(false),
                    launch: Arc::new(move |port| {
                        Ok(Model {
                            name: alias_owned.clone(),
                            required_available_bytes: 4096,
                            config: Config {
                                executable: std::env::current_exe().map_err(|_| down())?,
                                arguments: vec![
                                    "--exact".into(),
                                    "native::tests::owned_fake_child_streams_and_unloads".into(),
                                    "--nocapture".into(),
                                ],
                                environment: [
                                    ("SYNTHETIC_LOCAL_CHILD".into(), "1".into()),
                                    ("SYNTHETIC_LOCAL_PORT".into(), port.to_string().into()),
                                    ("SYNTHETIC_LOCAL_ALIAS".into(), alias_owned.clone().into()),
                                    ("SYNTHETIC_LOCAL_BUILD".into(), version.clone().into()),
                                ]
                                .into(),
                                startup_timeout: Duration::from_secs(5),
                                health_interval: Duration::from_millis(50),
                                unhealthy_limit: 2,
                                term_grace: Duration::from_millis(100),
                                restart_backoff: Duration::from_millis(100),
                                max_restarts: 0,
                            },
                        })
                    }),
                }
            })
            .collect();
        LocalEngine::with_source(
            models,
            env.work().join("private"),
            StreamOpts {
                connect_timeout_ms: 1000,
                silence_limit_ms: 2000,
            },
            policy,
            Arc::new(Memory {
                fail: AtomicBool::new(false),
            }),
        )
        .unwrap()
    }
    #[test]
    fn admission_refusal_reaches_caller_with_numbers_and_never_spawns() {
        let env = TestEnv::new();
        let mut engine = engine(&env, Policy::default());
        let model = engine.models.get_mut("fake-model").unwrap();
        let launch = model.launch.clone();
        model.launch = Arc::new(move |port| {
            let mut admitted = launch(port)?;
            admitted.required_available_bytes = 8589934593;
            Ok(admitted)
        });
        let error = engine
            .chat(
                &ChatRequest::new(&prefix(), &[], "fake-model"),
                &CancelToken::new(),
                &mut |_| panic!("unadmitted generation"),
            )
            .unwrap_err();
        assert_eq!(error.code, codes::ENGINE_DOWN);
        assert!(error.message.contains("available=8589934592"));
        assert!(error.message.contains("required=8589934593"));
        assert_eq!(engine.status().pid, None);
    }
    #[test]
    fn swap_policy_refusal_reaches_caller_with_predicates_and_never_spawns() {
        struct PolicyMemory(bool, u64);
        impl Sample for PolicyMemory {
            fn sample(&self) -> io::Result<MemorySample> {
                let mut sample = Memory {
                    fail: AtomicBool::new(false),
                }
                .sample()?;
                sample.swap_limit_zero = self.0;
                sample.swap_current_bytes = self.1;
                Ok(sample)
            }
        }
        for (zero, current, predicate) in [
            (false, 0, "memory.swap.max_zero=false (must be true)"),
            (true, 64, "memory.swap.current=64 (must be 0)"),
        ] {
            let env = TestEnv::new();
            let mut engine = engine(&env, Policy::default());
            Arc::get_mut(&mut engine.inner).unwrap().source = Arc::new(PolicyMemory(zero, current));
            let error = engine
                .chat(
                    &ChatRequest::new(&prefix(), &[], "fake-model"),
                    &CancelToken::new(),
                    &mut |_| panic!("unadmitted generation"),
                )
                .unwrap_err();
            assert_eq!(error.code, codes::ENGINE_DOWN);
            assert!(error.message.contains(predicate), "{}", error.message);
            assert_eq!(engine.status().pid, None);
        }
    }
    #[test]
    fn scope_policy_is_visible_and_refused_child_is_reaped_before_queue_reuse() {
        struct Scope {
            bad: bool,
            pid: std::sync::atomic::AtomicU32,
        }
        impl crate::platform::PlatformLauncher for Scope {
            fn attach(&self, pid: u32, _gated: bool, _: Instant) -> io::Result<crate::platform::Attachment> {
                self.pid.store(pid, Ordering::Release);
                let mut sample = Memory {
                    fail: AtomicBool::new(false),
                }
                .sample()?;
                sample.swap_limit_zero = !self.bad;
                Ok(crate::platform::Attachment {
                    sample,
                    scope: Some("synthetic.scope".into()),
                })
            }
        }
        for bad in [true, false] {
            let env = TestEnv::new();
            let engine = engine(&env, Policy::default());
            let scope = Arc::new(Scope {
                bad,
                pid: std::sync::atomic::AtomicU32::new(0),
            });
            engine.inner.state.lock().unwrap().lifecycle.platform = Some(scope.clone());
            let result = engine.chat(
                &ChatRequest::new(&prefix(), &[], "fake-model"),
                &CancelToken::new(),
                &mut |_| {},
            );
            if bad {
                assert!(result
                    .unwrap_err()
                    .message
                    .contains("engine child memory.swap.max == 0"));
                assert!(engine.gate.try_acquire().is_err()); // unloaded, not a stale ready permit
            } else {
                result.unwrap();
                assert_eq!(
                    engine
                        .local_memory_status("fake-model")
                        .map(|(pid, policy, _)| (pid, policy)),
                    Some((scope.pid.load(Ordering::Acquire), "scope"))
                );
                assert_eq!(engine.local_memory_status("other-model"), None);
                engine
                    .inner
                    .state
                    .lock()
                    .unwrap()
                    .lifecycle
                    .unload(Instant::now())
                    .unwrap();
                engine.inner.wake.notify_all();
                until(&engine, State::Stopped);
            }
            assert_eq!(engine.status().pid, None);
            assert_eq!(engine.local_memory_status("fake-model"), None);
            assert!(wait_operation(engine.gate.acquire_operation(), &CancelToken::new()).is_ok());
            assert!(!PathBuf::from(format!("/proc/{}", scope.pid.load(Ordering::Acquire))).exists());
        }
    }
    fn prefix() -> FrozenPrefix {
        FrozenPrefix {
            sections: vec![("system".into(), "Synthetic task".into())],
            tools_json: "[]".into(),
            sha256: [0; 32],
        }
    }
    fn until(engine: &LocalEngine, wanted: State) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while engine.status().state != wanted {
            assert!(Instant::now() < deadline, "state {:?}", engine.status().state);
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    #[test]
    fn owned_fake_child_streams_and_unloads() {
        if std::env::var("SYNTHETIC_LOCAL_CHILD").as_deref() == Ok("1") {
            let port = std::env::var("SYNTHETIC_LOCAL_PORT").unwrap().parse().unwrap();
            let model = std::env::var("SYNTHETIC_LOCAL_ALIAS").unwrap();
            let _server = FakeServer::start(
                vec![
                    ScriptTurn::tool("read-note", "read", "{\"path\":\"note.txt\"}"),
                    ScriptTurn::text("Synthetic task finished"),
                ],
                FakeOpts {
                    port,
                    model,
                    controller: false,
                    build_info: std::env::var("SYNTHETIC_LOCAL_BUILD")
                        .ok()
                        .filter(|s| !s.is_empty()),
                    ..FakeOpts::default()
                },
            )
            .unwrap();
            loop {
                std::thread::park();
            }
        }
        let env = TestEnv::new();
        let engine = engine(
            &env,
            Policy {
                idle_timeout: Some(Duration::from_millis(100)),
            },
        );
        assert_eq!(engine.status().state, State::Stopped);
        let info = engine.info(Some("other-model")).unwrap();
        assert_eq!(info.limits_for, "other-model");
        assert_eq!(info.caps.context_window, Some(1048576));
        assert!(!info.caps.remote);
        assert!(info.caps.hint_clear, "core cleanup must reach native clear_hint");
        assert!(engine.inner.worker.lock().unwrap().is_none());
        assert!(!env.work().join("private").exists());
        engine.clear_hint("synthetic-child").unwrap();
        let prefix = prefix();
        let first = engine
            .chat(
                &ChatRequest::new(&prefix, &[], "fake-model"),
                &CancelToken::new(),
                &mut |_| {
                    assert_eq!(engine.info(None).unwrap().model, "fake-model");
                    assert_eq!(engine.clear_hint("busy").unwrap_err().code, codes::BUSY);
                },
            )
            .unwrap();
        assert_eq!(first.finish, Finish::ToolCalls);
        let pid = engine.status().pid.unwrap();
        let continued = [
            WireMsg::user("Read the synthetic note"),
            first.msg,
            WireMsg::tool_result("read-note", "SYNTHETIC NOTE"),
        ];
        let second = engine
            .chat(
                &ChatRequest::new(&prefix, &continued, "fake-model"),
                &CancelToken::new(),
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(second.msg.content.as_deref(), Some("Synthetic task finished"));
        assert_eq!(engine.status().pid, Some(pid));
        engine.clear_hint("synthetic-child").unwrap();
        until(&engine, State::Stopped);
        assert!(!PathBuf::from(format!("/proc/{pid}")).exists());
        assert_eq!(std::fs::read_dir(env.work().join("private")).unwrap().count(), 0);
        drop(engine);
    }
    #[test]
    fn guest_admission_cancellation_leaves_owner_lease_held() {
        let env = TestEnv::new();
        let engine = engine(&env, Policy::default());
        let active = wait_operation(engine.gate.acquire_operation(), &CancelToken::new()).unwrap();
        let cancel = CancelToken::new();
        std::thread::scope(|scope| {
            let guest = scope.spawn(|| {
                engine.chat_guest(
                    &ChatRequest::new(&prefix(), &[], "fake-model"),
                    &cancel,
                    &mut |_| panic!("guest ran while the owner held the engine lease"),
                )
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            while engine.gate.waiting() == 0 && !guest.is_finished() && Instant::now() < deadline {
                std::thread::yield_now();
            }
            let waiting = engine.gate.waiting();
            cancel.cancel();
            assert_eq!(guest.join().unwrap().unwrap_err().code, codes::ABORTED);
            assert_eq!(waiting, 1, "guest must join the owner's engine queue");
        });
        assert_eq!(engine.gate.waiting(), 0);
        assert!(matches!(engine.gate.try_acquire(), Err(AdmissionError::Full)));
        assert_eq!(engine.status().state, State::Stopped);
        drop(active);
        assert!(wait_operation(engine.gate.acquire_operation(), &CancelToken::new()).is_ok());
    }

    #[test]
    fn cancellation_reaps_the_child_before_a_fresh_request_reuses_the_queue() {
        let env = TestEnv::new();
        let engine = engine(&env, Policy::default());
        let prefix = prefix();
        let cancel = CancelToken::new();
        let mut interrupted_pid = None;
        let result = engine.chat(
            &ChatRequest::new(&prefix, &[], "fake-model"),
            &cancel,
            &mut |_| {
                interrupted_pid = engine.status().pid;
                cancel.cancel();
            },
        );
        assert_eq!(result.unwrap_err().code, codes::ABORTED);
        let interrupted_pid = interrupted_pid.expect("stream callback observed the owned child");
        // Reuse the same gate while the interrupted child may still be stopping.
        let next = engine
            .chat(
                &ChatRequest::new(&prefix, &[], "fake-model"),
                &CancelToken::new(),
                &mut |_| {
                    assert!(!PathBuf::from(format!("/proc/{interrupted_pid}")).exists());
                },
            )
            .unwrap();
        assert_eq!(next.finish, Finish::ToolCalls);
        assert_ne!(engine.status().pid, Some(interrupted_pid));
        drop(engine);
        assert_eq!(std::fs::read_dir(env.work().join("private")).unwrap().count(), 0);
    }
    #[test]
    fn resident_model_reuses_the_frozen_launch_description() {
        use std::sync::atomic::AtomicUsize;
        let env = TestEnv::new();
        let mut engine = engine(&env, Policy::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let model = engine.models.get_mut("fake-model").unwrap();
        let launch = model.launch.clone();
        let count = calls.clone();
        model.launch = Arc::new(move |port| {
            count.fetch_add(1, Ordering::SeqCst);
            launch(port)
        });
        let prefix = prefix();
        for _ in 0..2 {
            engine
                .chat(
                    &ChatRequest::new(&prefix, &[], "fake-model"),
                    &CancelToken::new(),
                    &mut |_| {},
                )
                .unwrap();
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "resident work must not recheck changed launch files"
        );
    }
    #[test]
    fn unversioned_or_mismatched_child_is_refused_before_generation() {
        for version in ["", "oxe-incompatible"] {
            let env = TestEnv::new();
            let engine = engine_with_version(&env, Policy::default(), version);
            let prefix = prefix();
            let error = engine
                .chat(
                    &ChatRequest::new(&prefix, &[], "fake-model"),
                    &CancelToken::new(),
                    &mut |_| panic!("incompatible engine emitted generation"),
                )
                .expect_err("incompatible engine must be refused");
            assert_eq!(error.code, codes::ENGINE_DOWN);
            assert!(error.message.contains("version"));
            until(&engine, State::Stopped);
            drop(engine);
            assert_eq!(std::fs::read_dir(env.work().join("private")).unwrap().count(), 0);
        }
    }
    #[test]
    fn model_switch_reaps_the_previous_child_and_engine_drop_reaps_the_last() {
        let env = TestEnv::new();
        let engine = engine(&env, Policy::default());
        let prefix = prefix();
        engine
            .chat(
                &ChatRequest::new(&prefix, &[], "fake-model"),
                &CancelToken::new(),
                &mut |_| {},
            )
            .unwrap();
        let previous = engine.status().pid.unwrap();
        engine
            .chat(
                &ChatRequest::new(&prefix, &[], "other-model"),
                &CancelToken::new(),
                &mut |_| {},
            )
            .unwrap();
        let last = engine.status().pid.unwrap();
        assert_ne!(previous, last);
        assert!(!PathBuf::from(format!("/proc/{previous}")).exists());
        drop(engine);
        assert!(!PathBuf::from(format!("/proc/{last}")).exists());
        assert_eq!(std::fs::read_dir(env.work().join("private")).unwrap().count(), 0);
    }
}
