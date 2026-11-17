//! Caller-driven model lifecycle. No thread, executor, filesystem or implicit launch.
use crate::queue::{AdmissionError, Lease, RequestQueue};
use crate::{Config, State, Supervisor};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Resolved trusted launch profile, not a controller configuration or shell command.
#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub config: Config,
    /// Full resident weights plus anonymous allocations and explicit margin; derived at cold launch.
    /// No size is guessed from model file length.
    pub required_available_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Policy {
    /// None disables idle unload; memory stops remain enabled.
    pub idle_timeout: Option<Duration>,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            idle_timeout: Some(Duration::from_secs(7200)),
        }
    }
}

/// Fresh sample from the owned engine's no-swap cgroup and the host.
/// The caller must report read/parse failures as Err, never a zero-filled sample.
#[derive(Clone, Copy, Debug)]
pub struct MemorySample {
    pub available_bytes: u64,
    pub swap_free_bytes: u64,
    pub psi_full_avg10: f64,
    pub swap_current_bytes: u64,
    pub swap_limit_zero: bool,
    /// Sum of oom, oom_kill and oom_group_kill; checked against pre-launch baseline.
    pub oom_events: u64,
    pub host_oom_kills: u64,
    pub pids_max_events: u64,
}
#[derive(Debug)]
pub(crate) enum AdmissionRefusal {
    Allocation {
        available: u64,
        required: u64,
    },
    Policy {
        required: u64,
        swap_limit_zero: bool,
        swap_current_bytes: u64,
    },
}
impl std::fmt::Display for AdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation { available, required } => write!(f, "Local engine memory admission refused: available={available} bytes, required={required} bytes (allocation budget including margin); reduce configured context or free memory."),
            Self::Policy { required, swap_limit_zero, swap_current_bytes } => write!(f, "Local engine memory admission refused: required={required} (must be >0), memory.swap.max_zero={swap_limit_zero} (must be true), memory.swap.current={swap_current_bytes} (must be 0)."),
        }
    }
}
impl std::error::Error for AdmissionRefusal {}

impl MemorySample {
    fn validate(self) -> io::Result<Self> {
        if !self.psi_full_avg10.is_finite() || !(0.0..=100.0).contains(&self.psi_full_avg10) {
            return Err(io::Error::other("invalid memory pressure sample"));
        }
        Ok(self)
    }
    fn admit_host(self, required: u64) -> io::Result<Self> {
        let sample = self.validate()?;
        if required > 0 && sample.available_bytes < required {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                AdmissionRefusal::Allocation {
                    available: sample.available_bytes,
                    required,
                },
            ));
        }
        if required == 0 {
            return Err(io::Error::other(AdmissionRefusal::Policy {
                required,
                swap_limit_zero: sample.swap_limit_zero,
                swap_current_bytes: sample.swap_current_bytes,
            }));
        }
        Ok(sample)
    }
    fn admit(self, required: u64) -> io::Result<Self> {
        let sample = self.admit_host(required)?;
        if !sample.swap_limit_zero || sample.swap_current_bytes != 0 {
            return Err(io::Error::other(AdmissionRefusal::Policy {
                required,
                swap_limit_zero: sample.swap_limit_zero,
                swap_current_bytes: sample.swap_current_bytes,
            }));
        }
        Ok(sample)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    Explicit,
    Switch,
    Idle,
    MemoryPressure,
    MemoryUnavailable,
    RequestInterrupted,
}

/// Borrowed status never probes, starts a process, or opens a model file.
#[derive(Clone, Copy, Debug)]
pub struct Status<'a> {
    pub model: Option<&'a str>,
    pub state: State,
    pub pid: Option<u32>,
    pub waiting: usize,
    pub stop_reason: Option<StopReason>,
}

/// Generation owns the same exclusion as lifecycle operations. Drop only after
/// upstream completion/cancellation; release starts the idle interval.
pub struct RequestLease {
    lease: Option<Lease>,
    clock: Arc<Mutex<Clock>>,
}
impl Drop for RequestLease {
    fn drop(&mut self) {
        let mut clock = self.clock.lock().unwrap_or_else(|e| e.into_inner());
        clock.last_use = Instant::now().max(clock.observed);
    }
}
struct Clock {
    last_use: Instant,
    observed: Instant,
}

/// Cloneable admission handle; waiting does not borrow the lifecycle owner, so
/// its caller can keep driving health/memory/stop polls while requests queue.
#[derive(Clone)]
pub struct RequestGate {
    queue: RequestQueue,
    ready: Arc<AtomicBool>,
    clock: Arc<Mutex<Clock>>,
}
impl RequestGate {
    pub fn waiting(&self) -> usize {
        self.queue.waiting()
    }
    fn granted(&self, lease: Lease) -> Result<RequestLease, AdmissionError> {
        if !self.ready.load(Ordering::Acquire) {
            return Err(AdmissionError::Full);
        }
        Ok(self.operation(lease))
    }
    fn operation(&self, lease: Lease) -> RequestLease {
        RequestLease {
            lease: Some(lease),
            clock: self.clock.clone(),
        }
    }
    pub fn try_acquire(&self) -> Result<RequestLease, AdmissionError> {
        self.granted(self.queue.try_acquire()?)
    }
    pub async fn acquire(&self) -> Result<RequestLease, AdmissionError> {
        self.granted(self.queue.acquire().await?)
    }
    /// Reserve the same FIFO slot for load/switch followed by generation.
    /// Unlike acquire(), this permits an unloaded engine; hand it to load_reserved.
    pub async fn acquire_operation(&self) -> Result<RequestLease, AdmissionError> {
        Ok(self.operation(self.queue.acquire().await?))
    }
    /// Reserve the same load/generation lease at guest priority.
    pub async fn acquire_guest_operation(&self) -> Result<RequestLease, AdmissionError> {
        Ok(self.operation(self.queue.acquire_guest().await?))
    }
}

struct Guard {
    baseline: MemorySample,
    host_since: Option<Instant>,
    loading_since: Option<Instant>,
    previous: Instant,
    loading: bool,
}
impl Guard {
    fn new(baseline: MemorySample, now: Instant) -> Self {
        Self {
            baseline,
            host_since: None,
            loading_since: None,
            previous: now,
            loading: true,
        }
    }
    // GUARD-2b: host floors for 3s, ready PSI >20% for 3s,
    // loading PSI >60% for 5s; swap/OOM/pid-limit stops immediately.
    fn unsafe_sample(&mut self, now: Instant, s: MemorySample, loading: bool) -> io::Result<bool> {
        let s = s.validate()?;
        if now < self.previous
            || s.oom_events < self.baseline.oom_events
            || s.host_oom_kills < self.baseline.host_oom_kills
            || s.pids_max_events < self.baseline.pids_max_events
        {
            return Err(io::Error::other("memory clock or counter regressed"));
        }
        self.previous = now;
        self.loading = loading;
        let immediate = !s.swap_limit_zero
            || s.swap_current_bytes != 0
            || s.oom_events > self.baseline.oom_events
            || s.host_oom_kills > self.baseline.host_oom_kills
            || s.pids_max_events > self.baseline.pids_max_events;
        let host = s.available_bytes.saturating_add(s.swap_free_bytes) < 2 * 1024 * 1024 * 1024
            || s.available_bytes < 256 * 1024 * 1024
            || (!loading && s.psi_full_avg10 > 20.0);
        let loading_pressure = loading && s.psi_full_avg10 > 60.0;
        let host_stop = sustained(&mut self.host_since, now, host, Duration::from_secs(3));
        let loading_stop = sustained(
            &mut self.loading_since,
            now,
            loading_pressure,
            Duration::from_secs(5),
        );
        Ok(immediate || host_stop || loading_stop)
    }
}
fn sustained(since: &mut Option<Instant>, now: Instant, high: bool, duration: Duration) -> bool {
    if !high {
        *since = None;
        return false;
    }
    now.duration_since(*since.get_or_insert(now)) >= duration
}

/// All commands and polls run on one owner. Request leases may outlive a poll.
/// The caller supplies bounded authenticated health I/O and fresh memory samples.
pub struct Lifecycle {
    queue: RequestQueue,
    pub(crate) platform: Option<Arc<dyn crate::platform::PlatformLauncher>>,
    ready: Arc<AtomicBool>,
    supervisor: Option<Supervisor>,
    model: Option<Model>,
    next: Option<Model>,
    control: Option<Lease>,
    retain_ready: bool,
    guard: Option<Guard>,
    scope: Option<String>,
    pub(crate) launcher_micros: u64,
    clock: Arc<Mutex<Clock>>,
    policy: Policy,
    stop_reason: Option<StopReason>,
}
impl Lifecycle {
    pub fn new(max_waiters: usize, policy: Policy) -> Self {
        let now = Instant::now();
        Self {
            queue: RequestQueue::new(max_waiters),
            platform: None,
            ready: Arc::new(AtomicBool::new(false)),
            supervisor: None,
            model: None,
            next: None,
            control: None,
            retain_ready: false,
            guard: None,
            scope: None,
            launcher_micros: 0,
            clock: Arc::new(Mutex::new(Clock {
                last_use: now,
                observed: now,
            })),
            policy,
            stop_reason: None,
        }
    }
    pub fn status(&self) -> Status<'_> {
        Status {
            model: self.model.as_ref().map(|m| m.name.as_str()),
            state: self.supervisor.as_ref().map_or(State::Stopped, Supervisor::state),
            pid: self.supervisor.as_ref().and_then(Supervisor::pid),
            waiting: self.queue.waiting(),
            stop_reason: self.stop_reason,
        }
    }
    pub(crate) fn loaded_model(&self) -> Option<Model> {
        self.model.clone()
    }
    pub fn try_request(&self) -> Result<RequestLease, AdmissionError> {
        self.requests().try_acquire()
    }
    pub fn requests(&self) -> RequestGate {
        RequestGate {
            queue: self.queue.clone(),
            ready: self.ready.clone(),
            clock: self.clock.clone(),
        }
    }
    pub(crate) fn owns(&self, request: &RequestLease) -> bool {
        Arc::ptr_eq(&self.clock, &request.clock) && request.lease.is_some()
    }
    /// Transfer interrupted generation ownership back to the lifecycle. Poll
    /// through Stopped before admitting more work; even a failed stop retains
    /// the permit so another request cannot race a still-running generation.
    pub fn interrupt(&mut self, mut request: RequestLease, now: Instant) -> io::Result<()> {
        if !self.owns(&request) || self.control.is_some() {
            return Err(io::Error::other("request lease does not own this engine"));
        }
        self.control = request.lease.take();
        self.retain_ready = false;
        self.next = None;
        let reason = match self.stop_reason {
            Some(reason @ (StopReason::MemoryPressure | StopReason::MemoryUnavailable)) => reason,
            _ => StopReason::RequestInterrupted,
        };
        self.stop(now, reason)?;
        if self.status().state == State::Stopped {
            self.supervisor = None;
            self.model = None;
            self.guard = None;
            self.control = None;
        }
        Ok(())
    }
    /// Load/switch only when exclusive. The owner keeps the lease through readiness.
    /// A loaded name reuses its frozen launch profile; unload before replacing it.
    pub fn load(&mut self, model: Model, now: Instant, memory: io::Result<MemorySample>) -> io::Result<()> {
        let lease = self.queue.try_acquire().map_err(io::Error::other)?;
        self.load_inner(lease, model, now, memory, false)
    }
    /// Keep the admitted ticket through model readiness, then take_ready_request
    /// transfers that very lease into generation without admitting a competitor.
    pub fn load_reserved(
        &mut self,
        mut request: RequestLease,
        model: Model,
        now: Instant,
        memory: io::Result<MemorySample>,
    ) -> io::Result<()> {
        if !self.owns(&request) {
            return Err(io::Error::other("foreign model-operation lease"));
        }
        let lease = request.lease.take().unwrap();
        self.load_inner(lease, model, now, memory, true)
    }
    pub fn take_ready_request(&mut self) -> Option<RequestLease> {
        if self.status().state != State::Ready || !self.retain_ready {
            return None;
        }
        self.retain_ready = false;
        let lease = self.control.take()?;
        Some(RequestLease {
            lease: Some(lease),
            clock: self.clock.clone(),
        })
    }
    fn load_inner(
        &mut self,
        lease: Lease,
        model: Model,
        now: Instant,
        memory: io::Result<MemorySample>,
        retain_ready: bool,
    ) -> io::Result<()> {
        if self.status().state == State::Ready && self.status().model == Some(model.name.as_str()) {
            if retain_ready {
                self.control = Some(lease);
                self.retain_ready = true;
            }
            return Ok(());
        }
        if self.control.is_some() {
            return Err(io::Error::other("local model operation in progress"));
        }
        if model.name.is_empty() || model.name.len() > 128 || model.name.chars().any(char::is_control) {
            return Err(io::Error::other("invalid local model name"));
        }
        let sample = self.admit(memory?, model.required_available_bytes)?;
        // Validate launch policy before evicting the old model.
        let candidate = Supervisor::new(model.config.clone())?;
        self.control = Some(lease);
        self.retain_ready = retain_ready;
        self.stop_reason = None;
        if self.supervisor.as_ref().is_some_and(|s| s.pid().is_some()) {
            self.next = Some(model);
            self.stop(now, StopReason::Switch)?;
        } else {
            self.install(model, candidate, sample, now)?;
        }
        Ok(())
    }
    fn admit(&self, sample: MemorySample, required: u64) -> io::Result<MemorySample> {
        if self.platform.is_some() {
            sample.admit_host(required)
        } else {
            sample.admit(required)
        }
    }
    fn attach(&mut self, now: Instant) -> io::Result<()> {
        if let (Some(platform), Some(pid)) = (&self.platform, self.status().pid) {
            let gated = self.supervisor.as_ref().is_some_and(Supervisor::gated);
            let ready = self.supervisor.as_mut().unwrap().await_gate();
            match ready
                .and_then(|()| platform.attach(pid, gated, self.supervisor.as_ref().unwrap().deadline))
                .and_then(|attached| {
                    crate::platform::no_swap(attached.sample)?;
                    self.supervisor.as_mut().unwrap().release()?;
                    Ok(attached)
                }) {
                Ok(attached) => {
                    self.guard = Some(Guard::new(attached.sample, now));
                    self.scope = attached.scope;
                    self.launcher_micros = self
                        .supervisor
                        .as_ref()
                        .unwrap()
                        .launch_started
                        .elapsed()
                        .as_micros()
                        .min(u64::MAX as u128) as u64;
                }
                Err(error) => {
                    self.supervisor.take(); // owned Drop kills the group and reaps before lease release
                    self.model = None;
                    self.guard = None;
                    self.control = None;
                    return Err(
                        if error
                            .get_ref()
                            .is_some_and(|e| e.is::<crate::platform::Predicate>())
                        {
                            error
                        } else {
                            crate::platform::refused("engine child cgroup v2 memory/pids metadata readable")
                        },
                    );
                }
            }
        }
        Ok(())
    }
    fn install(
        &mut self,
        model: Model,
        mut supervisor: Supervisor,
        sample: MemorySample,
        now: Instant,
    ) -> io::Result<()> {
        self.ready.store(false, Ordering::Release);
        self.model = Some(model);
        supervisor.platform = self.platform.clone();
        self.supervisor = Some(supervisor);
        self.guard = Some(Guard::new(sample, now));
        self.stop_reason = None;
        let supervisor = self.supervisor.as_mut().unwrap();
        let result = supervisor.start(now);
        if supervisor.state() == State::Failed {
            self.control = None;
        }
        result?;
        self.attach(now)
    }
    pub fn unload(&mut self, now: Instant) -> io::Result<()> {
        self.retain_ready = false;
        if self.control.is_none() {
            self.control = Some(self.queue.try_acquire().map_err(io::Error::other)?);
        }
        self.next = None;
        self.stop(now, StopReason::Explicit)?;
        if self.status().state == State::Stopped {
            self.supervisor = None;
            self.model = None;
            self.guard = None;
            self.control = None;
        }
        Ok(())
    }
    fn stop(&mut self, now: Instant, reason: StopReason) -> io::Result<()> {
        self.ready.store(false, Ordering::Release);
        self.stop_reason = Some(reason);
        if let Some(s) = &mut self.supervisor {
            s.stop(now)?;
        }
        Ok(())
    }
    pub(crate) fn memory_policy(&self) -> Option<&'static str> {
        (self.platform.is_some() && self.status().pid.is_some()).then_some(if self.scope.is_some() {
            "scope"
        } else {
            "inherited"
        })
    }
    pub(crate) fn reap(&mut self, now: Instant) -> io::Result<()> {
        if let Some(supervisor) = &mut self.supervisor {
            supervisor.reap(now)?;
        }
        Ok(())
    }
    pub fn poll(
        &mut self,
        now: Instant,
        memory: io::Result<MemorySample>,
        healthy: impl FnMut() -> io::Result<bool>,
    ) -> io::Result<Status<'_>> {
        self.clock.lock().unwrap_or_else(|e| e.into_inner()).observed = now;
        let sample = memory.and_then(MemorySample::validate);
        let state = self.status().state;
        if !(self.platform.is_some() && self.status().pid.is_none())
            && matches!(
                state,
                State::Starting | State::Ready | State::Backoff | State::Stopping
            )
        {
            let decision = match (&mut self.guard, &sample) {
                (Some(guard), Ok(sample)) => {
                    let loading = if state == State::Stopping {
                        guard.loading
                    } else {
                        state != State::Ready
                    };
                    guard.unsafe_sample(now, *sample, loading)
                }
                _ => Err(io::Error::other("memory sample unavailable")),
            };
            let reason = match decision {
                Ok(true) => Some(StopReason::MemoryPressure),
                Err(_) => Some(StopReason::MemoryUnavailable),
                _ => None,
            };
            if let Some(reason) = reason {
                self.next = None;
                self.stop(now, reason)?;
            }
        }
        // Restarts cannot overlap cancellation cleanup from the previous request.
        if self.status().state == State::Backoff && self.control.is_none() {
            let Ok(lease) = self.queue.try_acquire() else {
                return Ok(self.status());
            };
            self.control = Some(lease);
        }
        if self.status().state == State::Backoff {
            let admitted = sample.as_ref().ok().is_some_and(|s| {
                self.model
                    .as_ref()
                    .is_some_and(|m| self.admit(*s, m.required_available_bytes).is_ok())
            });
            if !admitted {
                self.stop(now, StopReason::MemoryUnavailable)?;
            }
        }
        let previous_pid = self.status().pid;
        if let Some(s) = &mut self.supervisor {
            let result = s.poll(now, healthy);
            if s.state() == State::Failed {
                self.control = None;
            }
            result?;
        }
        if self.status().pid.is_some() && self.status().pid != previous_pid {
            self.attach(now)?;
        }
        match self.status().state {
            State::Stopped => {
                self.supervisor = None;
                self.model = None;
                self.guard = None;
                if let Some(next) = self.next.take() {
                    let admitted = sample.and_then(|s| self.admit(s, next.required_available_bytes));
                    match admitted {
                        Ok(sample) => {
                            let supervisor = Supervisor::new(next.config.clone())?;
                            self.install(next, supervisor, sample, now)?;
                        }
                        Err(error) => {
                            self.control = None;
                            self.stop_reason = Some(StopReason::MemoryUnavailable);
                            return Err(error);
                        }
                    }
                } else {
                    self.control = None;
                    self.retain_ready = false;
                }
            }
            State::Ready => {
                self.ready.store(true, Ordering::Release);
                if self.control.is_some() && !self.retain_ready {
                    self.clock.lock().unwrap_or_else(|e| e.into_inner()).last_use = now;
                    self.control = None;
                }
                if let Some(timeout) = self.policy.idle_timeout {
                    if let Some(lease) = idle_lease(now, &self.clock, timeout, || self.queue.try_acquire()) {
                        self.control = Some(lease);
                        self.stop(now, StopReason::Idle)?;
                    }
                }
            }
            State::Failed => {
                self.control = None;
            }
            _ => (),
        }
        self.ready
            .store(self.status().state == State::Ready, Ordering::Release);
        Ok(self.status())
    }
}

impl Drop for Lifecycle {
    fn drop(&mut self) {
        self.ready.store(false, Ordering::Release);
        self.queue.close();
    }
}

// Acquire is a reservation boundary: another request may finish immediately
// before it grants exclusive ownership. The deadline must reflect that release.
fn idle_lease(
    now: Instant,
    clock: &Mutex<Clock>,
    timeout: Duration,
    acquire: impl FnOnce() -> Result<Lease, AdmissionError>,
) -> Option<Lease> {
    let lease = acquire().ok()?;
    let last = clock.lock().unwrap_or_else(|e| e.into_inner()).last_use;
    (now.saturating_duration_since(last) >= timeout).then_some(lease)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_at_idle_reservation_resets_the_deadline() {
        let last = Instant::now();
        let now = last + Duration::from_secs(20);
        let queue = RequestQueue::new(0);
        let clock = Arc::new(Mutex::new(Clock {
            last_use: last,
            observed: now,
        }));
        let active = RequestLease {
            lease: Some(queue.try_acquire().unwrap()),
            clock: clock.clone(),
        };
        // Deterministically schedule a real request release immediately before
        // the queue grants the poll its idle reservation.
        let reservation = idle_lease(now, &clock, Duration::from_secs(10), || {
            drop(active);
            queue.try_acquire()
        });
        assert!(
            reservation.is_none(),
            "request release must start a fresh idle interval"
        );
        assert!(queue.try_acquire().is_ok());
    }
}
