//! Caller-driven ownership of one local engine. No daemon or background thread.
#![cfg(unix)]

mod admission;
mod child_auth;
mod configured;
pub use configured::{configured_engine, LOCAL_SETTING};
pub mod launch_profile;
pub mod lifecycle;
pub mod memory;
pub mod native;
pub mod platform;

pub mod queue;
pub mod transport;

use nix::errno::Errno;
use nix::sys::signal::{killpg, Signal};
use nix::unistd::Pid;
use std::ffi::OsString;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Trusted launch configuration. Do not put credentials in arguments.
#[derive(Clone)]
pub struct Config {
    pub executable: PathBuf,
    pub arguments: Vec<OsString>,
    pub environment: std::collections::BTreeMap<OsString, OsString>,
    pub startup_timeout: Duration,
    pub health_interval: Duration,
    pub unhealthy_limit: u32,
    pub term_grace: Duration,
    pub restart_backoff: Duration,
    /// Restarts after the initial attempt; never reset by successful health.
    pub max_restarts: u32,
}
impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("executable", &"[configured]")
            .field("argument_count", &self.arguments.len())
            .field("environment_count", &self.environment.len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum State {
    Stopped,
    Starting,
    Ready,
    Stopping,
    Backoff,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Failure {
    Spawn,
    Exited,
    StartupTimeout,
    Unhealthy,
}

/// One child, including its process group. Mutable access serializes control.
///
/// Call `stop` and continue polling until Stopped for bounded graceful teardown.
/// Drop is an emergency kill-and-reap fallback, not a graceful shutdown API.
pub struct Supervisor {
    config: Config,
    child: Option<Child>,
    pub(crate) platform: Option<std::sync::Arc<dyn platform::PlatformLauncher>>,
    state: State,
    deadline: Instant,
    next_health: Instant,
    pub(crate) launch_started: Instant,
    bad_health: u32,
    restarts: u32,
    wanted: bool,
    killed: bool,
    failure: Option<Failure>,
}

impl Supervisor {
    pub fn new(config: Config) -> io::Result<Self> {
        if !config.executable.is_absolute()
            || config.startup_timeout.is_zero()
            || config.health_interval.is_zero()
            || config.unhealthy_limit == 0
            || config.restart_backoff.is_zero()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid engine supervision policy",
            ));
        }
        let now = Instant::now();
        Ok(Self {
            config,
            child: None,
            platform: None,
            state: State::Stopped,
            deadline: now,
            next_health: now,
            launch_started: now,
            bad_health: 0,
            restarts: 0,
            wanted: false,
            killed: false,
            failure: None,
        })
    }

    pub fn state(&self) -> State {
        self.state
    }
    pub fn last_failure(&self) -> Option<Failure> {
        self.failure
    }
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    /// Idempotent while desired. An explicit start after Failed gets a new budget.
    pub fn start(&mut self, now: Instant) -> io::Result<()> {
        if self.child.is_some() || self.wanted {
            return Ok(());
        }
        self.wanted = true;
        self.restarts = 0;
        self.failure = None;
        self.spawn(now)
    }

    fn spawn(&mut self, now: Instant) -> io::Result<()> {
        self.launch_started = Instant::now();
        let result = (|| {
            let gated = self
                .platform
                .as_ref()
                .map(|p| p.requires_scope())
                .transpose()?
                .unwrap_or(false);
            let mut command = if gated {
                let mut command = gate_command()?;
                command.arg(platform::CHILD_GATE).arg(&self.config.executable);
                command
            } else {
                Command::new(&self.config.executable)
            };
            command
                .args(&self.config.arguments)
                // Ambient model/provider settings and credentials must not cross this boundary.
                .env_clear()
                .envs(&self.config.environment)
                .stdin(if gated { Stdio::piped() } else { Stdio::null() })
                .stdout(if gated { Stdio::piped() } else { Stdio::null() })
                .stderr(Stdio::null())
                .process_group(0)
                .spawn()
        })();
        match result {
            Ok(child) => {
                self.child = Some(child);
                self.state = State::Starting;
                self.deadline = now + self.config.startup_timeout;
                self.next_health = now;
                self.bad_health = 0;
                self.killed = false;
                Ok(())
            }
            Err(error) => {
                self.failure = Some(Failure::Spawn);
                self.after_exit(now);
                Err(error)
            }
        }
    }

    pub(crate) fn gated(&self) -> bool {
        self.child.as_ref().is_some_and(|c| c.stdin.is_some())
    }
    pub(crate) fn await_gate(&mut self) -> io::Result<()> {
        use nix::fcntl::{fcntl, FcntlArg, OFlag};
        use std::io::Read;
        if let Some(mut pipe) = self.child.as_mut().and_then(|child| child.stdout.take()) {
            let flags = OFlag::from_bits_truncate(fcntl(&pipe, FcntlArg::F_GETFL)?);
            fcntl(&pipe, FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK))?;
            let mut ready = [0];
            loop {
                match pipe.read(&mut ready) {
                    Ok(1) if ready == [0x6b] => return Ok(()),
                    Err(error)
                        if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < self.deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(1))
                    }
                    _ => {
                        return Err(platform::refused(
                            "owned child gate ready before startup deadline",
                        ))
                    }
                }
            }
        }
        Ok(())
    }
    pub(crate) fn release(&mut self) -> io::Result<()> {
        use std::io::Write;
        platform::before_startup_deadline(self.deadline)?;
        if let Some(pipe) = self.child.as_mut().and_then(|child| child.stdin.take()) {
            let mut pipe = pipe;
            pipe.write_all(&[0xa7])
                .map_err(|_| platform::refused("owned child scope gate released"))?;
        }
        Ok(())
    }

    /// Cancels automatic restarts, including during backoff.
    pub fn stop(&mut self, now: Instant) -> io::Result<()> {
        self.wanted = false;
        if self.child.is_none() {
            self.state = State::Stopped;
            return Ok(());
        }
        self.begin_stop(now)
    }

    fn signal(&self, signal: Signal) -> io::Result<()> {
        if let Some(child) = &self.child {
            match killpg(Pid::from_raw(child.id() as i32), signal) {
                Ok(()) | Err(Errno::ESRCH) => (),
                Err(error) => return Err(io::Error::from_raw_os_error(error as i32)),
            }
        }
        Ok(())
    }

    fn begin_stop(&mut self, now: Instant) -> io::Result<()> {
        if self.state != State::Stopping {
            // Keep the old state on a failed signal so a subsequent poll retries.
            self.signal(Signal::SIGTERM)?;
            self.state = State::Stopping;
            self.deadline = now + self.config.term_grace;
        }
        Ok(())
    }

    fn after_exit(&mut self, now: Instant) {
        if !self.wanted {
            self.state = State::Stopped;
        } else if self.restarts < self.config.max_restarts {
            self.restarts += 1;
            self.state = State::Backoff;
            self.deadline = now + self.config.restart_backoff;
        } else {
            self.wanted = false;
            self.state = State::Failed;
        }
    }

    pub(crate) fn reap(&mut self, now: Instant) -> io::Result<()> {
        if let Some(child) = self.child.as_mut() {
            if child.try_wait()?.is_some() {
                // Clean up descendants even if the engine leader died first. Do this
                // before discarding ownership or starting a replacement process.
                self.signal(Signal::SIGKILL)?;
                self.child = None;
                if self.state != State::Stopping {
                    self.failure = Some(Failure::Exited);
                }
                self.after_exit(now);
            }
        }
        Ok(())
    }

    /// Advances lifecycle, checking exit before health. Probe errors count as unhealthy.
    /// Never retries a generation request. Caller must bound probe I/O time.
    pub fn poll(&mut self, now: Instant, mut healthy: impl FnMut() -> io::Result<bool>) -> io::Result<State> {
        self.reap(now)?;
        match self.state {
            State::Backoff if now >= self.deadline => self.spawn(now)?,
            State::Starting | State::Ready => {
                if !self.wanted {
                    self.begin_stop(now)?;
                } else if self.state == State::Starting && now >= self.deadline {
                    self.failure = Some(Failure::StartupTimeout);
                    self.begin_stop(now)?;
                } else if now >= self.next_health {
                    self.next_health = now + self.config.health_interval;
                    if healthy().unwrap_or(false) {
                        self.bad_health = 0;
                        self.state = State::Ready;
                    } else if self.state == State::Ready {
                        self.bad_health = self.bad_health.saturating_add(1);
                        if self.bad_health >= self.config.unhealthy_limit {
                            self.failure = Some(Failure::Unhealthy);
                            self.begin_stop(now)?;
                        }
                    }
                }
            }
            State::Stopping if now >= self.deadline && !self.killed => {
                self.signal(Signal::SIGKILL)?;
                self.killed = true;
            }
            _ => (),
        }
        Ok(self.state)
    }
}

fn gate_command() -> io::Result<Command> {
    #[cfg(target_os = "linux")]
    {
        // Resolve in the forked child while it still runs this image. Never turn
        // this kernel reference back into a replaceable workspace pathname.
        Ok(Command::new("/proc/self/exe"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(platform::refused("Linux executable-image scope gate available"))
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        if self.child.is_some() {
            let _ = self.signal(Signal::SIGKILL);
            if let Some(child) = self.child.as_mut() {
                let _ = child.wait();
            }
        }
    }
}

#[cfg(test)]
mod platform_retry_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    struct Policy(AtomicBool);
    impl platform::PlatformLauncher for Policy {
        fn requires_scope(&self) -> io::Result<bool> {
            if self.0.load(Ordering::SeqCst) {
                Err(platform::refused("child policy metadata readable"))
            } else {
                Ok(false)
            }
        }
        fn attach(&self, _: u32, _: bool, _: Instant) -> io::Result<platform::Attachment> {
            panic!("no child may be attached after launch refusal")
        }
    }

    #[test]
    fn policy_lookup_failure_exhausts_backoff_instead_of_looping_forever() {
        let env = oxa_fake::TestEnv::new();
        let now = Instant::now();
        let backoff = Duration::from_secs(1);
        let mut s = Supervisor::new(Config {
            executable: env.work().join("nonexistent-engine"),
            arguments: Vec::new(),
            environment: Default::default(),
            startup_timeout: Duration::from_secs(1),
            health_interval: Duration::from_secs(1),
            unhealthy_limit: 1,
            term_grace: Duration::ZERO,
            restart_backoff: backoff,
            max_restarts: 1,
        })
        .unwrap();
        let policy = Arc::new(Policy(AtomicBool::new(false)));
        s.platform = Some(policy.clone());
        // The ordinary spawn-error path enters bounded backoff.
        assert!(s.start(now).is_err());
        assert_eq!(s.state(), State::Backoff);
        policy.0.store(true, Ordering::SeqCst);
        assert_eq!(s.poll(now, || panic!("no live engine")).unwrap(), State::Backoff);
        let error = s.poll(now + backoff, || panic!("no live engine")).unwrap_err();
        assert!(error.to_string().contains("child policy metadata readable"));
        assert_eq!(s.state(), State::Failed);
        assert_eq!(s.last_failure(), Some(Failure::Spawn));
        assert!(!s.wanted);
        assert_eq!(s.pid(), None);
        assert_eq!(
            s.poll(now + backoff * 2, || panic!("no live engine")).unwrap(),
            State::Failed
        );
    }
}

#[cfg(all(test, target_os = "linux"))]
mod gate_image_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn gate_reexec_keeps_running_image_after_replace_or_unlink() {
        const MODE: &str = "OXA_TEST_GATE_IMAGE";
        const TEST: &str = "gate_image_tests::gate_reexec_keeps_running_image_after_replace_or_unlink";
        if let Ok(mode) = std::env::var(MODE) {
            if mode == "probe" {
                return;
            }
            // Capture the actual shared launch command before the hostile pathname change.
            let mut command = gate_command().unwrap();
            let host = std::env::current_exe().unwrap();
            std::fs::remove_file(&host).unwrap();
            if mode == "replace" {
                std::fs::write(&host, "#!/bin/sh\nexit 91\n").unwrap();
                std::fs::set_permissions(&host, std::fs::Permissions::from_mode(0o700)).unwrap();
            }
            assert!(
                command
                    .args(["--exact", TEST])
                    .env_clear()
                    .env(MODE, "probe")
                    .status()
                    .unwrap()
                    .success(),
                "must execute the running image, never the replaced path"
            );
            return;
        }
        let env = oxa_fake::TestEnv::new();
        for mode in ["replace", "unlink"] {
            let host = env.work().join(format!("gate-host-{mode}"));
            std::fs::copy(std::env::current_exe().unwrap(), &host).unwrap();
            assert!(
                Command::new(&host)
                    .args(["--exact", TEST])
                    .env_clear()
                    .env(MODE, mode)
                    .status()
                    .unwrap()
                    .success(),
                "{mode} must retain the original executable image"
            );
        }
    }
}
