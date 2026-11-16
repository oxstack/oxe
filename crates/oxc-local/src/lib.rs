//! Caller-driven ownership of one local engine. No daemon or background thread.
#![cfg(unix)]

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
#[derive(Clone, Debug)]
pub struct Config {
    pub executable: PathBuf,
    pub arguments: Vec<OsString>,
    pub startup_timeout: Duration,
    pub health_interval: Duration,
    pub unhealthy_limit: u32,
    pub term_grace: Duration,
    pub restart_backoff: Duration,
    /// Restarts after the initial attempt; never reset by successful health.
    pub max_restarts: u32,
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
    state: State,
    deadline: Instant,
    next_health: Instant,
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
            state: State::Stopped,
            deadline: now,
            next_health: now,
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
        let result = Command::new(&self.config.executable)
            .args(&self.config.arguments)
            // Ambient model/provider settings and credentials must not cross this boundary.
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn();
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

    /// Advances lifecycle, checking exit before health. Probe errors count as unhealthy.
    /// Never retries a generation request. Caller must bound probe I/O time.
    pub fn poll(&mut self, now: Instant, mut healthy: impl FnMut() -> io::Result<bool>) -> io::Result<State> {
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
