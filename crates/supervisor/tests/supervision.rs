use oxe_supervisor::{Config, Failure, State, Supervisor};
use std::io;
use std::time::{Duration, Instant};

fn config(script: &str) -> Config {
    Config {
        environment: Default::default(),
        executable: "/bin/sh".into(),
        arguments: vec!["-c".into(), script.into()],
        startup_timeout: Duration::from_secs(10),
        health_interval: Duration::from_secs(1),
        unhealthy_limit: 2,
        term_grace: Duration::from_secs(2),
        restart_backoff: Duration::from_secs(3),
        max_restarts: 1,
    }
}

fn reap(supervisor: &mut Supervisor, now: Instant, target: State) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while supervisor.state() != target {
        supervisor.poll(now, || Ok(false)).unwrap();
        assert!(Instant::now() < deadline, "child never reached {target:?}");
        std::thread::yield_now();
    }
}

#[test]
fn no_process_or_probe_before_start() {
    let mut s = Supervisor::new(config("exec sleep 60")).unwrap();
    assert_eq!(s.pid(), None);
    assert_eq!(
        s.poll(Instant::now(), || panic!("idle probe")).unwrap(),
        State::Stopped
    );
}

#[test]
fn health_is_required_and_rate_limited_then_stop_cancels_restart() {
    let now = Instant::now();
    let mut s = Supervisor::new(config("exec /bin/sleep 60")).unwrap();
    s.start(now).unwrap();
    let pid = s.pid();
    s.start(now).unwrap();
    assert_eq!(s.pid(), pid);
    assert_eq!(s.poll(now, || Ok(false)).unwrap(), State::Starting);
    assert_eq!(
        s.poll(now, || panic!("duplicate probe")).unwrap(),
        State::Starting
    );
    assert_eq!(
        s.poll(now + Duration::from_secs(1), || Ok(true)).unwrap(),
        State::Ready
    );
    s.stop(now).unwrap();
    s.stop(now).unwrap();
    reap(&mut s, now + Duration::from_secs(3), State::Stopped);
    assert_eq!(s.pid(), None);
    assert_eq!(
        s.poll(now + Duration::from_secs(100), || panic!("stopped probe"))
            .unwrap(),
        State::Stopped
    );
}

#[test]
fn startup_timeout_retains_child_until_exit_and_respects_backoff() {
    let now = Instant::now();
    let mut s = Supervisor::new(config("exec /bin/sleep 60")).unwrap();
    s.start(now).unwrap();
    s.poll(now + Duration::from_secs(10), || panic!("expired probe"))
        .unwrap();
    assert_eq!(s.state(), State::Stopping);
    assert!(s.pid().is_some());
    assert_eq!(s.last_failure(), Some(Failure::StartupTimeout));
    reap(&mut s, now + Duration::from_secs(13), State::Backoff);
    assert_eq!(s.pid(), None);
    s.poll(now + Duration::from_secs(15), || panic!("backoff probe"))
        .unwrap();
    assert_eq!(s.state(), State::Backoff);
    s.stop(now).unwrap();
    s.poll(now + Duration::from_secs(100), || {
        panic!("cancelled restart")
    })
    .unwrap();
    assert_eq!(s.state(), State::Stopped);
}

#[test]
fn consecutive_unhealthy_limit_and_recovery() {
    let now = Instant::now();
    let mut s = Supervisor::new(config("exec /bin/sleep 60")).unwrap();
    s.start(now).unwrap();
    s.poll(now, || Ok(true)).unwrap();
    s.poll(now + Duration::from_secs(1), || {
        Err(io::Error::other("probe failure"))
    })
    .unwrap();
    assert_eq!(s.state(), State::Ready);
    s.poll(now + Duration::from_secs(2), || Ok(true)).unwrap();
    s.poll(now + Duration::from_secs(3), || Ok(false)).unwrap();
    assert_eq!(s.state(), State::Ready);
    s.poll(now + Duration::from_secs(4), || Ok(false)).unwrap();
    assert_eq!(s.state(), State::Stopping);
    assert_eq!(s.last_failure(), Some(Failure::Unhealthy));
    s.stop(now).unwrap();
    reap(&mut s, now + Duration::from_secs(7), State::Stopped);
}

#[test]
fn spawn_failures_have_a_finite_budget_and_explicit_start_resets_it() {
    let now = Instant::now();
    let mut cfg = config("");
    cfg.executable = "/nonexistent-oxc-local-test-engine".into();
    let mut s = Supervisor::new(cfg).unwrap();
    assert!(s.start(now).is_err());
    assert_eq!(s.state(), State::Backoff);
    assert_eq!(s.last_failure(), Some(Failure::Spawn));
    assert!(s
        .poll(now + Duration::from_secs(3), || panic!("no child"))
        .is_err());
    assert_eq!(s.state(), State::Failed);
    assert!(s.start(now).is_err());
    assert_eq!(s.state(), State::Backoff);
    s.stop(now).unwrap();
    assert_eq!(s.state(), State::Stopped);
}

#[test]
fn natural_exit_is_reaped_before_replacement_and_budget_is_bounded() {
    let now = Instant::now();
    let mut s = Supervisor::new(config("exit 7")).unwrap();
    s.start(now).unwrap();
    reap(&mut s, now, State::Backoff);
    assert_eq!(s.last_failure(), Some(Failure::Exited));
    s.poll(now + Duration::from_secs(3), || {
        panic!("spawn does not probe")
    })
    .unwrap();
    assert_eq!(s.state(), State::Starting);
    reap(&mut s, now + Duration::from_secs(3), State::Failed);
    assert_eq!(s.pid(), None);
}

/// PR_SET_PDEATHSIG is tied to the *spawning thread*, so the child must be created by a thread
/// that lives as long as the process. This test forks the engine from a worker that exits at
/// once: the child must survive that worker (a per-thread tie would kill it immediately) and
/// must die when the owning process is SIGKILLed.
#[cfg(target_os = "linux")]
#[test]
fn engine_child_outlives_its_spawning_thread_but_not_its_owner() {
    use std::io::{BufRead, BufReader, Read, Write};
    const MODE: &str = "OXA_TEST_OWNER_DEATH";
    const TEST: &str = "engine_child_outlives_its_spawning_thread_but_not_its_owner";
    if let Ok(mode) = std::env::var(MODE) {
        if mode == "probe" {
            return;
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut supervisor = Supervisor::new(config("exec /bin/sleep 300")).unwrap();
            supervisor.start(Instant::now()).unwrap();
            let pid = supervisor.pid().unwrap();
            sender.send((supervisor, pid)).unwrap();
        });
        // Ownership returns to this thread, which outlives the spawner worker that has already
        // returned. The child itself must not care which of them asked for it.
        let (_supervisor, pid) = receiver.recv().unwrap();
        // Write the raw handle: libtest captures `println!`, and the parent needs this line. The
        // leading newline ends libtest's unterminated `test <name> ... ` progress line.
        let mut stdout = std::io::stdout();
        writeln!(stdout, "\n{pid}").unwrap();
        stdout.flush().unwrap();
        std::thread::sleep(Duration::from_secs(60));
        return;
    }
    let mut owner = std::process::Command::new(std::env::current_exe().unwrap())
        // `--nocapture`: libtest captures both stdout and stderr, and this mode must stream the
        // pid line to the parent before it ever finishes.
        .args(["--exact", TEST, "--nocapture"])
        .env(MODE, "owner")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    // Skip libtest's "running 1 test" banner and read the pid line the owner prints.
    let mut reader = BufReader::new(owner.stdout.take().unwrap());
    let mut line = String::new();
    let pid: u32 = loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            let mut stderr = String::new();
            owner
                .stderr
                .take()
                .unwrap()
                .read_to_string(&mut stderr)
                .unwrap();
            panic!("the owner exited without a child pid: {stderr}");
        }
        if let Ok(pid) = line.trim().parse() {
            break pid;
        }
    };
    // A killed orphan stays a zombie until something reaps it; in a CI container PID 1 often does
    // not reap, and `kill(pid, 0)` still succeeds on a zombie. Count a zombie as dead.
    let alive = |pid: u32| {
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| {
                let state = stat.rsplit_once(')')?.1.trim_start().chars().next()?;
                Some(state != 'Z' && state != 'X')
            })
            .unwrap_or(false)
    };
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        alive(pid),
        "the child died with its short-lived spawning thread, not with the process"
    );
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(owner.id() as i32),
        nix::sys::signal::Signal::SIGKILL,
    )
    .unwrap();
    owner.wait().unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        !alive(pid),
        "the engine child outlived the process that owned it"
    );
}

#[test]
fn drop_kills_and_reaps_owned_process() {
    let now = Instant::now();
    let mut s = Supervisor::new(config("exec /bin/sleep 60")).unwrap();
    s.start(now).unwrap();
    let pid = s.pid().unwrap();
    drop(s);
    assert_eq!(
        nix::sys::wait::waitpid(nix::unistd::Pid::from_raw(pid as i32), None),
        Err(nix::errno::Errno::ECHILD)
    );
}

#[test]
fn term_grace_then_kill_for_an_uncooperative_child() {
    let now = Instant::now();
    let ready = std::env::temp_dir().join(format!("oxc-local-ignore-term-{}", std::process::id()));
    let mut cfg = config("");
    cfg.executable = "/usr/bin/python3".into();
    cfg.arguments = vec!["-I".into(), "-c".into(),
        "import signal,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); open(sys.argv[1], 'x').close(); signal.pause()".into(),
        ready.clone().into_os_string()];
    let mut s = Supervisor::new(cfg).unwrap();
    s.start(now).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !ready.exists() {
        assert!(
            Instant::now() < deadline,
            "child did not install TERM handler"
        );
        std::thread::yield_now();
    }
    s.stop(now).unwrap();
    assert_eq!(
        s.poll(now + Duration::from_secs(1), || panic!("stopping probe"))
            .unwrap(),
        State::Stopping
    );
    assert!(s.pid().is_some());
    reap(&mut s, now + Duration::from_secs(3), State::Stopped);
    std::fs::remove_file(ready).unwrap();
}

#[test]
fn term_allows_clean_shutdown_before_kill_deadline() {
    let now = Instant::now();
    let base = std::env::temp_dir().join(format!("oxc-local-graceful-{}", std::process::id()));
    let ready = base.with_extension("ready");
    let done = base.with_extension("done");
    let mut cfg = config("");
    cfg.executable = "/usr/bin/python3".into();
    cfg.arguments = vec!["-I".into(), "-c".into(),
        "import signal,sys; signal.signal(signal.SIGTERM, lambda *_: (open(sys.argv[2], 'x').close(), sys.exit(0))); open(sys.argv[1], 'x').close(); signal.pause()".into(),
        ready.clone().into_os_string(), done.clone().into_os_string()];
    let mut s = Supervisor::new(cfg).unwrap();
    s.start(now).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !ready.exists() {
        assert!(
            Instant::now() < deadline,
            "child did not install TERM handler"
        );
        std::thread::yield_now();
    }
    s.stop(now).unwrap();
    reap(&mut s, now, State::Stopped);
    assert!(done.exists(), "TERM handler was not allowed to finish");
    std::fs::remove_file(ready).unwrap();
    std::fs::remove_file(done).unwrap();
}

#[test]
fn healthy_restarts_do_not_reset_the_crash_budget() {
    let now = Instant::now();
    let mut s = Supervisor::new(config("exec /bin/sleep 60")).unwrap();
    s.start(now).unwrap();
    s.poll(now, || Ok(true)).unwrap();
    assert_eq!(s.state(), State::Ready);
    let kill = |pid: u32| {
        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        )
        .unwrap()
    };
    kill(s.pid().unwrap());
    reap(&mut s, now, State::Backoff);
    s.poll(now + Duration::from_secs(3), || Ok(true)).unwrap();
    s.poll(now + Duration::from_secs(3), || Ok(true)).unwrap();
    assert_eq!(s.state(), State::Ready);
    kill(s.pid().unwrap());
    reap(&mut s, now + Duration::from_secs(3), State::Failed);
    assert_eq!(s.pid(), None);
}
