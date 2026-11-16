//! Model-free lifecycle proof. Arguments: absolute oxe-server, synthetic GGUF.
//! Use engine/tools/ci/synth_qwen36_model.py, never an actual model.
use oxc_local::{Config, State, Supervisor};
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn health(address: SocketAddr) -> io::Result<bool> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(200)))
        .build()
        .new_agent();
    let mut response = agent
        .get(format!("http://{address}/health"))
        .call()
        .map_err(io::Error::other)?;
    let body = response
        .body_mut()
        .with_config()
        .limit(4096)
        .read_to_string()
        .map_err(io::Error::other)?;
    Ok(serde_json::from_str::<serde_json::Value>(&body).is_ok_and(|value| value["status"] == "ok"))
}

fn until(s: &mut Supervisor, address: SocketAddr, target: State) -> io::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if s.poll(Instant::now(), || health(address))? == target {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "lifecycle deadline"));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn main() -> io::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let executable = PathBuf::from(args.next().expect("absolute oxe-server path"));
    let model = args.next().expect("synthetic GGUF path");
    assert!(
        args.next().is_none(),
        "only executable and synthetic model are accepted"
    );
    let address = TcpListener::bind("127.0.0.1:0")?.local_addr()?;
    let mut s = Supervisor::new(Config {
        executable,
        arguments: vec![
            "--engine".into(),
            "stub".into(),
            "--model".into(),
            model,
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            address.port().to_string().into(),
        ],
        startup_timeout: Duration::from_secs(30),
        health_interval: Duration::from_millis(100),
        unhealthy_limit: 2,
        term_grace: Duration::from_secs(2),
        restart_backoff: Duration::from_millis(100),
        max_restarts: 1,
    })?;
    let started = Instant::now();
    s.start(started)?;
    until(&mut s, address, State::Ready)?;
    println!(
        "stub ready after {:?}; supervisor inline bytes {}",
        started.elapsed(),
        std::mem::size_of::<Supervisor>()
    );
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(s.pid().unwrap() as i32),
        nix::sys::signal::Signal::SIGKILL,
    )?;
    until(&mut s, address, State::Backoff)?;
    until(&mut s, address, State::Ready)?;
    s.stop(Instant::now())?;
    until(&mut s, address, State::Stopped)?;
    println!("PASS: readiness, crash/reap/restart, graceful stop; no real model or upstream");
    Ok(())
}
