//! Model-free lifecycle smoke. Only use a stub engine and synthetic GGUF.
//! Memory is explicitly faked here; this is not hardware qualification.
use oxa_engine::StreamOpts;
use oxa_proto::{CancelToken, ChatRequest, FrozenPrefix};
use oxc_local::lifecycle::{Lifecycle, MemorySample, Model, Policy, StopReason};
use oxc_local::transport::{Monitor, Transport};
use oxc_local::{Config, State};
use std::io;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn sample() -> MemorySample {
    MemorySample {
        available_bytes: 8 * 1024 * 1024 * 1024,
        swap_free_bytes: 0,
        psi_full_avg10: 0.0,
        swap_current_bytes: 0,
        swap_limit_zero: true,
        oom_events: 0,
        host_oom_kills: 0,
        pids_max_events: 0,
    }
}

fn main() -> io::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let executable = PathBuf::from(args.next().expect("absolute stub engine path"));
    let synthetic = args.next().expect("synthetic GGUF path");
    assert!(args.next().is_none());
    let address = TcpListener::bind("127.0.0.1:0")?.local_addr()?;
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(200)))
        .build()
        .new_agent();
    let health = || {
        let mut response = agent
            .get(format!("http://{address}/health"))
            .call()
            .map_err(io::Error::other)?;
        let text = response
            .body_mut()
            .with_config()
            .limit(4096)
            .read_to_string()
            .map_err(io::Error::other)?;
        Ok(serde_json::from_str::<serde_json::Value>(&text).is_ok_and(|v| v["status"] == "ok"))
    };
    let model = Model {
        name: "synthetic-stub".into(),
        required_available_bytes: 4096,
        config: Config {
            executable,
            arguments: vec![
                "--engine".into(),
                "stub".into(),
                "--model".into(),
                synthetic,
                "--host".into(),
                "127.0.0.1".into(),
                "--port".into(),
                address.port().to_string().into(),
                "--alias".into(),
                "synthetic-stub".into(),
            ],
            startup_timeout: Duration::from_secs(30),
            health_interval: Duration::from_millis(50),
            unhealthy_limit: 2,
            term_grace: Duration::from_millis(200),
            restart_backoff: Duration::from_millis(100),
            max_restarts: 1,
        },
    };
    let mut owner = Lifecycle::new(
        2,
        Policy {
            idle_timeout: Some(Duration::from_millis(100)),
        },
    );
    let wait = |owner: &mut Lifecycle, state: State, memory: MemorySample| -> io::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(40);
        loop {
            if owner.poll(Instant::now(), Ok(memory), health)?.state == state {
                return Ok(());
            }
            if Instant::now() > deadline {
                return Err(io::Error::other("stub lifecycle timed out"));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    };
    owner.load(model.clone(), Instant::now(), Ok(sample()))?;
    wait(&mut owner, State::Ready, sample())?;
    let prefix = FrozenPrefix {
        sections: vec![("system".into(), "Synthetic smoke".into())],
        tools_json: "[]".into(),
        sha256: [0; 32],
    };
    let request = ChatRequest {
        max_tokens: 4,
        ..ChatRequest::new(&prefix, &[], "synthetic-stub")
    };
    let lease = owner.try_request().map_err(io::Error::other)?;
    let transport = Transport::new(address, StreamOpts::default()).map_err(io::Error::other)?;
    let mut increments = 0;
    let reply = transport
        .chat(
            &mut owner,
            lease,
            &request,
            &mut Monitor {
                cancel: &CancelToken::new(),
                memory: &mut || Ok(sample()),
                healthy: &mut || health(),
            },
            &mut |_| {
                increments += 1;
            },
        )
        .map_err(io::Error::other)?;
    assert!(reply.usage.completion > 0);
    assert!(increments > 0);
    let active = owner.try_request().map_err(io::Error::other)?;
    std::thread::sleep(Duration::from_millis(150));
    owner.poll(Instant::now(), Ok(sample()), health)?;
    assert_eq!(owner.status().state, State::Ready);
    drop(active);
    wait(&mut owner, State::Stopped, sample())?;
    assert_eq!(owner.status().stop_reason, Some(StopReason::Idle));
    owner.load(model, Instant::now(), Ok(sample()))?;
    wait(&mut owner, State::Ready, sample())?;
    let mut pressure = sample();
    pressure.swap_current_bytes = 1;
    wait(&mut owner, State::Stopped, pressure)?;
    assert_eq!(owner.status().stop_reason, Some(StopReason::MemoryPressure));
    println!(
        "PASS: stub typed chat/stream, load, active lease, idle unload, reload, memory stop; owner inline bytes {}",
        std::mem::size_of::<Lifecycle>()
    );
    Ok(())
}
