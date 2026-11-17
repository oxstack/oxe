use oxc_local::lifecycle::{Lifecycle, MemorySample, Model, Policy, StopReason};
use oxc_local::{Config, State};
use std::time::{Duration, Instant};

fn model(name: &str) -> Model {
    Model {
        name: name.into(),
        required_available_bytes: 4096,
        config: Config {
            environment: Default::default(),
            executable: "/bin/sleep".into(),
            arguments: vec!["60".into()],
            startup_timeout: Duration::from_secs(10),
            health_interval: Duration::from_secs(1),
            unhealthy_limit: 2,
            term_grace: Duration::ZERO,
            restart_backoff: Duration::from_secs(1),
            max_restarts: 0,
        },
    }
}

fn memory() -> MemorySample {
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

fn poll(owner: &mut Lifecycle, now: Instant, sample: MemorySample) {
    owner.poll(now, Ok(sample), || Ok(true)).unwrap();
}

fn stopped(owner: &mut Lifecycle, now: Instant) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while owner.status().state != State::Stopped {
        poll(owner, now, memory());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}

#[test]
fn load_requires_memory_and_holds_lease_through_readiness_and_unload() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(2, Policy::default());
    let mut low = memory();
    low.available_bytes = 4095;
    assert!(owner.load(model("one"), now, Ok(low)).is_err());
    assert_eq!(owner.status().pid, None);
    owner.load(model("one"), now, Ok(memory())).unwrap();
    assert!(owner.try_request().is_err());
    poll(&mut owner, now, memory());
    let active = owner.try_request().unwrap();
    assert!(owner.unload(now).is_err());
    drop(active);
    owner.unload(now).unwrap();
    assert_eq!(owner.status().state, State::Stopping);
    assert!(owner.try_request().is_err());
    stopped(&mut owner, now);
    assert_eq!(owner.status().model, None);
}

#[test]
fn switching_waits_for_old_exit_and_uses_fresh_admission() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(1, Policy::default());
    owner.load(model("one"), now, Ok(memory())).unwrap();
    poll(&mut owner, now, memory());
    let old = owner.status().pid.unwrap();
    owner.load(model("two"), now, Ok(memory())).unwrap();
    assert_eq!(owner.status().pid, Some(old));
    assert_eq!(owner.status().model, Some("one"));
    let deadline = Instant::now() + Duration::from_secs(5);
    while owner.status().model != Some("two") {
        poll(&mut owner, now, memory());
        assert!(Instant::now() < deadline);
    }
    assert_ne!(owner.status().pid, Some(old));
    poll(&mut owner, now, memory());
    assert!(owner.try_request().is_ok());
}

#[test]
fn idle_unload_does_not_interrupt_a_request_and_starts_at_release() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(
        1,
        Policy {
            idle_timeout: Some(Duration::from_secs(10)),
        },
    );
    owner.load(model("one"), now, Ok(memory())).unwrap();
    poll(&mut owner, now, memory());
    let request = owner.try_request().unwrap();
    poll(&mut owner, now + Duration::from_secs(20), memory());
    assert_eq!(owner.status().state, State::Ready);
    drop(request);
    // Release records actual monotonic time, so an immediate poll cannot expire it.
    poll(&mut owner, now + Duration::from_secs(20), memory());
    assert_eq!(owner.status().state, State::Ready);
    poll(&mut owner, now + Duration::from_secs(31), memory());
    assert_eq!(owner.status().stop_reason, Some(StopReason::Idle));
    stopped(&mut owner, now + Duration::from_secs(40));
}

#[test]
fn memory_pressure_stops_even_active_work_and_cancels_restarts() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(1, Policy::default());
    let mut selected = model("one");
    selected.config.max_restarts = 5;
    owner.load(selected, now, Ok(memory())).unwrap();
    poll(&mut owner, now, memory());
    let active = owner.try_request().unwrap();
    let mut pressure = memory();
    pressure.psi_full_avg10 = 21.0;
    poll(&mut owner, now, pressure);
    poll(&mut owner, now + Duration::from_secs(2), pressure);
    assert_eq!(owner.status().state, State::Ready);
    poll(&mut owner, now + Duration::from_secs(3), pressure);
    assert_eq!(owner.status().stop_reason, Some(StopReason::MemoryPressure));
    assert!(owner.try_request().is_err());
    stopped(&mut owner, now + Duration::from_secs(4));
    drop(active);
    poll(&mut owner, now + Duration::from_secs(100), memory());
    assert_eq!(owner.status().pid, None);
}

#[test]
fn missing_memory_fails_closed_and_never_launches() {
    let mut owner = Lifecycle::new(0, Policy::default());
    assert!(owner
        .load(
            model("one"),
            Instant::now(),
            Err(std::io::Error::other("unavailable"))
        )
        .is_err());
    assert_eq!(owner.status().pid, None);
}

#[test]
fn loading_has_separate_psi_threshold_and_host_floor_timer() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(0, Policy::default());
    owner.load(model("one"), now, Ok(memory())).unwrap();
    let mut sample = memory();
    sample.psi_full_avg10 = 60.0;
    owner.poll(now, Ok(sample), || Ok(false)).unwrap();
    owner
        .poll(now + Duration::from_secs(4), Ok(sample), || Ok(false))
        .unwrap();
    assert_eq!(owner.status().state, State::Starting);
    sample.psi_full_avg10 = 61.0;
    owner
        .poll(now + Duration::from_secs(4), Ok(sample), || Ok(false))
        .unwrap();
    owner
        .poll(now + Duration::from_secs(8), Ok(sample), || Ok(false))
        .unwrap();
    assert_eq!(owner.status().state, State::Starting);
    owner
        .poll(now + Duration::from_secs(9), Ok(sample), || {
            panic!("stopping must not probe")
        })
        .unwrap();
    assert_eq!(owner.status().stop_reason, Some(StopReason::MemoryPressure));
}

#[test]
fn healthy_sample_resets_pressure_and_ready_uses_its_own_threshold() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(0, Policy::default());
    owner.load(model("one"), now, Ok(memory())).unwrap();
    poll(&mut owner, now, memory());
    let mut high = memory();
    high.psi_full_avg10 = 21.0;
    poll(&mut owner, now, high);
    poll(&mut owner, now + Duration::from_secs(2), memory());
    poll(&mut owner, now + Duration::from_secs(3), high);
    poll(&mut owner, now + Duration::from_secs(5), high);
    assert_eq!(owner.status().state, State::Ready);
    poll(&mut owner, now + Duration::from_secs(6), high);
    assert_eq!(owner.status().stop_reason, Some(StopReason::MemoryPressure));
}

#[test]
fn swap_oom_and_pid_limit_fail_immediately_relative_to_baseline() {
    for kind in 0..5 {
        let now = Instant::now();
        let mut owner = Lifecycle::new(0, Policy::default());
        let mut sample = memory();
        sample.oom_events = 5;
        sample.host_oom_kills = 7;
        sample.pids_max_events = 3;
        owner.load(model("one"), now, Ok(sample)).unwrap();
        poll(&mut owner, now, sample);
        assert_eq!(owner.status().state, State::Ready);
        match kind {
            0 => sample.swap_limit_zero = false,
            1 => sample.swap_current_bytes = 1,
            2 => sample.oom_events += 1,
            3 => sample.host_oom_kills += 1,
            _ => sample.pids_max_events += 1,
        }
        poll(&mut owner, now, sample);
        assert_eq!(owner.status().stop_reason, Some(StopReason::MemoryPressure));
    }
}

#[test]
fn invalid_or_missing_samples_stop_a_loaded_engine_and_forbid_restart() {
    for kind in 0..4 {
        let now = Instant::now();
        let mut owner = Lifecycle::new(0, Policy::default());
        owner.load(model("one"), now, Ok(memory())).unwrap();
        poll(&mut owner, now, memory());
        let mut sample = memory();
        let sample = match kind {
            0 => Err(std::io::Error::other("missing")),
            1 => {
                sample.psi_full_avg10 = f64::NAN;
                Ok(sample)
            }
            2 => {
                sample.psi_full_avg10 = 101.0;
                Ok(sample)
            }
            _ => {
                sample.psi_full_avg10 = -1.0;
                Ok(sample)
            }
        };
        owner
            .poll(now, sample, || panic!("no probe after failure"))
            .unwrap();
        assert_eq!(owner.status().stop_reason, Some(StopReason::MemoryUnavailable));
        stopped(&mut owner, now);
    }
}

#[test]
fn switch_rechecks_memory_after_exit_and_explicit_unload_cancels_switch() {
    for cancel in [false, true] {
        let now = Instant::now();
        let mut owner = Lifecycle::new(0, Policy::default());
        owner.load(model("one"), now, Ok(memory())).unwrap();
        poll(&mut owner, now, memory());
        owner.load(model("two"), now, Ok(memory())).unwrap();
        if cancel {
            owner.unload(now).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let mut sample = memory();
            sample.available_bytes = 1;
            let result = owner.poll(now, Ok(sample), || panic!("no probe during switch"));
            if result.is_err() || owner.status().state == State::Stopped {
                break;
            }
            assert!(Instant::now() < deadline);
        }
        assert_eq!(owner.status().pid, None);
        assert_eq!(owner.status().model, None);
        assert_eq!(
            owner.status().stop_reason,
            Some(if cancel {
                StopReason::Explicit
            } else {
                StopReason::MemoryUnavailable
            })
        );
    }
}

#[test]
fn idle_disabled_keeps_engine_but_not_under_memory_faults() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(0, Policy { idle_timeout: None });
    owner.load(model("one"), now, Ok(memory())).unwrap();
    poll(&mut owner, now, memory());
    poll(&mut owner, now + Duration::from_secs(10000), memory());
    assert_eq!(owner.status().state, State::Ready);
    let mut sample = memory();
    sample.swap_current_bytes = 1;
    poll(&mut owner, now + Duration::from_secs(10000), sample);
    assert_eq!(owner.status().stop_reason, Some(StopReason::MemoryPressure));
}

#[test]
fn admission_rejects_unconfined_swap_and_bad_launch_before_eviction() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(0, Policy::default());
    let mut sample = memory();
    sample.swap_limit_zero = false;
    assert!(owner.load(model("one"), now, Ok(sample)).is_err());
    owner.load(model("one"), now, Ok(memory())).unwrap();
    poll(&mut owner, now, memory());
    let pid = owner.status().pid;
    let mut invalid = model("two");
    invalid.config.executable = "relative-engine".into();
    assert!(owner.load(invalid, now, Ok(memory())).is_err());
    assert_eq!(owner.status().pid, pid);
    assert_eq!(owner.status().state, State::Ready);
}

#[test]
fn queued_requests_block_idle_unload_and_owner_drop_wakes_them_closed() {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};
    let mut cx = Context::from_waker(Waker::noop());
    let now = Instant::now();
    let mut owner = Lifecycle::new(
        2,
        Policy {
            idle_timeout: Some(Duration::from_secs(1)),
        },
    );
    owner.load(model("one"), now, Ok(memory())).unwrap();
    poll(&mut owner, now, memory());
    let active = owner.try_request().unwrap();
    let gate = owner.requests();
    let mut queued = Box::pin(gate.acquire());
    assert!(queued.as_mut().poll(&mut cx).is_pending());
    assert_eq!(owner.status().waiting, 1);
    drop(active);
    poll(&mut owner, now + Duration::from_secs(20), memory());
    assert_eq!(owner.status().state, State::Ready);
    let Poll::Ready(Ok(active)) = queued.as_mut().poll(&mut cx) else {
        panic!("grant lost")
    };
    let mut waiting = Box::pin(gate.acquire());
    assert!(waiting.as_mut().poll(&mut cx).is_pending());
    drop(owner);
    assert!(matches!(
        waiting.as_mut().poll(&mut cx),
        Poll::Ready(Err(oxc_local::queue::AdmissionError::Closed))
    ));
    drop(active);
    assert!(gate.try_acquire().is_err());
}

#[test]
fn memory_fault_during_switch_cancels_launch_even_if_sampling_recovers() {
    for fault_kind in 0..4 {
        let now = Instant::now();
        let ready = std::env::temp_dir().join(format!("local-switch-{}-{fault_kind}", std::process::id()));
        let mut initial = model("one");
        initial.config.executable = "/bin/sh".into();
        initial.config.arguments = vec![
            "-c".into(),
            "trap '' TERM; : > \"$1\"; exec /bin/sleep 60".into(),
            "fake-engine".into(),
            ready.clone().into_os_string(),
        ];
        initial.config.term_grace = Duration::from_secs(10);
        let mut owner = Lifecycle::new(0, Policy::default());
        owner.load(initial, now, Ok(memory())).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ready.exists() {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        poll(&mut owner, now, memory());
        owner.load(model("two"), now, Ok(memory())).unwrap();
        let mut fault = memory();
        match fault_kind {
            0 => fault.oom_events += 1,
            1 => fault.host_oom_kills += 1,
            2 => fault.pids_max_events += 1,
            _ => (),
        }
        owner
            .poll(
                now,
                if fault_kind == 3 {
                    Err(std::io::Error::other("unavailable"))
                } else {
                    Ok(fault)
                },
                || panic!("stopping probe"),
            )
            .unwrap();
        assert_eq!(
            owner.status().stop_reason,
            Some(if fault_kind == 3 {
                StopReason::MemoryUnavailable
            } else {
                StopReason::MemoryPressure
            })
        );
        stopped(&mut owner, now + Duration::from_secs(11));
        poll(&mut owner, now + Duration::from_secs(20), memory());
        assert_eq!(owner.status().pid, None);
        assert_eq!(owner.status().model, None);
        std::fs::remove_file(ready).unwrap();
    }
}

#[test]
fn repeated_unload_of_stopped_owner_allows_load_without_an_extra_poll() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(0, Policy::default());
    for _ in 0..2 {
        owner.unload(now).unwrap();
        owner.unload(now).unwrap();
        assert_eq!(owner.status().state, State::Stopped);
        owner.load(model("one"), now, Ok(memory())).unwrap();
        poll(&mut owner, now, memory());
        owner.unload(now).unwrap();
        stopped(&mut owner, now);
    }
}

#[test]
fn exhausted_spawn_failure_releases_control_before_returning_error() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(0, Policy::default());
    let mut invalid = model("missing");
    invalid.config.executable = "/nonexistent-local-engine-test".into();
    assert!(owner.load(invalid.clone(), now, Ok(memory())).is_err());
    assert_eq!(owner.status().state, State::Failed);
    owner.load(model("one"), now, Ok(memory())).unwrap();
    poll(&mut owner, now, memory());
    owner.unload(now).unwrap();
    stopped(&mut owner, now);
    invalid.config.max_restarts = 1;
    assert!(owner.load(invalid, now, Ok(memory())).is_err());
    assert_eq!(owner.status().state, State::Backoff);
    assert!(owner
        .poll(now + Duration::from_secs(2), Ok(memory()), || panic!("no child"))
        .is_err());
    assert_eq!(owner.status().state, State::Failed);
    owner
        .load(model("two"), now + Duration::from_secs(2), Ok(memory()))
        .unwrap();
}

#[test]
fn admitted_ticket_covers_cold_load_and_generation_without_a_fifo_gap() {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};
    let now = Instant::now();
    let mut owner = Lifecycle::new(1, Policy::default());
    let gate = owner.requests();
    let mut cx = Context::from_waker(Waker::noop());
    let mut admission = Box::pin(gate.acquire_operation());
    let Poll::Ready(Ok(lease)) = admission.as_mut().poll(&mut cx) else {
        panic!("cold admission")
    };
    owner
        .load_reserved(lease, model("one"), now, Ok(memory()))
        .unwrap();
    let mut waiting = Box::pin(gate.acquire_operation());
    assert!(waiting.as_mut().poll(&mut cx).is_pending());
    poll(&mut owner, now, memory());
    assert_eq!(owner.status().state, State::Ready);
    assert!(waiting.as_mut().poll(&mut cx).is_pending());
    let request = owner.take_ready_request().unwrap();
    assert!(waiting.as_mut().poll(&mut cx).is_pending());
    drop(request);
    assert!(matches!(waiting.as_mut().poll(&mut cx), Poll::Ready(Ok(_))));
}

#[test]
fn insufficient_allocation_budget_reports_available_and_required_before_spawn() {
    let now = Instant::now();
    let mut owner = Lifecycle::new(1, Policy::default());
    let mut low = memory();
    low.available_bytes = 4095;
    let error = owner.load(model("one"), now, Ok(low)).unwrap_err();
    assert!(error.to_string().contains("available=4095"));
    assert!(error.to_string().contains("required=4096"));
    assert_eq!(owner.status().pid, None);
}
