use oxa_engine::StreamOpts;
use oxa_fake::{FakeOpts, FakeServer, Fault, ScriptTurn};
use oxa_proto::{codes, CancelToken, ChatRequest, Finish, FrozenPrefix, StreamEvent, ToolCall};
use oxc_local::lifecycle::{Lifecycle, MemorySample, Model, Policy};
use oxc_local::transport::{Monitor, Transport};
use oxc_local::{Config, State};
use std::time::{Duration, Instant};

fn memory() -> std::io::Result<MemorySample> {
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
fn owner() -> Lifecycle {
    let now = Instant::now();
    let mut owner = Lifecycle::new(2, Policy::default());
    owner
        .load(
            Model {
                name: "fake-model".into(),
                required_available_bytes: 4096,
                config: Config {
                    executable: "/bin/sleep".into(),
                    arguments: vec!["60".into()],
                    startup_timeout: Duration::from_secs(5),
                    health_interval: Duration::from_millis(100),
                    unhealthy_limit: 2,
                    term_grace: Duration::ZERO,
                    restart_backoff: Duration::from_millis(10),
                    max_restarts: 1,
                },
            },
            now,
            memory(),
        )
        .unwrap();
    owner.poll(now, memory(), || Ok(true)).unwrap();
    owner
}
fn prefix() -> FrozenPrefix {
    FrozenPrefix {
        sections: vec![("system".into(), "Synthetic system".into())],
        tools_json: "[]".into(),
        sha256: [0; 32],
    }
}
fn transport(server: &FakeServer) -> Transport {
    Transport::new(
        server.url().trim_start_matches("http://").parse().unwrap(),
        StreamOpts {
            connect_timeout_ms: 1000,
            silence_limit_ms: 1000,
        },
    )
    .unwrap()
}
fn stopped(owner: &mut Lifecycle) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while owner.status().state != State::Stopped {
        owner.poll(Instant::now(), memory(), || Ok(false)).unwrap();
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}

#[test]
fn typed_request_streams_tools_usage_and_text_without_controller_routes() {
    let tool = ToolCall {
        id: "call-one".into(),
        name: "read".into(),
        args_raw: "{\"path\":\"note.txt\"}".into(),
    };
    let server = FakeServer::start(
        vec![ScriptTurn {
            text: Some("Local λ answer".into()),
            tool_calls: vec![tool.clone()],
            finish: Finish::ToolCalls,
            ..ScriptTurn::text("")
        }],
        FakeOpts {
            controller: false,
            ..FakeOpts::default()
        },
    )
    .unwrap();
    let mut owner = owner();
    let active = owner.try_request().unwrap();
    let prefix = prefix();
    let request = ChatRequest {
        max_tokens: 512,
        ..ChatRequest::new(&prefix, &[], "fake-model")
    };
    let caller = std::thread::current().id();
    let mut events = Vec::new();
    let reply = transport(&server)
        .chat(
            &mut owner,
            active,
            &request,
            &mut Monitor {
                cancel: &CancelToken::new(),
                memory: &mut memory,
                healthy: &mut || Ok(true),
            },
            &mut |event| {
                assert_eq!(std::thread::current().id(), caller);
                events.push(event);
            },
        )
        .unwrap();
    assert_eq!(reply.finish, Finish::ToolCalls);
    assert_eq!(reply.msg.tool_calls, vec![tool]);
    assert!(events.iter().any(|e| matches!(e, StreamEvent::Text(_))));
    assert!(reply.usage.prompt > 0);
    assert_eq!(server.requests().len(), 1);
    for (method, path) in [
        ("GET", "/_local/status"),
        ("POST", "/_local/cache/clear"),
        ("POST", "/_local/cancel"),
    ] {
        assert_eq!(server.hits(method, path), 0);
    }
    assert!(owner.try_request().is_ok());
}

#[test]
fn cancellation_holds_engine_exclusion_until_process_exit() {
    let server = FakeServer::start(
        vec![ScriptTurn {
            fault: Some(Fault::CommentsOnly { ms: 2000 }),
            ..ScriptTurn::text("unreached")
        }],
        FakeOpts::default(),
    )
    .unwrap();
    let mut owner = owner();
    let active = owner.try_request().unwrap();
    let cancel = CancelToken::new();
    let timer = cancel.clone();
    let cancel_thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        timer.cancel();
    });
    let prefix = prefix();
    let started = Instant::now();
    let error = transport(&server)
        .chat(
            &mut owner,
            active,
            &ChatRequest::new(&prefix, &[], "fake-model"),
            &mut Monitor {
                cancel: &cancel,
                memory: &mut memory,
                healthy: &mut || Ok(true),
            },
            &mut |_| {},
        )
        .unwrap_err();
    cancel_thread.join().unwrap();
    assert_eq!(error.code, codes::ABORTED);
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(owner.try_request().is_err());
    assert_eq!(owner.status().state, State::Stopping);
    stopped(&mut owner);
    assert_eq!(server.requests().len(), 1);
    assert_eq!(server.hits("POST", "/_local/cancel"), 0);
}

#[test]
fn refusal_and_cut_stream_are_not_replayed_and_keep_cleanup_owned() {
    for fault in [Fault::Busy { retry_after: Some(1) }, Fault::DropAt { bytes: 32 }] {
        let server = FakeServer::start(
            vec![ScriptTurn {
                fault: Some(fault),
                ..ScriptTurn::text("synthetic")
            }],
            FakeOpts::default(),
        )
        .unwrap();
        let mut owner = owner();
        let active = owner.try_request().unwrap();
        let prefix = prefix();
        assert!(transport(&server)
            .chat(
                &mut owner,
                active,
                &ChatRequest::new(&prefix, &[], "fake-model"),
                &mut Monitor {
                    cancel: &CancelToken::new(),
                    memory: &mut memory,
                    healthy: &mut || Ok(true)
                },
                &mut |_| {}
            )
            .is_err());
        assert_eq!(server.hits("POST", "/v1/chat/completions"), 1);
        assert!(owner.try_request().is_err());
        stopped(&mut owner);
    }
}

#[test]
fn memory_fault_stops_a_silent_stream_and_callback_panic_keeps_cleanup_owned() {
    for panic_callback in [false, true] {
        let fault = if panic_callback {
            None
        } else {
            Some(Fault::CommentsOnly { ms: 2000 })
        };
        let server = FakeServer::start(
            vec![ScriptTurn {
                fault,
                ..ScriptTurn::text("synthetic")
            }],
            FakeOpts::default(),
        )
        .unwrap();
        let mut owner = owner();
        let active = owner.try_request().unwrap();
        let prefix = prefix();
        let mut samples = 0;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            transport(&server).chat(
                &mut owner,
                active,
                &ChatRequest::new(&prefix, &[], "fake-model"),
                &mut Monitor {
                    cancel: &CancelToken::new(),
                    memory: &mut || {
                        samples += 1;
                        if samples > 1 {
                            Err(std::io::Error::other("missing"))
                        } else {
                            memory()
                        }
                    },
                    healthy: &mut || Ok(true),
                },
                &mut |_| {
                    if panic_callback {
                        panic!("synthetic callback panic");
                    }
                },
            )
        }));
        if panic_callback {
            assert!(outcome.is_err());
        } else {
            assert!(outcome.unwrap().is_err());
        }
        assert!(owner.try_request().is_err());
        stopped(&mut owner);
    }
}

#[test]
fn rejects_external_endpoint_wrong_model_and_foreign_lease_without_network() {
    assert!(Transport::new("192.0.2.1:8000".parse().unwrap(), StreamOpts::default()).is_err());
    let server = FakeServer::start(vec![], FakeOpts::default()).unwrap();
    let mut owner = owner();
    let foreign = self::owner();
    let active = foreign.try_request().unwrap();
    let prefix = prefix();
    assert!(transport(&server)
        .chat(
            &mut owner,
            active,
            &ChatRequest::new(&prefix, &[], "fake-model"),
            &mut Monitor {
                cancel: &CancelToken::new(),
                memory: &mut memory,
                healthy: &mut || Ok(true)
            },
            &mut |_| {}
        )
        .is_err());
    let active = owner.try_request().unwrap();
    assert!(transport(&server)
        .chat(
            &mut owner,
            active,
            &ChatRequest::new(&prefix, &[], "another-model"),
            &mut Monitor {
                cancel: &CancelToken::new(),
                memory: &mut memory,
                healthy: &mut || Ok(true)
            },
            &mut |_| {}
        )
        .is_err());
    assert_eq!(server.hits("POST", "/v1/chat/completions"), 0);
    assert_eq!(owner.status().state, State::Ready);
}

#[test]
fn oversized_events_and_error_bodies_fail_without_replaying() {
    for error_body in [false, true] {
        let text = "x".repeat(1024 * 1024 + 100);
        let turn = if error_body {
            ScriptTurn::fault(Fault::Http {
                status: 400,
                body: text,
            })
        } else {
            ScriptTurn::text(&text)
        };
        let server = FakeServer::start(
            vec![turn],
            FakeOpts {
                n_ctx: 4 * 1024 * 1024,
                ..FakeOpts::default()
            },
        )
        .unwrap();
        let mut owner = owner();
        let active = owner.try_request().unwrap();
        let prefix = prefix();
        let request = ChatRequest {
            max_tokens: 2 * 1024 * 1024,
            ..ChatRequest::new(&prefix, &[], "fake-model")
        };
        let error = transport(&server)
            .chat(
                &mut owner,
                active,
                &request,
                &mut Monitor {
                    cancel: &CancelToken::new(),
                    memory: &mut memory,
                    healthy: &mut || Ok(true),
                },
                &mut |_| {},
            )
            .unwrap_err();
        assert_eq!(error.code, codes::STREAM_BROKEN);
        assert!(error.message.len() < 256);
        assert_eq!(server.hits("POST", "/v1/chat/completions"), 1);
        assert!(owner.try_request().is_err());
        stopped(&mut owner);
    }
}

#[test]
fn pre_cancelled_request_has_no_network_or_process_teardown() {
    let server = FakeServer::start(vec![], FakeOpts::default()).unwrap();
    let mut owner = owner();
    let active = owner.try_request().unwrap();
    let prefix = prefix();
    let cancel = CancelToken::new();
    cancel.cancel();
    let error = transport(&server)
        .chat(
            &mut owner,
            active,
            &ChatRequest::new(&prefix, &[], "fake-model"),
            &mut Monitor {
                cancel: &cancel,
                memory: &mut memory,
                healthy: &mut || Ok(true),
            },
            &mut |_| {},
        )
        .unwrap_err();
    assert_eq!(error.code, codes::ABORTED);
    assert_eq!(server.hits("POST", "/v1/chat/completions"), 0);
    assert_eq!(owner.status().state, State::Ready);
    assert!(owner.try_request().is_ok());
}

#[test]
fn typed_tool_result_continues_the_same_owned_engine_without_a_controller() {
    let server = FakeServer::start(
        vec![
            ScriptTurn {
                chunk: 2,
                ..ScriptTurn::tool("read-one", "read", "{\"path\":\"note.txt\"}")
            },
            ScriptTurn::text("Synthetic completed task"),
        ],
        FakeOpts {
            controller: false,
            ..FakeOpts::default()
        },
    )
    .unwrap();
    let transport = transport(&server);
    let mut owner = owner();
    let pid = owner.status().pid;
    let prefix = prefix();
    let user = oxa_proto::WireMsg::user("Read the synthetic note");
    let first_request = [user.clone()];
    let lease = owner.try_request().unwrap();
    let first = transport
        .chat(
            &mut owner,
            lease,
            &ChatRequest::new(&prefix, &first_request, "fake-model"),
            &mut Monitor {
                cancel: &CancelToken::new(),
                memory: &mut memory,
                healthy: &mut || Ok(true),
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(first.finish, Finish::ToolCalls);
    let continued = [
        user,
        first.msg,
        oxa_proto::WireMsg::tool_result("read-one", "SYNTHETIC NOTE"),
    ];
    let lease = owner.try_request().unwrap();
    let second = transport
        .chat(
            &mut owner,
            lease,
            &ChatRequest::new(&prefix, &continued, "fake-model"),
            &mut Monitor {
                cancel: &CancelToken::new(),
                memory: &mut memory,
                healthy: &mut || Ok(true),
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(second.msg.content.as_deref(), Some("Synthetic completed task"));
    assert_eq!(owner.status().pid, pid);
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].messages, continued);
    assert_eq!(server.hits("GET", "/_local/status"), 0);
}

#[test]
fn local_overflow_uses_the_shared_error_classification() {
    let server = FakeServer::start(vec![ScriptTurn::fault(Fault::Overflow)], FakeOpts::default()).unwrap();
    let mut owner = owner();
    let lease = owner.try_request().unwrap();
    let prefix = prefix();
    let error = transport(&server)
        .chat(
            &mut owner,
            lease,
            &ChatRequest::new(&prefix, &[], "fake-model"),
            &mut Monitor {
                cancel: &CancelToken::new(),
                memory: &mut memory,
                healthy: &mut || Ok(true),
            },
            &mut |_| {},
        )
        .unwrap_err();
    assert_eq!(error.code, codes::CONTEXT_OVERFLOW);
    stopped(&mut owner);
}

#[test]
fn progress_before_overflow_preserves_typed_recovery_without_replaying() {
    let server = FakeServer::start(
        vec![ScriptTurn::fault(Fault::StreamError {
            queued: 1,
            status: 400,
            error_type: "exceed_context_size_error".into(),
            event: true,
        })],
        FakeOpts::default(),
    )
    .unwrap();
    let mut owner = owner();
    let lease = owner.try_request().unwrap();
    let prefix = prefix();
    let mut progress = false;
    let error = transport(&server)
        .chat(
            &mut owner,
            lease,
            &ChatRequest::new(&prefix, &[], "fake-model"),
            &mut Monitor {
                cancel: &CancelToken::new(),
                memory: &mut memory,
                healthy: &mut || Ok(true),
            },
            &mut |event| {
                progress |= matches!(event, StreamEvent::Queued { .. });
            },
        )
        .unwrap_err();
    assert!(progress);
    assert_eq!(error.code, codes::CONTEXT_OVERFLOW);
    assert!(!error.message.starts_with(oxa_proto::CUT_BY_REFUSAL));
    assert_eq!(server.hits("POST", "/v1/chat/completions"), 1);
    stopped(&mut owner);
}
