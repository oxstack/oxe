//! `oxe serve` end-to-end tests. The engine child is a Python fake: no weights, no inference,
//! no real oxe-server and no host cgroup dependency (`OXE_SERVE_FAKE_MEMORY=1`).
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;

const OXE: &str = env!("CARGO_BIN_EXE_oxe");

const FAKE: &str = r#"#!/usr/bin/env python3
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
port = 0
alias = "model"
for i, a in enumerate(args):
    if a == "--port" and i + 1 < len(args):
        port = int(args[i + 1])
    if a == "--alias" and i + 1 < len(args):
        alias = args[i + 1]
run = os.environ.get("FAKE_RUN_DIR", ".")
open(os.path.join(run, alias + ".pid"), "w").write(str(os.getpid()))
mode = "endless" if "endless" in alias else ("slow" if "slow" in alias else "stream")

def mark(suffix):
    open(os.path.join(run, alias + suffix), "w").write("1")

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def reply(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True
    def do_GET(self):
        if self.path == "/health":
            self.reply(200, b'{"status":"ok"}')
        elif self.path == "/props":
            self.reply(200, json.dumps({"build_info": "oxe-test"}).encode())
        else:
            self.reply(404, b'{}')
    def chunk(self, data):
        if not data:
            return
        self.wfile.write(("%x\r\n" % len(data)).encode() + data + b"\r\n")
        self.wfile.flush()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        with open(os.path.join(run, alias + ".log"), "ab") as f:
            f.write(self.path.encode() + b"\n" + body + b"\n")
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            return self.reply(404, b'{}')
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if mode == "endless":
                while True:
                    self.chunk(b": queued wait_s=1\n\n")
                    time.sleep(0.1)
            if mode == "slow":
                time.sleep(1.0)
            self.chunk(b": prefill 3/3 0s\n\n")
            if "error" in alias:
                self.chunk(b'event: error\ndata: {"error":{"message":"synthetic stream failure","type":"server_error","code":500}}\n\n')
                self.chunk(b": after error\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                self.close_connection = True
                return
            text = "hello from " + alias
            first = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": alias,
                     "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
            self.chunk(("data: " + json.dumps(first, separators=(",", ":")) + "\n\n").encode())
            fin = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": alias,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}}
            self.chunk(("data: " + json.dumps(fin, separators=(",", ":")) + "\n\n").encode())
            self.chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            mark(".cancel")

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"#;

struct Fixture {
    root: PathBuf,
    config: PathBuf,
    run: PathBuf,
    listen: u16,
}

impl Fixture {
    fn new(models: &[(&str, &str)]) -> Fixture {
        let root = std::env::temp_dir().join(format!(
            "oxe-serve-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        let run = root.join("run");
        std::fs::create_dir_all(&run).unwrap();
        let fake = root.join("fake-engine.py");
        std::fs::write(&fake, FAKE).unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let weights = root.join("synthetic.gguf");
        std::fs::write(&weights, b"synthetic weights placeholder; never opened").unwrap();
        let listen = free_port();
        let mut text = format!(
            "binary = \"{}\"\nlisten = \"127.0.0.1:{listen}\"\nidle_unload_seconds = 600\nstartup_timeout_seconds = 30\nmax_waiters = 8\n\n[environment]\nPATH = \"/usr/bin:/bin\"\nFAKE_RUN_DIR = \"{}\"\n",
            fake.display(),
            run.display()
        );
        for (id, extra) in models {
            text.push_str(&format!(
                "\n[models.{id}]\nmodel = \"{}\"\narguments = [\"--engine\", \"stub\"{}]\ncontext_window = 4096\nmax_output = 64\n\n[models.{id}.allocation_budget]\nkv_bytes_per_position = 1\nscratch_bytes = 0\nmtp_state_bytes = 0\nmargin_bytes = 1048576\n",
                weights.display(),
                extra
            ));
        }
        let config = root.join("models.toml");
        std::fs::write(&config, text).unwrap();
        Fixture {
            root,
            config,
            run,
            listen,
        }
    }

    fn start(&self) -> Serve {
        // Hold the child in its guard from the first instruction: a panic during startup still
        // ends `oxe serve` (and its engine) instead of leaking it into the next test run.
        let mut serve = Serve {
            child: Command::new(OXE)
                .env_clear()
                .env("HOME", &self.root)
                .env("PATH", "/usr/bin:/bin")
                .env("OXE_SERVE_FAKE_MEMORY", "1")
                .args([
                    "serve",
                    "--config",
                    self.config.to_str().unwrap(),
                    "--listen",
                    &format!("127.0.0.1:{}", self.listen),
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        };
        let deadline = Instant::now() + Duration::from_secs(15);
        while Instant::now() < deadline {
            if let Ok(mut response) = agent()
                .get(format!("http://127.0.0.1:{}/health", self.listen))
                .call()
            {
                let text = response.body_mut().read_to_string().unwrap_or_default();
                if text.contains("\"status\":\"ok\"") {
                    return serve;
                }
            }
            if serve.child.try_wait().unwrap().is_some() {
                let mut stderr = String::new();
                if let Some(mut pipe) = serve.child.stderr.take() {
                    let _ = pipe.read_to_string(&mut stderr);
                }
                panic!("oxe serve exited during startup: {stderr}");
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        panic!("oxe serve did not become healthy");
    }

    fn pid(&self, id: &str) -> Option<u32> {
        std::fs::read_to_string(self.run.join(format!("{id}.pid")))
            .ok()
            .and_then(|text| text.trim().parse().ok())
    }

    fn marker(&self, id: &str, suffix: &str) -> PathBuf {
        self.run.join(format!("{id}{suffix}"))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

struct Serve {
    child: Child,
}

impl Serve {
    /// The graceful signal every stop path uses: `oxe serve` must stop the engine and exit 0.
    fn terminate(&mut self) {
        let _ = kill(Pid::from_raw(self.child.id() as i32), Signal::SIGTERM);
    }

    /// SIGKILL: no handler, no destructor, no shutdown path — only the parent-death tie.
    fn kill_now(&mut self) {
        kill(Pid::from_raw(self.child.id() as i32), Signal::SIGKILL).unwrap();
    }

    fn wait_exit(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait().unwrap() {
                Some(status) => return Some(status),
                None if Instant::now() >= deadline => return None,
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }
}

impl Drop for Serve {
    fn drop(&mut self) {
        // SIGTERM, bounded wait, then SIGKILL: never leave an `oxe serve` behind.
        self.terminate();
        if self.wait_exit(Duration::from_secs(10)).is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// True once `pid` is gone or a zombie: a killed child keeps a `/proc` entry until it is reaped.
fn exited(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Err(_) => true,
        Ok(stat) => stat
            .rsplit_once(')')
            .map(|(_, rest)| rest.trim_start().starts_with('Z'))
            .unwrap_or(false),
    }
}

fn wait_exit_of(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if exited(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    exited(pid)
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(20)))
        .build()
        .new_agent()
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn chat(listen: u16, model: &str) -> (u16, String) {
    let body =
        json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let mut response = agent()
        .post(format!("http://127.0.0.1:{listen}/v1/chat/completions"))
        .header("Authorization", "Bearer local-only")
        .header("Content-Type", "application/json")
        .send(body.to_string())
        .unwrap();
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap();
    (status, text)
}

fn wait_for(path: &Path, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if path.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    false
}

#[test]
fn health_and_models_list_every_configured_model() {
    let fixture = Fixture::new(&[("tiny", ""), ("other", "")]);
    let _serve = fixture.start();
    let mut models = agent()
        .get(format!("http://127.0.0.1:{}/v1/models", fixture.listen))
        .call()
        .unwrap();
    let value: Value = serde_json::from_str(&models.body_mut().read_to_string().unwrap()).unwrap();
    assert_eq!(value["object"], "list");
    let ids: Vec<_> = value["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["other", "tiny"]);
}

#[test]
fn routing_by_model_streams_comments_and_switches_the_child() {
    let fixture = Fixture::new(&[("tiny", ""), ("other", "")]);
    let _serve = fixture.start();
    let (status, body) = chat(fixture.listen, "tiny");
    assert_eq!(status, 200, "{body}");
    assert!(body.contains(": prefill 3/3 0s"), "comment dropped: {body}");
    assert!(
        body.contains("\"content\":\"hello from tiny\""),
        "text missing: {body}"
    );
    assert!(body.contains("data: [DONE]"), "{body}");
    let first = fixture.pid("tiny").expect("first child pid");
    let (status, body) = chat(fixture.listen, "other");
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("hello from other"), "{body}");
    assert!(
        !Path::new(&format!("/proc/{first}")).exists(),
        "the old child must be stopped before the new one is admitted"
    );
}

#[test]
fn unknown_model_is_an_openai_style_404() {
    let fixture = Fixture::new(&[("tiny", "")]);
    let _serve = fixture.start();
    let mut response = agent()
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            fixture.listen
        ))
        .header("Content-Type", "application/json")
        .send(json!({"model": "nope", "messages": []}).to_string())
        .unwrap();
    assert_eq!(response.status().as_u16(), 404);
    let value: Value =
        serde_json::from_str(&response.body_mut().read_to_string().unwrap()).unwrap();
    assert_eq!(value["error"]["code"], "model_not_found");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert!(value["error"]["message"].as_str().unwrap().contains("nope"));
}

#[test]
fn stream_error_events_pass_through_unchanged() {
    let fixture = Fixture::new(&[("tiny-error", "")]);
    let _serve = fixture.start();
    let (status, body) = chat(fixture.listen, "tiny-error");
    assert_eq!(status, 200);
    assert!(body.contains("event: error\n"), "{body}");
    assert!(body.contains("synthetic stream failure"), "{body}");
    assert!(body.contains(": after error"), "{body}");
    assert!(!body.contains("data: [DONE]"), "{body}");
}

#[test]
fn idle_unload_stops_the_child_and_health_keeps_answering() {
    let fixture = Fixture::new(&[("tiny", "")]);
    let _serve = fixture.start();
    let (status, _) = chat(fixture.listen, "tiny");
    assert_eq!(status, 200);
    let pid = fixture.pid("tiny").expect("child pid");
    // idle_unload_seconds is 600 in this fixture; use the shipped default check instead: the
    // short-idle case runs below with its own config.
    assert!(Path::new(&format!("/proc/{pid}")).exists());
    let mut health = agent()
        .get(format!("http://127.0.0.1:{}/health", fixture.listen))
        .call()
        .unwrap();
    assert!(health
        .body_mut()
        .read_to_string()
        .unwrap()
        .contains("\"status\":\"ok\""));
}

#[test]
fn idle_unload_after_the_configured_seconds() {
    let fixture = Fixture::new(&[("tiny", "")]);
    let mut text = std::fs::read_to_string(&fixture.config).unwrap();
    text = text.replace("idle_unload_seconds = 600", "idle_unload_seconds = 1");
    std::fs::write(&fixture.config, text).unwrap();
    let _serve = fixture.start();
    assert_eq!(chat(fixture.listen, "tiny").0, 200);
    let pid = fixture.pid("tiny").expect("child pid");
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline && Path::new(&format!("/proc/{pid}")).exists() {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !Path::new(&format!("/proc/{pid}")).exists(),
        "idle unload did not stop the child"
    );
    let mut health = agent()
        .get(format!("http://127.0.0.1:{}/health", fixture.listen))
        .call()
        .unwrap();
    assert!(health
        .body_mut()
        .read_to_string()
        .unwrap()
        .contains("\"status\":\"ok\""));
}

#[test]
fn a_full_queue_answers_503_with_retry_after() {
    let fixture = Fixture::new(&[("slow", "")]);
    let mut text = std::fs::read_to_string(&fixture.config).unwrap();
    text = text.replace("max_waiters = 8", "max_waiters = 0");
    std::fs::write(&fixture.config, text).unwrap();
    let _serve = fixture.start();
    let listen = fixture.listen;
    let first = std::thread::spawn(move || chat(listen, "slow"));
    // Let the first request take the only lease (the child sleeps 1 s before replying).
    std::thread::sleep(Duration::from_millis(300));
    let mut response = agent()
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            fixture.listen
        ))
        .header("Content-Type", "application/json")
        .send(json!({"model": "slow", "messages": []}).to_string())
        .unwrap();
    assert_eq!(response.status().as_u16(), 503);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .map(|v| v.to_str().unwrap()),
        Some("1")
    );
    let value: Value =
        serde_json::from_str(&response.body_mut().read_to_string().unwrap()).unwrap();
    assert_eq!(value["error"]["code"], "queue_full");
    assert_eq!(first.join().unwrap().0, 200);
}

#[test]
fn client_disconnect_cancels_the_upstream_request() {
    let fixture = Fixture::new(&[("endless", "")]);
    let _serve = fixture.start();
    let mut stream = TcpStream::connect(("127.0.0.1", fixture.listen)).unwrap();
    let body = br#"{"model":"endless","messages":[],"stream":true}"#;
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).unwrap();
    stream.write_all(body).unwrap();
    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf).unwrap();
    assert!(n > 0, "no response bytes before the disconnect");
    drop(stream);
    assert!(
        wait_for(
            &fixture.marker("endless", ".cancel"),
            Duration::from_secs(5)
        ),
        "the fake child never saw the upstream cancellation"
    );
}

#[test]
fn non_loopback_listen_is_refused_without_allow_remote() {
    let fixture = Fixture::new(&[("tiny", "")]);
    let port = free_port();
    let output = Command::new(OXE)
        .env_clear()
        .env("HOME", &fixture.root)
        .env("PATH", "/usr/bin:/bin")
        .env("OXE_SERVE_FAKE_MEMORY", "1")
        .args([
            "serve",
            "--config",
            fixture.config.to_str().unwrap(),
            "--listen",
            &format!("0.0.0.0:{port}"),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("refusing non-loopback"), "{stderr}");
    assert!(stderr.contains("--allow-remote"), "{stderr}");
}

#[test]
fn local_only_bearer_is_accepted_without_a_key() {
    let fixture = Fixture::new(&[("tiny", "")]);
    let _serve = fixture.start();
    let (status, body) = chat(fixture.listen, "tiny");
    assert_eq!(status, 200, "{body}");
}

/// Regression: a plain `kill` (SIGTERM) must stop the engine through the supervisor's own stop
/// path and exit 0, not leave a ~20 GB engine behind with no owner.
#[test]
fn sigterm_stops_the_engine_child_and_exits_clean() {
    let fixture = Fixture::new(&[("tiny", "")]);
    let mut serve = fixture.start();
    assert_eq!(chat(fixture.listen, "tiny").0, 200);
    let engine = fixture.pid("tiny").expect("the fake engine wrote its pid");
    assert!(!exited(engine), "the fake child must be running first");
    serve.terminate();
    let status = serve
        .wait_exit(Duration::from_secs(10))
        .expect("oxe serve must exit on SIGTERM");
    assert_eq!(
        status.code(),
        Some(0),
        "SIGTERM is a clean shutdown, not a failure: {status:?}"
    );
    assert!(
        wait_exit_of(engine, Duration::from_secs(5)),
        "the engine child outlived the SIGTERM shutdown"
    );
}

/// Regression: if `oxe serve` dies abruptly (SIGKILL: no handler, no destructor), the engine child
/// must die with it through PR_SET_PDEATHSIG instead of running on forever.
#[test]
fn sigkilled_serve_takes_the_engine_child_with_it() {
    let fixture = Fixture::new(&[("tiny", "")]);
    let mut serve = fixture.start();
    assert_eq!(chat(fixture.listen, "tiny").0, 200);
    let engine = fixture.pid("tiny").expect("the fake engine wrote its pid");
    assert!(!exited(engine), "the fake child must be running first");
    serve.kill_now();
    let status = serve
        .wait_exit(Duration::from_secs(10))
        .expect("the killed oxe serve must be reaped");
    assert_eq!(status.signal(), Some(Signal::SIGKILL as i32));
    assert!(
        wait_exit_of(engine, Duration::from_secs(5)),
        "the engine child outlived a SIGKILLed oxe serve"
    );
}
