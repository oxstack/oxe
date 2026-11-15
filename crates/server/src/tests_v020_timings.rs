//! v0.2.0 request-path timings (`timings.queue_ms`, `render_ms`, `tokenize_ms`, `restore_ms`,
//! `ckpt_ms`, and the same five in the engine ledger line). Measurement only: model-free (stub
//! engine with a per-token prefill sleep, synthetic tokenizer, the chat template fixture).

use crate::engine::StubEngine;
use crate::generate::{GenParams, Worker};
use crate::sampling::SamplingParams;
use crate::service::{handle, Reply, State};
use crate::tests_hotfix::synthetic_tokenizer;
use crate::tests_state_v013::srv;
use serde_json::{json, Value as Json};
use std::sync::Arc;
use std::time::{Duration, Instant};

const FIELDS: [&str; 5] = [
    "queue_ms",
    "render_ms",
    "tokenize_ms",
    "restore_ms",
    "ckpt_ms",
];

/// One request through `handle`: the reply's JSON, or a stream's data frames as an array, and the
/// wall time of the whole call (ms).
async fn send(st: &Arc<State<StubEngine>>, path: &str, body: Json) -> (Json, f64) {
    let b = serde_json::to_vec(&body).unwrap();
    let t0 = Instant::now();
    let reply = match handle(st.clone(), "POST", path, "", &b).await {
        Reply::Json(c, j) => {
            assert_eq!(c, 200, "{path}: {j}");
            j
        }
        Reply::Stream(mut rx) => {
            let mut frames = Vec::new();
            while let Some(f) = tokio::time::timeout(Duration::from_secs(60), rx.recv())
                .await
                .expect("stream hung")
            {
                for line in f.lines() {
                    if let Some(d) = line.strip_prefix("data: ").filter(|d| *d != "[DONE]") {
                        frames.push(serde_json::from_str::<Json>(d).unwrap());
                    }
                }
            }
            Json::Array(frames)
        }
    };
    (reply, t0.elapsed().as_secs_f64() * 1e3)
}

/// The `timings` object of a reply, or of the one frame of a stream that carries it.
fn timings_of(j: &Json) -> &Json {
    match j {
        Json::Array(fs) => {
            let with: Vec<&Json> = fs.iter().filter(|f| f.get("timings").is_some()).collect();
            assert_eq!(with.len(), 1, "exactly one frame carries timings: {j}");
            &with[0]["timings"]
        }
        _ => &j["timings"],
    }
}

fn check_fields(t: &Json, what: &str) {
    for k in FIELDS {
        let v = t[k]
            .as_f64()
            .unwrap_or_else(|| panic!("{what}: {k} missing or not a number: {t}"));
        assert!(v.is_finite() && v >= 0.0, "{what}: {k} = {v}");
    }
}

fn greedy(n_predict: i64) -> GenParams {
    GenParams {
        n_predict,
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn timings_carry_the_five_fields_on_every_route_and_render_is_chat_only() {
    let st = srv(false, &[], None);
    let chat =
        json!({"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 4});
    let legacy = json!({"prompt": "hi", "temperature": 0, "max_tokens": 4});
    let native = json!({"prompt": "hi", "temperature": 0, "n_predict": 4});
    let cases: [(&str, Json, bool); 6] = [
        ("chat", chat.clone(), true),
        (
            "chat stream",
            json!({"stream": true, "messages": chat["messages"].clone(), "temperature": 0, "max_tokens": 4}),
            true,
        ),
        ("legacy", legacy.clone(), false),
        (
            "legacy stream",
            json!({"stream": true, "prompt": "hi", "temperature": 0, "max_tokens": 4}),
            false,
        ),
        ("native", native.clone(), false),
        (
            "native stream",
            json!({"stream": true, "prompt": "hi", "temperature": 0, "n_predict": 4}),
            false,
        ),
    ];
    for (name, body, is_chat) in cases {
        let path = if name.starts_with("chat") {
            "/v1/chat/completions"
        } else if name.starts_with("legacy") {
            "/v1/completions"
        } else {
            "/completion"
        };
        let (j, wall) = send(&st, path, body).await;
        let t = timings_of(&j);
        check_fields(t, name);
        if !is_chat {
            assert_eq!(
                t["render_ms"].as_f64(),
                Some(0.0),
                "{name}: render is a chat-only phase: {t}"
            );
        }
        let sum = ["queue_ms", "prompt_ms", "predicted_ms"]
            .iter()
            .map(|k| t[*k].as_f64().unwrap())
            .sum::<f64>();
        assert!(
            sum <= wall + 5.0,
            "{name}: queue + prompt + predicted = {sum} ms > the call's {wall} ms"
        );
    }
}

/// A slow prefill (`ms_per_token` each) and a request queued behind it: the queued one reports the
/// wait, the first one (already holding the worker) does not. Returns (first's queue, second's).
async fn queue_behind_a_slow_prefill(st: Arc<State<StubEngine>>) -> (f64, f64) {
    st.worker.lock().unwrap().engine.prefill_sleep_per_token = Duration::from_millis(2);
    let first_st = st.clone();
    // 100 prompt tokens at 2 ms: the first generation holds the worker for ~200 ms
    let first = tokio::spawn(async move {
        send(
            &first_st,
            "/completion",
            json!({"prompt": "x".repeat(100), "n_predict": 4, "temperature": 0}),
        )
        .await
    });
    // wait until the first generation holds the worker (`running` is set under it)
    let t0 = Instant::now();
    while st.running.lock().unwrap().is_none() {
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "the first generation never started"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let (second, _) = send(
        &st,
        "/completion",
        json!({"prompt": "y".repeat(10), "n_predict": 2, "temperature": 0}),
    )
    .await;
    let (f, _) = first.await.unwrap();
    let (qa, qb) = (
        timings_of(&f)["queue_ms"].as_f64().unwrap(),
        timings_of(&second)["queue_ms"].as_f64().unwrap(),
    );
    check_fields(timings_of(&f), "first");
    check_fields(timings_of(&second), "second");
    (qa, qb)
}

#[tokio::test(flavor = "multi_thread")]
async fn queue_ms_counts_the_wait_for_the_worker() {
    let st = srv(false, &[], None);
    let (qa, qb) = queue_behind_a_slow_prefill(st).await;
    assert!(
        qa < 50.0,
        "the first generation waited for nothing: queue_ms {qa}"
    );
    assert!(
        qb >= 100.0,
        "the second waited ~170 ms behind the first's prefill: queue_ms {qb}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn queue_ms_counts_the_wait_for_the_worker_with_fair_slicing_on() {
    let st = srv(false, &[], None);
    st.fair.set_quantum_ms(20);
    let (qa, qb) = queue_behind_a_slow_prefill(st).await;
    assert!(
        qa < 50.0,
        "the first generation took the turn at once: queue_ms {qa}"
    );
    assert!(
        qb >= 100.0,
        "the second waited for the first's prefill (a prefill is not sliced): queue_ms {qb}"
    );
}

/// The engine ledger line carries the five fields, and its `ckpt_ms` / `restore_ms` are the reply's.
#[test]
fn ledger_line_carries_restore_and_checkpoint_times() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/tests-v020-timings");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("ledger-{}.jsonl", std::process::id()));
    let tok = synthetic_tokenizer();
    let eng = StubEngine::new(
        4096,
        tok.n_vocab(),
        tok.eos.unwrap(),
        tok.encode("done", false, false),
    );
    let mut w = Worker::new(eng, 8);
    w.ledger = Some(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap(),
    );
    let p1 = tok.encode("hello there, friend", false, false);
    let mut p2 = p1[..10].to_vec();
    p2.extend(tok.encode(" and more words", false, false));
    let r1 = w.generate(&tok, &p1, &greedy(4), &mut |_| true).unwrap();
    let r2 = w.generate(&tok, &p2, &greedy(4), &mut |_| true).unwrap();
    drop(w);
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let lines: Vec<Json> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2, "{text}");
    for (line, r) in lines.iter().zip([&r1, &r2]) {
        for k in FIELDS {
            let v = line[k]
                .as_f64()
                .unwrap_or_else(|| panic!("ledger {k} missing: {line}"));
            assert!(v.is_finite() && v >= 0.0, "ledger {k} = {v}");
        }
        // the same numbers (serde_json's default float parser is off by an ulp at most)
        for (k, want) in [
            ("restore_ms", r.timings.restore_ms),
            ("ckpt_ms", r.timings.ckpt_ms),
        ] {
            let got = line[k].as_f64().unwrap();
            assert!(
                (got - want).abs() < 1e-9,
                "{k}: ledger {got} vs reply {want}: {line}"
            );
        }
    }
    let _ = std::fs::remove_dir(&dir);
}
