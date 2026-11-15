//! v0.2.0 fair time-slicing between generations (`fair.rs`). Model-free (stub engine with a
//! per-token decode time and scripts keyed by the sequence's first token, synthetic byte-level
//! tokenizer) except the mini-model checks (aarch64: real numerics, MTP off and on, no weights).
//!
//! Exactness = the same tokens, text, stop, stopping word, probability rows (streamed and final)
//! and the same cache state (tokens, checkpoint positions, the follow-up turn's cache_n; on the mini
//! model the engine's state bytes and draft counters) as each generation run alone, unsliced.

use crate::engine::{Engine, StubEngine};
use crate::generate::{
    GenError, GenParams, GenResult, Partial, Progress, StopType, TokenProbs, Worker,
};
use crate::sampling::SamplingParams;
use crate::service::{handle, Config, Reply, State};
use crate::tests_hotfix::synthetic_tokenizer;
use serde_json::{json, Value as Json};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokenizer::Tokenizer;

/// A long reply with multi-byte UTF-8 (byte tokens: every such character spans 2-4 tokens, so a
/// yield lands inside one), no EOS before its end.
fn script_a() -> String {
    "Héllo wörld, this is session A: ünïcode ✓ and 日本語 across yields. ".repeat(4)
}

fn script_b() -> String {
    "Bonjour ça va? Session B — ok ✓. ".repeat(3)
}

fn script_c() -> String {
    "Cheers from C, a shared guest. ".to_string()
}

fn first(tok: &Tokenizer, c: &str) -> u32 {
    tok.encode(c, false, false)[0]
}

fn stub(tok: &Tokenizer, n_ctx: usize) -> StubEngine {
    let mut eng = StubEngine::new(
        n_ctx,
        tok.n_vocab(),
        tok.eos.unwrap(),
        tok.encode("default", false, false),
    );
    eng.scripts_by_first = vec![
        (first(tok, "A"), tok.encode(&script_a(), false, false)),
        (first(tok, "B"), tok.encode(&script_b(), false, false)),
        (first(tok, "C"), tok.encode(&script_c(), false, false)),
    ];
    eng
}

fn stub_worker(tok: &Tokenizer, n_ctx: usize) -> Worker<StubEngine> {
    let mut w = Worker::new(stub(tok, n_ctx), 8);
    w.ubatch = 16;
    w.cache.park_budget = 1 << 30;
    // a finished session parked by a resume is found again by its next turn, however short
    w.cache.park_min = 1;
    w
}

fn prompt(tok: &Tokenizer, s: &str) -> Vec<u32> {
    tok.encode(s, false, false)
}

/// A probability row as comparable values (probabilities bit for bit).
type Row = (u32, Vec<u8>, u32, Vec<(u32, Vec<u8>, u32)>);

fn rows(p: &[TokenProbs]) -> Vec<Row> {
    p.iter()
        .map(|r| {
            (
                r.id,
                r.piece.clone(),
                r.prob.to_bits(),
                r.top
                    .iter()
                    .map(|e| (e.id, e.piece.clone(), e.prob.to_bits()))
                    .collect(),
            )
        })
        .collect()
}

#[derive(Debug, PartialEq)]
struct Out {
    tokens: Vec<u32>,
    text: String,
    stop: StopType,
    stopping_word: String,
    truncated: bool,
    n_decoded: usize,
    probs: Vec<Row>,
    /// what the partials carried (text, rows)
    streamed: (String, Vec<Row>),
    /// the engine state bytes right after the generation (empty for the stub)
    state: Vec<u8>,
    draft: (u64, u64),
}

fn out(r: &GenResult, streamed: (String, Vec<Row>), state: Vec<u8>) -> Out {
    Out {
        tokens: r.tokens.clone(),
        text: r.text.clone(),
        stop: r.stop,
        stopping_word: r.stopping_word.clone(),
        truncated: r.truncated,
        n_decoded: r.n_decoded,
        probs: rows(&r.probs),
        streamed,
        state,
        draft: (r.timings.draft_n, r.timings.draft_n_accepted),
    }
}

fn state_bytes<E: Engine>(e: &mut E, real: bool) -> Vec<u8> {
    if !real {
        return Vec::new();
    }
    let mut b = Vec::new();
    e.save_state(&mut b).unwrap();
    b
}

/// The unsliced control: `prompt` alone on `w` (a fresh worker).
fn alone<E: Engine>(
    w: &mut Worker<E>,
    tok: &Tokenizer,
    prompt: &[u32],
    p: &GenParams,
    real: bool,
) -> Out {
    let mut s = (String::new(), Vec::new());
    let r = w
        .generate(tok, prompt, p, &mut |pp: Partial| {
            s.0.push_str(&pp.text);
            s.1.extend(rows(&pp.probs));
            true
        })
        .unwrap();
    let st = state_bytes(&mut w.engine, real);
    out(&r, s, st)
}

/// Every job on one worker, interleaved slice by slice: each yields at every allowed point (a new
/// job starts while the others are suspended), round robin until all ended. Returns the outputs
/// and the yields of each.
fn interleaved<E: Engine>(
    w: &mut Worker<E>,
    tok: &Tokenizer,
    jobs: &[(Vec<u32>, GenParams)],
    real: bool,
) -> (Vec<Out>, Vec<u32>) {
    let n = jobs.len();
    let mut streams: Vec<(String, Vec<Row>)> = vec![(String::new(), Vec::new()); n];
    let mut runs: Vec<Option<crate::generate::SlicedRun<E::Snapshot>>> =
        (0..n).map(|_| None).collect();
    let mut outs: Vec<Option<Out>> = (0..n).map(|_| None).collect();
    let mut yields = vec![0u32; n];
    let mut started = 0;
    while outs.iter().any(Option::is_none) {
        for i in 0..n {
            if outs[i].is_some() {
                continue;
            }
            if runs[i].is_none() {
                if i != started {
                    continue; // jobs start in order, one per round
                }
                runs[i] = Some(w.generate_start(&jobs[i].0, &jobs[i].1).unwrap());
                started += 1;
            }
            let sr = runs[i].as_mut().unwrap();
            let s = &mut streams[i];
            let r = w
                .generate_slice(
                    tok,
                    &jobs[i].0,
                    &jobs[i].1,
                    sr,
                    &mut |pp: Partial| {
                        s.0.push_str(&pp.text);
                        s.1.extend(rows(&pp.probs));
                        true
                    },
                    &mut || true,
                )
                .unwrap();
            if let Some(r) = r {
                yields[i] = r.timings.fair_yields;
                let st = state_bytes(&mut w.engine, real);
                outs[i] = Some(out(&r, std::mem::take(&mut streams[i]), st));
                runs[i] = None;
            } else {
                assert!(runs[i].as_ref().unwrap().suspended());
                assert!(
                    w.cache.tokens.is_empty(),
                    "a suspended run leaves an empty engine"
                );
                assert_eq!(w.engine.n_past(), 0);
            }
        }
    }
    (outs.into_iter().map(Option::unwrap).collect(), yields)
}

/// (cache_n, the cache's tokens, its checkpoint positions) after a turn
type Turn = (usize, Vec<u32>, Vec<usize>);

/// The next turn of `prompt` + its reply.
fn follow_up<E: Engine>(
    w: &mut Worker<E>,
    tok: &Tokenizer,
    prompt: &[u32],
    reply: &[u32],
    p: &GenParams,
) -> Turn {
    let mut next = prompt.to_vec();
    next.extend(reply);
    next.extend(tok.encode(" and then?", false, false));
    if next.len() + 4 >= w.engine.n_ctx() {
        // a session at the context end: an edit inside its reply (resumes at a checkpoint)
        next = prompt.to_vec();
        next.extend(&reply[..reply.len() / 2]);
        next.extend(tok.encode("?", false, false));
    }
    let r = w
        .generate(
            tok,
            &next,
            &GenParams {
                n_predict: 4,
                ..p.clone()
            },
            &mut |_| true,
        )
        .unwrap();
    (
        r.timings.cache_n,
        w.cache.tokens.clone(),
        w.cache.snapshot_positions(),
    )
}

fn greedy(n: i64) -> GenParams {
    GenParams {
        n_predict: n,
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Two (or three) sessions interleaved == each alone, for one parameter set per session.
fn check_stub(name: &str, n_ctx: usize, jobs: Vec<(Vec<u32>, GenParams)>) {
    let tok = synthetic_tokenizer();
    let ctrl: Vec<(Out, Turn)> = jobs
        .iter()
        .map(|(pr, p)| {
            let mut w = stub_worker(&tok, n_ctx);
            let o = alone(&mut w, &tok, pr, p, false);
            let f = follow_up(&mut w, &tok, pr, &o.tokens, p);
            (o, f)
        })
        .collect();
    let mut w = stub_worker(&tok, n_ctx);
    let (got, yields) = interleaved(&mut w, &tok, &jobs, false);
    for (i, (g, (c, _))) in got.iter().zip(&ctrl).enumerate() {
        assert_eq!(g, c, "{name}: job {i} differs from its unsliced control");
        if c.n_decoded > 1 {
            assert!(yields[i] > 0, "{name}: job {i} yielded");
        }
    }
    // each session's next turn finds all of it, as alone (the last one live, the others parked)
    for (i, ((pr, p), (_, cf))) in jobs.iter().zip(&ctrl).enumerate() {
        let f = follow_up(&mut w, &tok, pr, &got[i].tokens, p);
        assert_eq!(
            &f, cf,
            "{name}: job {i}: the follow-up turn (cache_n, cache tokens, checkpoints)"
        );
    }
}

#[test]
fn worker_interleaved_equals_alone_greedy_and_utf8_across_yields() {
    let tok = synthetic_tokenizer();
    let a = prompt(&tok, "A: tell me something long, please. ");
    let b = prompt(&tok, "B: and something short. ");
    check_stub("greedy", 4096, vec![(a, greedy(200)), (b, greedy(60))]);
}

#[test]
fn worker_interleaved_equals_alone_seeded_sampling_and_n_probs() {
    let tok = synthetic_tokenizer();
    let a = prompt(&tok, "A: sample me. ");
    let b = prompt(&tok, "B: and me. ");
    let c = prompt(&tok, "C: guest. ");
    let sampled = |seed: u64| GenParams {
        n_predict: 90,
        sampling: SamplingParams {
            temperature: 2.5,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
            seed,
            ..Default::default()
        },
        n_probs: 3,
        ..Default::default()
    };
    let probs = GenParams {
        n_probs: 2,
        ..greedy(70)
    };
    check_stub(
        "sampled+probs",
        4096,
        vec![(a, sampled(11)), (b, probs), (c, sampled(12))],
    );
}

#[test]
fn worker_interleaved_equals_alone_stop_strings_and_context_end() {
    let tok = synthetic_tokenizer();
    let a = prompt(&tok, "A: stop at the word. ");
    let b = prompt(&tok, "B: run to the end of the context. ");
    let stop = GenParams {
        stop: vec!["日本".into(), "never".into()],
        ..greedy(-1)
    };
    // n_ctx just past B's prompt: B stops at the context end (truncated), A at its stop string
    let n_ctx = b.len().max(a.len()) + 40;
    let ctx = greedy(-1);
    let tokc = synthetic_tokenizer();
    let mut w = stub_worker(&tokc, n_ctx);
    let ca = alone(&mut w, &tokc, &b, &ctx, false);
    assert!(
        ca.truncated && ca.stop == StopType::Limit,
        "the control stops at the context end"
    );
    let mut w = stub_worker(&tokc, 4096);
    let sa = alone(&mut w, &tokc, &a, &stop, false);
    assert_eq!(sa.stop, StopType::Word);
    // A's stop string lies past n_ctx here: give A a separate check on a large context
    check_stub(
        "stop-string",
        4096,
        vec![(a.clone(), stop.clone()), (b.clone(), greedy(80))],
    );
    check_stub(
        "context-end",
        n_ctx,
        vec![
            (b, ctx.clone()),
            (prompt(&tok, "C: also to the end. "), ctx),
        ],
    );
}

#[test]
fn without_a_park_budget_a_run_never_yields_and_stays_exact() {
    let tok = synthetic_tokenizer();
    let a = prompt(&tok, "A: no room to hold me. ");
    let mut w = stub_worker(&tok, 4096);
    let ctrl = alone(&mut w, &tok, &a, &greedy(50), false);
    let mut w = stub_worker(&tok, 4096);
    w.cache.park_budget = 0;
    let mut sr = w.generate_start(&a, &greedy(50)).unwrap();
    let mut asked = 0;
    let r = w.generate_slice(&tok, &a, &greedy(50), &mut sr, &mut |_| true, &mut || {
        asked += 1;
        true
    });
    let r = r.unwrap().expect("ran to its end in one slice");
    assert_eq!(
        asked, 0,
        "nobody is asked to yield when the session cannot be held"
    );
    assert_eq!(r.timings.fair_yields, 0);
    assert_eq!(r.tokens, ctrl.tokens);
    assert_eq!(r.text, ctrl.text);
    // a state larger than the budget: the same
    let mut w = stub_worker(&tok, 4096);
    w.cache.park_budget = 16;
    let mut sr = w.generate_start(&a, &greedy(50)).unwrap();
    let r = w
        .generate_slice(&tok, &a, &greedy(50), &mut sr, &mut |_| true, &mut || true)
        .unwrap()
        .unwrap();
    assert_eq!((r.timings.fair_yields, r.tokens), (0, ctrl.tokens));
}

/// One RAM budget for the parked sessions and the sessions of suspended runs (state copy AND live
/// checkpoints): whatever the budget, parked + held never exceed it, a suspension evicts parked
/// sessions to make room, a run that cannot be held does not yield, and the output stays exact.
#[test]
fn held_and_parked_sessions_share_one_tight_budget() {
    let tok = synthetic_tokenizer();
    let pp = prompt(
        &tok,
        &"P: an earlier conversation that gets parked. ".repeat(6),
    );
    let a = prompt(
        &tok,
        &"A: the run that is sliced, with a checkpoint of its own. ".repeat(6),
    );
    // a parked session P, then A live with a checkpoint; returns the worker and A's run
    let scene = |budget: usize| {
        let mut w = stub_worker(&tok, 4096);
        alone(&mut w, &tok, &pp, &greedy(20), false);
        let sr = w.generate_start(&a, &greedy(60)).unwrap();
        w.cache
            .checkpoint_with(&mut w.engine, crate::cache::PRIO_TURN, false)
            .unwrap();
        let parked = w.cache.parked_bytes();
        let need = w.engine.state_size_hint() + w.cache.live_snap_bytes();
        w.cache.park_budget = budget;
        w.cache.make_room();
        (w, sr, parked, need)
    };
    // the control: the same scene with no budget at all (never held, one slice)
    let ctrl = {
        let (mut w, mut sr, _, _) = scene(0);
        w.generate_slice(&tok, &a, &greedy(60), &mut sr, &mut |_| true, &mut || true)
            .unwrap()
            .expect("one slice")
    };
    assert_eq!(ctrl.timings.fair_yields, 0);
    let (w, _, parked, need) = scene(1 << 30);
    assert!(parked > 0, "P was parked when A started");
    assert!(w.cache.live_snap_bytes() > 0, "A has a live checkpoint");
    let top = parked + 3 * need;
    let (mut held_any, mut refused_any, mut evicted_any, mut beside_any) =
        (false, false, false, false);
    for budget in
        (0..=48)
            .map(|k| top * k / 48)
            .chain([need - 1, need, parked + need - 1, parked + need])
    {
        let (mut w, mut sr, _, _) = scene(budget);
        let parked0 = w.cache.parked_bytes();
        let mut yields = 0u32;
        let r = loop {
            match w
                .generate_slice(&tok, &a, &greedy(60), &mut sr, &mut |_| true, &mut || true)
                .unwrap()
            {
                Some(r) => break r,
                None => {
                    yields += 1;
                    let (p, h) = (w.cache.parked_bytes(), w.cache.reserved);
                    assert!(h > 0, "budget {budget}: a suspended run holds its session");
                    assert!(
                        p + h <= budget,
                        "budget {budget}: parked {p} + held {h} exceed it"
                    );
                    evicted_any |= p < parked0;
                    beside_any |= p > 0;
                }
            }
        };
        assert_eq!(
            w.cache.reserved, 0,
            "budget {budget}: nothing held at the end"
        );
        assert!(
            w.cache.parked_bytes() <= budget,
            "budget {budget}: parked sessions fit at the end"
        );
        assert_eq!(
            (r.timings.fair_yields, &r.tokens, &r.text),
            (yields, &ctrl.tokens, &ctrl.text),
            "budget {budget}: exact"
        );
        held_any |= yields > 0;
        refused_any |= yields == 0;
    }
    assert!(
        held_any && refused_any,
        "the sweep covers budgets that hold and budgets that refuse"
    );
    assert!(
        evicted_any,
        "a suspension evicted a parked session to make room"
    );
    assert!(
        beside_any,
        "a parked session stayed beside a held one where both fit"
    );
}

// ---- service level ----

fn fair_state(quantum_ms: u64, decode_us: u64, park_budget: usize) -> Arc<State<StubEngine>> {
    let tok = synthetic_tokenizer();
    let mut eng = stub(&tok, 8192);
    eng.decode_sleep = Duration::from_micros(decode_us);
    let cfg = Config {
        model_name: "synthetic".into(),
        model_path: "/m.gguf".into(),
        chat_opts: Default::default(),
        fingerprint: "oxe-test".into(),
        slots: true,
        idle_reprefill: false,
        engine_status: None,
    };
    let mut w = Worker::new(eng, 8);
    w.ubatch = 16;
    w.cache.park_budget = park_budget;
    w.cache.park_min = 1;
    let mut st = State::new(cfg, tok, w).unwrap();
    st.sse_keepalive = Duration::from_millis(10);
    st.fair.set_quantum_ms(quantum_ms);
    Arc::new(st)
}

type Log = Arc<Mutex<Vec<(String, Instant)>>>;

fn logged(log: &Log, what: &str) -> Option<Instant> {
    log.lock()
        .unwrap()
        .iter()
        .find(|x| x.0 == what)
        .map(|x| x.1)
}

/// A streamed /completion: (the content text, the final frame, every frame)
async fn stream(
    st: Arc<State<StubEngine>>,
    body: Json,
    name: &'static str,
    log: Log,
) -> (String, Json, Vec<String>) {
    let Reply::Stream(mut rx) = handle(
        st,
        "POST",
        "/completion",
        "",
        &serde_json::to_vec(&body).unwrap(),
    )
    .await
    else {
        panic!("{name}: not a stream")
    };
    let mut frames = Vec::new();
    let mut text = String::new();
    let mut last = Json::Null;
    loop {
        match tokio::time::timeout(Duration::from_secs(60), rx.recv()).await {
            Ok(Some(f)) => {
                if let Some(d) = f.strip_prefix("data: ") {
                    let j: Json = serde_json::from_str(d.trim_end()).unwrap();
                    if text.is_empty() && logged(&log, &format!("{name} first")).is_none() {
                        log.lock()
                            .unwrap()
                            .push((format!("{name} first"), Instant::now()));
                    }
                    text.push_str(j["content"].as_str().unwrap_or(""));
                    last = j;
                }
                frames.push(f);
            }
            Ok(None) => break,
            Err(_) => panic!("{name}: stream did not end within 60 s"),
        }
    }
    log.lock()
        .unwrap()
        .push((format!("{name} done"), Instant::now()));
    (text, last, frames)
}

fn body(p: &str, n: i64) -> Json {
    json!({"prompt": p, "n_predict": n, "stream": true, "temperature": 0, "n_probs": 1})
}

const PA: &str = "A: the long story, please.";
const PB: &str = "B: a short one.";

/// A (long) streams; B (short) arrives after A's first token. Returns (A, B, log).
async fn a_then_b(
    st: Arc<State<StubEngine>>,
) -> (
    (String, Json, Vec<String>),
    (String, Json, Vec<String>),
    Log,
) {
    let log: Log = Default::default();
    let ha = tokio::spawn(stream(st.clone(), body(PA, 150), "A", log.clone()));
    let t0 = Instant::now();
    while logged(&log, "A first").is_none() {
        assert!(t0.elapsed() < Duration::from_secs(30), "A never started");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let hb = tokio::spawn(stream(st.clone(), body(PB, 20), "B", log.clone()));
    let b = hb.await.unwrap();
    let a = ha.await.unwrap();
    (a, b, log)
}

async fn control(p: &str, n: i64) -> String {
    let st = fair_state(0, 0, 1 << 30);
    stream(st, body(p, n), "ctrl", Default::default()).await.0
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn service_a_short_request_overtakes_a_long_stream_and_both_stay_exact() {
    let (ca, cb) = (control(PA, 150).await, control(PB, 20).await);
    let st = fair_state(40, 3000, 1 << 30);
    let ((ta, fa, frames_a), (tb, fb, _), log) = a_then_b(st.clone()).await;
    assert!(
        logged(&log, "B done").unwrap() < logged(&log, "A done").unwrap(),
        "B finished before A (no starvation)"
    );
    assert_eq!(ta, ca, "A's text equals its unsliced control");
    assert_eq!(tb, cb, "B's text equals its unsliced control");
    assert!(
        fa["timings"]["fair_yields"].as_u64().unwrap_or(0) >= 1,
        "A yielded: {}",
        fa["timings"]
    );
    assert!(fa["timings"]["fair_wait_ms"].as_f64().unwrap() > 0.0);
    assert_eq!(fa["stop_type"], "limit");
    assert_eq!(fb["stop_type"], "limit");
    // the keepalive kept A's connection busy while it was suspended (after its first data frame)
    let first_data = frames_a
        .iter()
        .position(|f| f.starts_with("data: "))
        .unwrap();
    assert!(
        frames_a[first_data..]
            .iter()
            .any(|f| f.starts_with(": queued wait_s=")),
        "a `: queued` comment while suspended"
    );
    assert_eq!(st.fair.suspended(), 0);
    assert!(st.fair.order().is_empty(), "every ticket left the queue");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn service_quantum_zero_keeps_first_come_first_served() {
    let st = fair_state(0, 3000, 1 << 30);
    let ((ta, fa, frames_a), (tb, _, _), log) = a_then_b(st).await;
    assert!(
        logged(&log, "B done").unwrap() > logged(&log, "A done").unwrap(),
        "off: B waits for A (the test discriminates)"
    );
    assert!(
        fa["timings"].get("fair_yields").is_none(),
        "no fair fields when nothing yielded"
    );
    assert_eq!((ta, tb), (control(PA, 150).await, control(PB, 20).await));
    let first_data = frames_a
        .iter()
        .position(|f| f.starts_with("data: "))
        .unwrap();
    assert!(
        !frames_a[first_data..]
            .iter()
            .any(|f| f.starts_with(": queued")),
        "off: no queued comment after the first token"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn service_without_a_park_budget_stays_fifo_and_exact() {
    let st = fair_state(40, 3000, 0);
    let ((ta, fa, _), (tb, _, _), log) = a_then_b(st).await;
    assert!(
        logged(&log, "B done").unwrap() > logged(&log, "A done").unwrap(),
        "no hold possible: no yield, FIFO"
    );
    assert!(fa["timings"].get("fair_yields").is_none());
    assert_eq!((ta, tb), (control(PA, 150).await, control(PB, 20).await));
}

// ---- driver level (fair::run_sliced on threads, as run_blocking calls it) ----

struct Job {
    gp: GenParams,
    flag: Arc<AtomicBool>,
    progress: Arc<Progress>,
    first: Arc<AtomicBool>,
}

fn job(gp: GenParams) -> Job {
    let flag = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(Progress::default());
    Job {
        gp: GenParams {
            cancel: Some(flag.clone()),
            progress: Some(progress.clone()),
            ..gp
        },
        flag,
        progress,
        first: Default::default(),
    }
}

fn spawn_job(
    st: &Arc<State<StubEngine>>,
    text: &str,
    j: &Job,
) -> std::thread::JoinHandle<Result<GenResult, GenError>> {
    let st = st.clone();
    let prompt = st.tok.encode(text, false, false);
    let mut gp = j.gp.clone();
    let first = j.first.clone();
    std::thread::spawn(move || {
        let queued = Instant::now();
        crate::fair::run_sliced(&st, &prompt, &mut gp, queued, &mut |_| {
            first.store(true, Ordering::SeqCst);
            true
        })
    })
}

fn wait_for(what: &str, f: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "timed out waiting for {what}"
        );
        std::thread::sleep(Duration::from_micros(200));
    }
}

fn alone_text(text: &str, gp: &GenParams) -> GenResult {
    let tok = synthetic_tokenizer();
    let mut w = stub_worker(&tok, 8192);
    w.generate(&tok, &tok.encode(text, false, false), gp, &mut |_| true)
        .unwrap()
}

#[test]
fn cancel_while_suspended_ends_that_run_only_and_its_session_still_works() {
    let st = fair_state(30, 2000, 1 << 30);
    let a = job(greedy(200));
    let b = job(greedy(150));
    let ha = spawn_job(&st, PA, &a);
    wait_for("A's first token", || a.first.load(Ordering::SeqCst));
    let hb = spawn_job(&st, PB, &b);
    wait_for("A suspended", || {
        a.progress.suspended.load(Ordering::SeqCst)
    });
    a.flag.store(true, Ordering::SeqCst);
    assert!(
        matches!(ha.join().unwrap(), Err(GenError::Cancelled)),
        "A ends cancelled"
    );
    let rb = hb.join().unwrap().unwrap();
    let cb = alone_text(PB, &greedy(150));
    assert_eq!((rb.tokens, rb.text), (cb.tokens, cb.text), "B unaffected");
    assert_eq!(st.fair.suspended(), 0);
    // the next request on A's session works and reuses its prompt (the held session was parked)
    let next = job(greedy(10));
    let r = spawn_job(&st, PA, &next).join().unwrap().unwrap();
    let pa = st.tok.encode(PA, false, false);
    assert_eq!(r.timings.cache_n, pa.len() - 1, "A's prompt found again");
    assert_eq!(r.tokens, alone_text(PA, &greedy(10)).tokens);
}

#[test]
fn sigterm_cancels_a_suspended_run_and_frees_the_worker() {
    let st = fair_state(30, 2000, 1 << 30);
    let a = job(greedy(300));
    let b = job(greedy(300));
    let ha = spawn_job(&st, PA, &a);
    wait_for("A's first token", || a.first.load(Ordering::SeqCst));
    let hb = spawn_job(&st, PB, &b);
    wait_for("A suspended", || {
        a.progress.suspended.load(Ordering::SeqCst)
    });
    // what main.rs's SIGTERM thread does
    let t0 = Instant::now();
    if let Some(f) = st.running.lock().unwrap().take() {
        f.store(true, Ordering::SeqCst);
    }
    st.fair.cancel_all();
    let w = loop {
        if let Ok(w) = st.worker.try_lock() {
            if st.fair.suspended() == 0 {
                break w;
            }
        }
        assert!(
            t0.elapsed() < Duration::from_secs(1),
            "the shutdown write gets the worker within 1 s"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    drop(w);
    assert!(
        matches!(ha.join().unwrap(), Err(GenError::Cancelled)),
        "the suspended run is cancelled"
    );
    assert!(
        matches!(hb.join().unwrap(), Err(GenError::Cancelled)),
        "the running one too"
    );
    assert!(st.fair.order().is_empty());
}

#[test]
fn an_ephemeral_request_runs_while_another_is_suspended() {
    let st = fair_state(30, 2000, 1 << 30);
    let a = job(greedy(200));
    let e = job(GenParams {
        ephemeral: true,
        ..greedy(30)
    });
    let ha = spawn_job(&st, PA, &a);
    wait_for("A's first token", || a.first.load(Ordering::SeqCst));
    let he = spawn_job(&st, "C: a request from another device.", &e);
    let re = he.join().unwrap().unwrap();
    let ce = alone_text(
        "C: a request from another device.",
        &GenParams {
            ephemeral: true,
            ..greedy(30)
        },
    );
    assert_eq!(
        (re.tokens, re.text, re.timings.cache_n),
        (ce.tokens, ce.text, 0)
    );
    let ra = ha.join().unwrap().unwrap();
    let ca = alone_text(PA, &greedy(200));
    assert_eq!((ra.tokens, ra.text), (ca.tokens, ca.text));
    assert!(
        ra.timings.fair_yields >= 1,
        "A gave the engine to the ephemeral request"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_endpoints_refuse_while_a_generation_is_suspended() {
    let dir = crate::tests_state::tmp("fair-clear");
    let tok = synthetic_tokenizer();
    let st = {
        let mut w = stub_worker(&tok, 8192);
        w.states = Some(crate::tests_state::open(
            &dir,
            crate::tests_state::ident("sha256:m", ""),
            1 << 40,
        ));
        let cfg = Config {
            model_name: "synthetic".into(),
            model_path: "/m.gguf".into(),
            chat_opts: Default::default(),
            fingerprint: "oxe-test".into(),
            slots: true,
            idle_reprefill: false,
            engine_status: None,
        };
        let st = State::new(cfg, tok, w).unwrap();
        st.fair.set_quantum_ms(10);
        Arc::new(st)
    };
    let pa = st.tok.encode(PA, false, false);
    let gp = greedy(60);
    // suspend a run by hand, the way the driver does (counted with the worker held)
    let mut sr = {
        let mut w = st.worker.lock().unwrap();
        let mut sr = w.generate_start(&pa, &gp).unwrap();
        assert!(w
            .generate_slice(&st.tok, &pa, &gp, &mut sr, &mut |_| true, &mut || true)
            .unwrap()
            .is_none());
        st.fair.note_suspend();
        sr
    };
    for (path, b) in [
        ("/state-cache/clear", json!({"scope": "all"})),
        ("/state-cache/budget", json!({"bytes": 1 << 30})),
    ] {
        match handle(
            st.clone(),
            "POST",
            path,
            "",
            &serde_json::to_vec(&b).unwrap(),
        )
        .await
        {
            Reply::Json(c, j) => assert_eq!(c, 409, "{path} while suspended: {j}"),
            _ => panic!("{path}: not JSON"),
        }
    }
    // the run goes on and ends exactly; then the clear is accepted
    let r = {
        let mut w = st.worker.lock().unwrap();
        st.fair.note_resume();
        let mut r = None;
        while r.is_none() {
            r = w
                .generate_slice(&st.tok, &pa, &gp, &mut sr, &mut |_| true, &mut || false)
                .unwrap();
        }
        r.unwrap()
    };
    assert_eq!(r.tokens, alone_text(PA, &gp).tokens);
    assert_eq!(r.timings.fair_yields, 1);
    match handle(
        st.clone(),
        "POST",
        "/state-cache/clear",
        "",
        &serde_json::to_vec(&json!({"scope": "all"})).unwrap(),
    )
    .await
    {
        Reply::Json(c, j) => assert_eq!(c, 200, "{j}"),
        _ => panic!("not JSON"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn post_reply_jobs_skip_when_another_generation_waits_and_run_after_a_slice_with_nobody_queued() {
    let st = fair_state(20, 2000, 1 << 30);
    let req_a = st.preempt_idle();
    let a = job(GenParams {
        req_no: Some(req_a),
        ..greedy(150)
    });
    let ha = spawn_job(&st, PA, &a);
    wait_for("A's first token", || a.first.load(Ordering::SeqCst));
    let req_b = st.preempt_idle();
    let b = job(GenParams {
        req_no: Some(req_b),
        ..greedy(5)
    });
    let rb = spawn_job(&st, PB, &b).join().unwrap().unwrap();
    assert_eq!(rb.timings.fair_yields, 0, "B ran within one slice");
    let ra = ha.join().unwrap().unwrap();
    assert!(ra.timings.fair_yields >= 1);
    // B ended while A waited: B's jobs never run (A's resume wrote B's state when it put it away)
    assert_eq!(st.fair.post_reply_req(req_b), crate::fair::SKIP);
    // A ended with nobody queued: its jobs compare the count at its end, not its stale number
    assert_eq!(st.fair.post_reply_req(req_a), req_b);
    assert_eq!(
        st.fair.post_reply_req(req_a),
        req_b,
        "both jobs of the request get the same answer"
    );
}

/// Over HTTP with a state store: B ends while A is suspended, A's resume writes B's state (B's hint),
/// A's own post-reply write runs at its end (A's hint); no state is ever written under the other
/// session's hint.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn state_writes_keep_each_session_under_its_own_hint() {
    let dir = crate::tests_state::tmp("fair-hints");
    let tok = synthetic_tokenizer();
    let st = {
        let mut w = stub_worker(&tok, 8192);
        w.engine.decode_sleep = Duration::from_millis(2);
        w.states = Some(crate::tests_state::open(
            &dir,
            crate::tests_state::ident("sha256:m", ""),
            1 << 40,
        ));
        let cfg = Config {
            model_name: "synthetic".into(),
            model_path: "/m.gguf".into(),
            chat_opts: Default::default(),
            fingerprint: "oxe-test".into(),
            slots: true,
            idle_reprefill: false,
            engine_status: None,
        };
        let st = State::new(cfg, tok, w).unwrap();
        st.fair.set_quantum_ms(20);
        Arc::new(st)
    };
    // prompts past the store's 256-token minimum
    let pa = format!("A{}", " the long session".repeat(20));
    let pb = format!("B{}", " the short session".repeat(20));
    let go = |p: String, n: i64, hint: &'static str| {
        let st = st.clone();
        async move {
            let b = serde_json::to_vec(&json!({"prompt": p, "n_predict": n, "temperature": 0}))
                .unwrap();
            match crate::service::handle_session(
                st,
                "POST",
                "/completion",
                "",
                &b,
                Some(hint.into()),
            )
            .await
            {
                Reply::Json(200, j) => j,
                Reply::Json(c, j) => panic!("{hint}: {c} {j}"),
                _ => panic!("{hint}: a stream"),
            }
        }
    };
    let ha = tokio::spawn(go(pa.clone(), 150, "sa"));
    let t0 = Instant::now();
    while st.worker.try_lock().is_ok() || st.fair.order().is_empty() {
        assert!(t0.elapsed() < Duration::from_secs(30));
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    tokio::time::sleep(Duration::from_millis(30)).await; // A is decoding
    let jb = go(pb.clone(), 20, "sb").await;
    let ja = ha.await.unwrap();
    assert!(
        ja["timings"]["fair_yields"].as_u64().unwrap_or(0) >= 1,
        "A yielded: {}",
        ja["timings"]
    );
    let (ta, tb) = (
        st.tok.encode(&pa, false, false),
        st.tok.encode(&pb, false, false),
    );
    // the cache holds the prompt and every generated token but the last (never fed back)
    let (na, nb) = (ta.len() + 149, tb.len() + 19);
    let t0 = Instant::now();
    let want = |h: &str| Some(crate::state_store::hint_hash(h));
    loop {
        let found = {
            let w = st.worker.lock().unwrap();
            let s = w.states.as_ref().unwrap();
            // whose conversation a manifest holds: its first block's chain hash
            let (ha0, hb0) = (
                crate::state_store::chain(&s.ident.hash, &ta)[0].clone(),
                crate::state_store::chain(&s.ident.hash, &tb)[0].clone(),
            );
            let of = |m: &crate::state_store::Manifest| m.blocks.first().map(|b| b.0.clone());
            for m in s.manifests.values() {
                if of(m) == Some(ha0.clone()) {
                    assert_eq!(
                        m.hint,
                        want("sa"),
                        "A's state is written under A's hint only"
                    );
                }
                if of(m) == Some(hb0.clone()) {
                    assert_eq!(
                        m.hint,
                        want("sb"),
                        "B's state is written under B's hint only"
                    );
                }
            }
            let has = |n: usize, h: &str, b0: &str| {
                s.manifests
                    .values()
                    .any(|m| m.n_tokens == n && m.hint == want(h) && of(m).as_deref() == Some(b0))
            };
            (has(na, "sa", &ha0), has(nb, "sb", &hb0))
        };
        if found == (true, true) {
            break;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "written: A {} B {} (B {} tokens, A {})",
            found.0,
            found.1,
            nb,
            na
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(jb["stop_type"], "limit");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- the mini model: real numerics (KV, recurrent and MTP state through hold / restore) ----

#[cfg(target_arch = "aarch64")] // the model engine (x86_64 builds the stub engine only)
#[test]
fn mini_model_interleaved_equals_alone_bitwise_with_mtp_off_and_on() {
    use crate::tests_state_v013::{mini_dir, mini_worker};
    let dir = mini_dir("v020-fair");
    let tok = crate::tests_echo::synth_tokenizer();
    let pa: Vec<u32> = (0..150).map(|i| (i * 13 + 7) % 64).collect();
    let pb: Vec<u32> = (0..90).map(|i| (i * 5 + 3) % 64).collect();
    let pc: Vec<u32> = (0..70).map(|i| (i * 11 + 1) % 64).collect();
    let sampled = GenParams {
        n_predict: 20,
        sampling: SamplingParams {
            temperature: 0.9,
            seed: 7,
            ..Default::default()
        },
        n_probs: 2,
        ..Default::default()
    };
    for depth in [0usize, 2] {
        let jobs = vec![
            (pa.clone(), greedy(28)),
            (
                pb.clone(),
                GenParams {
                    n_probs: 2,
                    ..greedy(24)
                },
            ),
            (pc.clone(), sampled.clone()),
        ];
        // the next turn of each session: greedy (MTP drafts at depth 2) from the end of its reply
        let next = |p: &[u32], o: &Out| {
            let mut n = p.to_vec();
            n.extend(&o.tokens);
            n.extend([5u32, 9, 17, 33]);
            n
        };
        let ctrl: Vec<(Out, Out)> = jobs
            .iter()
            .map(|(p, g)| {
                let mut w = mini_worker(&dir, depth, None);
                let o = alone(&mut w, &tok, p, g, true);
                let f = alone(&mut w, &tok, &next(p, &o), &greedy(16), true);
                (o, f)
            })
            .collect();
        let mut w = mini_worker(&dir, depth, None);
        w.cache.park_budget = 1 << 30;
        w.cache.park_min = 1;
        let (got, yields) = interleaved(&mut w, &tok, &jobs, true);
        for (i, (g, (c, _))) in got.iter().zip(&ctrl).enumerate() {
            assert_eq!(g.tokens, c.tokens, "depth {depth} job {i}: tokens");
            assert_eq!(g.probs, c.probs, "depth {depth} job {i}: probability rows");
            assert_eq!(
                g.streamed, c.streamed,
                "depth {depth} job {i}: streamed text and rows"
            );
            // The state bytes: KV, recurrent and the MTP driver export. One exception: a sampled
            // request with MTP on decodes through `decode`, which leaves MTP catch-up pairs pending;
            // the MTP KV rows of those pending positions are written by the catch-up before anything
            // reads them and until then hold whatever an earlier sequence left in that memory (a
            // fresh engine: zeros). The same request after another one, unsliced, differs from a
            // fresh engine there too, so those bytes are no exactness signal: the next turn below is.
            if depth == 0 || jobs[i].1.sampling.temperature <= 0.0 {
                assert!(
                    g.state == c.state,
                    "depth {depth} job {i}: engine state bytes (KV, recurrent, MTP driver) differ"
                );
            }
            assert_eq!(
                g.draft, c.draft,
                "depth {depth} job {i}: drafted / accepted"
            );
            assert!(yields[i] > 0, "depth {depth} job {i}: yielded");
            if depth > 0 && i == 0 {
                assert!(c.draft.0 > 0, "MTP drafted on the greedy fast path");
            }
        }
        for (i, ((p, _), (c, cf))) in jobs.iter().zip(&ctrl).enumerate() {
            let f = alone(&mut w, &tok, &next(p, c), &greedy(16), true);
            assert_eq!(
                f.tokens, cf.tokens,
                "depth {depth} job {i}: next turn tokens"
            );
            assert_eq!(f.draft, cf.draft, "depth {depth} job {i}: next turn drafted / accepted (the MTP state came back exact)");
            assert!(
                f.state == cf.state,
                "depth {depth} job {i}: state bytes after the next turn"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// MTP acceptance guard wired into the decode loop (real numerics): a guard that trips on the
/// first rejected draft changes how many tokens are drafted, never the tokens, the probability
/// rows or the reply; a guard that can never trip changes nothing at all.
#[cfg(target_arch = "aarch64")]
#[test]
fn mini_model_mtp_guard_changes_drafting_only() {
    use crate::tests_state_v013::{mini_dir, mini_worker};
    let dir = mini_dir("v020-guard");
    let tok = crate::tests_echo::synth_tokenizer();
    let prompt: Vec<u32> = (0..150).map(|i| (i * 13 + 7) % 64).collect();
    let g = greedy(96);
    let mut w = mini_worker(&dir, 2, None);
    let ctrl = alone(&mut w, &tok, &prompt, &g, true);
    assert!(ctrl.draft.0 > 0, "MTP drafted in the control");
    // never trips: acceptance can not be below a ratio no window can miss
    let mut w = mini_worker(&dir, 2, None);
    w.mtp_guard = Some(crate::mtp_guard::MtpGuard::new(
        f32::MIN_POSITIVE,
        1 << 20,
        8,
    ));
    let same = alone(&mut w, &tok, &prompt, &g, true);
    assert_eq!(
        (&same.tokens, same.draft),
        (&ctrl.tokens, ctrl.draft),
        "a guard that never trips changes nothing"
    );
    assert_eq!(w.mtp_guard_trips, 0);
    // strict: every window of 2 drafted tokens must be fully accepted, else 8 tokens without drafts
    let mut w = mini_worker(&dir, 2, None);
    w.mtp_guard = Some(crate::mtp_guard::MtpGuard::new(1.0, 2, 8));
    let strict = alone(&mut w, &tok, &prompt, &g, true);
    assert_eq!(strict.tokens, ctrl.tokens, "tokens");
    assert_eq!(strict.streamed, ctrl.streamed, "streamed text");
    if ctrl.draft.1 < ctrl.draft.0 {
        // the control had rejected drafts, so the strict guard tripped and drafted less
        assert!(w.mtp_guard_trips > 0, "tripped");
        assert!(
            strict.draft.0 < ctrl.draft.0,
            "fewer drafts: {:?} vs {:?}",
            strict.draft,
            ctrl.draft
        );
    } else {
        assert_eq!(w.mtp_guard_trips, 0);
        assert_eq!(strict.draft, ctrl.draft);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
