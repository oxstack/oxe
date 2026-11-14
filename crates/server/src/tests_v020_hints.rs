//! v0.2.0 engine hints (HARNESS-RS-PLAN sec. 3; API doc `docs/lab-notes/SERVER.md`, "Hints"):
//! - H6 `max_tokens: 0`: prefill only. Zero generated tokens, no sampling, no ECHO record; the prompt
//!   is the conversation without the assistant opener, so the next request continues it exactly
//!   (`cache_n` = this request's `prompt_tokens`) and generates what it would have without the
//!   prefill.
//! - H4 `X-Oxe-Snapshot: keep`: a checkpoint exactly at the end of the last message, reported as
//!   `timings.snap_pos`; the newest 2 per session; null when none can be kept.
//! - No behaviour change without the hints.
//!
//! Model-free (stub engine, synthetic tokenizer, the Qwen3.x template fixture) plus the tiny
//! synthetic hybrid model (`model::testkit`, no real weights) for the next-turn output check. No
//! prompt text or token ids leave the test.

use crate::cache::{PRIO_SNAP, PRIO_TAIL};
use crate::engine::StubEngine;
use crate::generate::{GenParams, StopType, Worker};
use crate::sampling::SamplingParams;
use crate::service::{handle_hints, Hints, Reply, State};
use crate::tests_echo::synth_tokenizer;
use crate::tests_state_v013::srv;
use serde_json::{json, Value as Json};
use std::sync::Arc;
use std::time::Duration;

const SYS: &str = "You are a test system. Keep answers short.";

fn greedy(n_predict: i64) -> GenParams {
    GenParams { n_predict, sampling: SamplingParams { temperature: 0.0, ..Default::default() }, ..Default::default() }
}

fn req(msgs: &Json, thinking: bool, extra: &[(&str, Json)]) -> Vec<u8> {
    let mut b = json!({"messages": msgs, "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": true}});
    for (k, v) in extra {
        b[*k] = v.clone();
    }
    serde_json::to_vec(&b).unwrap()
}

/// One non-streamed chat request with session "h" and the given snapshot hint.
async fn chat(st: &Arc<State<StubEngine>>, b: &[u8], snap: bool) -> Json {
    match handle_hints(st.clone(), "POST", "/v1/chat/completions", "", b, Some("h".into()), None, false, Hints { snapshot_keep: snap }).await {
        Reply::Json(c, j) => {
            assert_eq!(c, 200, "{j}");
            j
        }
        Reply::Stream(_) => panic!("stream"),
    }
}

/// The `data:` JSON frames of one streamed chat request, and whether it ended with `[DONE]`.
async fn chat_stream(st: &Arc<State<StubEngine>>, b: &[u8], snap: bool) -> (Vec<Json>, bool) {
    let reply = handle_hints(st.clone(), "POST", "/v1/chat/completions", "", b, Some("h".into()), None, false, Hints { snapshot_keep: snap }).await;
    let Reply::Stream(mut rx) = reply else { panic!("expected a stream") };
    let (mut frames, mut done) = (Vec::new(), false);
    while let Some(f) = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await.expect("stream hung") {
        for line in f.lines() {
            let Some(d) = line.strip_prefix("data: ") else { continue };
            if d == "[DONE]" {
                done = true;
            } else {
                frames.push(serde_json::from_str(d).unwrap());
            }
        }
    }
    (frames, done)
}

fn kv(st: &State<StubEngine>) -> Vec<u32> {
    st.worker.lock().unwrap().engine.tokens.clone()
}

fn decodes(st: &State<StubEngine>) -> usize {
    st.worker.lock().unwrap().engine.calls.iter().filter(|c| c.0 == "decode").count()
}

fn u(j: &Json, p: &str) -> usize {
    j.pointer(p).and_then(Json::as_u64).unwrap_or_else(|| panic!("{p} missing in {j}")) as usize
}

/// H6 through the service: an empty reply (finish_reason length, 0 completion tokens, no decode
/// step), the prompt is the conversation without the opener and stays live, no ECHO record; the
/// next ordinary request resumes exactly there and returns what a server without the prefill
/// returns, from the same fed prompt.
#[tokio::test(flavor = "multi_thread")]
async fn h6_prefills_without_generating_and_the_next_turn_resumes_at_its_end() {
    let m1 = json!([{"role":"system","content":SYS},{"role":"user","content":"hi"}]);
    let m2 = json!([{"role":"system","content":SYS},{"role":"user","content":"hi"},{"role":"assistant","content":"Hello there."},{"role":"user","content":"and more"}]);
    for thinking in [false, true] {
        for next in [&m1, &m2] {
            let s = srv(thinking, &["Hello there.", "Second answer."], None);
            let recs0 = s.echo.lock().unwrap().n_records();
            let r0 = chat(&s, &req(&m1, thinking, &[("max_tokens", json!(0))]), false).await;
            let ch = &r0["choices"][0];
            assert_eq!(ch["finish_reason"], "length", "{r0}");
            assert_eq!(ch["message"]["content"], "", "{r0}");
            assert!(ch["message"].get("reasoning_content").is_none() && ch["message"].get("tool_calls").is_none(), "{r0}");
            assert_eq!(u(&r0, "/usage/completion_tokens"), 0);
            assert_eq!(u(&r0, "/timings/predicted_n"), 0);
            assert!(r0["timings"].get("snap_pos").is_none(), "no hint, no snap_pos: {r0}");
            let p = u(&r0, "/usage/prompt_tokens");
            assert_eq!(u(&r0, "/usage/total_tokens"), p);
            assert_eq!(u(&r0, "/usage/prompt_tokens_details/cached_tokens"), u(&r0, "/timings/cache_n"));
            assert_eq!(decodes(&s), 0, "nothing was decoded");
            assert_eq!(s.echo.lock().unwrap().n_records(), recs0, "no ECHO record");
            // the prompt is the conversation itself: the opener a generating request adds is not in it
            let live = kv(&s);
            assert_eq!(live.len(), p, "the prompt stays live, nothing after it");
            let opener = s.tok.encode("<|im_start|>assistant\n", false, true);
            assert!(!live.ends_with(&opener), "no generation opener in an H6 prompt");
            assert!(s.worker.lock().unwrap().cache.snapshot_info().iter().any(|x| x.0 == p), "the prompt end is checkpointed");

            let r1 = chat(&s, &req(next, thinking, &[]), false).await;
            assert_eq!(u(&r1, "/timings/cache_n"), p, "thinking {thinking}: the next turn resumes at the H6 prompt end");
            // the same request on a server that never saw the prefill
            let cold = srv(thinking, &["Hello there.", "Second answer."], None);
            let rc = chat(&cold, &req(next, thinking, &[]), false).await;
            assert_eq!(r1["choices"][0]["message"], rc["choices"][0]["message"], "thinking {thinking}: same reply as without the prefill");
            assert_eq!(r1["choices"][0]["finish_reason"], rc["choices"][0]["finish_reason"]);
            assert_eq!(u(&r1, "/usage/prompt_tokens"), u(&rc, "/usage/prompt_tokens"));
            assert_eq!(u(&r1, "/usage/completion_tokens"), u(&rc, "/usage/completion_tokens"));
            assert_eq!(kv(&s), kv(&cold), "the same prompt was fed and the same tokens generated");
        }
    }
}

/// H6 streamed: role chunk, the finish chunk (length), the usage chunk (0 completion tokens) with
/// timings, `[DONE]`; no content or reasoning delta.
#[tokio::test(flavor = "multi_thread")]
async fn h6_stream_has_no_delta_and_reports_usage() {
    let s = srv(true, &["Hello there."], None);
    let m1 = json!([{"role":"system","content":SYS},{"role":"user","content":"hi"}]);
    let b = req(&m1, true, &[("max_tokens", json!(0)), ("stream", json!(true)), ("stream_options", json!({"include_usage": true}))]);
    let (frames, done) = chat_stream(&s, &b, false).await;
    assert!(done, "ends with [DONE]");
    assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant", "{frames:?}");
    let deltas: Vec<&Json> = frames.iter().filter_map(|f| f["choices"].get(0)).map(|c| &c["delta"]).collect();
    assert!(deltas.iter().all(|d| d.get("content").is_none_or(Json::is_null) && d.get("reasoning_content").is_none()), "{frames:?}");
    let fin: Vec<&Json> = frames.iter().filter_map(|f| f["choices"].get(0)).filter_map(|c| c.get("finish_reason")).filter(|x| !x.is_null()).collect();
    assert_eq!(fin, vec![&json!("length")]);
    let last = frames.last().unwrap();
    assert_eq!(u(last, "/usage/completion_tokens"), 0);
    assert_eq!(u(last, "/usage/prompt_tokens"), kv(&s).len());
    assert_eq!(u(last, "/timings/predicted_n"), 0);
    assert_eq!(decodes(&s), 0);
}

/// Worker level: n_predict 0 samples nothing (no decode call, no token, stop = limit) and leaves the
/// whole prompt live; n_predict 1 still generates exactly one token (only 0 is special).
#[test]
fn h6_on_the_worker_and_the_one_token_limit_unchanged() {
    let tok = synth_tokenizer();
    let prompt: Vec<u32> = (0..150).map(|k| (k * 7 % 250) as u32 + 1).collect();
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![7, 8, 9]), 8);
    w.ubatch = 16;
    let r = w.generate(&tok, &prompt, &greedy(0), &mut |_| panic!("no partial for a prefill")).unwrap();
    assert_eq!((r.n_decoded, r.tokens.len(), r.text.as_str(), r.stop), (0, 0, "", StopType::Limit));
    assert_eq!(w.cache.tokens, prompt);
    assert!(!w.engine.calls.iter().any(|c| c.0 == "decode"));
    assert!(w.cache.snapshot_info().contains(&(prompt.len(), PRIO_TAIL)));
    let mut next = prompt.clone();
    next.extend([11, 12, 13]);
    let r = w.generate(&tok, &next, &greedy(1), &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, prompt.len());
    assert_eq!((r.n_decoded, r.tokens.as_slice()), (1, &[7u32][..]), "the scripted reply starts where it would have");
}

/// H6 on the mini model (real numerics, MTP off and on): the next turn after a prefill-only request
/// generates the same tokens as without it, and resumes at the H6 prompt end.
#[cfg(target_arch = "aarch64")] // the model engine (O-174: x86_64 builds the stub engine only)
#[test]
fn h6_next_turn_output_unchanged_on_the_mini_model() {
    use crate::tests_state_v013::{mini_dir, mini_worker, run_mini};
    let dir = mini_dir("v020-h6");
    let p1: Vec<u32> = (0..260).map(|i| (i * 13 + 7) % 64).collect();
    let mut p2 = p1.clone();
    p2.extend((0..40).map(|i| (i * 5 + 3) % 64));
    for depth in [0usize, 2] {
        let mut w = mini_worker(&dir, depth, None);
        let gp = |n: i64| GenParams { first_user: Some(70), ..greedy(n) };
        let pre = run_mini(&mut w, &p1, &gp(0));
        assert!(pre.0.is_empty(), "depth {depth}: H6 generated nothing");
        let with = run_mini(&mut w, &p2, &gp(8));
        assert_eq!(with.3, p1.len(), "depth {depth}: resumed at the H6 prompt end");
        let mut cold = mini_worker(&dir, depth, None);
        let without = run_mini(&mut cold, &p2, &gp(8));
        assert_eq!(with.0, without.0, "depth {depth}: the next turn's tokens");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The arithmetic an engine ran: every ubatch it evaluated (a prefill call is chunked by the ubatch,
/// 16 in these tests, from the call's start, so splitting a call at a ubatch boundary of it changes
/// nothing) and every decode step. Snapshot / restore bookkeeping is left out.
fn arith(calls: &[(&'static str, usize, usize)]) -> Vec<(&'static str, usize, usize)> {
    let mut v = Vec::new();
    for &(k, pos, n) in calls {
        match k {
            "prefill" => {
                let mut i = 0;
                while i < n {
                    v.push(("ubatch", pos + i, (n - i).min(16)));
                    i += 16;
                }
            }
            "decode" => v.push((k, pos, n)),
            _ => {}
        }
    }
    v
}

/// H4 truth table (last message user / tool / assistant, thinking off / on), service level.
/// - With H6 the point is exactly the end of the last message = `prompt_tokens`, with no extra call.
/// - On a generating request `snap_pos` is the deepest ubatch-grid point (16 here) at or before the
///   end of the last message and not before its start, else null with a reason; the engine calls
///   (prefill and decode) are exactly those of the request without the hint, and so is the reply.
/// - A generating request ending with an assistant message answers with a new turn (F9); the H6
///   form takes one as a finished turn, and the next message continues it exactly.
#[tokio::test(flavor = "multi_thread")]
async fn h4_snap_pos_truth_table() {
    let call = json!([{"id":"c1","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]);
    let tools = json!([{"type":"function","function":{"name":"read","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]);
    let long: String = (0..12).map(|i| format!("Line {i:02} of a longer user message. ")).collect();
    let cases = [
        ("user", json!([{"role":"system","content":SYS},{"role":"user","content":"hi"}])),
        ("long user", json!([{"role":"system","content":SYS},{"role":"user","content":long}])),
        ("tool", json!([{"role":"system","content":SYS},{"role":"user","content":"hi"},{"role":"assistant","content":"","tool_calls":call},{"role":"tool","tool_call_id":"c1","content":long}])),
        ("assistant", json!([{"role":"system","content":SYS},{"role":"user","content":"hi"},{"role":"assistant","content":"Partial answer."}])),
    ];
    let mut kept = 0;
    for thinking in [false, true] {
        for (last, msgs) in &cases {
            let with_tools = |extra: &[(&str, Json)]| {
                let mut e = vec![("tools", tools.clone())];
                e.extend(extra.iter().cloned());
                req(msgs, thinking, &e)
            };
            // H4 + H6: exactly the end of the last message; the same calls as H6 alone
            let oracle = srv(thinking, &["Done."], None);
            let h6 = chat(&oracle, &with_tools(&[("max_tokens", json!(0))]), false).await;
            let end = u(&h6, "/usage/prompt_tokens");
            let both = srv(thinking, &["Done."], None);
            let rb = chat(&both, &with_tools(&[("max_tokens", json!(0))]), true).await;
            assert_eq!(u(&rb, "/timings/snap_pos"), end, "{last}/{thinking}");
            assert!(both.worker.lock().unwrap().cache.snapshot_info().contains(&(end, PRIO_SNAP)));
            assert_eq!(arith(&both.worker.lock().unwrap().engine.calls), arith(&oracle.worker.lock().unwrap().engine.calls), "{last}/{thinking}");
            if *last == "assistant" {
                // F9 (OpenAI v1): a generating request ending with an assistant message takes it as
                // history and answers with a new turn (the snapshot hint does not change that)
                let s = srv(thinking, &["Done."], None);
                let r = handle_hints(s, "POST", "/v1/chat/completions", "", &with_tools(&[]), None, None, false, Hints { snapshot_keep: true }).await;
                match r {
                    Reply::Json(200, j) => {
                        // thinking on: the stub's text lands in the open think block of the new turn
                        let m = &j["choices"][0]["message"];
                        let text = if thinking { &m["reasoning_content"] } else { &m["content"] };
                        assert_eq!(text, "Done.", "{last}/{thinking}: a new turn: {j}");
                    }
                    Reply::Json(c, j) => panic!("{last}/{thinking}: {c} {j}"),
                    Reply::Stream(_) => panic!("stream"),
                }
                let text = both.tok.decode(&kv(&both), true);
                assert!(text.trim_end().ends_with("Partial answer.<|im_end|>"), "{last}/{thinking}: a closed turn: {text:?}");
                let mut next = msgs.as_array().unwrap().clone();
                next.push(json!({"role":"user","content":"go on"}));
                let r1 = chat(&both, &req(&Json::Array(next), thinking, &[("tools", tools.clone())]), false).await;
                assert_eq!(u(&r1, "/timings/cache_n"), end, "{last}/{thinking}: the next message continues it");
                continue;
            }
            // a generating request: hint on vs off
            let on = srv(thinking, &["Done."], None);
            let r = chat(&on, &with_tools(&[]), true).await;
            let off = srv(thinking, &["Done."], None);
            let r0 = chat(&off, &with_tools(&[]), false).await;
            assert_eq!(r["choices"], r0["choices"], "{last}/{thinking}: same reply");
            assert_eq!(arith(&on.worker.lock().unwrap().engine.calls), arith(&off.worker.lock().unwrap().engine.calls), "{last}/{thinking}: same engine calls");
            let fed = kv(&on);
            assert_eq!(fed[..end], kv(&oracle)[..], "{last}/{thinking}: the end of the last message is the H6 prompt end");
            let start = s_last_start(&on, &fed[..end]);
            let floor = end / 16 * 16;
            let want = (floor >= start && floor > 0).then_some(floor);
            match want {
                Some(w) => {
                    assert_eq!(u(&r, "/timings/snap_pos"), w, "{last}/{thinking}: the grid floor of the end");
                    assert!(on.worker.lock().unwrap().cache.snapshot_info().contains(&(w, PRIO_SNAP)));
                    kept += 1;
                }
                None => {
                    assert!(r["timings"]["snap_pos"].is_null(), "{last}/{thinking}: {r}");
                    assert_eq!(r["timings"]["snap_reason"], "no_boundary_in_last_message", "{last}/{thinking}");
                }
            }
        }
    }
    assert!(kept >= 2, "the long cases keep a point");
}

/// The start (`<|im_start|>`) of the last message in a fed prompt that ends with that message.
fn s_last_start(s: &State<StubEngine>, toks: &[u32]) -> usize {
    let d = s.tok.encode("<|im_start|>", false, true);
    toks.windows(d.len()).rposition(|w| w == &d[..]).unwrap()
}

/// H4 bound and eviction (worker level): the newest 2 snap points per session are kept, a third
/// evicts the least recently used one; a request that cuts back to a kept point resumes exactly
/// there, one cut back to an evicted point resumes below it. A point inside the reused prefix is an
/// existing checkpoint in the last message or none (null, `inside_reused_prefix`); no slot: null,
/// `no_slot`.
#[test]
fn h4_keeps_the_newest_two_per_session() {
    let tok = synth_tokenizer();
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![7, 8, 9]), 8);
    w.ubatch = 16;
    let snap = |at: usize, start: usize| GenParams { snap_keep: true, snap_at: Some(at), msg_starts: vec![start], ..greedy(3) };
    let mut prompt: Vec<u32> = (0..96).map(|k| (k * 7 % 250) as u32 + 1).collect();
    let mut points = Vec::new();
    for turn in 0..3u32 {
        // the last message: 37 tokens from the live end, so a grid point (live end + 32) lies in it
        let start = prompt.len();
        prompt.extend((0..37).map(|k| (k * 3 + turn * 17) % 250 + 1));
        let m = prompt.len();
        prompt.extend([251, 252, 253]);
        let r = w.generate(&tok, &prompt, &snap(m, start), &mut |_| true).unwrap();
        let want = if turn == 0 { m / 16 * 16 } else { start + 32 };
        assert_eq!(r.timings.snap_pos, Some(want), "turn {turn}");
        points.push(want);
        prompt = w.cache.tokens.clone();
    }
    let kept: Vec<usize> = w.cache.snapshot_info().iter().filter(|x| x.1 == PRIO_SNAP).map(|x| x.0).collect();
    assert_eq!(kept, points[1..].to_vec(), "the newest two");
    let mut cut = prompt[..points[1]].to_vec();
    cut.extend([5, 6, 7, 8]);
    let r = w.generate(&tok, &cut, &greedy(2), &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, points[1], "cut back to a kept point");
    assert!(!r.timings.snap_keep && r.timings.snap_pos.is_none() && r.timings.snap_reason.is_none());
    let mut cut = prompt[..points[0]].to_vec();
    cut.extend([5, 6, 7, 8]);
    let r = w.generate(&tok, &cut, &greedy(2), &mut |_| true).unwrap();
    assert!(r.timings.cache_n < points[0], "an evicted point: {} vs {}", r.timings.cache_n, points[0]);
    // inside the reused prefix, no checkpoint in the last message: none
    let base = w.cache.tokens.clone();
    let mut next = base.clone();
    next.extend([9, 9, 9]);
    let have = w.cache.snapshot_positions();
    let at = (2..base.len()).rev().find(|&x| !have.contains(&x) && !have.contains(&(x - 1))).unwrap();
    let r = w.generate(&tok, &next, &snap(at, at - 1), &mut |_| true).unwrap();
    assert_eq!((r.timings.snap_pos, r.timings.snap_reason), (None, Some("inside_reused_prefix")));
    // ... with one there: promoted, no new snapshot
    let tail = next.len() - 1; // the previous request's n-1 checkpoint
    assert!(w.cache.snapshot_positions().contains(&tail));
    let mut again = w.cache.tokens.clone();
    again.extend([4, 4]);
    let snaps0 = w.engine.calls.len();
    let r = w.generate(&tok, &again, &snap(tail, tail - 1), &mut |_| true).unwrap();
    assert_eq!(r.timings.snap_pos, Some(tail), "an existing checkpoint takes the class");
    assert!(!w.engine.calls[snaps0..].iter().any(|c| c.0 == "snapshot" && c.1 == tail), "promoted, not copied");
    // no slot for the class
    w.cache.max_snap = 0;
    let mut more = w.cache.tokens.clone();
    let start = more.len();
    more.extend((0..40).map(|k| k + 1));
    let r = w.generate(&tok, &more, &snap(start + 37, start), &mut |_| true).unwrap();
    assert_eq!((r.timings.snap_pos, r.timings.snap_reason), (None, Some("no_slot")));
    let j = crate::api::timings_json(&r.timings);
    assert!(j.get("snap_pos").is_some_and(Json::is_null) && j["snap_reason"] == "no_slot", "{j}");
}

/// The hint never changes the arithmetic (worker level, stub): for prompts of many lengths and
/// end-of-message positions, the engine's prefill / decode calls and the output are identical with
/// and without it, over a three-request session.
#[test]
fn h4_adds_no_split_anywhere() {
    let tok = synth_tokenizer();
    for opener in [1usize, 2, 3, 5] {
        for len in [20usize, 33, 47, 64, 95, 130] {
            let run = |hint: bool| {
                let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![7, 8, 9]), 8);
                w.ubatch = 16;
                let mut prompt: Vec<u32> = (0..len).map(|k| (k * 7 % 250) as u32 + 1).collect();
                let mut outs = Vec::new();
                for turn in 0..3usize {
                    let start = if turn == 0 { 0 } else { prompt.len() };
                    prompt.extend((0..len / 2 + turn).map(|k| (k * 5 % 250) as u32 + 2));
                    let m = prompt.len();
                    prompt.extend((0..opener).map(|k| 251 + k as u32));
                    let gp = GenParams { snap_keep: hint, snap_at: hint.then_some(m), msg_starts: vec![start], ..greedy(4) };
                    outs.push(w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap().tokens);
                    prompt = w.cache.tokens.clone();
                }
                (outs, arith(&w.engine.calls))
            };
            assert_eq!(run(true), run(false), "opener {opener} len {len}");
        }
    }
}

/// On the mini model (real numerics, MTP off and on): a session with the hint generates the same
/// tokens and leaves the same state bytes (KV + recurrent + MTP) and drafts as without it, and the
/// hint keeps a point inside the last message.
#[cfg(target_arch = "aarch64")] // the model engine (O-174: x86_64 builds the stub engine only)
#[test]
fn h4_output_and_state_bitwise_unchanged_on_the_mini_model() {
    use crate::tests_state_v013::{mini_dir, mini_worker, run_mini};
    let dir = mini_dir("v020-h4");
    for depth in [0usize, 2] {
        let mut runs = Vec::new();
        for hint in [true, false] {
            let mut w = mini_worker(&dir, depth, None);
            let mut out = Vec::new();
            let mut prompt: Vec<u32> = (0..250).map(|i| (i * 13 + 7) % 64).collect();
            let mut m = 250;
            let mut start = 130;
            for turn in 0..3u32 {
                let gp = GenParams { first_user: Some(70), msg_starts: vec![0, 70, start], snap_keep: hint, snap_at: hint.then_some(m), ..greedy(6) };
                prompt.extend([60u32, 61, 62]); // the opener
                out.push(run_mini(&mut w, &prompt, &gp));
                if hint {
                    assert!(w.cache.snapshot_info().iter().any(|x| x.1 == PRIO_SNAP), "depth {depth} turn {turn}: a point is kept");
                }
                prompt = w.cache.tokens.clone();
                start = prompt.len();
                prompt.extend((0..90).map(|i| (i * 5 + turn * 11 + 3) % 64));
                m = prompt.len();
            }
            runs.push(out);
        }
        for (i, (a, b)) in runs[0].iter().zip(&runs[1]).enumerate() {
            assert_eq!(a.0, b.0, "depth {depth} request {i}: tokens");
            assert!(a.1 == b.1, "depth {depth} request {i}: state bytes differ");
            assert_eq!(a.2, b.2, "depth {depth} request {i}: drafts");
            assert_eq!(a.3, b.3, "depth {depth} request {i}: cache_n");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The direct check on the mini model (real numerics, MTP off and on): the same request with and
/// without the hint (through the service's position rule: the end of a long last message, its grid
/// floor kept) gives bit-identical logits at every generated step, the same tokens and the same state
/// bytes after the turn, on a cold server and on a continued session.
#[cfg(target_arch = "aarch64")] // the model engine (O-174: x86_64 builds the stub engine only)
#[test]
fn h4_logits_bitwise_identical_on_the_mini_model() {
    use crate::tests_state_v013::{mini_dir, mini_worker, state_bytes};
    let dir = mini_dir("v020-h4-logits");
    let tok = synth_tokenizer();
    for depth in [0usize, 2] {
        let mut runs = Vec::new();
        for hint in [true, false] {
            let mut w = mini_worker(&dir, depth, None);
            w.trace_logits = Some(Vec::new());
            let mut seen = Vec::new();
            let mut prompt: Vec<u32> = (0..250).map(|i| (i * 13 + 7) % 64).collect();
            let (mut start, mut m) = (130usize, 250usize);
            for turn in 0..2u32 {
                prompt.extend([60u32, 61, 62]);
                // n_probs > 0: the sampler reads (and the trace records) every full logits row
                let gp = GenParams { first_user: Some(70), msg_starts: vec![0, 70, start], snap_keep: hint, snap_at: hint.then_some(m), n_probs: 4, ..greedy(8) };
                let r = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
                if hint {
                    assert!(r.timings.snap_pos.is_some_and(|c| c >= start && c <= m), "depth {depth} turn {turn}: {:?}", r.timings.snap_pos);
                }
                let rows = w.trace_logits.replace(Vec::new()).unwrap();
                assert_eq!(rows.len(), r.tokens.len(), "one row per generated step");
                let bits: Vec<Vec<u32>> = rows.iter().map(|row| row.iter().map(|x| x.to_bits()).collect()).collect();
                seen.push((r.tokens.clone(), r.text.clone(), bits, state_bytes(&mut w.engine)));
                prompt = w.cache.tokens.clone();
                start = prompt.len();
                prompt.extend((0..90).map(|i| (i * 5 + turn * 11 + 3) % 64));
                m = prompt.len();
            }
            runs.push(seen);
        }
        for (i, (a, b)) in runs[0].iter().zip(&runs[1]).enumerate() {
            assert_eq!(a.0, b.0, "depth {depth} turn {i}: tokens");
            assert_eq!(a.1, b.1, "depth {depth} turn {i}: generated bytes");
            assert!(a.2 == b.2, "depth {depth} turn {i}: logits differ (bitwise)");
            assert!(a.3 == b.3, "depth {depth} turn {i}: state bytes differ");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A session parked by another conversation keeps its snap points: the bound is per session.
#[test]
fn h4_points_travel_with_a_parked_session() {
    let tok = synth_tokenizer();
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![7, 8, 9]), 8);
    w.ubatch = 16;
    w.cache.park_budget = 1 << 30;
    w.cache.park_min = 16;
    let snap = |at: usize| GenParams { snap_keep: true, snap_at: Some(at), msg_starts: vec![150], ..greedy(2) };
    let mut a: Vec<u32> = (0..192).map(|k| (k * 7 % 250) as u32 + 1).collect();
    a.extend([251, 252, 253]);
    let r = w.generate(&tok, &a, &snap(192), &mut |_| true).unwrap();
    assert_eq!(r.timings.snap_pos, Some(192));
    let mut b: Vec<u32> = (0..192).map(|k| (k * 11 % 250) as u32 + 2).collect();
    b.extend([251, 252, 253]);
    let r = w.generate(&tok, &b, &snap(192), &mut |_| true).unwrap();
    assert_eq!(r.timings.snap_pos, Some(192));
    let mut a2 = a[..192].to_vec();
    a2.extend([3, 3, 3]);
    let r = w.generate(&tok, &a2, &greedy(2), &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, 192, "A resumes at its own snap point");
}

type Seen = (Vec<Json>, Vec<(&'static str, usize, usize)>, Vec<(usize, u8)>);

/// Two turns on a fresh stub server; `hints`: through `handle_hints` with this snapshot hint, else
/// through the pre-v0.2.0 entry point `handle_request`.
async fn two_turns(hints: Option<bool>) -> Seen {
    let m1 = json!([{"role":"system","content":SYS},{"role":"user","content":"hi"}]);
    let m2 = json!([{"role":"system","content":SYS},{"role":"user","content":"hi"},{"role":"assistant","content":"Hello there."},{"role":"user","content":"and more"}]);
    let s = srv(false, &["Hello there.", "Second."], None);
    let mut out = Vec::new();
    for m in [&m1, &m2] {
        let b = req(m, false, &[("max_tokens", json!(16))]);
        let reply = match hints {
            Some(h) => handle_hints(s.clone(), "POST", "/v1/chat/completions", "", &b, Some("h".into()), None, false, Hints { snapshot_keep: h }).await,
            None => crate::service::handle_request(s.clone(), "POST", "/v1/chat/completions", "", &b, Some("h".into()), None).await,
        };
        let Reply::Json(200, r) = reply else { panic!("not a 200 reply") };
        assert_eq!(r["timings"].get("snap_pos").is_some(), hints == Some(true), "{r}");
        out.push(r["choices"][0]["message"].clone());
    }
    let w = s.worker.lock().unwrap();
    (out, w.engine.calls.clone(), w.cache.snapshot_info())
}

/// No hints = no change: without `X-Oxe-Snapshot` and with `max_tokens` > 0 the engine sees exactly
/// the calls of the pre-v0.2.0 entry point, keeps the same checkpoints and the reply has no
/// `snap_pos`; with the hint the replies and the arithmetic are the same. The header parser accepts
/// only `keep`; `/props` lists the hints.
#[tokio::test(flavor = "multi_thread")]
async fn no_hint_no_change() {
    let (a, calls_a, snaps_a) = two_turns(Some(false)).await;
    let (old, calls_old, snaps_old) = two_turns(None).await;
    assert_eq!(a, old);
    assert_eq!(calls_a, calls_old, "same engine calls");
    assert_eq!(snaps_a, snaps_old, "same checkpoints");
    let (b, calls_b, _) = two_turns(Some(true)).await;
    assert_eq!(a, b);
    assert_eq!(arith(&calls_a), arith(&calls_b), "the hint adds no prefill split");
    let s = srv(false, &[], None);
    let Reply::Json(200, props) = handle_hints(s, "GET", "/props", "", b"", None, None, false, Hints::default()).await else { panic!("props") };
    assert_eq!(props["oxe_hints"], json!({"snapshot_keep": 2, "prefill_only": true, "hint_clear": true,
                                          "sampling": ["presence_penalty", "frequency_penalty", "logit_bias"], "count_only": true}));
    // H5: the turn label as the ledger keeps it
    assert_eq!(crate::service::ledger_label("branch_summary"), "branch_summary");
    assert_eq!(crate::service::ledger_label("two words!"), "twowords");
    let mut h = hyper::header::HeaderMap::new();
    assert!(!crate::http::snapshot_keep(&h));
    h.insert("x-oxe-snapshot", "drop".parse().unwrap());
    assert!(!crate::http::snapshot_keep(&h));
    h.insert("x-oxe-snapshot", " Keep ".parse().unwrap());
    assert!(crate::http::snapshot_keep(&h));
}
