//! v0.1.4 session levers (V014-SESSION-LEVERS.md): the user-message-start checkpoint class
//! (PRIO_USER, cache.rs). Model-free (stub engine) plus the tiny synthetic hybrid model
//! (`model::testkit`, no real weights) for the "numerics unchanged" check. No prompt text or token
//! ids leave the test.
//!
//! The shape is a coding agent's: a user turn, a tool loop, a second user message right after the
//! previous reply, a longer tool loop, then the compaction request [system, history to the cut,
//! instruction] that cuts at that second user message (clients that compact keep the last user turns within
//! `compaction.keep` verbatim). In v0.1.3 the point at that user start was a turn point (or, with
//! no grid position between the reply's end and the user start, only the reply's end-of-generation
//! tail point), and the next tool hops evicted it, so the compaction request resumed at an older
//! query point and prefilled again the whole tool loop before the cut.

use crate::cache::{PRIO_QUERY, PRIO_TURN, PRIO_USER};
use crate::engine::StubEngine;
use crate::generate::{GenParams, Worker};
use crate::sampling::SamplingParams;
use crate::tests_echo::synth_tokenizer;

fn greedy(n_predict: i64) -> GenParams {
    GenParams { n_predict, sampling: SamplingParams { temperature: 0.0, ..Default::default() }, ..Default::default() }
}

const SYS: usize = 100;
const INSTR: u32 = 299;

/// `n` filler tokens (1..=250, never the instruction id), varied by `seed`.
fn fill(n: usize, seed: usize) -> Vec<u32> {
    (0..n).map(|k| ((k * 7 + seed * 13) % 250) as u32 + 1).collect()
}

/// One request's message layout: message starts, user starts, last message a user message.
#[derive(Clone)]
struct Layout {
    msg_starts: Vec<usize>,
    user_starts: Vec<usize>,
}

fn params(l: &Layout, query: bool, kind: Option<&str>) -> GenParams {
    let mut gp = greedy(3);
    gp.first_user = Some(SYS);
    gp.msg_starts = l.msg_starts.clone();
    gp.user_starts = l.user_starts.clone();
    gp.query_start = query;
    gp.turn_kind = kind.map(str::to_string);
    gp.session = Some("s".into());
    gp
}

/// What one run of the session saw: per request (cache_n, prefill calls), the user point of the
/// second user message, and the compaction request's cache_n.
struct Run {
    requests: Vec<(usize, Vec<(usize, usize)>)>,
    pt: usize,
    compaction_cache: usize,
    max_counts: (usize, usize, usize),
}

/// The agent-client-shaped session on a stub worker (ubatch 16, 8 RAM slots, 4 turn slots).
fn session(max_user: usize, hops2: usize) -> Run {
    let tok = synth_tokenizer();
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![7, 8, 9]), 8);
    w.ubatch = 16;
    w.cache.max_user = max_user;
    let mut requests = Vec::new();
    let mut max_counts = (0, 0, 0);
    let mut l = Layout { msg_starts: vec![0, SYS], user_starts: vec![SYS] };
    let mut run = |w: &mut Worker<StubEngine>, prompt: &[u32], gp: &GenParams| -> usize {
        let c0 = w.engine.calls.len();
        let r = w.generate(&tok, prompt, gp, &mut |_| true).unwrap();
        let pre: Vec<(usize, usize)> = w.engine.calls[c0..].iter().filter(|c| c.0 == "prefill").map(|c| (c.1, c.2)).collect();
        requests.push((r.timings.cache_n, pre));
        let info = w.cache.snapshot_info();
        let users = info.iter().filter(|x| x.1 == PRIO_USER).count();
        let turns = info.iter().filter(|x| x.1 == PRIO_TURN).count();
        max_counts = (max_counts.0.max(users), max_counts.1.max(turns), max_counts.2.max(info.len()));
        r.timings.cache_n
    };
    // turn 1: [system | user 1], then three tool hops
    let mut prompt: Vec<u32> = (0..SYS).map(|k| (k % 250) as u32 + 1).collect();
    prompt.extend(fill(300, 1));
    run(&mut w, &prompt, &params(&l, true, None));
    for h in 0..3 {
        let mut next = w.cache.tokens.clone();
        l.msg_starts.push(next.len());
        next.extend(fill(120, 10 + h));
        run(&mut w, &next, &params(&l, false, None));
    }
    // user 2 right after the reply: the reply's turn end (2 tokens), then the user message
    let reply_end = w.cache.tokens.len();
    let mut next = w.cache.tokens.clone();
    next.extend([251, 252]);
    let u2 = next.len();
    l.msg_starts.push(u2);
    l.user_starts.push(u2);
    next.extend(fill(150, 2));
    run(&mut w, &next, &params(&l, true, None));
    let history_to_cut = next[..u2].to_vec();
    let layout_to_cut = l.clone();
    // turn 2: a longer tool loop
    for h in 0..hops2 {
        let mut next = w.cache.tokens.clone();
        l.msg_starts.push(next.len());
        next.extend(fill(120, 30 + h));
        run(&mut w, &next, &params(&l, false, None));
    }
    // the compaction request: the history to the cut (before user 2), then the instruction message
    let mut comp = history_to_cut;
    comp.extend([INSTR; 30]);
    let compaction_cache = run(&mut w, &comp, &params(&layout_to_cut, true, Some("compaction")));
    Run { requests, pt: reply_end, compaction_cache, max_counts }
}

/// The compaction request resumes at the second user message's point (the previous reply's end:
/// no grid position between it and the user start), with v0.1.4's class; the v0.1.3 rule
/// (`--user-snapshots 0`) lost that point to the tool hops and fell back to user 1's query point.
#[test]
fn compaction_request_resumes_at_the_cut_after_a_long_tool_loop() {
    for hops2 in [2usize, 6, 12] {
        let on = session(2, hops2);
        let off = session(0, hops2);
        assert_eq!(on.compaction_cache, on.pt, "hops {hops2}: v0.1.4 resumes at the user point");
        assert!(off.compaction_cache + 64 < off.pt, "hops {hops2}: v0.1.3 fell back far: {} vs point {}", off.compaction_cache, off.pt);
        // the class caps and the slot budget hold all along
        let (users, turns, total) = on.max_counts;
        assert!(users <= 2 && turns <= 4 && total <= 8, "hops {hops2}: {:?}", on.max_counts);
    }
}

/// Numerics unchanged: every request before the compaction runs the same prefill calls (same
/// ubatch boundaries) and reuses the same prefix with and without the class; the class only
/// changes which checkpoint is kept.
#[test]
fn user_points_add_no_prefill_split() {
    let on = session(2, 8);
    let off = session(0, 8);
    let k = on.requests.len() - 1;
    assert_eq!(k, off.requests.len() - 1);
    for i in 0..k {
        assert_eq!(on.requests[i], off.requests[i], "request {i}: cache_n and prefill calls");
    }
}

/// The point taken where no grid position lies between the reused prefix and the user start is
/// the reused prefix's end; with a grid position in between it is the user start's grid floor; a
/// user start inside the reused prefix (an older message) gets no new point; and only the newest
/// `max_user` points stay.
#[test]
fn user_point_placement_and_cap() {
    let tok = synth_tokenizer();
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![7, 8, 9]), 8);
    w.ubatch = 16;
    w.cache.max_user = 2;
    // cold request with three user messages: grid floors of the two newest (cap 2)
    let mut prompt: Vec<u32> = (0..SYS).map(|k| (k % 250) as u32 + 1).collect();
    prompt.extend(fill(500, 5));
    let mut gp = greedy(3);
    gp.first_user = Some(SYS);
    gp.user_starts = vec![SYS, 203, 350, 470];
    gp.msg_starts = vec![0, SYS, 203, 300, 350, 470];
    gp.query_start = true;
    w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    let users: Vec<usize> = w.cache.snapshot_info().iter().filter(|x| x.1 == PRIO_USER).map(|x| x.0).collect();
    assert_eq!(users, vec![336, 464], "grid floors of the two newest user starts");
    // a user message 5 tokens after the live end: the point is the live end itself
    let live = w.cache.tokens.len();
    let mut next = w.cache.tokens.clone();
    next.extend(fill(5, 6));
    let u = next.len();
    next.extend(fill(40, 7));
    let mut gp2 = gp.clone();
    gp2.user_starts.push(u);
    gp2.msg_starts.push(u);
    let snaps_before = w.engine.calls.iter().filter(|c| c.0 == "snapshot").count();
    w.generate(&tok, &next, &gp2, &mut |_| true).unwrap();
    let info = w.cache.snapshot_info();
    let users: Vec<usize> = info.iter().filter(|x| x.1 == PRIO_USER).map(|x| x.0).collect();
    assert!(users.contains(&live), "the live end is the new user point: {info:?}");
    assert_eq!(users.len(), 2, "cap 2: {info:?}");
    assert!(!users.contains(&336), "the oldest user point went: {info:?}");
    // the live end's own end-of-generation checkpoint was promoted, not copied again
    let snaps: Vec<usize> = w.engine.calls.iter().filter(|c| c.0 == "snapshot").map(|c| c.1).skip(snaps_before).collect();
    assert!(!snaps.contains(&live), "no second snapshot at the live end: {snaps:?}");
    // the query point of that request is a query point, not a user point
    assert!(info.iter().any(|x| x.1 == PRIO_QUERY && x.0 > u), "{info:?}");
}

/// On the mini model (real numerics, MTP off and on): the same session with and without the class
/// generates the same tokens with the same state bytes and drafts for every request before the
/// compaction, and the compaction request resumes at the user point only with the class.
#[cfg(target_arch = "aarch64")] // the model engine (O-174: x86_64 builds the stub engine only)
#[test]
fn user_class_keeps_outputs_bitwise_on_the_mini_model() {
    use crate::tests_state_v013::{mini_dir, mini_worker, run_mini};
    let dir = mini_dir("v014-user");
    for depth in [0usize, 2] {
        let mut runs = Vec::new();
        for max_user in [2usize, 0] {
            let mut w = mini_worker(&dir, depth, None);
            w.cache.max_user = max_user;
            let mut out = Vec::new();
            let mut l = Layout { msg_starts: vec![0, 70], user_starts: vec![70] };
            let gp = |l: &Layout, q: bool, kind: Option<&str>| {
                let mut g = greedy(6);
                g.first_user = Some(70);
                g.msg_starts = l.msg_starts.clone();
                g.user_starts = l.user_starts.clone();
                g.query_start = q;
                g.turn_kind = kind.map(str::to_string);
                g
            };
            let p1: Vec<u32> = (0..260).map(|i| (i * 13 + 7) % 64).collect();
            out.push(run_mini(&mut w, &p1, &gp(&l, true, None)));
            for h in 0..2u32 {
                let mut next = w.cache.tokens.clone();
                l.msg_starts.push(next.len());
                next.extend((0..90).map(|i| (i * 5 + h * 11 + 3) % 64));
                out.push(run_mini(&mut w, &next, &gp(&l, false, None)));
            }
            let pt = w.cache.tokens.len();
            let mut next = w.cache.tokens.clone();
            next.extend([1, 2]);
            let u2 = next.len();
            l.msg_starts.push(u2);
            l.user_starts.push(u2);
            next.extend((0..70).map(|i| (i * 3 + 1) % 64));
            out.push(run_mini(&mut w, &next, &gp(&l, true, None)));
            let cut = next[..u2].to_vec();
            let lc = l.clone();
            for h in 0..6u32 {
                let mut next = w.cache.tokens.clone();
                l.msg_starts.push(next.len());
                next.extend((0..90).map(|i| (i * 9 + h * 7 + 5) % 64));
                out.push(run_mini(&mut w, &next, &gp(&l, false, None)));
            }
            let mut comp = cut;
            comp.extend([63u32; 12]);
            out.push(run_mini(&mut w, &comp, &gp(&lc, true, Some("compaction"))));
            runs.push((out, pt));
        }
        let (on, pt_on) = &runs[0];
        let (off, pt_off) = &runs[1];
        assert_eq!(pt_on, pt_off);
        let k = on.len() - 1;
        for i in 0..k {
            assert_eq!(on[i].0, off[i].0, "depth {depth} request {i}: tokens");
            assert!(on[i].1 == off[i].1, "depth {depth} request {i}: state bytes differ");
            assert_eq!(on[i].2, off[i].2, "depth {depth} request {i}: drafted / accepted");
            assert_eq!(on[i].3, off[i].3, "depth {depth} request {i}: cache_n");
        }
        assert_eq!(on[k].3, *pt_on, "depth {depth}: the compaction resumes at the user point");
        assert!(off[k].3 < *pt_off, "depth {depth}: v0.1.3 resumed earlier ({} vs {})", off[k].3, pt_off);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
