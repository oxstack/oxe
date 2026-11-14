//! CACHE-PERFECT P2b, Codex P2/P3 review 1-3 (model-free, stub engine, temp dirs only): a session
//! clear purges every tier, budget 0 turns off every conversation disk writer (state store, echo
//! files, legacy prefix store / session cache) and a clear(all) covers the legacy stores too.

use crate::engine::Engine;
use crate::state_store::hint_hash;
use crate::tests_echo::synth_tokenizer;
use crate::tests_state::{call, files, sys, tmp};
use crate::tests_state_p2b::{chat_h, model_dir, server_echo, toks};
use serde_json::json;
use std::path::Path;

/// Codex P2/P3 review 1: clearing one session purges it from every tier (manifest, echo file and
/// RAM echo records, the live and parked RAM state, legacy store entries) so nothing can bring it
/// back; the other session (and the blocks it shares) still restores after a restart.
#[tokio::test(flavor = "multi_thread")]
async fn session_clear_purges_every_tier_and_keeps_the_other_session() {
    let tok = synth_tokenizer();
    let dir = tmp("p2b-clear");
    let a = server_echo(&dir, vec![tok.encode("Answer A.", false, true), tok.encode("Answer B.", false, true)], 1 << 40);
    let ma = json!([{"role":"system","content":sys()},{"role":"user","content":"conversation A asks this"}]);
    let mb = json!([{"role":"system","content":sys()},{"role":"user","content":"conversation B asks that"}]);
    chat_h(&a, mb.clone(), "conv-b").await;
    chat_h(&a, ma.clone(), "conv-a").await; // A is the live state
    let a_tokens = crate::tests_state::kv(&a);
    let md = model_dir(&a);
    let (sid_a, n_echo_b) = {
        let w = a.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        let sid = s.manifests.values().find(|m| m.hint.as_deref() == Some(hint_hash("conv-a").as_str())).unwrap().sid.clone();
        (sid, a.echo.lock().unwrap().n_session_records("conv-b"))
    };
    assert!(n_echo_b >= 1);
    assert!(md.join("echo").join(format!("{}.jsonl", hint_hash("conv-a"))).is_file());
    let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "session", "session": sid_a})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!(j["ram_states"], 1, "A's live RAM state dropped: {j}");
    assert!(j["echo_records"].as_u64().unwrap() >= 1, "{j}");
    assert!(!md.join("echo").join(format!("{}.jsonl", hint_hash("conv-a"))).exists(), "A's echo file gone");
    assert_eq!(a.echo.lock().unwrap().n_session_records("conv-a"), 0, "A's RAM echo records gone");
    assert_eq!(a.echo.lock().unwrap().n_session_records("conv-b"), n_echo_b, "B's echo records stay");
    assert!(crate::tests_state::kv(&a).is_empty(), "A's live state is gone");
    assert!(md.join("echo").join(format!("{}.jsonl", hint_hash("conv-b"))).is_file(), "B's echo file stays");
    drop(a);
    // restart: A cannot be restored (no resurrection), B still can; the shared system blocks stay
    let b = server_echo(&dir, vec![tok.encode("x", false, true), tok.encode("y", false, true)], 1 << 40);
    let ma2 = json!([{"role":"system","content":sys()},{"role":"user","content":"conversation A asks this"},{"role":"assistant","content":"Answer A."},{"role":"user","content":"more"}]);
    let r = chat_h(&b, ma2, "conv-a").await;
    let n = r["timings"]["cache_n"].as_u64().unwrap() as usize;
    assert!(n < a_tokens.len(), "A's own state is gone: resumed only {n} (shared prefix at most)");
    {
        let mut w = b.worker.lock().unwrap();
        let w = &mut *w;
        w.cache.reset(&mut w.engine);
        let _ = w.cache.drain_parked();
    }
    let mb2 = json!([{"role":"system","content":sys()},{"role":"user","content":"conversation B asks that"},{"role":"assistant","content":"Answer B."},{"role":"user","content":"more"}]);
    let r = chat_h(&b, mb2, "conv-b").await;
    assert_eq!(r["timings"]["resume_src"], "disk", "B restores from disk: {}", r["timings"]);
    assert!(r["timings"]["cache_n"].as_u64().unwrap() as usize > n);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Codex P2/P3 review 2 and 3: budget 0 means no conversation state goes to disk from then on:
/// no state files, no echo lines, no legacy prefix-store / session-cache entries; raising the
/// budget brings them back. A clear(all) also clears the legacy stores' entries.
#[tokio::test(flavor = "multi_thread")]
async fn budget_zero_turns_off_every_disk_writer_and_clear_covers_the_legacy_stores() {
    let tok = synth_tokenizer();
    let dir = tmp("p2b-off");
    let a = server_echo(&dir, vec![tok.encode("One.", false, true), tok.encode("Two.", false, true), tok.encode("Three.", false, true)], 1 << 40);
    // a legacy prefix store next to it (as --prefix-cache)
    let legacy = dir.join("legacy");
    {
        let mut w = a.worker.lock().unwrap();
        let mut ps = crate::prefix_store::PrefixStore::open(&legacy, "legacy-id".into(), 1 << 30, 64).unwrap();
        ps.save_async(toks(300, 1), vec![1, 2, 3, 4]);
        ps.flush();
        w.store = Some(ps);
    }
    let (c, j) = call(&a, "POST", "/state-cache/budget", json!({"bytes": 0})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!(j["legacy_stores"], "off");
    let snapshot = |d: &Path| {
        let mut v: Vec<(std::path::PathBuf, u64)> = files(d, "").into_iter().map(|p| {
            let n = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            (p, n)
        }).collect();
        v.sort();
        v
    };
    let before = snapshot(&dir);
    let r = chat_h(&a, json!([{"role":"system","content":sys()},{"role":"user","content":"while off"}]), "conv-off").await;
    assert_eq!(r["choices"][0]["message"]["content"], "One.");
    assert_eq!(snapshot(&dir), before, "budget 0: nothing on disk changed (state, echo, legacy)");
    assert!(a.echo.lock().unwrap().n_session_records("conv-off") >= 1, "RAM echo still works");
    {
        let w = a.worker.lock().unwrap();
        assert!(w.store.is_none() && w.legacy_off.is_some(), "the legacy store is off while the budget is 0");
    }
    // back on: every writer returns
    let (_, j) = call(&a, "POST", "/state-cache/budget", json!({"bytes": 1u64 << 40})).await;
    assert_eq!(j["legacy_stores"], "on");
    chat_h(&a, json!([{"role":"system","content":sys()},{"role":"user","content":"while on"}]), "conv-on").await;
    assert!(model_dir(&a).join("echo").join(format!("{}.jsonl", hint_hash("conv-on"))).is_file(), "echo on disk again");
    assert!(a.worker.lock().unwrap().store.is_some());
    // clear all: the legacy store's entries go too
    let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "all"})).await;
    assert_eq!(c, 200, "{j}");
    assert!(j["legacy_entries"].as_u64().unwrap() >= 1, "{j}");
    assert_eq!(a.worker.lock().unwrap().store.as_ref().unwrap().n_entries(), 0);
    assert!(files(&legacy, ".bin").is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// v0.2.0 EG5: `{"scope": "hint", "hint": ...}` clears exactly the sessions created under that
/// request hint (a dotted child id such as `parent.s1` is a valid hint). The parent's session and
/// every block it names (the shared system prompt included) stay and still restore after a
/// restart; an invalid hint is refused; a second clear finds nothing and is still a 200.
#[tokio::test(flavor = "multi_thread")]
async fn hint_clear_removes_only_that_hints_sessions_and_keeps_the_parent() {
    let tok = synth_tokenizer();
    let dir = tmp("eg5-hint");
    let a = server_echo(&dir, vec![tok.encode("Parent answer.", false, true), tok.encode("Child answer.", false, true)], 1 << 40);
    let parent = json!([{"role":"system","content":sys()},{"role":"user","content":"the parent conversation asks this"}]);
    let child = json!([{"role":"system","content":sys()},{"role":"user","content":"a child task asks that"}]);
    chat_h(&a, parent.clone(), "parent").await;
    chat_h(&a, child.clone(), "parent.s1").await; // the child is the live state; the parent was parked/persisted
    let (parent_sid, parent_blocks, child_sids) = {
        let w = a.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        let p = s.manifests.values().find(|m| m.hint.as_deref() == Some(hint_hash("parent").as_str())).unwrap();
        (p.sid.clone(), p.blocks.clone(), s.sessions_with_hint(&hint_hash("parent.s1")))
    };
    assert_eq!(child_sids.len(), 1, "the child was persisted under its own hint");
    assert_ne!(child_sids[0], parent_sid);
    for bad in ["", "parent s1", "parent/s1", &"x".repeat(129)] {
        let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "hint", "hint": bad})).await;
        assert_eq!(c, 400, "{bad:?}: {j}");
    }
    let (c, _) = call(&a, "POST", "/state-cache/clear", json!({"scope": "hint"})).await;
    assert_eq!(c, 400, "a hint scope without a hint is refused (never a wider clear)");
    {
        let w = a.worker.lock().unwrap();
        assert_eq!(w.states.as_ref().unwrap().manifests.len(), 2, "a refused clear removed nothing");
    }
    let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "hint", "hint": "parent.s1"})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!(j["sessions"], 1, "{j}");
    assert_eq!(j["ram_states"], 1, "the child's live RAM state dropped: {j}");
    assert!(!model_dir(&a).join("echo").join(format!("{}.jsonl", hint_hash("parent.s1"))).exists(), "the child's echo file is gone");
    assert!(model_dir(&a).join("echo").join(format!("{}.jsonl", hint_hash("parent"))).is_file(), "the parent's echo file stays");
    {
        let w = a.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        assert!(s.sessions_with_hint(&hint_hash("parent.s1")).is_empty());
        assert!(s.manifest(&parent_sid).is_some(), "the parent's session stays");
        for (h, c) in &parent_blocks {
            assert!(s.has_block(h, c), "the parent's block {h} (shared or its own) stays");
        }
    }
    // the removals are durable before the clear answered: the sessions and echo directories were fsync'd
    {
        let synced = crate::state_store::DIR_SYNCS.lock().unwrap().clone();
        for sub in ["sessions", "echo"] {
            assert!(synced.contains(&model_dir(&a).join(sub)), "{sub} directory fsync'd after its removal");
        }
    }
    // idempotent: nothing left under that hint
    let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "hint", "hint": "parent.s1"})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!((j["sessions"].as_u64(), j["freed_bytes"].as_u64()), (Some(0), Some(0)), "{j}");
    drop(a);
    // restart: the parent restores from disk
    let b = server_echo(&dir, vec![tok.encode("x", false, true)], 1 << 40);
    let parent2 = json!([{"role":"system","content":sys()},{"role":"user","content":"the parent conversation asks this"},{"role":"assistant","content":"Parent answer."},{"role":"user","content":"more"}]);
    let r = chat_h(&b, parent2, "parent").await;
    assert_eq!(r["timings"]["resume_src"], "disk", "the parent restores from disk: {}", r["timings"]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// v0.2.0 EG5 (Codex review b53f400): a conversation started under the parent's hint and continued
/// (the identical prompt, or a prefix-extending one) under a child's hint is one stored lineage owned
/// by both hints. Clearing the child's hint only drops its ownership: the parent's session, its
/// blocks and its echo file stay, and the parent restores from disk after a restart. Clearing the
/// parent's hint afterwards (the sole owner now) removes it.
#[tokio::test(flavor = "multi_thread")]
async fn hint_clear_keeps_a_lineage_another_hint_still_owns() {
    let tok = synth_tokenizer();
    for extend in [false, true] {
        let dir = tmp(if extend { "eg5-own-ext" } else { "eg5-own-same" });
        let a = server_echo(&dir, vec![tok.encode("Parent answer.", false, true), tok.encode("Child answer.", false, true)], 1 << 40);
        let parent = json!([{"role":"system","content":sys()},{"role":"user","content":"the shared conversation asks this"}]);
        chat_h(&a, parent.clone(), "parent").await;
        let child = if extend {
            json!([{"role":"system","content":sys()},{"role":"user","content":"the shared conversation asks this"},{"role":"assistant","content":"Parent answer."},{"role":"user","content":"and the child goes on"}])
        } else {
            parent.clone()
        };
        chat_h(&a, child, "parent.s1").await;
        let (sids, owners) = {
            let w = a.worker.lock().unwrap();
            let s = w.states.as_ref().unwrap();
            let sids = s.sessions_with_hint(&hint_hash("parent"));
            let owners: Vec<Vec<String>> = sids.iter().map(|x| s.manifest(x).unwrap().owners.clone()).collect();
            (sids, owners)
        };
        assert!(!sids.is_empty(), "extend={extend}");
        if extend {
            // the child continued the parent's stored lineage: one session owned by both, creator first
            assert!(owners.iter().any(|o| o == &vec![hint_hash("parent"), hint_hash("parent.s1")]), "{owners:?}");
        } else {
            // the same prompt with another reply is a fork: the child has its own session, the parent's
            // stays its own (the shared prompt blocks are shared on disk)
            assert!(owners.iter().all(|o| o == &vec![hint_hash("parent")]), "{owners:?}");
            assert!(!a.worker.lock().unwrap().states.as_ref().unwrap().sessions_with_hint(&hint_hash("parent.s1")).is_empty());
        }
        let blocks: Vec<(String, String)> = {
            let w = a.worker.lock().unwrap();
            let s = w.states.as_ref().unwrap();
            sids.iter().flat_map(|x| s.manifest(x).unwrap().blocks.clone()).collect()
        };
        let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "hint", "hint": "parent.s1"})).await;
        assert_eq!(c, 200, "{j}");
        assert_eq!(j["released"].as_u64().unwrap() >= 1, extend, "extend={extend}: {j}");
        assert_eq!(j["sessions"].as_u64().unwrap() >= 1, !extend, "extend={extend}: the fork alone is purged: {j}");
        {
            let w = a.worker.lock().unwrap();
            let s = w.states.as_ref().unwrap();
            assert!(s.sessions_with_hint(&hint_hash("parent.s1")).is_empty(), "the child owns nothing now");
            for x in &sids {
                let m = s.manifest(x).expect("the parent's session stays");
                assert_eq!(m.owners, vec![hint_hash("parent")]);
                assert_eq!(m.hint.as_deref(), Some(hint_hash("parent").as_str()));
            }
            for (h, c) in &blocks {
                assert!(s.has_block(h, c), "block {h} stays");
            }
        }
        assert!(model_dir(&a).join("echo").join(format!("{}.jsonl", hint_hash("parent"))).is_file(), "the parent's echo file stays");
        drop(a);
        let b = server_echo(&dir, vec![tok.encode("x", false, true)], 1 << 40);
        {
            let w = b.worker.lock().unwrap();
            let s = w.states.as_ref().unwrap();
            assert!(sids.iter().all(|x| s.manifest(x).is_some_and(|m| m.owners == vec![hint_hash("parent")])), "owners survive a restart");
        }
        let parent2 = json!([{"role":"system","content":sys()},{"role":"user","content":"the shared conversation asks this"},{"role":"assistant","content":"Parent answer."},{"role":"user","content":"more"}]);
        let r = chat_h(&b, parent2, "parent").await;
        assert_eq!(r["timings"]["resume_src"], "disk", "extend={extend}: the parent restores from disk: {}", r["timings"]);
        // the parent's own clear (sole owner now) removes it
        {
            let mut w = b.worker.lock().unwrap();
            let w = &mut *w;
            w.cache.reset(&mut w.engine);
            let _ = w.cache.drain_parked();
        }
        let (c, j) = call(&b, "POST", "/state-cache/clear", json!({"scope": "hint", "hint": "parent"})).await;
        assert_eq!(c, 200, "{j}");
        assert!(j["sessions"].as_u64().unwrap() >= 1, "{j}");
        assert!(b.worker.lock().unwrap().states.as_ref().unwrap().sessions_with_hint(&hint_hash("parent")).is_empty());
        drop(b);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A chat request with an optional session hint (None: the request sends no `X-Oxe-Session`);
/// returns after its post-reply state write.
async fn chat_opt(st: &std::sync::Arc<crate::service::State<crate::engine::StubEngine>>, msgs: serde_json::Value, hint: Option<&str>) -> serde_json::Value {
    use std::sync::atomic::Ordering;
    let n0 = st.persist_done.load(Ordering::SeqCst);
    let b = crate::tests_state::body(msgs);
    let r = match crate::service::handle_session(st.clone(), "POST", "/v1/chat/completions", "", &serde_json::to_vec(&b).unwrap(), hint.map(str::to_string)).await {
        crate::service::Reply::Json(c, j) => {
            assert_eq!(c, 200, "{j}");
            j
        }
        crate::service::Reply::Stream(_) => panic!("stream"),
    };
    let t0 = std::time::Instant::now();
    while st.persist_done.load(Ordering::SeqCst) == n0 {
        assert!(t0.elapsed().as_secs() < 30, "persist did not finish");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    r
}

/// v0.2.0 EG5 (Codex review, unhinted parent): a conversation started WITHOUT a session hint and
/// continued under a child's hint is owned by both (the unhinted request owns as the anonymous
/// owner). Clearing the child's hint releases only its ownership: the parent's session and blocks
/// stay, after a restart too, and the parent (still unhinted) restores from disk. A legacy manifest
/// with no hint and no owners field counts as anonymously owned.
#[tokio::test(flavor = "multi_thread")]
async fn hint_clear_keeps_an_unhinted_parents_lineage() {
    use crate::state_store::ANON_OWNER;
    let tok = synth_tokenizer();
    let dir = tmp("eg5-anon");
    let a = server_echo(&dir, vec![tok.encode("Parent answer.", false, true), tok.encode("Child answer.", false, true)], 1 << 40);
    let parent = json!([{"role":"system","content":sys()},{"role":"user","content":"an unhinted conversation asks this"}]);
    chat_opt(&a, parent.clone(), None).await;
    let child = json!([{"role":"system","content":sys()},{"role":"user","content":"an unhinted conversation asks this"},{"role":"assistant","content":"Parent answer."},{"role":"user","content":"and a hinted child goes on"}]);
    chat_opt(&a, child, Some("p.s1")).await;
    let (sids, blocks) = {
        let w = a.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        let sids = s.sessions_with_hint(&hint_hash("p.s1"));
        for x in &sids {
            assert_eq!(s.manifest(x).unwrap().owners, vec![ANON_OWNER.to_string(), hint_hash("p.s1")], "the unhinted creator owns it too");
        }
        let blocks: Vec<(String, String)> = s.manifests.values().flat_map(|m| m.blocks.clone()).collect();
        (sids, blocks)
    };
    assert!(!sids.is_empty());
    let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "hint", "hint": "p.s1"})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!((j["sessions"].as_u64(), j["released"].as_u64()), (Some(0), Some(sids.len() as u64)), "{j}");
    {
        let w = a.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        for x in &sids {
            let m = s.manifest(x).expect("the unhinted parent's session stays");
            assert_eq!((m.owners.clone(), m.hint.clone()), (vec![ANON_OWNER.to_string()], None));
        }
        for (h, c) in &blocks {
            assert!(s.has_block(h, c), "block {h} stays");
        }
    }
    // a legacy manifest (no hint, no owners field) reads as anonymously owned
    let md = model_dir(&a);
    drop(a);
    for x in &sids {
        let path = md.join("sessions").join(format!("{x}.json"));
        let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        v.as_object_mut().unwrap().remove("owners");
        std::fs::write(&path, v.to_string()).unwrap();
    }
    let b = server_echo(&dir, vec![tok.encode("x", false, true)], 1 << 40);
    {
        let w = b.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        assert!(sids.iter().all(|x| s.manifest(x).is_some_and(|m| m.owners == vec![ANON_OWNER.to_string()])));
    }
    let (c, j) = call(&b, "POST", "/state-cache/clear", json!({"scope": "hint", "hint": "p.s1"})).await;
    assert_eq!((c, j["sessions"].as_u64(), j["released"].as_u64()), (200, Some(0), Some(0)), "{j}");
    let parent2 = json!([{"role":"system","content":sys()},{"role":"user","content":"an unhinted conversation asks this"},{"role":"assistant","content":"Parent answer."},{"role":"user","content":"more"}]);
    let r = chat_opt(&b, parent2, None).await;
    assert_eq!(r["timings"]["resume_src"], "disk", "the unhinted parent restores from disk: {}", r["timings"]);
    drop(b);
    let _ = std::fs::remove_dir_all(&dir);
}
