//! History rewrite (a client's compaction) then appends: the first pure append after the rewritten
//! turn must reuse every fed token of it (cached == prompt + completion - 1), like any append.
//! On the stub engine and on the tiny synthetic HYBRID model (`model::testkit`, recurrent + attention
//! layers, MTP head, random weights, the synthetic tokenizer's vocabulary), with and without the
//! state store, with and without the client's H6 idle prefill. Model-free.

use crate::engine::{Engine, StubEngine};
use crate::generate::Worker;
use crate::service::{handle_hints, Config, Hints, Reply, State};
use crate::state_store::StateStore;
use crate::tests_echo::{synth_tokenizer, synth_tokenizer_padded};
use crate::tests_state::tmp;
use crate::tests_state_v013::srv;
use serde_json::{json, Value as Json};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

const SYS: &str =
    "You are a coding agent. Tools: read, edit, bash. Be brief and exact in every answer you give.";

/// One request as oxa sends it (session "s", a turn kind, H4 hint); returns the reply JSON.
async fn req<E: Engine>(
    st: &Arc<State<E>>,
    msgs: &Json,
    max: i64,
    kind: &str,
    snap: bool,
    mode: Mode,
) -> Json {
    let n0 = st.persist_done.load(Ordering::SeqCst);
    let b = json!({"messages": msgs, "temperature": 0, "cache_prompt": true, "max_tokens": max, "stream": mode.stream,
                   "stream_options": {"include_usage": true},
                   "chat_template_kwargs": {"preserve_thinking": mode.preserve, "enable_thinking": mode.thinking}});
    let r = match handle_hints(
        st.clone(),
        "POST",
        "/v1/chat/completions",
        "",
        &serde_json::to_vec(&b).unwrap(),
        Some("s".into()),
        Some(kind.into()),
        false,
        Hints {
            snapshot_keep: snap,
        },
    )
    .await
    {
        Reply::Json(200, j) => j,
        Reply::Json(c, j) => panic!("{c} {j}"),
        Reply::Stream(mut rx) => {
            // the streamed reply as a client assembles it
            let (mut content, mut reasoning, mut usage, mut timings) =
                (String::new(), String::new(), Json::Null, Json::Null);
            while let Some(f) = tokio::time::timeout(std::time::Duration::from_secs(60), rx.recv())
                .await
                .expect("stream hung")
            {
                for line in f.lines() {
                    let Some(d) = line.strip_prefix("data: ").filter(|d| *d != "[DONE]") else {
                        continue;
                    };
                    let j: Json = serde_json::from_str(d).unwrap();
                    if let Some(c) = j.pointer("/choices/0/delta/content").and_then(Json::as_str) {
                        content.push_str(c);
                    }
                    if let Some(c) = j
                        .pointer("/choices/0/delta/reasoning_content")
                        .and_then(Json::as_str)
                    {
                        reasoning.push_str(c);
                    }
                    if j["usage"].is_object() {
                        usage = j["usage"].clone();
                    }
                    if j["timings"].is_object() {
                        timings = j["timings"].clone();
                    }
                }
            }
            json!({"choices": [{"message": {"role": "assistant", "content": content, "reasoning_content": reasoning}}], "usage": usage, "timings": timings})
        }
    };
    if st.state_cache {
        let t0 = Instant::now();
        while st.persist_done.load(Ordering::SeqCst) == n0 {
            assert!(t0.elapsed().as_secs() < 60, "persist did not finish");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
    r
}

fn n(j: &Json, p: &str) -> u64 {
    j.pointer(p)
        .and_then(Json::as_u64)
        .unwrap_or_else(|| panic!("{p} in {j}"))
}

fn user(s: &str) -> Json {
    json!({"role": "user", "content": s})
}

/// The reply's message as the client sends it back (no reasoning: ECHO restores it).
fn back(r: &Json) -> Json {
    let mut m = r["choices"][0]["message"].clone();
    if let Some(o) = m.as_object_mut() {
        o.remove("reasoning_content");
    }
    m
}

/// The oxa compaction sequence: two turns, a summary request, the rewritten history (summary as a
/// user message + kept tail), optionally the H6 idle prefill (H4 hint), the post-compaction turn,
/// then pure appends: each append reuses every fed token of the turn before.
#[derive(Clone, Copy, Debug)]
struct Mode {
    stream: bool,
    thinking: bool,
    preserve: bool,
}

const MODES: [Mode; 4] = [
    Mode {
        stream: false,
        thinking: false,
        preserve: true,
    },
    Mode {
        stream: true,
        thinking: false,
        preserve: true,
    },
    Mode {
        stream: true,
        thinking: true,
        preserve: true,
    },
    Mode {
        stream: true,
        thinking: true,
        preserve: false,
    },
];

async fn rewrite_then_append<E: Engine>(st: Arc<State<E>>, idle: bool, mode: Mode, tag: &str) {
    let tag = format!("{tag} {mode:?}");
    let long = |t: &str| {
        format!(
            "{t} {}",
            "lorem ipsum dolor sit amet consectetur ".repeat(12)
        )
    };
    let mut h = vec![
        json!({"role":"system","content":SYS}),
        user(&long("read the file")),
    ];
    let r = req(&st, &json!(h), 8, "chat", true, mode).await;
    h.push(back(&r));
    h.push(user(&long("now fix it")));
    let r = req(&st, &json!(h), 8, "chat", true, mode).await;
    h.push(back(&r));
    let mut s = h.clone();
    s.push(user("Summarize the conversation so far."));
    let sum = req(&st, &json!(s), 8, "compaction", false, mode).await;
    let summary = sum["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let mut w = vec![json!({"role":"system","content":SYS}),
                     user(&format!("The conversation history before this point was compacted into the following summary:\n<summary>\n{summary}\n</summary>")),
                     h[3].clone(), h[4].clone()];
    if idle {
        req(&st, &json!(w), 0, "prefill", true, mode).await;
    }
    w.push(user("run the tests"));
    let mut prev = req(&st, &json!(w), 8, "chat", true, mode).await;
    for (i, next_user) in ["and commit", "thanks", "one more"].iter().enumerate() {
        w.push(back(&prev));
        w.push(user(next_user));
        let r = req(&st, &json!(w), 8, "chat", true, mode).await;
        let want = n(&prev, "/usage/prompt_tokens") + n(&prev, "/usage/completion_tokens") - 1;
        assert!(
            n(&r, "/usage/prompt_tokens_details/cached_tokens") >= want,
            "{tag}: append {i} after the rewrite: want >= {want}: before {} now {}",
            prev["timings"],
            r["timings"]
        );
        prev = r;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn first_append_after_a_rewrite_keeps_every_token_stub() {
    for (store, idle) in [(false, true), (true, true), (true, false), (false, false)] {
        let dir = tmp("rewrite");
        for mode in MODES {
            let think = |a: &str| {
                if mode.thinking {
                    format!("let me think about {a}\n</think>\n\n{a}")
                } else {
                    a.to_string()
                }
            };
            let scripts: Vec<String> = [
                "First answer.",
                "Second answer.",
                "Summary of the work so far.",
                "Third answer.",
                "Fourth answer.",
                "Fifth answer.",
                "Sixth.",
            ]
            .iter()
            .map(|a| think(a))
            .collect();
            let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
            let st = srv(mode.thinking, &scripts, store.then_some(dir.as_path()));
            rewrite_then_append(st, idle, mode, &format!("stub store {store} idle {idle}")).await;
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

fn mini_state(
    dir: &Path,
    depth: usize,
    store: bool,
    thinking: bool,
) -> Arc<State<crate::oxe_engine::OxeEngine>> {
    let base = synth_tokenizer().n_vocab();
    let tok = synth_tokenizer_padded(base.next_multiple_of(8) - base);
    let model = dir.join("mini-vocab.gguf");
    if !model.exists() {
        model::testkit::mini_model_vocab(&model, true, tok.n_vocab() as u64);
    }
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let n = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    let eng = crate::oxe_engine::OxeEngine::open_cache(
        &model,
        None,
        &[n - 1],
        None,
        64,
        4096,
        depth,
        Some(dir.join("cache")),
    )
    .unwrap();
    let mut w = Worker::new(eng, 8);
    w.ubatch = 64;
    if store {
        let id = crate::state_store::Identity::new(
            "sha256:mini",
            &w.engine.state_id().unwrap(),
            64,
            &w.engine.arith_id(),
            "b",
            false,
        );
        let mut s = StateStore::open(&dir.join("store"), id, 1 << 40, 3600, true).unwrap();
        s.min_tokens = 0;
        s.min_free_bytes = 0;
        s.min_mem_bytes = 0;
        w.states = Some(s);
    }
    let cfg = Config {
        model_name: "mini".into(),
        model_path: model.display().to_string(),
        chat_opts: chat::ServerChatOptions {
            enable_thinking: thinking,
            ..Default::default()
        },
        fingerprint: "oxe-test".into(),
        slots: false,
        idle_reprefill: false,
        engine_status: None,
    };
    Arc::new(State::new(cfg, tok, w).unwrap())
}

#[cfg(target_arch = "aarch64")] // the model engine (O-174: x86_64 builds the stub engine only)
#[tokio::test(flavor = "multi_thread")]
async fn first_append_after_a_rewrite_keeps_every_token_hybrid() {
    for (depth, store, idle) in [
        (0, false, false),
        (0, false, true),
        (2, false, true),
        (2, true, true),
        (0, true, false),
    ] {
        let dir = tmp("rewrite-hybrid");
        std::fs::create_dir_all(dir.join("cache")).unwrap();
        std::fs::set_permissions(
            dir.join("cache"),
            std::os::unix::fs::PermissionsExt::from_mode(0o700),
        )
        .unwrap();
        // thinking off only: random weights never close a think block, and a length-cut reasoning-only
        // turn is never kept by ECHO (V014-SL-1), which is a different, intended miss
        for mode in MODES.into_iter().filter(|m| !m.thinking) {
            let st = mini_state(&dir, depth, store, mode.thinking);
            rewrite_then_append(
                st,
                idle,
                mode,
                &format!("hybrid mtp {depth} store {store} idle {idle}"),
            )
            .await;
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The oxa compaction shape that showed the miss on the tiny model (T1.5, A/B A2): the summary is a
/// user message and the kept tail ends with tool turns, so the post-compaction request has NO new
/// user message: the kept-tail assistant turns come AFTER the last user query there, and BEFORE it
/// once the next user message arrives. A template without `preserve_thinking` (the tiny model's
/// Qwen3.5) renders a turn after the last query with its think block and one before it without, so
/// the next request's render of the kept tail differs from what was fed, unless ECHO keeps the fed
/// form (their generated-turn records hang off the pre-compaction prefix). The first append must
/// reuse every fed token: cached == prompt + completion - 1. Also with the H6 idle prefill and the
/// state store, streamed and not.
#[tokio::test(flavor = "multi_thread")]
async fn append_after_a_tool_tail_rewrite_keeps_every_token() {
    const TOOLS: &str = r#"[{"type":"function","function":{"name":"read_file","description":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]"#;
    const CALL: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>";
    let tools: Json = serde_json::from_str(TOOLS).unwrap();
    for (store, idle, stream) in [
        (false, false, false),
        (false, false, true),
        (false, true, true),
        (true, true, true),
        (true, false, false),
    ] {
        let tag = format!("store {store} idle {idle} stream {stream}");
        let dir = tmp("rewrite-tool");
        let st = srv(
            false,
            &[
                "I'll read it.\n\n".to_string() + CALL,
                "Done reading.".into(),
                "Let me check.\n\n".to_string() + CALL,
                "Summary: read a.txt twice.".into(),
                "Fixed it.".into(),
                "Committed.".into(),
                "Bye.".into(),
            ]
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
            store.then_some(dir.as_path()),
        );
        let send = |msgs: Vec<Json>, max: i64, kind: &'static str| {
            let st = st.clone();
            let tools = tools.clone();
            async move {
                let n0 = st.persist_done.load(Ordering::SeqCst);
                let b = json!({"messages": msgs, "tools": tools, "temperature": 0, "max_tokens": max, "stream": stream,
                               "stream_options": {"include_usage": true},
                               "chat_template_kwargs": {"preserve_thinking": false, "enable_thinking": false}});
                let r = handle_hints(
                    st.clone(),
                    "POST",
                    "/v1/chat/completions",
                    "",
                    &serde_json::to_vec(&b).unwrap(),
                    Some("s".into()),
                    Some(kind.into()),
                    false,
                    Hints {
                        snapshot_keep: true,
                    },
                )
                .await;
                let out = match r {
                    Reply::Json(200, j) => {
                        json!({"message": j["choices"][0]["message"], "usage": j["usage"], "timings": j["timings"]})
                    }
                    Reply::Json(c, j) => panic!("{c} {j}"),
                    Reply::Stream(mut rx) => {
                        let (mut content, mut calls, mut usage, mut timings) =
                            (String::new(), Vec::<Json>::new(), Json::Null, Json::Null);
                        while let Some(f) = rx.recv().await {
                            for line in f.lines() {
                                let Some(d) =
                                    line.strip_prefix("data: ").filter(|d| *d != "[DONE]")
                                else {
                                    continue;
                                };
                                let j: Json = serde_json::from_str(d).unwrap();
                                if let Some(c) =
                                    j.pointer("/choices/0/delta/content").and_then(Json::as_str)
                                {
                                    content.push_str(c);
                                }
                                if let Some(tc) = j
                                    .pointer("/choices/0/delta/tool_calls")
                                    .and_then(Json::as_array)
                                {
                                    for t in tc {
                                        let i = t["index"].as_u64().unwrap_or(0) as usize;
                                        while calls.len() <= i {
                                            calls.push(json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}));
                                        }
                                        if let Some(id) = t["id"].as_str() {
                                            calls[i]["id"] = json!(id);
                                        }
                                        if let Some(nm) =
                                            t.pointer("/function/name").and_then(Json::as_str)
                                        {
                                            calls[i]["function"]["name"] = json!(nm);
                                        }
                                        if let Some(a) =
                                            t.pointer("/function/arguments").and_then(Json::as_str)
                                        {
                                            let prev = calls[i]["function"]["arguments"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_string();
                                            calls[i]["function"]["arguments"] = json!(prev + a);
                                        }
                                    }
                                }
                                if j["usage"].is_object() {
                                    usage = j["usage"].clone();
                                }
                                if j["timings"].is_object() {
                                    timings = j["timings"].clone();
                                }
                            }
                        }
                        let mut m = json!({"role": "assistant", "content": content});
                        if !calls.is_empty() {
                            m["tool_calls"] = json!(calls);
                        }
                        json!({"message": m, "usage": usage, "timings": timings})
                    }
                };
                if st.state_cache {
                    let t0 = Instant::now();
                    while st.persist_done.load(Ordering::SeqCst) == n0 {
                        assert!(t0.elapsed().as_secs() < 30, "persist did not finish");
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    }
                }
                out
            }
        };
        let back = |r: &Json| {
            let mut m = r["message"].clone();
            if let Some(o) = m.as_object_mut() {
                o.remove("reasoning_content");
            }
            m
        };
        let tool_result = |r: &Json, text: &str| json!({"role": "tool", "tool_call_id": r["message"]["tool_calls"][0]["id"], "content": text});
        let mut h = vec![
            json!({"role":"system","content":SYS}),
            user("Read a.txt and tell me what it says."),
        ];
        let r1 = send(h.clone(), 256, "chat").await;
        assert!(r1["message"]["tool_calls"].is_array(), "{tag}: {r1}");
        h.push(back(&r1));
        h.push(tool_result(&r1, &"line of a.txt ".repeat(30)));
        let r2 = send(h.clone(), 256, "chat").await;
        h.push(back(&r2));
        h.push(user("Fix the bug in it."));
        let r3 = send(h.clone(), 256, "chat").await;
        h.push(back(&r3));
        h.push(tool_result(&r3, &"second read ".repeat(30)));
        let mut sumreq = h.clone();
        sumreq.push(user("Summarize the conversation so far."));
        let sum = send(sumreq, 256, "compaction").await;
        let kept: Vec<Json> = h[h.len() - 2..].to_vec(); // the tool turn: assistant call + its result
        let mut w = vec![json!({"role":"system","content":SYS}),
                         user(&format!("The conversation history before this point was compacted into the following summary:\n<summary>\n{}\n</summary>", sum["message"]["content"].as_str().unwrap()))];
        w.extend(kept);
        if idle {
            send(w.clone(), 0, "prefill").await;
        }
        let post = send(w.clone(), 256, "chat").await; // no new user message: the agent loop continues
        w.push(back(&post));
        w.push(user("Now commit it."));
        let next = send(w.clone(), 256, "chat").await;
        let want = n(&post, "/usage/prompt_tokens") + n(&post, "/usage/completion_tokens") - 1;
        assert_eq!(
            n(&next, "/usage/prompt_tokens_details/cached_tokens"),
            want,
            "{tag}: first append after the rewrite: post {} next {}",
            post["timings"],
            next["timings"]
        );
        w.push(back(&next));
        w.push(user("Thanks."));
        let after = send(w.clone(), 256, "chat").await;
        let want = n(&next, "/usage/prompt_tokens") + n(&next, "/usage/completion_tokens") - 1;
        assert_eq!(
            n(&after, "/usage/prompt_tokens_details/cached_tokens"),
            want,
            "{tag}: second append"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
