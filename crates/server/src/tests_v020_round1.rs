//! v0.2.0 round-1 leftovers of the engine server: the start-up check of the new OXE_* knobs and the MTP
//! guard (`OXE_MTP_GUARD`), and `/apply-template` `"count_only": true` (the exact prompt length
//! without the prompt text). Model-free: stub engine, synthetic tokenizer, the Qwen3.x template
//! fixture (the same helpers as the other `tests_v020_*` modules).

use crate::engine::StubEngine;
use crate::service::{handle, Reply, State};
use crate::startup_check::check_env;
use crate::tests_state_v013::srv;
use serde_json::{json, Value as Json};
use std::sync::Arc;
use std::time::Duration;

const SYS: &str = "You are a test system.";

async fn call(st: &Arc<State<StubEngine>>, method: &str, path: &str, body: Json) -> (u16, Json) {
    let b = if body.is_null() { Vec::new() } else { serde_json::to_vec(&body).unwrap() };
    match handle(st.clone(), method, path, "", &b).await {
        Reply::Json(c, j) => (c, j),
        Reply::Stream(mut rx) => {
            let mut frames = Vec::new();
            while let Some(f) = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await.expect("stream hung") {
                for line in f.lines() {
                    if let Some(d) = line.strip_prefix("data: ").filter(|d| *d != "[DONE]") {
                        frames.push(serde_json::from_str::<Json>(d).unwrap());
                    }
                }
            }
            (200, Json::Array(frames))
        }
    }
}

fn tools() -> Json {
    json!([{"type": "function", "function": {"name": "run", "description": "Run a command.",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}])
}

/// A chat body with an assistant turn in the history (so the echo splice has something to do).
fn body(thinking: bool, with_tools: bool, extra: Json) -> Json {
    let mut b = json!({"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"}],
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking}});
    if with_tools {
        b["tools"] = tools();
    }
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

fn env1(k: &str, v: &str) -> crate::startup_check::EnvReport {
    check_env(&[(k.to_string(), v.to_string())])
}

#[test]
fn known_l1_knobs_accept_the_values_their_readers_accept() {
    let ok: &[(&str, &[&str])] = &[
        ("OXE_MTP_GUARD", &["off", "0", "0.45", "1", "0.5:128:64", " 0.5:16:0 ", ""]),
    ];
    for (k, vs) in ok {
        for v in *vs {
            let r = env1(k, v);
            assert!(r.errors.is_empty(), "{k}={v:?} must pass: {:?}", r.errors);
            assert!(r.effective.contains_key(*k), "{k}={v:?} has no effective entry");
        }
    }
}

#[test]
fn known_l1_knobs_refuse_what_their_readers_refuse_and_name_the_variable() {
    // the readers match exactly (trimmed): case and near-miss words are refused, as the engine would ignore them
    let bad: &[(&str, &[&str])] = &[
        ("OXE_MTP_GUARD", &["1.5", "x", "0.5:0:1", "0.5:8:1:9", "0.5:8:-1", "nan"]),
    ];
    for (k, vs) in bad {
        for v in *vs {
            let r = env1(k, v);
            assert_eq!(r.errors.len(), 1, "{k}={v:?} must be refused: {:?}", r.errors);
            assert!(r.errors[0].starts_with(&format!("{k}=")), "the error must name the variable: {:?}", r.errors);
            assert!(!r.effective.contains_key(*k));
        }
    }
}

#[test]
fn known_l1_knobs_read_an_empty_value_as_unset() {
    for k in ["OXE_MTP_GUARD"] {
        assert!(env1(k, "").errors.is_empty(), "{k}= (empty) is read as the default");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn count_only_reports_the_length_of_the_return_tokens_render() {
    for thinking in [false, true] {
        for with_tools in [false, true] {
            let st = srv(thinking, &[], None);
            let (c, full) = call(&st, "POST", "/apply-template", body(thinking, with_tools, json!({"return_tokens": true}))).await;
            assert_eq!(c, 200, "{full}");
            let n_tokens = full["tokens"].as_array().unwrap().len();
            let (c, n) = call(&st, "POST", "/apply-template", body(thinking, with_tools, json!({"count_only": true}))).await;
            assert_eq!(c, 200, "{n}");
            let case = format!("thinking={thinking} tools={with_tools}");
            assert_eq!(n["n_prompt_tokens"].as_u64().unwrap() as usize, n_tokens, "{case}");
            assert_eq!(n["n_ctx"], st.n_ctx, "{case}");
            // nothing else: no prompt text, no token array
            let mut keys: Vec<&str> = n.as_object().unwrap().keys().map(|k| k.as_str()).collect();
            keys.sort();
            assert_eq!(keys, ["n_ctx", "n_prompt_tokens"], "{case}: {n}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn count_only_equals_the_prompt_tokens_of_a_real_chat_request() {
    for thinking in [false, true] {
        for with_tools in [false, true] {
            let st = srv(thinking, &["Done."], None);
            let (_, n) = call(&st, "POST", "/apply-template", body(thinking, with_tools, json!({"count_only": true}))).await;
            let n = n["n_prompt_tokens"].as_u64().unwrap();
            let fresh = srv(thinking, &["Done."], None);
            let (c, reply) = call(&fresh, "POST", "/v1/chat/completions", body(thinking, with_tools, json!({}))).await;
            assert_eq!(c, 200, "{reply}");
            assert_eq!(reply["usage"]["prompt_tokens"].as_u64().unwrap(), n, "thinking={thinking} tools={with_tools}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn count_only_with_echo_reports_the_spliced_length() {
    let st = srv(false, &[], None);
    let (_, full) = call(&st, "POST", "/apply-template", body(false, false, json!({"return_tokens": true, "echo": true}))).await;
    let want = full["echo_tokens"].as_array().unwrap().len();
    let (c, n) = call(&st, "POST", "/apply-template", body(false, false, json!({"count_only": true, "echo": true}))).await;
    assert_eq!(c, 200, "{n}");
    assert_eq!(n["n_echo_tokens"].as_u64().unwrap() as usize, want);
    assert_eq!(n["n_prompt_tokens"].as_u64().unwrap() as usize, full["tokens"].as_array().unwrap().len());
    let mut keys: Vec<&str> = n.as_object().unwrap().keys().map(|k| k.as_str()).collect();
    keys.sort();
    assert_eq!(keys, ["n_ctx", "n_echo_tokens", "n_prompt_tokens"], "{n}");
    // without echo there is no echo count
    let (_, plain) = call(&st, "POST", "/apply-template", body(false, false, json!({"count_only": true}))).await;
    assert!(plain.get("n_echo_tokens").is_none(), "{plain}");
}

#[tokio::test(flavor = "multi_thread")]
async fn count_only_bad_bodies_fail_like_the_plain_route() {
    let st = srv(false, &[], None);
    let bad = [json!({"messages": "not a list"}), json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})];
    for b in bad {
        let (c1, j1) = call(&st, "POST", "/apply-template", b.clone()).await;
        let mut with = b.clone();
        with["count_only"] = json!(true);
        let (c2, j2) = call(&st, "POST", "/apply-template", with).await;
        assert_eq!(c2, c1, "{b}");
        assert_eq!(j2, j1, "{b}");
        assert!(c1 >= 400, "{b} must be refused: {c1} {j1}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn count_only_absent_or_false_keeps_today_s_reply() {
    let st = srv(false, &[], None);
    let (_, plain) = call(&st, "POST", "/apply-template", body(false, false, json!({}))).await;
    assert!(plain.get("prompt").is_some() && plain.get("n_prompt_tokens").is_none(), "{plain}");
    let (_, f) = call(&st, "POST", "/apply-template", body(false, false, json!({"count_only": false}))).await;
    assert_eq!(f, plain);
}

#[tokio::test(flavor = "multi_thread")]
async fn props_advertise_count_only_and_keep_the_other_hints() {
    let st = srv(false, &[], None);
    let (c, p) = call(&st, "GET", "/props", Json::Null).await;
    assert_eq!(c, 200);
    assert_eq!(p["oxe_hints"]["count_only"], true, "{p}");
    assert_eq!(p["oxe_hints"]["prefill_only"], true, "{p}");
    assert_eq!(p["oxe_hints"]["sampling"], json!(["presence_penalty", "frequency_penalty", "logit_bias"]), "{p}");
}
