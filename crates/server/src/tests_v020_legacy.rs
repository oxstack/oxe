//! v0.2.0 OpenAI parity of the engine port: the legacy `POST /v1/completions` (`text_completion`,
//! streamed and not), the `n` contract on the chat and legacy routes, and the `x-request-id` header
//! of every HTTP reply. Model-free: stub engine, synthetic tokenizer. The socket tests run the real
//! front end on 127.0.0.1.

use crate::engine::StubEngine;
use crate::http::Access;
use crate::service::{default_max_tokens_for, handle, handle_hints, Hints, Reply, State};
use crate::tests_hotfix::start;
use crate::tests_state_v013::srv;
use serde_json::{json, Value as Json};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::Duration;

const REPLY: &str = "Hello there friend, how are you today";

fn merged(mut base: Json, extra: Json) -> Json {
    for (k, v) in extra.as_object().unwrap() {
        base[k] = v.clone();
    }
    base
}

fn legacy(extra: Json) -> Json {
    merged(json!({"model": "synth", "prompt": "Say hi", "temperature": 0}), extra)
}

fn chat_body(extra: Json) -> Json {
    merged(json!({"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}), extra)
}

/// A JSON reply (a stream is a panic: the caller expects a body).
async fn call(st: &Arc<State<StubEngine>>, path: &str, body: Json) -> (u16, Json) {
    match handle(st.clone(), "POST", path, "", &serde_json::to_vec(&body).unwrap()).await {
        Reply::Json(c, j) => (c, j),
        Reply::Stream(_) => panic!("{path}: a stream, not a JSON reply"),
    }
}

/// The raw SSE frames of a streamed reply, in order.
async fn frames(st: &Arc<State<StubEngine>>, path: &str, body: Json) -> Vec<String> {
    let Reply::Stream(mut rx) = handle(st.clone(), "POST", path, "", &serde_json::to_vec(&body).unwrap()).await else {
        panic!("{path}: not a stream");
    };
    let mut out = Vec::new();
    while let Some(f) = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await.expect("stream hung") {
        out.push(f);
    }
    out
}

/// The parsed `data:` chunks of a stream (the `[DONE]` marker and comments left out).
fn chunks(frames: &[String]) -> Vec<Json> {
    frames
        .iter()
        .filter_map(|f| f.strip_prefix("data: "))
        .filter(|d| d.trim() != "[DONE]")
        .map(|d| serde_json::from_str(d.trim_end()).unwrap())
        .collect()
}

fn assert_error(j: &Json, code: &str, param: &str) {
    let e = &j["error"];
    let keys: Vec<&str> = e.as_object().unwrap().keys().map(|k| k.as_str()).take(4).collect();
    assert_eq!(keys, ["code", "message", "param", "type"], "{j}");
    assert_eq!((&e["code"], &e["param"]), (&json!(code), &json!(param)), "{j}");
    assert!(e["message"].as_str().is_some_and(|m| !m.is_empty()), "{j}");
}

/// The concatenated logprobs rows of a streamed reply, in the non-streamed object's shape.
fn joined_logprobs(chunks: &[Json]) -> Json {
    let mut out = json!({"tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []});
    for c in chunks {
        if let Some(lp) = c["choices"][0]["logprobs"].as_object() {
            for k in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
                out[k].as_array_mut().unwrap().extend(lp[k].as_array().unwrap().iter().cloned());
            }
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_reply_has_the_text_completion_shape() {
    let st = srv(false, &[REPLY; 3], None);
    let (c, r) = call(&st, "/v1/completions", legacy(json!({"max_tokens": 4}))).await;
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["object"], "text_completion");
    let id = r["id"].as_str().unwrap();
    assert!(id.starts_with("cmpl-") && id.len() == 29 && id[5..].bytes().all(|b| b.is_ascii_hexdigit()), "{id}");
    assert_eq!((&r["model"], &r["system_fingerprint"]), (&json!("synth"), &json!("oxe-test")), "{r}");
    assert!(r["created"].is_u64(), "{r}");
    let ch = &r["choices"][0];
    assert_eq!((&ch["index"], &ch["logprobs"], &ch["finish_reason"]), (&json!(0), &Json::Null, &json!("length")), "{r}");
    assert!(ch["text"].as_str().is_some_and(|t| !t.is_empty()), "{r}");
    let u = &r["usage"];
    assert_eq!(u["completion_tokens"], 4, "{r}");
    let prompt = u["prompt_tokens"].as_u64().unwrap();
    assert!(prompt > 0 && u["total_tokens"].as_u64() == Some(prompt + 4), "{r}");
    assert!(u["prompt_tokens_details"]["cached_tokens"].is_u64(), "{r}");
    assert!(r["timings"].is_object(), "{r}");
    // the script runs to its end: EOS, finish `stop`, the whole text
    let (_, r) = call(&st, "/v1/completions", legacy(json!({"max_tokens": 64}))).await;
    assert_eq!((&r["choices"][0]["finish_reason"], &r["choices"][0]["text"]), (&json!("stop"), &json!(REPLY)), "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_stream_text_equals_the_whole_reply_and_ends_with_done() {
    let st = srv(false, &[REPLY; 3], None);
    let (_, full) = call(&st, "/v1/completions", legacy(json!({}))).await;
    let st = srv(false, &[REPLY; 3], None);
    let f = frames(&st, "/v1/completions", legacy(json!({"stream": true, "stream_options": {"include_usage": true}}))).await;
    assert_eq!(f.last().unwrap(), "data: [DONE]\n\n", "{f:?}");
    let all = chunks(&f);
    let (usage, body) = all.split_last().unwrap();
    assert!(usage["choices"].as_array().unwrap().is_empty() && usage["usage"]["completion_tokens"].as_u64().unwrap() > 0, "{usage}");
    assert!(usage["timings"].is_object(), "{usage}");
    let last = body.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop", "{last}");
    assert!(body[..body.len() - 1].iter().all(|c| c["choices"][0]["finish_reason"].is_null()), "finish only on the last choice chunk");
    assert!(body.iter().any(|c| !c["choices"][0]["text"].as_str().unwrap().is_empty()), "at least one non-empty piece");
    let text: String = body.iter().map(|c| c["choices"][0]["text"].as_str().unwrap()).collect();
    assert_eq!(text, full["choices"][0]["text"]);
    assert!(body.iter().all(|c| c["object"] == "text_completion" && c["id"] == body[0]["id"] && c["choices"][0]["index"] == 0), "{body:?}");

    // without include_usage there is no usage chunk: the finish chunk is the last data frame
    let st = srv(false, &[REPLY; 3], None);
    let f = frames(&st, "/v1/completions", legacy(json!({"stream": true}))).await;
    let all = chunks(&f);
    assert_eq!(all.last().unwrap()["choices"][0]["finish_reason"], "stop", "{all:?}");
    assert!(all.iter().all(|c| !c["choices"].as_array().unwrap().is_empty()));
    assert_eq!(f.last().unwrap(), "data: [DONE]\n\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn echo_and_logprobs_stream_rows_equal_the_non_stream_rows() {
    let st = srv(false, &[REPLY; 3], None);
    let body = legacy(json!({"echo": true, "logprobs": 2, "max_tokens": 4}));
    let (c, r) = call(&st, "/v1/completions", body.clone()).await;
    assert_eq!(c, 200, "{r}");
    let text = r["choices"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("Say hi"), "echo prefixes the prompt: {text:?}");
    let lp = r["choices"][0]["logprobs"].clone();
    let n = lp["tokens"].as_array().unwrap().len();
    assert!(n > 0, "{r}");
    for k in ["token_logprobs", "top_logprobs", "text_offset"] {
        assert_eq!(lp[k].as_array().unwrap().len(), n, "{k}: {r}");
    }
    let offs: Vec<u64> = lp["text_offset"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(offs[0], 6, "offsets continue after the echoed prompt (6 characters)");
    for i in 0..n - 1 {
        assert_eq!(offs[i + 1] - offs[i], lp["tokens"][i].as_str().unwrap().chars().count() as u64, "row {i}");
    }
    assert!(lp["top_logprobs"][0].is_object() && lp["token_logprobs"][0].is_number(), "{r}");

    let st = srv(false, &[REPLY; 3], None);
    let mut s = body.clone();
    s["stream"] = json!(true);
    let f = frames(&st, "/v1/completions", s).await;
    let all = chunks(&f);
    assert!(all[0]["choices"][0]["text"].as_str().unwrap().starts_with("Say hi"), "the echo rides on the first chunk");
    assert_eq!(joined_logprobs(&all), lp, "streamed rows, concatenated, equal the non-streamed rows");
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_stop_string_ends_the_text() {
    let st = srv(false, &["Hello there friend"], None);
    let (c, r) = call(&st, "/v1/completions", legacy(json!({"stop": ["there"], "max_tokens": 16}))).await;
    assert_eq!(c, 200, "{r}");
    let text = r["choices"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("Hello") && !text.contains("there"), "{text:?}");
    assert_eq!(r["choices"][0]["finish_reason"], "stop", "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn prompt_forms_string_array_and_token_ids_are_one_prompt() {
    let tok = crate::tests_echo::synth_tokenizer();
    let ids = tok.encode("Say hi", true, true);
    let st = srv(false, &[REPLY; 4], None);
    let mut texts = Vec::new();
    for prompt in [json!("Say hi"), json!(["Say hi"]), json!(ids), json!([ids])] {
        let (c, r) = call(&st, "/v1/completions", legacy(json!({"prompt": prompt, "max_tokens": 4}))).await;
        assert_eq!(c, 200, "{r}");
        assert!(r["usage"]["completion_tokens"].as_u64().unwrap() > 0, "{r}");
        texts.push(r["choices"][0]["text"].clone());
    }
    assert!(texts.iter().all(|t| t == &texts[0]), "{texts:?}");
    // an empty suffix is no suffix
    let (c, r) = call(&st, "/v1/completions", legacy(json!({"suffix": "", "max_tokens": 2}))).await;
    assert_eq!(c, 200, "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_refusals_name_the_field() {
    let st = srv(false, &[], None);
    let cases = [
        (json!({"prompt": ["a", "b"]}), "unsupported_value", "prompt"),
        (json!({"prompt": [[1, 2], [3]]}), "unsupported_value", "prompt"),
        (json!({"prompt": []}), "invalid_value", "prompt"),
        (json!({"prompt": 5}), "invalid_value", "prompt"),
        (json!({"prompt": [999_999_999]}), "invalid_value", "prompt"),
        (json!({"prompt": "a", "suffix": "x"}), "unsupported_value", "suffix"),
        (json!({"prompt": "a", "logprobs": 6}), "invalid_value", "logprobs"),
        (json!({"prompt": "a", "logprobs": "2"}), "invalid_value", "logprobs"),
        (json!({"prompt": "a", "echo": "yes"}), "invalid_value", "echo"),
        (json!({"prompt": "a", "n": 2}), "unsupported_value", "n"),
    ];
    for (extra, code, param) in cases {
        let (c, e) = call(&st, "/v1/completions", merged(json!({"model": "synth", "prompt": "a"}), extra.clone())).await;
        assert_eq!(c, 400, "{extra}: {e}");
        assert_error(&e, code, param);
    }
    // a refused body with no prompt at all
    let (c, e) = call(&st, "/v1/completions", json!({"max_tokens": 2})).await;
    assert_eq!(c, 400, "{e}");
    assert_error(&e, "invalid_value", "prompt");
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_default_output_limit_applies_without_a_length_field() {
    let mut st = srv(false, &[REPLY; 3], None);
    Arc::get_mut(&mut st).unwrap().default_max_tokens = 3;
    let (_, r) = call(&st, "/v1/completions", legacy(json!({}))).await;
    assert_eq!((&r["usage"]["completion_tokens"], &r["choices"][0]["finish_reason"]), (&json!(3), &json!("length")), "{r}");
    assert_eq!(default_max_tokens_for(Some("0"), 8192), -1);
    let (_, r) = call(&st, "/v1/completions", legacy(json!({"max_tokens": 2}))).await;
    assert_eq!(r["usage"]["completion_tokens"], 2, "an explicit value wins: {r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_header_gives_a_cache_hit_on_the_repeat() {
    let st = srv(false, &[REPLY; 3], None);
    let b = serde_json::to_vec(&legacy(json!({"max_tokens": 4}))).unwrap();
    let mut replies = Vec::new();
    for _ in 0..2 {
        match handle_hints(st.clone(), "POST", "/v1/completions", "", &b, Some("legacy-s1".into()), None, false, Hints::default()).await {
            Reply::Json(200, j) => replies.push(j),
            Reply::Json(c, j) => panic!("{c}: {j}"),
            Reply::Stream(_) => panic!("not a stream"),
        }
    }
    let cached = replies[1]["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
    assert!(cached > 0, "the repeat reuses the session's prefix: {}", replies[1]);
}

#[tokio::test(flavor = "multi_thread")]
async fn n_is_one_choice_or_a_400_on_both_routes() {
    let st = srv(false, &[REPLY; 40], None);
    for path in ["/v1/chat/completions", "/v1/completions"] {
        let base = || if path.starts_with("/v1/chat") { chat_body(json!({"max_tokens": 4})) } else { legacy(json!({"max_tokens": 4})) };
        let (c, r) = call(&st, path, base()).await;
        assert_eq!(c, 200, "{path} absent: {r}");
        for ok in [Json::Null, json!(1)] {
            let (c, r) = call(&st, path, merged(base(), json!({"n": ok}))).await;
            assert_eq!(c, 200, "{path} n={ok}: {r}");
        }
        for n in [json!(2), json!(5)] {
            let (c, e) = call(&st, path, merged(base(), json!({"n": n}))).await;
            assert_eq!(c, 400, "{path} n={n}: {e}");
            assert_error(&e, "unsupported_value", "n");
            assert!(e["error"]["message"].as_str().unwrap().contains("one choice per request"), "{e}");
        }
        for n in [json!(0), json!(-1), json!("1"), json!(true), json!(1.5)] {
            let (c, e) = call(&st, path, merged(base(), json!({"n": n}))).await;
            assert_eq!(c, 400, "{path} n={n}: {e}");
            assert_error(&e, "invalid_value", "n");
        }
    }
}

// ------------------------------------------------------------------------------ x-request-id

/// One raw HTTP/1.1 exchange: (status, lower-case header block, body).
fn exchange(port: u16, method: &str, path: &str, headers: &[&str], body: &str) -> (u16, String, String) {
    let mut req = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\nContent-Length: {}\r\n", body.len());
    for h in headers {
        req += h;
        req += "\r\n";
    }
    req += "\r\n";
    req += body;
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    s.write_all(req.as_bytes()).unwrap();
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    let text = String::from_utf8_lossy(&out).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    (status, head.to_ascii_lowercase(), body.to_string())
}

/// The `x-request-id` value of a reply head, checked for its shape.
fn request_id_of(head: &str) -> String {
    let id = head.lines().find_map(|l| l.strip_prefix("x-request-id: ")).unwrap_or_else(|| panic!("no x-request-id: {head}")).trim().to_string();
    let hex = id.strip_prefix("req_").unwrap_or_default();
    assert!(hex.len() == 32 && hex.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)), "{id}");
    id
}

#[test]
fn request_ids_are_new_and_well_formed_on_every_kind_of_reply() {
    let (port, _rt) = start(srv(false, &[REPLY; 8], None), None);
    let host = format!("Host: 127.0.0.1:{port}");
    let json_body = legacy(json!({"max_tokens": 2})).to_string();
    let stream_body = legacy(json!({"max_tokens": 2, "stream": true})).to_string();
    let mut ids = Vec::new();
    // a 200 JSON reply, twice: two different ids; a client-sent id is not echoed back
    for _ in 0..2 {
        let (c, h, _) = exchange(port, "POST", "/v1/completions", &[&host, "Content-Type: application/json", "x-request-id: client-chosen"], &json_body);
        assert_eq!(c, 200, "{h}");
        ids.push(request_id_of(&h));
    }
    assert_ne!(ids[0], ids[1]);
    assert!(!ids[0].contains("client-chosen"));
    // a stream
    let (c, h, b) = exchange(port, "POST", "/v1/completions", &[&host, "Content-Type: application/json"], &stream_body);
    assert_eq!(c, 200, "{h}");
    assert!(h.contains("content-type: text/event-stream"), "{h}");
    assert!(b.contains("data: [DONE]"), "{b}");
    ids.push(request_id_of(&h));
    // a 404
    let (c, h, _) = exchange(port, "GET", "/no-such-route", &[&host], "");
    assert_eq!(c, 404);
    ids.push(request_id_of(&h));
    // a 403 from the access check (a foreign Host)
    let (c, h, _) = exchange(port, "GET", "/props", &["Host: evil.example"], "");
    assert_eq!(c, 403, "{h}");
    ids.push(request_id_of(&h));
    // the 204 answer to a preflight
    let (c, h, _) = exchange(port, "OPTIONS", "/props", &[&host], "");
    assert_eq!(c, 204, "{h}");
    ids.push(request_id_of(&h));
    let distinct: HashSet<&String> = ids.iter().collect();
    assert_eq!(distinct.len(), ids.len(), "{ids:?}");
}

#[test]
fn a_401_reply_carries_a_request_id_too() {
    let access = Access { keys: vec![b"secret".to_vec()], ..Default::default() };
    let (port, _rt) = start(srv(false, &[REPLY], None), Some(access));
    let host = format!("Host: 127.0.0.1:{port}");
    let (c, h, _) = exchange(port, "GET", "/props", &[&host], "");
    assert_eq!(c, 401, "{h}");
    let first = request_id_of(&h);
    let (c, h, _) = exchange(port, "GET", "/props", &[&host, "Authorization: Bearer secret"], "");
    assert_eq!(c, 200, "{h}");
    assert_ne!(first, request_id_of(&h));
}

#[test]
fn request_id_source_gives_distinct_well_formed_values() {
    let ids: HashSet<String> = (0..64).map(|_| crate::http::request_id()).collect();
    assert_eq!(ids.len(), 64);
    for id in &ids {
        let hex = id.strip_prefix("req_").unwrap();
        assert_eq!(hex.len(), 32);
        assert!(hex.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)), "{id}");
    }
}

// ------------------------------------------------------------------------- the chat route

#[tokio::test(flavor = "multi_thread")]
async fn chat_route_still_answers_the_same_for_n_one() {
    let st = srv(false, &[REPLY], None);
    let (c, r) = call(&st, "/v1/chat/completions", chat_body(json!({"max_tokens": 4, "n": 1}))).await;
    assert_eq!(c, 200, "{r}");
    assert_eq!(r["object"], "chat.completion");
}

/// An explicit `logprobs: 0` is not an absent field: the chosen token's logprob is reported with an
/// empty top list (stream and non-stream, same rows); absent or null gives no logprobs object.
#[tokio::test(flavor = "multi_thread")]
async fn logprobs_zero_reports_the_chosen_tokens_without_candidates() {
    let st = srv(false, &[REPLY; 4], None);
    let (c, zero) = call(&st, "/v1/completions", legacy(json!({"logprobs": 0, "max_tokens": 4}))).await;
    assert_eq!(c, 200, "{zero}");
    let lp = zero["choices"][0]["logprobs"].clone();
    let n = lp["tokens"].as_array().expect("a logprobs object for an explicit 0").len();
    assert!(n > 0, "{zero}");
    for k in ["token_logprobs", "top_logprobs", "text_offset"] {
        assert_eq!(lp[k].as_array().unwrap().len(), n, "{k}: {zero}");
    }
    for i in 0..n {
        assert!(lp["token_logprobs"][i].is_number(), "row {i}: {zero}");
        assert_eq!(lp["top_logprobs"][i], json!({}), "row {i}: no candidates for logprobs 0");
    }
    // the same chosen tokens and logprobs as a request that asks for candidates
    let (_, two) = call(&st, "/v1/completions", legacy(json!({"logprobs": 2, "max_tokens": 4}))).await;
    assert_eq!(two["choices"][0]["text"], zero["choices"][0]["text"]);
    for k in ["tokens", "token_logprobs", "text_offset"] {
        assert_eq!(two["choices"][0]["logprobs"][k], lp[k], "{k}");
    }
    assert_eq!(two["choices"][0]["logprobs"]["top_logprobs"][0].as_object().unwrap().len(), 2);
    // streamed: the rows, concatenated, equal the non-streamed rows
    let f = frames(&st, "/v1/completions", legacy(json!({"logprobs": 0, "max_tokens": 4, "stream": true}))).await;
    assert_eq!(joined_logprobs(&chunks(&f)), lp, "streamed rows");
    // absent and null: no logprobs object
    for extra in [json!({"max_tokens": 4}), json!({"max_tokens": 4, "logprobs": null})] {
        let st = srv(false, &[REPLY], None);
        let (_, r) = call(&st, "/v1/completions", legacy(extra)).await;
        assert_eq!(r["choices"][0]["logprobs"], Json::Null, "{r}");
    }
}
