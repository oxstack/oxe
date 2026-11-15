//! v0.2.0 OpenAI conformance of oxe-server itself (handoff OPENAI-CONFORMANCE.md, engine side):
//! the default output limit (M2), `/v1/models/{id}`, the legacy `/v1/completions` answer, the
//! OpenAI error object (string `code`, `param`), image parts on a text-only model,
//! `usage.completion_tokens_details.reasoning_tokens`, and the sampler's `presence_penalty` /
//! `frequency_penalty` / `logit_bias` (exact when absent). Model-free: stub engine, synthetic
//! tokenizer, the Qwen3.x template fixture.

use crate::engine::StubEngine;
use crate::service::{default_max_tokens_for, handle, Reply, State};
use crate::tests_state_v013::srv;
use serde_json::{json, Value as Json};
use std::sync::Arc;
use std::time::Duration;

const SYS: &str = "You are a test system.";

async fn call(st: &Arc<State<StubEngine>>, method: &str, path: &str, body: Json) -> (u16, Json) {
    let b = if body.is_null() {
        Vec::new()
    } else {
        serde_json::to_vec(&body).unwrap()
    };
    match handle(st.clone(), method, path, "", &b).await {
        Reply::Json(c, j) => (c, j),
        Reply::Stream(mut rx) => {
            let mut frames = Vec::new();
            while let Some(f) = tokio::time::timeout(Duration::from_secs(30), rx.recv())
                .await
                .expect("stream hung")
            {
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

fn chat_body(thinking: bool, extra: Json) -> Json {
    let mut b = json!({"messages": [{"role":"system","content":SYS},{"role":"user","content":"hi"}], "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": thinking}});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

fn with_default(st: Arc<State<StubEngine>>, n: i64) -> Arc<State<StubEngine>> {
    let mut st = st;
    Arc::get_mut(&mut st)
        .expect("fresh state")
        .default_max_tokens = n;
    st
}

fn assert_error(j: &Json, code: &str, param: Option<&str>) {
    let e = &j["error"];
    let keys: Vec<&str> = e
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .take(4)
        .collect();
    assert_eq!(keys, ["code", "message", "param", "type"], "{j}");
    assert_eq!(e["code"], code, "{j}");
    assert_eq!(e["param"], param.map_or(Json::Null, |p| json!(p)), "{j}");
    assert!(e["message"].as_str().is_some_and(|m| !m.is_empty()), "{j}");
}

/// M2: a request without a length field stops at the server's default (finish `length`); an explicit
/// `max_tokens` / `max_completion_tokens` wins (max_completion_tokens first); `0` disables the
/// default (the request runs to its end). The default is min(16384, n_ctx / 4) and is published.
#[tokio::test(flavor = "multi_thread")]
async fn default_output_limit_applies_only_without_a_length_field() {
    assert_eq!(default_max_tokens_for(None, 1 << 16), 16384);
    assert_eq!(default_max_tokens_for(None, 8192), 2048);
    assert_eq!(default_max_tokens_for(Some("0"), 8192), -1);
    assert_eq!(default_max_tokens_for(Some("777"), 8192), 777);
    assert_eq!(default_max_tokens_for(Some("junk"), 4), 1);
    let long = "Hello there, this is a much longer stub reply text";
    let st = with_default(srv(false, &[long; 6], None), 3);
    let (c, r) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        chat_body(false, json!({})),
    )
    .await;
    assert_eq!(c, 200, "{r}");
    assert_eq!(
        (
            r["usage"]["completion_tokens"].as_u64(),
            &r["choices"][0]["finish_reason"]
        ),
        (Some(3), &json!("length")),
        "{r}"
    );
    let (_, r) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        chat_body(false, json!({"max_tokens": 5})),
    )
    .await;
    assert_eq!(
        r["usage"]["completion_tokens"], 5,
        "explicit max_tokens wins"
    );
    let (_, r) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        chat_body(false, json!({"max_completion_tokens": 2, "max_tokens": 5})),
    )
    .await;
    assert_eq!(
        r["usage"]["completion_tokens"], 2,
        "max_completion_tokens is the OpenAI name and wins"
    );
    let (_, r) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        chat_body(false, json!({"max_tokens": null})),
    )
    .await;
    assert_eq!(r["usage"]["completion_tokens"], 3, "null = absent");
    let (_, r) = call(
        &st,
        "POST",
        "/completion",
        json!({"prompt": "hi", "temperature": 0}),
    )
    .await;
    assert_eq!(r["tokens_predicted"], 3, "the native route too: {r}");
    let st = with_default(srv(false, &[long], None), -1);
    let (_, r) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        chat_body(false, json!({})),
    )
    .await;
    assert_eq!(
        r["choices"][0]["finish_reason"], "stop",
        "no default: runs to the end: {r}"
    );

    let st = with_default(srv(false, &[], None), 1234);
    let (_, m) = call(&st, "GET", "/v1/models", Json::Null).await;
    assert_eq!(m["data"][0]["max_output_tokens"], 1234);
    assert_eq!(m["data"][0]["meta"]["max_output_tokens"], 1234);
    assert_eq!(
        m["data"][0]["context_length"],
        m["data"][0]["meta"]["n_ctx"]
    );
    let (_, p) = call(&st, "GET", "/props", Json::Null).await;
    assert_eq!(p["default_generation_settings"]["max_output_tokens"], 1234);
    assert_eq!(
        p["default_generation_settings"]["params"]["n_predict"],
        1234
    );
}

/// `GET /v1/models/{id}` (OpenAI models.retrieve), the legacy completions route, an unknown route.
#[tokio::test(flavor = "multi_thread")]
async fn models_retrieve_and_route_errors() {
    let st = srv(false, &[], None);
    let (c, m) = call(&st, "GET", "/v1/models/synth", Json::Null).await;
    assert_eq!(c, 200, "{m}");
    assert_eq!((&m["id"], &m["object"]), (&json!("synth"), &json!("model")));
    assert!(m["created"].is_u64() && m["owned_by"].is_string(), "{m}");
    let (c, e) = call(&st, "GET", "/v1/models/gpt-4o", Json::Null).await;
    assert_eq!(c, 404);
    assert_error(&e, "model_not_found", Some("model"));
    // the legacy completions route answers (tests_v020_legacy); a malformed one is a 400 naming its field
    let (c, e) = call(
        &st,
        "POST",
        "/v1/completions",
        json!({"prompt": ["a", "b"]}),
    )
    .await;
    assert_eq!(c, 400, "{e}");
    assert_error(&e, "unsupported_value", Some("prompt"));
    let (c, e) = call(&st, "GET", "/v1/nope", Json::Null).await;
    assert_eq!(c, 404);
    assert_error(&e, "not_found", None);
    assert_eq!(e["error"]["type"], "not_found_error");
}

/// The OpenAI error object: string `code`, `param`; overflow keeps llama-server's type + text.
#[tokio::test(flavor = "multi_thread")]
async fn error_object_shape() {
    let st = srv(false, &[], None);
    let (c, e) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        json!({"messages": "x"}),
    )
    .await;
    assert_eq!(c, 400);
    assert_error(&e, "invalid_request", None);
    assert_eq!(e["error"]["type"], "invalid_request_error");
    // image part on this text-only model
    let b = json!({"messages": [{"role":"user","content":[{"type":"text","text":"what is this"},
                   {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]});
    let (c, e) = call(&st, "POST", "/v1/chat/completions", b).await;
    assert_eq!(c, 400);
    assert_error(&e, "unsupported_content_type", Some("messages"));
    assert!(
        e["error"]["message"]
            .as_str()
            .unwrap()
            .contains("text only"),
        "{e}"
    );
    // context overflow
    let mut st = srv(false, &[], None);
    Arc::get_mut(&mut st).unwrap().n_ctx = 64;
    let big = "word ".repeat(200);
    let (c, e) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        json!({"messages": [{"role":"user","content": big}]}),
    )
    .await;
    assert_eq!(c, 400, "{e}");
    assert_error(&e, "context_length_exceeded", Some("messages"));
    assert_eq!(e["error"]["type"], "exceed_context_size_error");
    assert!(e["error"]["message"]
        .as_str()
        .unwrap()
        .contains("exceeds the available context size"));
}

/// `usage.completion_tokens_details.reasoning_tokens`: the think block's tokens (thinking on), 0 off;
/// in the non-streamed reply and the streamed usage chunk.
#[tokio::test(flavor = "multi_thread")]
async fn usage_reports_reasoning_tokens() {
    let st = srv(
        true,
        &["plan it\n</think>\n\nDone", "plan it\n</think>\n\nDone"],
        None,
    );
    let want = st.tok.encode("plan it\n</think>", false, true).len() as u64;
    let (_, r) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        chat_body(true, json!({})),
    )
    .await;
    assert_eq!(
        r["choices"][0]["message"]["reasoning_content"], "plan it",
        "{r}"
    );
    assert_eq!(
        r["usage"]["completion_tokens_details"]["reasoning_tokens"], want,
        "{r}"
    );
    assert!(r["usage"]["completion_tokens"].as_u64().unwrap() > want);
    assert!(r["usage"]["prompt_tokens_details"]["cached_tokens"].is_u64());
    let (_, frames) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        chat_body(
            true,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        ),
    )
    .await;
    let u = frames
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["usage"].is_object())
        .expect("usage chunk");
    assert_eq!(
        u["usage"]["completion_tokens_details"]["reasoning_tokens"], want,
        "{u}"
    );
    let st = srv(false, &["Done"], None);
    let (_, r) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        chat_body(false, json!({})),
    )
    .await;
    assert_eq!(
        r["usage"]["completion_tokens_details"]["reasoning_tokens"], 0,
        "{r}"
    );
}

/// Penalties / logit_bias: absent or zero = the same reply and the same engine calls as before
/// (greedy fast path kept); a bias changes the pick; out-of-range values are a 400.
#[tokio::test(flavor = "multi_thread")]
async fn sampler_penalties_and_logit_bias() {
    let run = |mut extra: Json| async move {
        extra["max_tokens"] = json!(16);
        let st = srv(false, &["Hello there"], None);
        let (c, r) = call(&st, "POST", "/v1/chat/completions", chat_body(false, extra)).await;
        let calls = st.worker.lock().unwrap().engine.calls.clone();
        (c, r, calls)
    };
    let (_, base, base_calls) = run(json!({})).await;
    for z in [
        json!({"presence_penalty": 0, "frequency_penalty": 0, "logit_bias": {}}),
        json!({"logit_bias": {"5": 0, "7": 0.0}}),
        json!({"logit_bias": [[5, 0]]}),
    ] {
        let (_, zero, zero_calls) = run(z.clone()).await;
        assert_eq!(base["choices"], zero["choices"], "{z}");
        assert_eq!(
            base_calls, zero_calls,
            "{z}: no adjustment (an all-zero bias too): the same engine calls, fast path kept"
        );
    }
    // the first scripted token is banned: the stub's runner-up (id + 1) is picked instead
    let tok = crate::tests_echo::synth_tokenizer();
    let first = tok.encode("Hello there", false, true)[0];
    let (c, biased, _) = run(json!({"logit_bias": {first.to_string(): -100}})).await;
    assert_eq!(c, 200, "{biased}");
    assert_ne!(
        biased["choices"][0]["message"]["content"], base["choices"][0]["message"]["content"],
        "{biased}"
    );
    let (c, pen, _) = run(json!({"presence_penalty": 1.5, "frequency_penalty": 0.5})).await;
    assert_eq!(c, 200, "{pen}");
    for bad in [
        json!({"logit_bias": {"1": 101}}),
        json!({"logit_bias": {"x": 1}}),
        json!({"logit_bias": 3}),
        json!({"presence_penalty": 3}),
    ] {
        let (c, e, _) = run(bad.clone()).await;
        assert_eq!(c, 400, "{bad}: {e}");
        assert_error(&e, "invalid_request", None);
    }
}

/// Codex HOLD (1): with a logit_bias, logprobs (non-streamed and streamed) report the distribution
/// the token was drawn from, not the engine's raw row.
#[tokio::test(flavor = "multi_thread")]
async fn logprobs_follow_the_adjusted_row() {
    let tok = crate::tests_echo::synth_tokenizer();
    let first = tok.encode("Hello there", false, true)[0];
    let runner = first + 1; // the stub's second-best token (logit 5 vs 10)
    let extra = json!({"max_tokens": 1, "logprobs": true, "top_logprobs": 2, "logit_bias": {runner.to_string(): 10}});
    for stream in [false, true] {
        let st = srv(false, &["Hello there"], None);
        let mut e = extra.clone();
        e["stream"] = json!(stream);
        let (c, r) = call(&st, "POST", "/v1/chat/completions", chat_body(false, e)).await;
        assert_eq!(c, 200, "{r}");
        let content = if stream {
            r.as_array()
                .unwrap()
                .iter()
                .find_map(|f| f["choices"][0]["logprobs"]["content"].as_array().cloned())
                .expect("streamed logprobs")
        } else {
            r["choices"][0]["logprobs"]["content"]
                .as_array()
                .cloned()
                .expect("logprobs")
        };
        let t = &content[0];
        assert_eq!(
            t["id"], runner,
            "stream {stream}: the biased token was drawn: {t}"
        );
        let lp = t["logprob"].as_f64().unwrap();
        assert!(
            lp > (0.5f64).ln(),
            "stream {stream}: reported from the adjusted row (raw p would be ~e^-5): {t}"
        );
        assert_eq!(
            t["top_logprobs"][0]["id"], runner,
            "stream {stream}: top-1 of the adjusted row: {t}"
        );
        assert!((t["top_logprobs"][0]["logprob"].as_f64().unwrap() - lp).abs() < 1e-6);
    }
}

/// CI-TRUTH: `--spec-type draft-mtp` on a model WITHOUT an MTP block loads (no crash), warns and runs
/// without speculative decoding; /props.speculative reports it (requested draft-mtp, active none,
/// the reason). The same model decodes exactly as with speculation off; a model WITH an MTP block
/// still gets it.
#[cfg(target_arch = "aarch64")] // the model engine (O-174: x86_64 builds the stub engine only)
#[test]
fn draft_mtp_without_an_mtp_block_degrades_to_none() {
    use crate::engine::Engine;
    use crate::oxe_engine::OxeEngine;
    let dir = std::env::temp_dir().join(format!("oxe-nomtp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cache")).unwrap();
    std::fs::set_permissions(
        dir.join("cache"),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    model::testkit::mini_model(&dir.join("plain.gguf"));
    model::testkit::mini_model_ext(&dir.join("mtp.gguf"), true);
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let n = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    let open = |name: &str, depth: usize| {
        OxeEngine::open_cache(
            &dir.join(name),
            None,
            &[n - 1],
            None,
            64,
            2048,
            depth,
            Some(dir.join("cache")),
        )
    };

    let mut asked = open("plain.gguf", 2).expect("draft-mtp on a model without MTP must load");
    assert_eq!(asked.spec_max_depth(), 0);
    let why = asked.spec_fallback().expect("the fallback is reported");
    assert!(why.contains("MTP"), "{why}");
    let mut plain = open("plain.gguf", 0).unwrap();
    assert!(plain.spec_fallback().is_none());
    // the same greedy output as without speculation
    let prompt: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 64).collect();
    let (mut a, mut b) = (
        crate::sampling::argmax(asked.prefill(&prompt).unwrap()),
        crate::sampling::argmax(plain.prefill(&prompt).unwrap()),
    );
    for _ in 0..8 {
        assert_eq!(a, b);
        assert!(
            asked.decode_spec_argmax(a).unwrap().is_none(),
            "no speculation"
        );
        a = asked.decode_argmax(a).unwrap();
        b = plain.decode_argmax(b).unwrap();
    }
    let p = crate::service::speculative_json(asked.spec_max_depth(), asked.spec_fallback());
    assert_eq!(
        (&p["requested"], &p["active"], &p["depth"]),
        (&json!("draft-mtp"), &json!("none"), &json!(0)),
        "{p}"
    );
    assert!(
        p["fallback"].as_str().is_some_and(|f| f.contains("MTP")),
        "{p}"
    );

    // /props of a server on that engine
    let tok = crate::tests_echo::synth_tokenizer();
    let cfg = crate::service::Config {
        model_name: "plain".into(),
        model_path: "/plain.gguf".into(),
        chat_opts: chat::ServerChatOptions::default(),
        fingerprint: "oxe-test".into(),
        slots: false,
        idle_reprefill: false,
        engine_status: None,
    };
    let st = Arc::new(State::new(cfg, tok, crate::generate::Worker::new(asked, 4)).unwrap());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let props = match rt.block_on(handle(st, "GET", "/props", "", b"")) {
        Reply::Json(200, j) => j,
        _ => panic!("/props"),
    };
    assert_eq!(
        props["speculative"]["active"], "none",
        "{}",
        props["speculative"]
    );
    assert_eq!(props["speculative"]["requested"], "draft-mtp");

    // with an MTP block it is used
    let with = open("mtp.gguf", 2).unwrap();
    assert_eq!(with.spec_max_depth(), 2);
    assert!(with.spec_fallback().is_none());
    let p = crate::service::speculative_json(with.spec_max_depth(), with.spec_fallback());
    assert_eq!(
        (&p["active"], &p["depth"], &p["fallback"]),
        (&json!("draft-mtp"), &json!(2), &Json::Null)
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Codex HOLD (UTF-8 logprobs): a token that ends inside a multibyte character sends no partial, so
/// its row must ride on the next chunk (and rows after the last partial on the finish chunk). The
/// streamed rows, concatenated, equal the non-streamed reply's rows: all of them, in order, for
/// replies with 2-, 3- and 4-byte characters (several byte tokens each) mid-text and at the very end;
/// chat and the native /completion route.
#[tokio::test(flavor = "multi_thread")]
async fn streamed_logprobs_equal_non_streamed_rows_with_multibyte_text() {
    for reply in ["Héllo € 🦀 ok", "ünï 🦀", "€", "plain ascii"] {
        let tok = crate::tests_echo::synth_tokenizer();
        let n_tok = tok.encode(reply, false, true).len();
        let extra = json!({"max_tokens": 64, "logprobs": true, "top_logprobs": 3});
        let st = srv(false, &[reply], None);
        let (_, r) = call(
            &st,
            "POST",
            "/v1/chat/completions",
            chat_body(false, extra.clone()),
        )
        .await;
        let want = r["choices"][0]["logprobs"]["content"]
            .as_array()
            .cloned()
            .expect("logprobs");
        assert_eq!(r["choices"][0]["message"]["content"], reply);
        assert!(
            want.len() > n_tok - 1,
            "{reply}: one row per generated token ({} rows, {n_tok} text tokens)",
            want.len()
        );
        let st = srv(false, &[reply], None);
        let mut e = extra.clone();
        e["stream"] = json!(true);
        let (_, frames) = call(&st, "POST", "/v1/chat/completions", chat_body(false, e)).await;
        let got: Vec<Json> = frames
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["choices"][0]["logprobs"]["content"].as_array().cloned())
            .flatten()
            .collect();
        assert_eq!(
            got, want,
            "{reply}: streamed rows differ from the non-streamed ones"
        );
        let text: String = frames
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, reply);

        // native /completion: n_probs rows
        let b = json!({"prompt": "hi", "temperature": 0, "n_predict": 64, "n_probs": 2});
        let st = srv(false, &[reply], None);
        let (_, r) = call(&st, "POST", "/completion", b.clone()).await;
        let want = r["completion_probabilities"]
            .as_array()
            .cloned()
            .expect("native rows");
        let st = srv(false, &[reply], None);
        let mut b = b;
        b["stream"] = json!(true);
        let (_, frames) = call(&st, "POST", "/completion", b).await;
        let got: Vec<Json> = frames
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["completion_probabilities"].as_array().cloned())
            .flatten()
            .collect();
        assert_eq!(got, want, "{reply}: native streamed rows differ");
    }
}

/// `reasoning_effort` on the engine port: `none` / `minimal` = thinking off, `low` .. `xhigh` = on,
/// whatever the server default; an explicit `enable_thinking` wins; absent, null or an unknown word
/// = the server default; a non-string is a 400.
#[tokio::test(flavor = "multi_thread")]
async fn reasoning_effort_is_the_thinking_switch() {
    const SCRIPT: &str = "plan it\n</think>\n\nDone";
    let body = |extra: Json| {
        let mut b = json!({"messages": [{"role":"system","content":SYS},{"role":"user","content":"hi"}], "temperature": 0});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b
    };
    // (server default, request fields, thinking expected)
    let cases = [
        (false, json!({"reasoning_effort": "none"}), false),
        (false, json!({"reasoning_effort": "minimal"}), false),
        (false, json!({"reasoning_effort": "low"}), true),
        (false, json!({"reasoning_effort": "medium"}), true),
        (false, json!({"reasoning_effort": "high"}), true),
        (false, json!({"reasoning_effort": "xhigh"}), true),
        (true, json!({"reasoning_effort": "none"}), false),
        (true, json!({"reasoning_effort": "minimal"}), false),
        (true, json!({"reasoning_effort": "high"}), true),
        // the explicit template switch wins, both ways
        (
            false,
            json!({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": false}}),
            false,
        ),
        (
            false,
            json!({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": true}}),
            true,
        ),
        (
            false,
            json!({"reasoning_effort": "minimal", "chat_template_kwargs": {"enable_thinking": true}}),
            true,
        ),
        (
            true,
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": false}}),
            false,
        ),
        // absent, null, unknown: the server default
        (false, json!({}), false),
        (true, json!({}), true),
        (true, json!({"reasoning_effort": null}), true),
        (false, json!({"reasoning_effort": "ultra"}), false),
        (true, json!({"reasoning_effort": "ultra"}), true),
    ];
    for (default, extra, thinking) in cases {
        let st = srv(default, &[SCRIPT], None);
        let (c, r) = call(&st, "POST", "/v1/chat/completions", body(extra.clone())).await;
        assert_eq!(c, 200, "{extra} {r}");
        let m = &r["choices"][0]["message"];
        let n = r["usage"]["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .unwrap();
        if thinking {
            assert_eq!(
                m["reasoning_content"], "plan it",
                "default {default} {extra}: {r}"
            );
            assert!(n > 0, "default {default} {extra}: {r}");
        } else {
            assert!(
                m.get("reasoning_content")
                    .is_none_or(|v| v.is_null() || v == ""),
                "default {default} {extra}: {r}"
            );
            assert!(
                m["content"].as_str().is_some_and(|t| !t.is_empty()),
                "default {default} {extra}: {r}"
            );
            assert_eq!(n, 0, "default {default} {extra}: {r}");
        }
    }
    // absent = the same prompt as before, byte for byte (the field alone renders nothing)
    let st = srv(false, &[], None);
    let (_, plain) = call(&st, "POST", "/apply-template", body(json!({}))).await;
    let (_, null) = call(
        &st,
        "POST",
        "/apply-template",
        body(json!({"reasoning_effort": null})),
    )
    .await;
    assert_eq!(plain["prompt"], null["prompt"]);
    let (c, e) = call(
        &st,
        "POST",
        "/v1/chat/completions",
        body(json!({"reasoning_effort": 3})),
    )
    .await;
    assert_eq!(c, 400, "{e}");
}
