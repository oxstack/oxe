//! Request budgets exercise the actual generation/cache/parser path, without model files.
use crate::engine::{Engine, Result};
use crate::generate::Worker;
use crate::service::{handle, Config, Reply, State};
use crate::tests_echo::synth_tokenizer;
use serde_json::{json, Value};
use std::sync::Arc;
use tokenizer::Token;

/// Never ends its reasoning voluntarily. Only EVALUATING </think> produces answer A;
/// inserting text into a response without advancing the engine cannot pass these tests.
struct ThinkForever {
    tokens: Vec<Token>,
    logits: Vec<f32>,
    close: Token,
    start: Token,
    literal_answer: bool,
    eos: Token,
    spec: bool,
    spec_start: usize,
    spec_calls: usize,
}
impl Engine for ThinkForever {
    type Snapshot = Vec<Token>;
    fn n_ctx(&self) -> usize {
        8192
    }
    fn n_vocab(&self) -> usize {
        self.logits.len()
    }
    fn n_past(&self) -> usize {
        self.tokens.len()
    }
    fn reset(&mut self) {
        self.tokens.clear();
    }
    fn prefill(&mut self, tokens: &[Token]) -> Result<&[f32]> {
        self.tokens.extend_from_slice(tokens);
        let next = match self.tokens.last().copied() {
            Some(t) if t == self.close && self.literal_answer => self.start,
            Some(t) if t == self.start && self.literal_answer => b'X' as Token,
            Some(t) if t == b'X' as Token && self.literal_answer => self.eos,
            Some(t) if t == self.close => b'A' as Token,
            Some(t) if t == b'A' as Token => self.eos,
            _ => b'a' as Token,
        };
        self.logits.fill(-100.0);
        self.logits[next as usize] = 10.0;
        Ok(&self.logits)
    }
    fn snapshot(&mut self) -> Result<Self::Snapshot> {
        Ok(self.tokens.clone())
    }
    fn restore(&mut self, snapshot: &Self::Snapshot) -> Result<()> {
        self.tokens.clear();
        if !snapshot.is_empty() {
            self.prefill(snapshot)?;
        }
        Ok(())
    }
    fn spec_max_depth(&self) -> usize {
        usize::from(self.spec)
    }
    fn decode_spec(&mut self, t: Token) -> Result<Option<(Vec<Token>, Vec<Vec<f32>>)>> {
        if !self.spec {
            return Ok(None);
        }
        self.spec_calls += 1;
        self.spec_start = self.tokens.len();
        self.prefill(&[t])?;
        let mut rows = vec![self.logits.clone()];
        let next = crate::sampling::argmax(&self.logits);
        let mut extra = Vec::new();
        if next != self.eos {
            self.prefill(&[next])?;
            extra.push(next);
            rows.push(self.logits.clone());
        }
        Ok(Some((extra, rows)))
    }
    fn spec_rewind(&mut self, keep: usize) -> Result<()> {
        let saved = self.tokens[..self.spec_start + keep].to_vec();
        self.restore(&saved)
    }
}
fn state(spec: bool) -> Arc<State<ThinkForever>> {
    let tok = synth_tokenizer();
    let engine = ThinkForever {
        tokens: vec![],
        logits: vec![0.; tok.n_vocab()],
        close: tok.token_id("</think>").unwrap(),
        start: tok.token_id("<think>").unwrap(),
        literal_answer: false,
        eos: tok.eos.unwrap(),
        spec,
        spec_start: 0,
        spec_calls: 0,
    };
    let cfg = Config {
        model_name: "synthetic".into(),
        model_path: "/model.gguf".into(),
        chat_opts: chat::ServerChatOptions {
            enable_thinking: true,
            ..Default::default()
        },
        fingerprint: "budget-test".into(),
        slots: false,
        idle_reprefill: false,
        engine_status: None,
    };
    Arc::new(State::new(cfg, tok, Worker::new(engine, 8)).unwrap())
}
async fn chat(
    st: Arc<State<ThinkForever>>,
    budget: Option<Value>,
    stream: bool,
    total: usize,
) -> (u16, Value) {
    let mut body = json!({"messages":[{"role":"user","content":"answer"}],
        "temperature":0,"max_tokens":total,"stream":stream,"stream_options":{"include_usage":true}});
    if let Some(b) = budget {
        body["thinking_budget"] = b;
    }
    match handle(
        st,
        "POST",
        "/v1/chat/completions",
        "",
        &serde_json::to_vec(&body).unwrap(),
    )
    .await
    {
        Reply::Json(c, j) => (c, j),
        Reply::Stream(mut rx) => {
            let mut frames = vec![];
            while let Some(frame) =
                tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
                    .await
                    .unwrap()
            {
                for line in frame.lines() {
                    if let Some(data) = line.strip_prefix("data: ").filter(|s| *s != "[DONE]") {
                        frames.push(serde_json::from_str::<Value>(data).unwrap());
                    }
                }
            }
            (200, Value::Array(frames))
        }
    }
}
#[tokio::test(flavor = "multi_thread")]
async fn thinking_budget_forces_an_evaluated_boundary_and_an_answer_with_or_without_mtp() {
    for spec in [false, true] {
        let st = state(spec);
        let (code, r) = chat(st.clone(), Some(json!(3)), false, 16).await;
        assert_eq!(code, 200, "{r}");
        assert_eq!(r["choices"][0]["message"]["reasoning_content"], "aaa");
        assert_eq!(r["choices"][0]["message"]["content"], "A");
        assert_eq!(
            r["usage"]["completion_tokens_details"]["reasoning_tokens"],
            4
        );
        let worker = st.worker.lock().unwrap();
        assert!(worker.engine.tokens.ends_with(&[
            b'a' as Token,
            b'a' as Token,
            b'a' as Token,
            worker.engine.close,
            b'A' as Token
        ]));
        assert_eq!(worker.engine.tokens, worker.cache.tokens);
        if spec {
            assert!(worker.engine.spec_calls > 0);
        }
    }
}
#[tokio::test(flavor = "multi_thread")]
async fn thinking_budget_stream_separates_reasoning_from_answer_and_total_limit_still_wins() {
    let (_, frames) = chat(state(true), Some(json!(99)), true, 5).await;
    let frames = frames.as_array().unwrap();
    let reasoning: String = frames
        .iter()
        .filter_map(|f| {
            f.pointer("/choices/0/delta/reasoning_content")
                .and_then(Value::as_str)
        })
        .collect();
    let answer: String = frames
        .iter()
        .filter_map(|f| {
            f.pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(reasoning, "aaa");
    assert_eq!(answer, "A");
    assert_eq!(
        frames.iter().find_map(|f| f
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)),
        Some(5)
    );
}
#[tokio::test(flavor = "multi_thread")]
async fn thinking_budget_zero_closes_immediately_and_omitted_retains_length_stop() {
    let (_, zero) = chat(state(false), Some(json!(0)), false, 8).await;
    assert_eq!(zero["choices"][0]["message"]["content"], "A");
    assert_eq!(
        zero["choices"][0]["message"]["reasoning_content"],
        Value::Null
    );
    let (_, unbounded) = chat(state(false), None, false, 8).await;
    assert_eq!(unbounded["choices"][0]["finish_reason"], "length");
    assert_eq!(
        unbounded["choices"][0]["message"]["reasoning_content"],
        "aaaaaaaa"
    );
}
#[tokio::test(flavor = "multi_thread")]
async fn thinking_budget_rejects_invalid_values_without_poisoning_the_next_request() {
    let st = state(false);
    for budget in [json!(-1), json!(1.5), json!(true), json!("5")] {
        assert_eq!(chat(st.clone(), Some(budget), false, 16).await.0, 400);
    }
    assert_eq!(chat(st, Some(json!(2)), false, 16).await.0, 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn thinking_budget_prefill_only_stays_empty_and_one_token_limit_is_refused() {
    let (code, prefill) = chat(state(false), Some(json!(0)), false, 0).await;
    assert_eq!(code, 200);
    assert_eq!(prefill["usage"]["completion_tokens"], 0);
    assert_eq!(chat(state(false), Some(json!(0)), false, 1).await.0, 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn thinking_budget_off_template_does_not_change_plain_answer_bytes() {
    let body = json!({"messages":[{"role":"user","content":"answer"}],"temperature":0,
        "max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}});
    let plain = match handle(
        state(true),
        "POST",
        "/v1/chat/completions",
        "",
        &serde_json::to_vec(&body).unwrap(),
    )
    .await
    {
        Reply::Json(200, j) => j,
        _ => panic!("plain request failed"),
    };
    let mut bounded = body;
    bounded["thinking_budget"] = json!(1);
    let bounded = match handle(
        state(true),
        "POST",
        "/v1/chat/completions",
        "",
        &serde_json::to_vec(&bounded).unwrap(),
    )
    .await
    {
        Reply::Json(200, j) => j,
        _ => panic!("budget request failed"),
    };
    assert_eq!(plain["choices"], bounded["choices"]);
    assert_eq!(plain["usage"], bounded["usage"]);
}

#[test]
fn thinking_budget_cancel_at_forced_marker_keeps_speculative_state_reusable() {
    use crate::generate::{GenError, GenParams};
    let st = state(true);
    let prompt = st.tok.encode("<think>\n", false, true);
    let start = st.tok.token_id("<think>").unwrap();
    let end = st.tok.token_id("</think>").unwrap();
    let params = GenParams {
        n_predict: 16,
        thinking_budget: Some(3),
        thinking_markers: Some(crate::thinking_budget::Markers {
            start,
            end,
            opened: true,
        }),
        sampling: crate::sampling::SamplingParams {
            temperature: 0.,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut worker = st.worker.lock().unwrap();
    let mut emitted = vec![];
    let r = worker.generate(&st.tok, &prompt, &params, &mut |p| {
        emitted.push(p.token);
        p.token != end
    });
    assert!(matches!(r, Err(GenError::Cancelled)));
    assert_eq!(worker.engine.tokens, worker.cache.tokens);
    let follow = [prompt, emitted].concat();
    let r = worker
        .generate(
            &st.tok,
            &follow,
            &GenParams {
                n_predict: 4,
                sampling: crate::sampling::SamplingParams {
                    temperature: 0.,
                    ..Default::default()
                },
                ..Default::default()
            },
            &mut |_| true,
        )
        .unwrap();
    assert_eq!(r.text, "A");
}

#[tokio::test(flavor = "multi_thread")]
async fn thinking_budget_discovery_and_unsupported_template_are_fail_closed() {
    let mut st = state(false);
    let props = handle(st.clone(), "GET", "/props", "", b"").await;
    assert!(matches!(props, Reply::Json(200, ref p) if p["thinking_budget"] == true));
    Arc::get_mut(&mut st)
        .unwrap()
        .tmpl
        .as_mut()
        .unwrap()
        .supports_thinking = false;
    let props = handle(st.clone(), "GET", "/props", "", b"").await;
    assert!(matches!(props, Reply::Json(200, ref p) if p["thinking_budget"] == false));
    assert_eq!(chat(st.clone(), Some(json!(3)), false, 16).await.0, 400);
    assert!(st.worker.lock().unwrap().engine.tokens.is_empty());
    let raw = handle(
        st,
        "POST",
        "/completion",
        "",
        br#"{"prompt":"answer","thinking_budget":3}"#,
    )
    .await;
    assert!(matches!(raw, Reply::Json(400, _)));
}

#[tokio::test(flavor = "multi_thread")]
async fn thinking_budget_does_not_rewrite_literal_think_tags_in_the_answer() {
    let mut st = state(true);
    Arc::get_mut(&mut st)
        .unwrap()
        .worker
        .get_mut()
        .unwrap()
        .engine
        .literal_answer = true;
    let (_, reply) = chat(st, Some(json!(3)), false, 16).await;
    assert_eq!(reply["choices"][0]["finish_reason"], "stop");
    assert_eq!(reply["choices"][0]["message"]["reasoning_content"], "aaa");
    assert_eq!(reply["choices"][0]["message"]["content"], "<think>X");
}
