//! #591: per-request leftovers. The diagnostic env knobs are read once (at `Worker::new`, copied
//! into `State`), not on every request; a request body is collected into a buffer sized from its
//! `Content-Length`.

use crate::engine::{Engine, StubEngine, StubSnapshot};
use crate::generate::{DiagKnobs, GenParams, Worker};
use crate::tests_hotfix::synthetic_tokenizer;

/// Speculating engine (depth 0 drafts) that tells which decode entry point the loop used.
struct Probe {
    inner: StubEngine,
    argmax_calls: usize,
    rows_calls: usize,
}

impl Engine for Probe {
    type Snapshot = StubSnapshot;
    fn n_ctx(&self) -> usize {
        self.inner.n_ctx()
    }
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_past(&self) -> usize {
        self.inner.n_past()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> crate::engine::Result<&[f32]> {
        self.inner.prefill(t)
    }
    fn decode(&mut self, t: u32) -> crate::engine::Result<&[f32]> {
        self.inner.decode(t)
    }
    fn snapshot(&mut self) -> crate::engine::Result<StubSnapshot> {
        self.inner.snapshot()
    }
    fn restore(&mut self, s: &StubSnapshot) -> crate::engine::Result<()> {
        self.inner.restore(s)
    }
    fn spec_max_depth(&self) -> usize {
        2
    }
    fn decode_spec(&mut self, t: u32) -> crate::engine::Result<Option<(Vec<u32>, Vec<Vec<f32>>)>> {
        self.rows_calls += 1;
        let row = self.inner.decode(t)?.to_vec();
        Ok(Some((Vec::new(), vec![row])))
    }
    fn decode_spec_argmax(
        &mut self,
        t: u32,
    ) -> crate::engine::Result<Option<(Vec<u32>, Vec<u32>)>> {
        self.argmax_calls += 1;
        let row = self.inner.decode(t)?;
        Ok(Some((Vec::new(), vec![crate::sampling::argmax(row)])))
    }
}

fn run(legacy: bool) -> Probe {
    let tok = synthetic_tokenizer();
    let script = tok.encode("abcdefghij", false, false);
    let prompt = tok.encode("hello", false, false);
    let inner = StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script);
    let mut w = Worker::new(
        Probe {
            inner,
            argmax_calls: 0,
            rows_calls: 0,
        },
        4,
    );
    w.diag.legacy = legacy;
    let gp = GenParams {
        n_predict: 8,
        ignore_eos: true,
        sampling: crate::sampling::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    w.engine
}

#[test]
fn legacy_knob_on_the_worker_selects_the_row_loop() {
    let fast = run(false);
    assert!(fast.argmax_calls > 0 && fast.rows_calls == 0, "fast path");
    let legacy = run(true);
    assert!(
        legacy.rows_calls > 0 && legacy.argmax_calls == 0,
        "legacy loop {} {}",
        legacy.rows_calls,
        legacy.argmax_calls
    );
}

#[test]
fn diag_knobs_parse_once_from_a_lookup() {
    let none = DiagKnobs::from_lookup(|_| None);
    assert!(!none.legacy && !none.timing && none.timing_warmup == 16);
    let all = DiagKnobs::from_lookup(|k| match k {
        "OXE_SERVER_LEGACY" | "OXE_TIMING_DIAG" => Some(String::new()),
        "OXE_TIMING_DIAG_WARMUP" => Some("3".into()),
        _ => None,
    });
    assert!(all.legacy && all.timing && all.timing_warmup == 3);
    let bad = DiagKnobs::from_lookup(|k| (k == "OXE_TIMING_DIAG_WARMUP").then(|| "x".into()));
    assert_eq!(bad.timing_warmup, 16);
}

#[test]
fn state_carries_the_worker_knobs() {
    let tok = synthetic_tokenizer();
    let script = tok.encode("ab", false, false);
    let eng = StubEngine::new(256, tok.n_vocab(), tok.eos.unwrap(), script);
    let mut w = Worker::new(eng, 4);
    w.diag.legacy = true;
    let cfg = crate::service::Config {
        model_name: "m".into(),
        model_path: "/m.gguf".into(),
        chat_opts: Default::default(),
        fingerprint: "oxe-test".into(),
        slots: true,
        idle_reprefill: false,
        engine_status: None,
    };
    let st = crate::service::State::new(cfg, tok, w).unwrap();
    assert!(st.diag.legacy);
}

#[test]
fn body_buffer_follows_content_length_up_to_a_cap() {
    use crate::http::body_capacity;
    assert_eq!(body_capacity(None), 0);
    assert_eq!(body_capacity(Some("1234")), 1234);
    assert_eq!(body_capacity(Some("junk")), 0);
    assert_eq!(
        body_capacity(Some("999999999999")),
        crate::http::PREALLOC_MAX
    );
}
