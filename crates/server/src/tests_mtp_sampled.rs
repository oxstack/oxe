//! #462 MTP speculation under sampled decoding (`--mtp-sampled`, default off): a speculated token
//! is used only when the seeded sampler draws that same token from the row, so at a fixed seed the
//! output and the engine state equal decoding without speculation. Model-free: a stub engine with
//! a two-peak row (the sampler leaves the draft path often at a high temperature).

use crate::engine::{Engine, StubEngine, StubSnapshot};
use crate::generate::{GenParams, Worker};
use crate::sampling::SamplingParams;
use crate::tests_hotfix::synthetic_tokenizer;

/// Speculates like the MTP adapter: `decode_spec` appends the token and up to `depth` argmax
/// drafts with their rows; `spec_rewind` drops the unused tail.
struct Spec {
    inner: StubEngine,
    depth: usize,
    before: Option<StubSnapshot>,
    appended: Vec<u32>,
    calls: usize,
    rewinds: usize,
}

impl Engine for Spec {
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
    fn decode_spec(&mut self, t: u32) -> crate::engine::Result<Option<(Vec<u32>, Vec<Vec<f32>>)>> {
        self.calls += 1;
        self.before = Some(self.inner.snapshot()?);
        let mut rows = vec![self.inner.decode(t)?.to_vec()];
        let mut extra = Vec::new();
        self.appended = vec![t];
        while extra.len() < self.depth && self.inner.n_past() + 1 < self.inner.n_ctx {
            let d = crate::sampling::argmax(rows.last().unwrap());
            rows.push(self.inner.decode(d)?.to_vec());
            extra.push(d);
            self.appended.push(d);
        }
        Ok(Some((extra, rows)))
    }
    fn spec_rewind(&mut self, keep: usize) -> crate::engine::Result<()> {
        self.rewinds += 1;
        let s = self.before.take().unwrap();
        self.inner.restore(&s)?;
        for &t in &self.appended[..keep] {
            self.inner.decode(t)?;
        }
        Ok(())
    }
}

fn sampled(seed: u64) -> GenParams {
    GenParams {
        n_predict: 48,
        ignore_eos: true,
        sampling: SamplingParams {
            // the stub's rows are 10 / 5 logits: at 4.0 the second peak wins about a fifth of
            // the draws, so drafts are both used and rejected
            temperature: 4.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            seed,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn sampled_speculation_matches_plain_at_fixed_seed() {
    let tok = synthetic_tokenizer();
    let script = tok.encode("abcdefghijklmnopqrstuvwxyz0123456789", false, false);
    let prompt = tok.encode("hello", false, false);
    let mk = || StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone());
    let (mut total_calls, mut total_rewinds) = (0, 0);
    for seed in [1u64, 7, 42, 1234] {
        let gp = sampled(seed);
        let mut plain = Worker::new(mk(), 4);
        let want = plain.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
        for depth in 1..=3 {
            let mut w = Worker::new(
                Spec {
                    inner: mk(),
                    depth,
                    before: None,
                    appended: vec![],
                    calls: 0,
                    rewinds: 0,
                },
                4,
            );
            w.mtp_sampled = true;
            let got = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
            assert_eq!(got.tokens, want.tokens, "seed {seed}, depth {depth}");
            assert_eq!(got.text, want.text, "seed {seed}, depth {depth}");
            assert_eq!(got.stop, want.stop);
            assert_eq!(
                w.engine.inner.tokens, plain.engine.tokens,
                "seed {seed}, depth {depth}: engine state"
            );
            assert!(
                w.engine.calls > 0,
                "seed {seed}, depth {depth}: no speculation"
            );
            total_calls += w.engine.calls;
            total_rewinds += w.engine.rewinds;
        }
    }
    assert!(total_calls > 0 && total_rewinds > 0, "both paths exercised");
}

/// Default off: a sampled request does not speculate unless `--mtp-sampled` is given.
#[test]
fn sampled_speculation_is_off_by_default() {
    let tok = synthetic_tokenizer();
    let script = tok.encode("abcdefghij", false, false);
    let prompt = tok.encode("hello", false, false);
    let inner = StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script);
    let mut w = Worker::new(
        Spec {
            inner,
            depth: 2,
            before: None,
            appended: vec![],
            calls: 0,
            rewinds: 0,
        },
        4,
    );
    w.generate(&tok, &prompt, &sampled(3), &mut |_| true)
        .unwrap();
    assert_eq!(
        w.engine.calls, 0,
        "decode_spec was called for a sampled request"
    );
}
