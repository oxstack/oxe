//! The dense `qwen35` architecture (the Qwen3.5 small models) on the mini dense GGUF, model-free:
//! descriptor and tied head, the dense FFN and the head against an f32 reference of their
//! definition, and the engine invariants the MoE model has (thread count, small prefill ==
//! decode, batched prefill, verify lanes, snapshot / state round trips).

use super::*;
use crate::engine::EngineOptions;

fn dense_paths(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("oxe-dense-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dense.gguf");
    mini_dense_model(&path);
    (dir, path)
}

fn cfg(trace: bool) -> EngineConfig {
    EngineConfig {
        n_ctx: 512,
        logits: true,
        trace,
        profile: false,
    }
}

/// Every row of a 2-D tensor of the original GGUF, dequantised to f32.
fn rows_f32(g: &gguf::Gguf, name: &str) -> Vec<Vec<f32>> {
    let t = g.header.tensor(name).unwrap();
    let (k, rows) = (t.shape[0] as usize, t.shape[1] as usize);
    let q = crate::desc::qtype(t.ty).unwrap();
    let rb = q.row_bytes(k);
    let data = g.tensor_data(t);
    (0..rows)
        .map(|r| {
            let mut out = vec![0f32; k];
            kernels_sys::dequantize_row(q, &data[r * rb..(r + 1) * rb], &mut out);
            out
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>() as f32
}

/// `x` through the activation quantisation the GEMV kernels apply to their input (Q8_K: f32
/// scale + 256 int8 per block; Q8_0: the weight block layout), back to f32.
fn act_roundtrip(at: kernels_sys::ActType, x: &[f32]) -> Vec<f32> {
    let mut q = vec![0u8; at.bytes(x.len())];
    kernels_sys::quantize_act(at, x, &mut q);
    let mut out = vec![0f32; x.len()];
    match at {
        kernels_sys::ActType::Q8K => {
            let qk = kernels_sys::QK_K;
            let bb = q.len() / (x.len() / qk);
            for (b, o) in q.chunks_exact(bb).zip(out.chunks_exact_mut(qk)) {
                let d = f32::from_le_bytes(b[0..4].try_into().unwrap());
                for (v, &c) in o.iter_mut().zip(&b[4..4 + qk]) {
                    *v = d * (c as i8) as f32;
                }
            }
        }
        kernels_sys::ActType::Q8_0 => kernels_sys::dequantize_row(QType::Q8_0, &q, &mut out),
    }
    out
}

fn act_of(g: &gguf::Gguf, name: &str) -> kernels_sys::ActType {
    crate::desc::qtype(g.header.tensor(name).unwrap().ty)
        .unwrap()
        .act()
}

/// down(silu(gate . x) * (up . x)) in f32 (f64 sums), with `hq` applied to h before the down rows.
fn ffn_ref(
    gate: &[Vec<f32>],
    up: &[Vec<f32>],
    down: &[Vec<f32>],
    x: &[f32],
    hq: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let h: Vec<f32> = gate
        .iter()
        .zip(up)
        .map(|(gr, ur)| {
            let (a, b) = (dot(gr, x), dot(ur, x));
            a / (1.0 + (-a).exp()) * b
        })
        .collect();
    let h = hq(&h);
    down.iter().map(|r| dot(r, &h)).collect()
}

/// max |a - b| / max |b|
fn rel_gap(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sc = b.iter().fold(1e-30f32, |m, x| m.max(x.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / sc
}

/// The descriptor reads `feed_forward_length` and no expert keys, the head is the token
/// embedding (no `output.weight` in the file), each layer's FFN is one gate/up/down triple, and
/// the MoE-only MTP head is refused (the server then runs without speculation).
#[test]
fn mini_dense_descriptor_tied_head_and_no_mtp() {
    let (dir, path) = dense_paths("desc");
    let g = gguf::Gguf::open(&path).unwrap();
    let d = ModelDesc::from_header(&g.header).unwrap();
    assert!(d.is_dense());
    assert_eq!(d.arch, "qwen35");
    assert_eq!(d.ff_dense as u64, MINI_DENSE_FF);
    assert_eq!(
        (d.n_expert, d.n_expert_used, d.ff_exp, d.ff_shexp),
        (0, 0, 0, 0)
    );
    assert_eq!(d.n_vocab, 64);
    let plan = Plan::build(&g.header, &d, 3, 512).unwrap();
    plan.check_coverage().unwrap();
    let (e, o) = (&g.header.tensors[plan.embd], &g.header.tensors[plan.output]);
    assert_eq!(e.name, "token_embd.weight");
    assert_eq!((&o.shape, o.ty, o.offset), (&e.shape, e.ty, e.offset));
    for l in &plan.layers {
        assert!(matches!(l.ffn, crate::plan::Ffn::Dense(_)));
    }
    assert!(plan.phases.iter().all(|p| !matches!(
        p.kind,
        PhaseKind::MoeRouter | PhaseKind::MoeGateUp | PhaseKind::MoeDown
    )));
    // a dense FFN width that is not whole 256-row blocks is refused with its reason
    let mut h = g.header.clone();
    for (k, v) in h.metadata.iter_mut() {
        if k == "qwen35.feed_forward_length" {
            *v = Value::U32(384);
        }
    }
    assert!(ModelDesc::from_header(&h).is_err());
    let opt = mini_opts(&dir);
    let e = crate::mtp::MtpWeights::load(&path, &plan, &opt)
        .err()
        .unwrap();
    assert!(e.contains("dense"), "{e}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn mini_dense_rejects_f32_down_projection_before_engine_construction() {
    let (dir, path) = dense_paths("f32-down");
    let g = gguf::Gguf::open(&path).unwrap();
    let d = ModelDesc::from_header(&g.header).unwrap();
    let plan = Plan::build(&g.header, &d, 3, 512).unwrap();
    let mut tensors = g.header.tensors.clone();
    let crate::plan::Ffn::Dense(ffn) = &plan.layers[0].ffn else {
        unreachable!()
    };
    let down = ffn.down;
    tensors[down].ty = gguf::GgmlType::F32;

    let err = WeightStore::check_supported(&plan, &tensors).unwrap_err();
    assert!(err.contains("blk.0.ffn_down.weight"), "{err}");
    assert!(err.contains("F32 dense down projection"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The dense FFN of every layer and the tied head against f32 math on the original GGUF rows
/// (down(silu(gate . xn) * (up . xn)), embedding rows . result_norm), from the engine's own
/// intermediates. Pure f32 differs by the activation quantisation of the GEMVs (a few percent on
/// these random weights); with the GEMV inputs passed through that quantisation only summation
/// order is left. A wrong row, block, gate or head tensor is off by O(1).
#[test]
fn mini_dense_ffn_and_tied_head_match_f32_reference() {
    let (dir, path) = dense_paths("f32");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (plan, ws, pool) = mini_engine_parts(&path, &opt, 2);
    let g = gguf::Gguf::open(&path).unwrap();
    let embd = rows_f32(&g, "token_embd.weight");
    let head_at = act_of(&g, "output.weight");
    let layers: Vec<_> = (0..plan.desc.n_layer)
        .map(|l| {
            let n = |s: &str| format!("blk.{l}.{s}");
            let r = |s: &str| rows_f32(&g, &n(s));
            let ats = (
                act_of(&g, &n("ffn_gate.weight")),
                act_of(&g, &n("ffn_down.weight")),
            );
            (
                r("ffn_gate.weight"),
                r("ffn_up.weight"),
                r("ffn_down.weight"),
                ats,
            )
        })
        .collect();
    let eng = Engine::new(&plan, &ws, cfg(true)).unwrap();
    let (mut ffn, mut ffn_q, mut head, mut head_q) = (0f32, 0f32, 0f32, 0f32);
    for t in [5u32, 17, 33, 2, 60, 9, 41, 23] {
        eng.step(&pool, t);
        let m = eng.trace.as_ref().unwrap().take();
        for (l, (gate, up, down, (gu_at, dn_at))) in layers.iter().enumerate() {
            let xn = &m[&format!("attn_post_norm-{l}")];
            let got = &m[&format!("ffn_out-{l}")];
            assert!(got.iter().all(|v| v.is_finite()));
            let want = ffn_ref(gate, up, down, xn, &|h| h.to_vec());
            ffn = ffn.max(rel_gap(got, &want));
            let xq = act_roundtrip(*gu_at, xn);
            let want = ffn_ref(gate, up, down, &xq, &|h| act_roundtrip(*dn_at, h));
            ffn_q = ffn_q.max(rel_gap(got, &want));
        }
        let hn = &m["result_norm"];
        let want: Vec<f32> = embd.iter().map(|r| dot(r, hn)).collect();
        head = head.max(rel_gap(eng.logits(), &want));
        let hq = act_roundtrip(head_at, hn);
        let want: Vec<f32> = embd.iter().map(|r| dot(r, &hq)).collect();
        head_q = head_q.max(rel_gap(eng.logits(), &want));
    }
    eprintln!(
        "dense mini, max relative gap: FFN {ffn:.2e} to f32, {ffn_q:.2e} with quantised GEMV \
         inputs; tied head {head:.2e}, {head_q:.2e}"
    );
    assert!(ffn < 0.2 && head < 0.2, "vs f32: FFN {ffn}, head {head}");
    assert!(
        ffn_q < 5e-3,
        "dense FFN vs the quantised-input reference: {ffn_q}"
    );
    assert!(
        head_q < 5e-3,
        "tied head vs the quantised-input reference: {head_q}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Decode is bit-identical at 1-4 threads (a thread may own no FFN block), argmax matches the
/// logits, and the barriers per token are the plan's (two per dense FFN, plus the tiled
/// attention's two inner ones).
#[test]
fn mini_dense_decode_is_thread_count_invariant() {
    let (dir, path) = dense_paths("decode");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let g = gguf::Gguf::open(&path).unwrap();
    let d = ModelDesc::from_header(&g.header).unwrap();
    let plan = Plan::build(&g.header, &d, 1, 512).unwrap();
    let ffn_barriers = plan
        .phases
        .iter()
        .filter(|p| matches!(p.kind, PhaseKind::DenseGateUp | PhaseKind::DenseDown))
        .count();
    assert_eq!(ffn_barriers, 2 * d.n_layer);
    let (ids1, lg1, b1) = run_mini(&path, &opt, 1, 12);
    assert_eq!(b1 as usize, plan.barriers_per_token() + 2);
    for (i, lg) in lg1.iter().enumerate() {
        assert!(lg.iter().all(|v| v.is_finite()));
        assert_eq!(ids1[i] as usize, argmax(lg));
    }
    for n in [2usize, 3, 4] {
        let (ids, lg, b) = run_mini(&path, &opt, n, 12);
        assert_eq!(b, b1, "{n} threads: barriers");
        assert_eq!(ids, ids1, "{n} threads: ids differ");
        for (s, (a, b)) in lg.iter().zip(&lg1).enumerate() {
            assert!(same_bits(a, b), "{n} threads: logits of step {s} differ");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// A prefill of 1-3 tokens (GEMV rows) equals the same tokens decoded one by one, logits and
/// state; a batched prefill (GEMM rows, ubatches of 9) is thread-count invariant and decode
/// continues identically.
#[test]
fn mini_dense_prefill_equals_decode_and_is_thread_invariant() {
    let (dir, path) = dense_paths("prefill");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, 3);
        let dec = Engine::new(&plan, &ws, cfg(false)).unwrap();
        let pf = Engine::new(&plan, &ws, cfg(false)).unwrap();
        let toks: Vec<u32> = vec![5, 17, 33, 2, 60, 9];
        let mut i = 0;
        for len in [1usize, 2, 3] {
            for &t in &toks[i..i + len] {
                dec.step(&pool, t);
            }
            let am = pf
                .prefill(&pool, &toks[i..i + len], crate::engine::UBATCH)
                .unwrap();
            i += len;
            assert!(
                same_bits(dec.logits(), pf.logits()),
                "prefill of {len}: logits"
            );
            assert!(
                same_state(&dec.dump_state(), &pf.dump_state()),
                "prefill of {len}: state"
            );
            assert_eq!(am as usize, argmax(pf.logits()));
        }
    }

    let toks: Vec<u32> = (0..23).map(|i| (i * 13 + 7) % 64).collect();
    let mut want: Option<(Vec<f32>, crate::engine::StateDump, Vec<Vec<f32>>)> = None;
    for n in [1usize, 2, 3] {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
        let eng = Engine::new(&plan, &ws, cfg(false)).unwrap();
        let mut next = eng.prefill(&pool, &toks, 9).unwrap();
        let lg = eng.logits().to_vec();
        assert!(lg.iter().all(|v| v.is_finite()));
        let st = eng.dump_state();
        let mut dec = Vec::new();
        for _ in 0..3 {
            next = eng.step(&pool, next);
            dec.push(eng.logits().to_vec());
        }
        match &want {
            None => want = Some((lg, st, dec)),
            Some((l1, s1, d1)) => {
                assert!(same_bits(&lg, l1), "{n} threads: prefill logits differ");
                assert!(same_state(&st, s1), "{n} threads: prefill state differs");
                for (a, b) in dec.iter().zip(d1) {
                    assert!(same_bits(a, b), "{n} threads: decode after prefill differs");
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// verify(k) == k one-token steps bit for bit for every lane (the dense FFN runs all lanes over
/// the same rows), with partial commits rolling KV and recurrent state back, at 1-3 threads.
#[test]
fn mini_dense_verify_lanes_are_batch_invariant_with_rollback() {
    let (dir, path) = dense_paths("lanes");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    for nt in [1usize, 2, 3] {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, nt);
        let seq: Vec<u32> = (0..40).map(|i| (i * 13 + 7) % 64).collect();
        let r = Engine::new(&plan, &ws, cfg(false)).unwrap();
        let mut want = Vec::new();
        for &t in &seq {
            r.step(&pool, t);
            want.push(r.logits().to_vec());
        }
        drop(r);
        let e = Engine::with_options(
            &plan,
            &ws,
            cfg(false),
            EngineOptions {
                verify_lanes: 4,
                mtp: None,
                kv_type: None,
            },
        )
        .unwrap();
        let (mut p, mut it) = (0usize, 0usize);
        while p < seq.len() {
            let n = (1 + (it * 7) % 4).min(seq.len() - p);
            let m = 1 + (it * 5 + 3) % n;
            let mut toks: Vec<u32> = seq[p..p + n].to_vec();
            for (i, t) in toks.iter_mut().enumerate().skip(m) {
                *t = ((it * 31 + i * 17 + 1) % 64) as u32;
            }
            let g = e.verify(&pool, &toks);
            for ln in 0..m {
                assert!(
                    same_bits(e.logits_lane(ln), &want[p + ln]),
                    "{nt} threads: lane {ln} at pos {} differs",
                    p + ln
                );
                assert_eq!(g[ln] as usize, argmax(e.logits_lane(ln)));
            }
            e.commit(m);
            p += m;
            it += 1;
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Snapshot / restore of the recurrent state and the full state file (KV + recurrent) round
/// trip: replaying from either gives the original logits bit for bit.
#[test]
fn mini_dense_snapshot_and_state_file_replay_exactly() {
    let (dir, path) = dense_paths("snap");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (plan, ws, pool) = mini_engine_parts(&path, &opt, 2);
    let eng = Engine::new(&plan, &ws, cfg(false)).unwrap();
    let toks: Vec<u32> = (0..10).map(|i| (i * 11 + 5) % 64).collect();
    let (mut want, mut snap, mut file) = (Vec::new(), None, Vec::new());
    for (i, &t) in toks.iter().enumerate() {
        if i == 4 {
            snap = Some(eng.save_recurrent());
            eng.write_state(&mut file).unwrap();
        }
        eng.step(&pool, t);
        want.push(eng.logits().to_vec());
    }
    eng.restore_recurrent(&snap.unwrap());
    assert_eq!(eng.pos(), 4);
    for (i, &t) in toks.iter().enumerate().skip(4) {
        eng.step(&pool, t);
        assert!(same_bits(eng.logits(), &want[i]), "step {i} after restore");
    }
    let other = Engine::new(&plan, &ws, cfg(false)).unwrap();
    other.read_state(&mut &file[..], 4).unwrap();
    assert_eq!(other.pos(), 4);
    for (i, &t) in toks.iter().enumerate().skip(4) {
        other.step(&pool, t);
        assert!(
            same_bits(other.logits(), &want[i]),
            "step {i} after the state file"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
