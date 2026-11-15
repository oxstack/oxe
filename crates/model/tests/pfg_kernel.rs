//! v0.1.4 PFG path A (pfg-f16) model-free numerics tests of `oxe_pfg_unit` through
//! `model::engine::pfattn::Tile::run_g` (the code the engine runs per unit) at the production
//! shape: head_dim 256, 16 query heads / 2 KV heads (8 query heads per KV head), units of 16 tokens
//! = 128 query columns (and 8 / 4 / 1 tokens, 32 tokens = 256 columns).
//!
//! - exact f64 reference over the same f16 inputs, today's tiled prefill unit (replica of
//!   `pf_attention_tiled`'s f16 steps) and fa2 on every case: random data at kv_len around every
//!   boundary (0, the 8-row panels, the 16 / 32 folds, the 32..256 chunks, 4096, ~8K, deep) +-1,
//!   outlier channels (massive activations) and +40..60 large logits. The pfg error vs f64 must
//!   stay inside the f16-class envelope of the shipped kernels on the SAME case (2.5 x
//!   max(tiled, fa2) + 2e-3, the fa2x bound) and nothing may be non-finite;
//! - exact-count fixtures (uniform attention / +128 logit spikes over one-hot V: every product
//!   and sum exact) at 1e-6 for every fold / chunk / micro-kernel: a masked row read, a dropped or
//!   doubled chunk, a wrong causal mask inside the unit, a broken online rescale or a wrong GQA
//!   column mapping is O(1);
//! - invariances, bit for bit: unit size (32 / 16 / 8 / 4 / 1 tokens), LD1R vs by-element
//!   micro-kernel, each column alone vs the shared GQA pass;
//! - nothing beyond the unit's last position is read: K rows and V^T columns beyond it are f16
//!   NaN (any read makes an output NaN), and the result equals the finite-cache result;
//! - bad arguments are refused (rc -1).
//!
//! `cargo test --release -p model --test pfg_kernel -- --nocapture`

use kernels_sys::ffi;
use model::engine::padded_n_kv;
use model::engine::pfattn::{GParams, PgMk, Scratch, Tile, PFA_MAX_COLS};

const HD: usize = 256;
const GROUP: usize = 8;
/// tokens of one case (one pfg unit at the default parameters)
const TT: usize = 16;
const NCOL: usize = TT * GROUP;
const VPAD: usize = 32;
const F16_NAN: u16 = 0x7e00;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// uniform in [-1, 1]
    fn f(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn f16s(v: &[f32]) -> Vec<u16> {
    let mut o = vec![0u16; v.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::oxe_fp32_to_fp16_row(v.as_ptr(), o.as_mut_ptr(), v.len() as i64) };
    o
}

fn h2f(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        0 => s * m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                s * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// One KV head's cache and TT tokens x GROUP query heads, laid out as the engine holds them.
#[derive(Clone)]
struct Case {
    n_ctx: usize,
    pos0: usize,
    /// K rows [position][HD]: finite up to the last token, NaN beyond
    k: Vec<u16>,
    /// V^T rows [dim][v_stride]: finite on [0, n_ctx) (the tiled replica's tail rule reads there),
    /// NaN in the pad
    vt: Vec<u16>,
    v_stride: usize,
    /// query columns [col][HD], col = token * GROUP + head
    q16: Vec<u16>,
}

enum Kind {
    Random(f32),
    /// q = 0: uniform attention; V one-hot (V[p][d] = 1 iff d == p % HD)
    Uniform,
    /// q = 2, K = 4 at the spike positions (score 128 after scaling), 0 elsewhere; V one-hot
    Spikes(Vec<usize>),
}

impl Case {
    fn new(pos0: usize, kind: Kind, seed: u64) -> Case {
        let last = pos0 + TT - 1;
        let n_ctx = (last + 1).div_ceil(256) * 256 + 256;
        let v_stride = n_ctx + VPAD;
        let mut r = Lcg(seed);
        let mut k = vec![F16_NAN; n_ctx * HD];
        let mut vt = vec![F16_NAN; HD * v_stride];
        let q: Vec<f32>;
        match &kind {
            Kind::Random(qs) => {
                let kf: Vec<f32> = (0..(last + 1) * HD).map(|_| r.f()).collect();
                k[..(last + 1) * HD].copy_from_slice(&f16s(&kf));
                for d in 0..HD {
                    let row: Vec<f32> = (0..n_ctx).map(|_| r.f()).collect();
                    vt[d * v_stride..d * v_stride + n_ctx].copy_from_slice(&f16s(&row));
                }
                q = (0..NCOL * HD).map(|_| r.f() * qs).collect();
            }
            Kind::Uniform | Kind::Spikes(_) => {
                let sp: &[usize] = if let Kind::Spikes(s) = &kind { s } else { &[] };
                for p in 0..=last {
                    let x = if sp.contains(&p) { 4.0f32 } else { 0.0 };
                    k[p * HD..(p + 1) * HD].copy_from_slice(&f16s(&vec![x; HD]));
                }
                for d in 0..HD {
                    for p in 0..n_ctx {
                        vt[d * v_stride + p] = if p % HD == d { 0x3c00 } else { 0 };
                    }
                }
                let qv = if sp.is_empty() { 0.0 } else { 2.0 };
                q = vec![qv; NCOL * HD];
            }
        }
        Case {
            n_ctx,
            pos0,
            k,
            vt,
            v_stride,
            q16: f16s(&q),
        }
    }

    fn last(&self) -> usize {
        self.pos0 + TT - 1
    }

    /// Columns [c0, c0 + ncol) as one unit (c0 a multiple of GROUP).
    fn tile(&self, c0: usize, ncol: usize) -> Tile {
        Tile {
            hd: HD,
            ncol,
            group: GROUP,
            pos0: self.pos0 + c0 / GROUP,
            scale: 1.0 / (HD as f32).sqrt(),
            k: self.k.as_ptr(),
            vt: self.vt.as_ptr(),
            v_stride: self.v_stride,
            n_ctx: self.n_ctx,
            q16: self.q16[c0 * HD..].as_ptr(),
        }
    }

    /// The same case with every K row and V^T column beyond `last` set to NaN.
    fn poisoned(&self) -> Case {
        let mut c = self.clone();
        let rows = self.last() + 1;
        for x in c.k[rows * HD..].iter_mut() {
            *x = F16_NAN;
        }
        for d in 0..HD {
            for x in c.vt[d * c.v_stride + rows..(d + 1) * c.v_stride].iter_mut() {
                *x = F16_NAN;
            }
        }
        c
    }
}

/// pfg: outputs [col][HD], units of `per` tokens.
fn pfg(c: &Case, per: usize, gp: GParams) -> Vec<f32> {
    let mut out = vec![0f32; NCOL * HD];
    let mut scr = Scratch::new_pfg(HD, GROUP);
    for t0 in (0..TT).step_by(per) {
        let ncol = per.min(TT - t0) * GROUP;
        let tile = c.tile(t0 * GROUP, ncol);
        // SAFETY: the case holds every row the tile reads (see the field docs).
        unsafe { tile.run_g(&mut scr, gp, None) };
        for col in 0..ncol {
            let g = t0 * GROUP + col;
            out[g * HD..(g + 1) * HD].copy_from_slice(scr.out(col));
        }
    }
    out
}

/// fa2 (lane A): units of 8 tokens (its 64-column limit).
fn fa2(c: &Case) -> Vec<f32> {
    let mut out = vec![0f32; NCOL * HD];
    let mut scr = Scratch::new(PFA_MAX_COLS, HD);
    for t0 in (0..TT).step_by(8) {
        let ncol = 8.min(TT - t0) * GROUP;
        let tile = c.tile(t0 * GROUP, ncol);
        // SAFETY: as pfg (fa2's tail rule reads V^T up to n_ctx: finite there).
        unsafe { tile.run(&mut scr) };
        for col in 0..ncol {
            let g = t0 * GROUP + col;
            out[g * HD..(g + 1) * HD].copy_from_slice(scr.out(col));
        }
    }
    out
}

/// Replica of today's tiled prefill unit (`pf_attention_tiled`, f16 KV), units of 8 tokens: scores
/// of every column over all rows, one FA-off softmax row per column over the padded n_kv (f16
/// probabilities), V^T . p over round_up(rows, 32) with the tail rule.
fn tiled(c: &Case) -> Vec<f32> {
    let n_ctx = c.n_ctx;
    let mut out = vec![0f32; NCOL * HD];
    let scale = 1.0 / (HD as f32).sqrt();
    for t0 in (0..TT).step_by(8) {
        let nc = 8 * GROUP;
        let pos0 = c.pos0 + t0;
        let last = pos0 + 7;
        let rows = last + 1;
        let n32 = rows.div_ceil(32) * 32;
        let acs = n_ctx;
        let nkv = padded_n_kv(last, n_ctx);
        let mut akq = vec![0f32; nc * acs];
        let mut ap16 = vec![0u16; nc * acs];
        let (mut wp, mut dp) = (vec![0f32; n_ctx], vec![0f32; n_ctx]);
        let mut aot = vec![0f32; HD * nc];
        let nf = [nkv as i32; PFA_MAX_COLS];
        let q = c.q16[t0 * GROUP * HD..].as_ptr();
        // SAFETY: buffers sized for the case; K rows < rows finite, V^T finite up to n_ctx >= nkv.
        unsafe {
            ffi::oxe_dot_f16_tile(
                HD as i32,
                c.k.as_ptr(),
                HD as i64,
                rows as i32,
                q,
                HD as i64,
                nc as i32,
                akq.as_mut_ptr(),
                1,
                acs as i64,
                std::ptr::null(),
            );
            for col in 0..nc {
                let pos = pos0 + col / GROUP;
                ffi::oxe_attn_softmax_row_f16(
                    akq.as_ptr().add(col * acs),
                    pos as i32,
                    nkv as i32,
                    scale,
                    wp.as_mut_ptr(),
                    dp.as_mut_ptr(),
                    ap16.as_mut_ptr().add(col * acs),
                    n32 as i32,
                );
            }
            ffi::oxe_dot_f16_tile(
                n32 as i32,
                c.vt.as_ptr(),
                c.v_stride as i64,
                HD as i32,
                ap16.as_ptr(),
                acs as i64,
                nc as i32,
                aot.as_mut_ptr(),
                nc as i64,
                1,
                nf.as_ptr(),
            );
        }
        for col in 0..nc {
            for d in 0..HD {
                out[(t0 * GROUP + col) * HD + d] = aot[d * nc + col];
            }
        }
    }
    out
}

/// Exact attention per column in f64 over the same f16 inputs.
fn ref_f64(c: &Case) -> Vec<f64> {
    let rows = c.last() + 1;
    let k: Vec<f64> = c.k[..rows * HD].iter().map(|&h| h2f(h)).collect();
    let scale = 1.0 / (HD as f64).sqrt();
    let mut out = vec![0f64; NCOL * HD];
    let mut s = vec![0f64; rows];
    for col in 0..NCOL {
        let n = c.pos0 + col / GROUP + 1;
        let q: Vec<f64> = c.q16[col * HD..(col + 1) * HD]
            .iter()
            .map(|&h| h2f(h))
            .collect();
        let mut m = f64::NEG_INFINITY;
        for p in 0..n {
            s[p] = k[p * HD..(p + 1) * HD]
                .iter()
                .zip(&q)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                * scale;
            m = m.max(s[p]);
        }
        let mut sum = 0f64;
        for x in s[..n].iter_mut() {
            *x = (*x - m).exp();
            sum += *x;
        }
        for d in 0..HD {
            let v = &c.vt[d * c.v_stride..d * c.v_stride + n];
            out[col * HD + d] = s[..n].iter().zip(v).map(|(a, &b)| a * h2f(b)).sum::<f64>() / sum;
        }
    }
    out
}

/// Max over columns of ||x - ref||_2 / ||ref||_2.
fn rel_err(x: &[f32], r: &[f64]) -> f64 {
    x.chunks(HD)
        .zip(r.chunks(HD))
        .map(|(xs, rs)| {
            let e2: f64 = xs
                .iter()
                .zip(rs)
                .map(|(&a, &b)| (a as f64 - b) * (a as f64 - b))
                .sum();
            let r2: f64 = rs.iter().map(|b| b * b).sum();
            (e2 / r2.max(1e-300)).sqrt()
        })
        .fold(0f64, f64::max)
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn gp(fold_qk: usize, fold_pv: usize, ch: usize, mk: PgMk) -> GParams {
    GParams::new(fold_qk, fold_pv, ch, 16, mk).unwrap()
}

/// Every fold pair (QK, PV) at the default chunk.
fn folds() -> Vec<GParams> {
    [(32usize, 32usize), (16, 16), (16, 32), (32, 16)]
        .iter()
        .map(|&(a, b)| gp(a, b, 128, PgMk::Ld1r))
        .collect()
}

/// kv_len of the case's columns = pos0 + 1 .. pos0 + 16: 1 (pos0 0), the 8-row panels, the 16 / 32
/// folds, the 32 / 64 / 128 / 256 chunks, 4096 and ~8K, each boundary +-1 for the first and the
/// last column of the unit.
const POS0S: [usize; 22] = [
    0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 111, 112, 113, 127, 128, 129, 239, 240, 241, 4080, 8176,
];

/// Random data at every boundary (and deep, a chunk boundary inside the unit): pfg inside the f16
/// class envelope of tiled and fa2 on the same case, for every fold pair and both micro-kernels;
/// the LD1R and by-element forms give the same bits; every fold pair stays within the f64 bound
/// used for fa2x's lossy folds.
#[test]
fn pfg_random_envelope_vs_f64_tiled_fa2_all_boundaries() {
    let mut worst = (0f64, String::new());
    let mut pos0s: Vec<usize> = POS0S.to_vec();
    pos0s.extend([40_000 - 7, 70_001 - 4]);
    for &pos0 in &pos0s {
        for &qs in &[1.0f32, 4.0, 16.0] {
            if pos0 > 10_000 && qs == 1.0 {
                continue;
            }
            let c = Case::new(
                pos0,
                Kind::Random(qs),
                0x9f6 ^ (pos0 as u64) << 4 ^ ((qs as u64) << 40),
            );
            let r = ref_f64(&c);
            let (t, f) = (tiled(&c), fa2(&c));
            assert!(
                finite(&t) && finite(&f),
                "pos0 {pos0}: tiled / fa2 non-finite"
            );
            let (et, ef) = (rel_err(&t, &r), rel_err(&f, &r));
            for g in folds() {
                let x = pfg(&c, TT, g);
                assert!(
                    finite(&x),
                    "pos0 {pos0} qs {qs} {g:?}: NON-FINITE pfg output"
                );
                let xe = pfg(
                    &c,
                    TT,
                    GParams {
                        mk: PgMk::Elem,
                        ..g
                    },
                );
                assert!(
                    same_bits(&x, &xe),
                    "pos0 {pos0} qs {qs} {g:?}: LD1R and by-element differ"
                );
                let ex = rel_err(&x, &r);
                let ratio = ex / et.max(ef).max(1e-7);
                let l = format!("pfg {} kv_len {:>5}..{:<5} qs {qs:>4}: rel L2 vs f64 pfg {ex:.2e} tiled {et:.2e} fa2 {ef:.2e} (x{ratio:.2})", g.tag(), pos0 + 1, pos0 + TT);
                println!("{l}");
                assert!(
                    ex <= 2.5 * et.max(ef) + 2e-3,
                    "pfg outside the f16 class envelope: {l}"
                );
                let tol = 3e-3 * (qs as f64 / 4.0).max(1.0) * 2.0;
                assert!(ex <= tol, "pfg vs f64 above {tol:.1e}: {l}");
                if g == GParams::default() && ratio > worst.0 {
                    worst = (ratio, l);
                }
            }
        }
    }
    println!(
        "pfg default worst error ratio vs max(tiled, fa2): {:.2} ({})",
        worst.0, worst.1
    );
}

/// Uniform attention over one-hot V and +128 logit spikes (exact fixtures): every fold pair,
/// chunks 32 / 96 / 128 / 256, both micro-kernels, units of 16 / 8 / 1 tokens, at 1e-6 of f64.
#[test]
fn pfg_exact_fixtures_all_params() {
    let mut gps: Vec<GParams> = Vec::new();
    for (fq, fp, ch) in [
        (32usize, 32usize, 32usize),
        (32, 32, 128),
        (16, 16, 96),
        (16, 32, 256),
        (32, 16, 64),
    ] {
        for mk in [PgMk::Ld1r, PgMk::Elem] {
            gps.push(gp(fq, fp, ch, mk));
        }
    }
    let check = |c: &Case, what: &str, pers: &[usize]| {
        let r = ref_f64(c);
        for &g in &gps {
            for &per in pers {
                let f = pfg(c, per, g);
                for (i, (&x, &y)) in f.iter().zip(&r).enumerate() {
                    assert!(
                        (x as f64 - y).abs() <= 1e-6 * y.abs() + 1e-12,
                        "{what} pos0 {} {g:?} per {per}: out[{i}] = {x} want {y}",
                        c.pos0
                    );
                }
            }
        }
    };
    for &pos0 in &[
        0usize, 1, 7, 8, 9, 25, 31, 32, 33, 95, 96, 120, 127, 128, 129, 250, 255, 256, 509, 8_184,
        40_000,
    ] {
        check(&Case::new(pos0, Kind::Uniform, 3), "uniform", &[TT, 8, 1]);
    }
    for &pos0 in &[0usize, 120, 250, 8_188, 40_000] {
        let mut sp = vec![
            0usize, 7, 8, 31, 32, 95, 96, 127, 128, 255, 256, 257, 8_191, 8_192, 39_999,
        ];
        sp.extend([pos0 + 3, pos0 + 9, pos0 + TT - 1]);
        sp.retain(|&p| p < pos0 + TT);
        for inside in [false, true] {
            let s: Vec<usize> = if inside { vec![pos0 + 9] } else { sp.clone() };
            check(&Case::new(pos0, Kind::Spikes(s), 5), "spikes", &[TT, 1]);
        }
    }
}

/// A column's pfg output is the same bits whether its unit holds 16, 8, 4, 2 or 1 tokens, and a
/// 32-token unit (256 columns) gives the 16-token bits for its columns: the engine's unit grid
/// and the thread that runs a unit never change a result.
#[test]
fn pfg_unit_split_is_bitwise_invariant() {
    for &pos0 in &[3usize, 125, 253, 9_000, 40_000 - 3] {
        let c = Case::new(pos0, Kind::Random(4.0), 77 ^ pos0 as u64);
        for g in folds() {
            let a = pfg(&c, TT, g);
            assert!(finite(&a));
            for per in [8usize, 4, 2, 1] {
                assert!(
                    same_bits(&pfg(&c, per, g), &a),
                    "pos0 {pos0} {g:?}: units of {per} tokens differ from units of {TT}"
                );
            }
        }
    }
    // 32 tokens = 256 columns in one unit: the first 16 tokens' columns equal the 16-token unit
    let pos0 = 1_000;
    let c32 = {
        let mut c = Case::new(pos0, Kind::Random(4.0), 0x3232);
        // extend the case by 16 tokens (K rows / queries; V^T is already finite up to n_ctx)
        let mut r = Lcg(0x3233);
        let last = pos0 + 2 * TT - 1;
        let kf: Vec<f32> = (0..TT * HD).map(|_| r.f()).collect();
        c.k[(pos0 + TT) * HD..(last + 1) * HD].copy_from_slice(&f16s(&kf));
        let qf: Vec<f32> = (0..NCOL * HD).map(|_| r.f() * 4.0).collect();
        c.q16.extend(f16s(&qf));
        c
    };
    let g = GParams::default();
    let mut scr = Scratch::new_pfg(HD, GROUP);
    let t = Tile {
        ncol: 2 * NCOL,
        ..c32.tile(0, NCOL)
    };
    // SAFETY: K rows up to pos0 + 31 and 256 query columns are set; V^T finite up to n_ctx.
    unsafe { t.run_g(&mut scr, g, None) };
    let a = pfg(&c32, TT, g);
    for col in 0..NCOL {
        assert!(
            same_bits(scr.out(col), &a[col * HD..(col + 1) * HD]),
            "column {col}: 256-column unit differs from the 128-column unit"
        );
    }
    assert!(finite(
        &(NCOL..2 * NCOL)
            .flat_map(|col| scr.out(col).to_vec())
            .collect::<Vec<_>>()
    ));
}

/// GQA reuse: every column equals the same query run alone (a 1-column unit, group 1: its own
/// pass over K / V^T) bit for bit, so no column reads another column's query, mask or running max.
#[test]
fn pfg_gqa_shared_pass_equals_single_column_runs() {
    for &pos0 in &[5usize, 254, 9_001] {
        let c = Case::new(pos0, Kind::Random(8.0), 0x6aa ^ pos0 as u64);
        let g = GParams::default();
        let all = pfg(&c, TT, g);
        let mut scr = Scratch::new_pfg(HD, GROUP);
        for col in 0..NCOL {
            let mut t = c.tile(0, NCOL);
            t.ncol = 1;
            t.group = 1;
            t.pos0 = c.pos0 + col / GROUP;
            t.q16 = c.q16[col * HD..].as_ptr();
            // SAFETY: one column of the case (rows up to its own position).
            unsafe { t.run_g(&mut scr, g, None) };
            assert!(
                same_bits(scr.out(0), &all[col * HD..(col + 1) * HD]),
                "pos0 {pos0}: column {col} (head {} of token {}) differs alone",
                col % GROUP,
                col / GROUP
            );
        }
    }
}

/// Nothing beyond the unit's last position is read: with K rows and V^T columns beyond it NaN the
/// output is finite and the same bits as with a finite cache (no tail rule, no V^T padding
/// contract), for every chunk / fold / micro-kernel and unit size.
#[test]
fn pfg_reads_nothing_beyond_the_last_position() {
    for &pos0 in &[0usize, 9, 17, 113, 127, 241, 4080] {
        let c = Case::new(pos0, Kind::Random(4.0), 0xbad ^ pos0 as u64);
        let p = c.poisoned();
        for g in [
            gp(32, 32, 32, PgMk::Ld1r),
            gp(32, 32, 128, PgMk::Elem),
            gp(16, 16, 256, PgMk::Ld1r),
            gp(16, 32, 96, PgMk::Elem),
        ] {
            for per in [TT, 1] {
                let (a, b) = (pfg(&c, per, g), pfg(&p, per, g));
                assert!(
                    finite(&b),
                    "pos0 {pos0} {g:?} per {per}: a read beyond the last position"
                );
                assert!(same_bits(&a, &b), "pos0 {pos0} {g:?} per {per}: the cache beyond the last position changed the result");
            }
        }
    }
}

/// Outlier channels (massive activations: a block of consecutive head dims where q and k are both
/// large and of one sign) and +40..60 logit spikes: never non-finite; per case within the bug
/// bound 2.5 x max(tiled, fa2) + 2.5e-2 (a mask / fold / mapping bug is O(1e-1..1); tiled itself
/// is at 1e-1 at mag 16); in aggregate (mean ratio over all outlier cases) inside the f16-class
/// envelope, x2.0 for the default (QK fold 16, head dims in the tiled lane order since e3f9f78+1). CI 43375423636: fold-32 QK x1.3..5.2 (why the
/// default QK fold is 16), fold-16 QK x0.8..1.9 except mag 32 (x4.4, abs 2e-2, tiled 4.5e-3).
/// The fold-32 QK variant is printed alongside.
#[test]
fn pfg_envelope_outliers_and_large_logits() {
    let g = GParams::default();
    let g16 = GParams { fold_qk: 32, ..g };
    let (mut sum, mut sum16, mut cnt) = (0f64, 0f64, 0usize);
    let blocks: [(&str, Vec<usize>); 3] = [
        ("block8x2", (24..32).chain(56..64).collect()),
        ("block32", (0..32).collect()),
        ("scatter16", (0..16).map(|i| i * 16 + 3).collect()),
    ];
    let mut worst = 0f64;
    for (bname, dims) in &blocks {
        for &mag in &[4.0f32, 8.0, 16.0, 32.0] {
            for &pos0 in &[0usize, 15, 16, 17, 127, 128, 4089] {
                let mut c = Case::new(
                    pos0,
                    Kind::Random(4.0),
                    0xa771 ^ (pos0 as u64) << 8 ^ mag as u64,
                );
                let mut r = Lcg(0x07 ^ pos0 as u64);
                let mut bump = || f16s(&[mag * (1.0 + 0.1 * r.f())])[0];
                for p in 0..=c.last() {
                    for &d in dims {
                        c.k[p * HD + d] = bump();
                    }
                }
                for col in 0..NCOL {
                    for &d in dims {
                        c.q16[col * HD + d] = bump();
                    }
                }
                let rf = ref_f64(&c);
                let (t, f, x, x16) = (tiled(&c), fa2(&c), pfg(&c, TT, g), pfg(&c, TT, g16));
                assert!(
                    finite(&t) && finite(&f),
                    "{bname} mag {mag} pos0 {pos0}: tiled / fa2 non-finite"
                );
                assert!(
                    finite(&x) && finite(&x16),
                    "{bname} mag {mag} pos0 {pos0}: pfg NON-FINITE where tiled and fa2 are finite"
                );
                let (et, ef, ex, e16) = (
                    rel_err(&t, &rf),
                    rel_err(&f, &rf),
                    rel_err(&x, &rf),
                    rel_err(&x16, &rf),
                );
                let base = et.max(ef).max(1e-7);
                let ratio = ex / base;
                worst = worst.max(ratio);
                (sum, sum16, cnt) = (sum + ratio, sum16 + e16 / base, cnt + 1);
                let l = format!("pfg-envelope {bname:<9} mag {mag:>4} kv_len {:>5}..: tiled {et:.2e} fa2 {ef:.2e} pfg {ex:.2e} (x{ratio:.2}) pfg-q32 {e16:.2e} (x{:.2})", pos0 + 1, e16 / base);
                println!("{l}");
                assert!(
                    ex <= 2.5 * et.max(ef) + 2.5e-2,
                    "pfg beyond the bug bound: {l}"
                );
            }
        }
    }
    for &pos0 in &[0usize, 120, 121, 248, 4089] {
        for &spike in &[640.0f32, 960.0] {
            let mut c = Case::new(pos0, Kind::Random(1.0), 0x5b1 ^ pos0 as u64);
            for &p in &[pos0 / 2, pos0 + 3, pos0 + 12] {
                for d in 0..16 {
                    c.k[p * HD + d * 16] = f16s(&[spike / 16.0])[0];
                }
            }
            for col in 0..NCOL {
                for d in 0..16 {
                    c.q16[col * HD + d * 16] = f16s(&[1.0])[0];
                }
            }
            let rf = ref_f64(&c);
            let (et, ex) = (rel_err(&tiled(&c), &rf), rel_err(&pfg(&c, TT, g), &rf));
            println!(
                "pfg-envelope spike {spike:>4} kv_len {:>5}..: tiled {et:.2e} pfg {ex:.2e}",
                pos0 + 1
            );
            assert!(
                ex.is_finite() && ex <= 4.0 * et + 2e-3,
                "spike {spike} pos0 {pos0}: tiled {et:.2e} pfg {ex:.2e}"
            );
        }
    }
    let (mean, mean16) = (sum / cnt as f64, sum16 / cnt as f64);
    println!("pfg-envelope outliers: default worst x{worst:.2} mean x{mean:.2}; fold-32 QK mean x{mean16:.2} ({cnt} cases)");
    assert!(
        mean <= 2.0,
        "pfg outlier error outside the f16-class envelope on average: x{mean:.2}"
    );
}

/// The C entry point refuses what it cannot do (and writes nothing): head dim not a multiple of
/// 32, more than 256 columns, a bad fold / chunk / micro-kernel, the last position at n_ctx, a
/// short scratch.
#[test]
fn pfg_refuses_bad_arguments() {
    let c = Case::new(40, Kind::Random(1.0), 1);
    let n = ffi_bytes(HD, NCOL, 128);
    let mut scr = vec![0u8; n as usize];
    let mut out = vec![7f32; NCOL * HD];
    let call = |hd: i32,
                ncol: i32,
                n_ctx: i32,
                ch: i32,
                fq: i32,
                fp: i32,
                mk: i32,
                bytes: i64,
                scr: &mut [u8],
                out: &mut [f32]|
     -> i32 {
        // SAFETY: the case's buffers; the C side validates before touching anything.
        unsafe {
            ffi::oxe_pfg_unit(
                hd,
                ncol,
                GROUP as i32,
                c.pos0 as i32,
                0.0625,
                c.k.as_ptr(),
                c.vt.as_ptr(),
                c.v_stride as i64,
                n_ctx,
                c.q16.as_ptr(),
                ch,
                fq,
                fp,
                mk,
                scr.as_mut_ptr(),
                bytes,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    };
    let nc = c.n_ctx as i32;
    assert_eq!(
        call(
            HD as i32,
            NCOL as i32,
            nc,
            128,
            32,
            32,
            0,
            n,
            &mut scr[..],
            &mut out[..]
        ),
        0
    );
    out.fill(7.0);
    for (what, rc) in [
        (
            "hd 200",
            call(
                200,
                NCOL as i32,
                nc,
                128,
                32,
                32,
                0,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "ncol 257",
            call(
                HD as i32,
                257,
                nc,
                128,
                32,
                32,
                0,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "ncol 0",
            call(
                HD as i32,
                0,
                nc,
                128,
                32,
                32,
                0,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "ch 48",
            call(
                HD as i32,
                NCOL as i32,
                nc,
                48,
                32,
                32,
                0,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "ch 512",
            call(
                HD as i32,
                NCOL as i32,
                nc,
                512,
                32,
                32,
                0,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "fold 8",
            call(
                HD as i32,
                NCOL as i32,
                nc,
                128,
                8,
                32,
                0,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "fold pv 64",
            call(
                HD as i32,
                NCOL as i32,
                nc,
                128,
                32,
                64,
                0,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "mk 2",
            call(
                HD as i32,
                NCOL as i32,
                nc,
                128,
                32,
                32,
                2,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "last >= n_ctx",
            call(
                HD as i32,
                NCOL as i32,
                (c.pos0 + 8) as i32 / 32 * 32,
                128,
                32,
                32,
                0,
                n,
                &mut scr[..],
                &mut out[..],
            ),
        ),
        (
            "short scratch",
            call(
                HD as i32,
                NCOL as i32,
                nc,
                128,
                32,
                32,
                0,
                n - 64,
                &mut scr[..],
                &mut out[..],
            ),
        ),
    ] {
        assert_eq!(rc, -1, "{what} accepted");
    }
    assert!(
        out.iter().all(|&x| x == 7.0),
        "a refused call wrote its output"
    );
}

fn ffi_bytes(hd: usize, ncol: usize, ch: usize) -> i64 {
    // SAFETY: pure size computation.
    unsafe { ffi::oxe_pfg_scratch_bytes(hd as i32, ncol as i32, ch as i32) }
}
