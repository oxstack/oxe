//! v0.1.3 APPEND-PREFILL micro-benchmark (model-free; needs a QUIET machine: hold the build
//! locks, ~3-6 min): the prefill attention core of ONE full-attention layer for an append of A
//! tokens at depth D, old tiled kernel (replica of `pf_attention_tiled`'s f16 unit at the default
//! mask: scores over all rows, FA-off softmax rows, V^T . p without position blocking) vs fa2
//! (`pfattn::Tile::run`, lane A) vs fa2x (`pfattn::Tile::run_x`, this lane), on a `sched::Pool`
//! with dynamically scheduled units as the engine runs them.
//!
//! Shapes: head_dim 256, 2 KV heads x 8 query heads, units of 8 tokens (64 columns), K rows
//! [kvh][n_ctx][256] and V^T rows with the engine's stride (n_ctx + 32), f16. K / V^T of the
//! layer (~290 MB at 136K) do not fit any cache, as in the engine.
//! Work: per (D, A) a sample of `OXE_PFBENCH_UNITS` units (default 96, spread evenly over the
//! append's units and both KV heads) is timed; us per append token = sample wall x (all units /
//! sampled) / A for ONE layer (x 10 for Ornith's attention layers).
//!
//! Run (Snapdragon, quiet):
//! `cargo test --release -p model --test pfattn_kbench -- --ignored --nocapture --test-threads 1`
//! Optional: OXE_PFBENCH_THREADS (default 12), OXE_PFBENCH_CPUS (cpu list), OXE_PFBENCH_DEPTH
//! (default "32768,65536,131072"), OXE_PFBENCH_APPEND (default "512,2048,8192"),
//! OXE_PFBENCH_X ("fold:ch" list, default "8:128,16:128,8:256"), OXE_PFBENCH_REPS (default 3),
//! OXE_PFBENCH_UNITS, OXE_PFBENCH_OLD=0 (skip the tiled replica: it needs ~54 MB scratch per thread).
//!
//! Kill (APPEND-PREFILL): at D 64K, A 2048, fa2x (fold 8, ch 128) < 1.5x the old tiled kernel ->
//! stop the lane; fa2x < 1.2x fa2 -> fa2x is not worth a second numerics class over fa2.

mod common;

use common::*;
use kernels_sys::ffi;
use model::engine::padded_n_kv;
use model::engine::pfattn::{Scratch, Tile, XParams};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

const TT: usize = 8;
const NCOL: usize = TT * GROUP;
/// the engine's PF_ATTN_PAD column offset (default mask has it)
const ATT_PAD: usize = 208;

fn env_list(k: &str, d: &str) -> Vec<usize> {
    std::env::var(k)
        .unwrap_or_else(|_| d.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kern {
    Old,
    Fa2,
    Fa2x(XParams),
}

impl Kern {
    fn name(self) -> String {
        match self {
            Kern::Old => "tiled".into(),
            Kern::Fa2 => "fa2".into(),
            Kern::Fa2x(x) => format!("fa2x f{} c{}", x.fold, x.ch),
        }
    }
}

/// Per-thread scratch of all kernels.
struct Thr {
    fa: Scratch,
    akq: Vec<f32>,
    ap16: Vec<u16>,
    wp: Vec<f32>,
    dp: Vec<f32>,
    aot: Vec<f32>,
}

struct Layer {
    n_ctx: usize,
    v_stride: usize,
    k: Vec<u16>,
    vt: Vec<u16>,
    q16: Vec<u16>,
}

impl Layer {
    fn new(n_ctx: usize) -> Layer {
        let v_stride = n_ctx + VPAD;
        let mut r = Lcg(0xa99e_d0);
        let blk = f16s(&(0..(1 << 16) + 97).map(|_| r.f()).collect::<Vec<_>>());
        let fill = |n: usize, salt: usize| -> Vec<u16> {
            let mut v = vec![0u16; n];
            let mut o = salt % 97;
            for c in v.chunks_mut(1 << 16) {
                c.copy_from_slice(&blk[o..o + c.len()]);
                o = (o + 31) % 97;
            }
            v
        };
        let q16 = f16s(&(0..NCOL * HD).map(|_| r.f() * 4.0).collect::<Vec<_>>());
        Layer {
            n_ctx,
            v_stride,
            k: fill(NKV * n_ctx * HD, 1),
            vt: fill(NKV * HD * v_stride, 2),
            q16,
        }
    }

    fn tile(&self, kvh: usize, pos0: usize) -> Tile {
        Tile {
            hd: HD,
            ncol: NCOL,
            group: GROUP,
            pos0,
            scale: 1.0 / (HD as f32).sqrt(),
            k: self.k[kvh * self.n_ctx * HD..].as_ptr(),
            vt: self.vt[kvh * HD * self.v_stride..].as_ptr(),
            v_stride: self.v_stride,
            n_ctx: self.n_ctx,
            q16: self.q16.as_ptr(),
        }
    }

    /// One unit of the old tiled path into thr.aot ([d][col]).
    unsafe fn old_unit(&self, t: &Tile, thr: &mut Thr) {
        // SAFETY: the caller's contract (tile within the layer, scratch sized for n_ctx).
        unsafe {
            let rows = t.last() + 1;
            let n32 = rows.div_ceil(32) * 32;
            let cs = self.n_ctx + ATT_PAD;
            let nkv = padded_n_kv(t.last(), self.n_ctx);
            let nf = [nkv as i32; NCOL];
            ffi::oxe_dot_f16_tile(
                HD as i32,
                t.k,
                HD as i64,
                rows as i32,
                t.q16,
                HD as i64,
                NCOL as i32,
                thr.akq.as_mut_ptr(),
                1,
                cs as i64,
                std::ptr::null(),
            );
            for col in 0..NCOL {
                let pos = t.pos0 + col / GROUP;
                ffi::oxe_attn_softmax_row_f16(
                    thr.akq.as_ptr().add(col * cs),
                    pos as i32,
                    nkv as i32,
                    t.scale,
                    thr.wp.as_mut_ptr(),
                    thr.dp.as_mut_ptr(),
                    thr.ap16.as_mut_ptr().add(col * cs),
                    n32 as i32,
                );
            }
            ffi::oxe_dot_f16_tile(
                n32 as i32,
                t.vt,
                t.v_stride as i64,
                HD as i32,
                thr.ap16.as_ptr(),
                cs as i64,
                NCOL as i32,
                thr.aot.as_mut_ptr(),
                NCOL as i64,
                1,
                nf.as_ptr(),
            );
        }
    }

    unsafe fn run(&self, k: Kern, t: &Tile, thr: &mut Thr) {
        // SAFETY: the caller's contract.
        unsafe {
            match k {
                Kern::Old => self.old_unit(t, thr),
                Kern::Fa2 => t.run(&mut thr.fa),
                Kern::Fa2x(x) => t.run_x(&mut thr.fa, x),
            }
        }
    }

    /// Output column-major [col][hd] of the unit (spot check).
    fn out(&self, k: Kern, thr: &Thr) -> Vec<f32> {
        match k {
            Kern::Old => (0..NCOL * HD)
                .map(|i| thr.aot[(i % HD) * NCOL + i / HD])
                .collect(),
            _ => (0..NCOL).flat_map(|c| thr.fa.out(c).to_vec()).collect(),
        }
    }
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut e, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        e += (x as f64 - y as f64).powi(2);
        n += (y as f64).powi(2);
    }
    (e / n.max(1e-300)).sqrt()
}

#[test]
#[ignore]
fn kbench_append_prefill_attention() {
    let nth = env_list("OXE_PFBENCH_THREADS", "12")[0].max(1);
    let depths = env_list("OXE_PFBENCH_DEPTH", "32768,65536,131072");
    let appends = env_list("OXE_PFBENCH_APPEND", "512,2048,8192");
    let reps = env_list("OXE_PFBENCH_REPS", "3")[0].max(1);
    let sample = env_list("OXE_PFBENCH_UNITS", "96")[0].max(1);
    let old_on = std::env::var("OXE_PFBENCH_OLD")
        .map(|v| v.trim() != "0")
        .unwrap_or(true);
    let xs: Vec<XParams> = std::env::var("OXE_PFBENCH_X")
        .unwrap_or_else(|_| "8:128,16:128,8:256".into())
        .split(',')
        .filter_map(|s| {
            let (f, c) = s.trim().split_once(':')?;
            XParams::new(f.parse().ok()?, c.parse().ok()?)
        })
        .collect();
    let mut kerns = Vec::new();
    if old_on {
        kerns.push(Kern::Old);
    }
    kerns.push(Kern::Fa2);
    kerns.extend(xs.iter().map(|&x| Kern::Fa2x(x)));
    let n_ctx =
        (depths.iter().max().unwrap() + appends.iter().max().unwrap()).div_ceil(256) * 256 + 256;
    let cpus = std::env::var("OXE_PFBENCH_CPUS")
        .ok()
        .map(|s| sched::parse_cpu_list(&s).expect("OXE_PFBENCH_CPUS"))
        .unwrap_or_else(|| cpus(nth));
    let cpus = cpus[..nth.min(cpus.len())].to_vec();
    assert_eq!(
        cpus.len(),
        nth,
        "OXE_PFBENCH_CPUS has fewer cpus than threads"
    );
    println!("APPEND-PREFILL kbench: head_dim {HD}, {NKV} KV heads x {GROUP}, units of {TT} tokens, n_ctx {n_ctx}, threads {nth} on {cpus:?}, reps {reps}, sampled units {sample}");
    let layer = Layer::new(n_ctx);
    let thrs: Vec<Shared<Thr>> = (0..nth)
        .map(|_| {
            let big = if old_on { NCOL * (n_ctx + ATT_PAD) } else { 0 };
            Shared::new(vec![Thr {
                fa: Scratch::new(NCOL, HD),
                akq: vec![0.0; big],
                ap16: vec![0; big],
                wp: vec![0.0; if old_on { n_ctx } else { 0 }],
                dp: vec![0.0; if old_on { n_ctx } else { 0 }],
                aot: vec![0.0; HD * NCOL],
            }])
        })
        .collect();
    let mut table: Vec<(usize, usize, Kern, f64)> = Vec::new();
    std::thread::scope(|s| {
        s.spawn(|| {
            let pool = sched::Pool::new(&cpus, Duration::from_micros(200)).unwrap();
            for &d in &depths {
                for &a in &appends {
                    let total = NKV * a.div_ceil(TT);
                    let ns = sample.min(total);
                    let units: Vec<(usize, usize)> = (0..ns).map(|j| j * total / ns).map(|u| (u % NKV, d + (u / NKV) * TT)).collect();
                    let ctr = AtomicUsize::new(0);
                    let run = |k: Kern| {
                        ctr.store(0, Relaxed);
                        pool.run(|ctx| loop {
                            let i = ctr.fetch_add(1, Relaxed);
                            if i >= units.len() {
                                break;
                            }
                            let (kvh, pos0) = units[i];
                            // SAFETY: slot tid is this thread's; the layer holds every row read.
                            unsafe { layer.run(k, &layer.tile(kvh, pos0), &mut *thrs[ctx.tid].ptr()) };
                        });
                    };
                    // spot check on one unit (thread 0's scratch, outside the pool)
                    let spot: Vec<(Kern, Vec<f32>)> = kerns
                        .iter()
                        .map(|&k| {
                            // SAFETY: the pool is idle; slot 0 used by this thread only.
                            let thr = unsafe { &mut *thrs[0].ptr() };
                            unsafe { layer.run(k, &layer.tile(0, units[ns / 2].1), thr) };
                            (k, layer.out(k, thr))
                        })
                        .collect();
                    let base = &spot.iter().find(|(k, _)| *k == Kern::Fa2).unwrap().1;
                    let mut times = vec![Vec::new(); kerns.len()];
                    for k in &kerns {
                        run(*k); // warm: page faults, scratch
                    }
                    for r in 0..reps {
                        for j in 0..kerns.len() {
                            // rotate the order so drift hits every kernel
                            let ki = (j + r) % kerns.len();
                            let t0 = Instant::now();
                            run(kerns[ki]);
                            times[ki].push(t0.elapsed().as_secs_f64());
                        }
                    }
                    for (ki, k) in kerns.iter().enumerate() {
                        let w = median(&mut times[ki]);
                        let us_tok = w * 1e6 * (total as f64 / ns as f64) / a as f64;
                        // multiply-adds of the sampled units (scores + V^T . p over each column's rows)
                        let fma: f64 = units.iter().map(|&(_, p0)| (0..TT).map(|t| (p0 + t + 1) as f64).sum::<f64>() * GROUP as f64 * HD as f64 * 2.0).sum();
                        let dsp = rel(&spot[ki].1, base);
                        println!("D {d:>6} A {a:>5} {:<16} {us_tok:>10.2} us/append-token/layer  {:>7.1} GFMA/s  (vs fa2 rel {dsp:.1e})", k.name(), fma / w / 1e9);
                        table.push((d, a, *k, us_tok));
                    }
                }
            }
        })
        .join()
        .unwrap()
    });
    println!("speedups (us per append token, one layer; x10 = Ornith's attention per token):");
    let get = |d: usize, a: usize, k: Kern| {
        table
            .iter()
            .find(|r| r.0 == d && r.1 == a && r.2 == k)
            .map(|r| r.3)
    };
    for &d in &depths {
        for &a in &appends {
            for &x in &xs {
                let fx = get(d, a, Kern::Fa2x(x)).unwrap();
                let vs_old = get(d, a, Kern::Old)
                    .map(|o| format!("{:.2}x", o / fx))
                    .unwrap_or("-".into());
                let vs_fa2 = get(d, a, Kern::Fa2).unwrap() / fx;
                println!(
                    "D {d:>6} A {a:>5} fa2x f{} c{}: vs tiled {vs_old}, vs fa2 {vs_fa2:.2}x",
                    x.fold, x.ch
                );
            }
        }
    }
    let dx = XParams::default();
    if let (Some(o), Some(f), Some(x)) = (
        get(65536, 2048, Kern::Old),
        get(65536, 2048, Kern::Fa2),
        get(65536, 2048, Kern::Fa2x(dx)),
    ) {
        let (so, sf) = (o / x, f / x);
        println!(
            "KILL CHECK D 64K A 2048 fa2x(f{} c{}): vs tiled {so:.2}x -> {}; vs fa2 {sf:.2}x -> {}",
            dx.fold,
            dx.ch,
            if so < 1.5 {
                "KILL (< 1.5x): stop the lane"
            } else {
                "pass (>= 1.5x)"
            },
            if sf < 1.2 {
                "fa2x not worth a numerics class over fa2 (< 1.2x)"
            } else {
                "pass (>= 1.2x)"
            }
        );
    } else {
        println!("KILL CHECK needs D 65536, A 2048, the tiled replica and fa2x {dx:?} in the run");
    }
}

// =============================================================================================
// v0.1.4 PFG path A: per-phase kernel bench at real shapes (V014-DIRECTION section 5 item 2).
//
// `cargo test --release -p model --test pfattn_kbench kbench_pfg -- --ignored --nocapture --test-threads 1`
// (quiet Snapdragon, ~1-3 min, ~0.9 GB RAM: the K / V^T of one layer at the deepest depth plus
// one tiled scratch per thread).
//
// One full-attention layer, head_dim 256, 2 KV heads x 8 query heads, an append of A tokens at
// depth D. Work = a sample of `OXE_PFBENCH_BLOCKS` (default 24) blocks of 32 tokens spread over
// the append and both KV heads; every kernel runs the SAME blocks split into ITS units (tiled /
// fa2 / fa2x: 8 tokens = 64 columns; pfg: `tb` tokens), dynamically scheduled on a `sched::Pool`
// as the engine does (24 blocks = 96 / 48 / 24 units: whole rounds of 12 threads). FMA = the
// useful multiply-adds (QK + PV over each column's causal rows), the same for every kernel.
//
// Per kernel: median wall of `OXE_PFBENCH_REPS` (3) unprofiled reps -> TFMA/s and % of the
// measured 12-core FMLA.8h roof (1.28 TFMA/s, ISA-ROOF 2026-11-13); then ONE profiled rep for the
// phase split (thread-time %): tiled replica QK / softmax / PV; fa2 replica QK / fused
// softmax+PV chunk / online merge; pfg (kernel counters) packing / QK / softmax / PV / out; fa2x
// total only (one fused kernel). Spot check: every kernel's output on one block vs fa2.
//
// Env: OXE_PFBENCH_THREADS (12), OXE_PFBENCH_CPUS, OXE_PFBENCH_DEPTH ("32768,65536"),
// OXE_PFBENCH_APPEND ("2048"), OXE_PFBENCH_REPS (3), OXE_PFBENCH_BLOCKS (24),
// OXE_PFBENCH_G ("fq:fp:ch:tb:mk" list, default
// "16:32:128:16:ld1r,16:32:128:16:elem,32:32:128:16:ld1r,16:16:128:16:ld1r,16:32:256:16:ld1r,16:32:128:32:ld1r,16:32:64:16:ld1r"),
// OXE_PFBENCH_OLD=0 (skip the tiled replica), OXE_PFBENCH_FA2X=0 (skip fa2x).
//
// Gates (registered in V014-DIRECTION, before the run): pfg (default params) >= 0.60 TFMA/s on 12
// threads at D 32K (kill < 0.50); the best variant is reported too.
// =============================================================================================

use model::engine::pfattn::{pfg_tick_hz, GParams, PgMk, PFA_CH};

/// ISA-ROOF 2026-11-13: 60 s sustained FMLA.8h register roof on 12 Oryon cores (TFMA/s).
const ROOF_TFMA: f64 = 1.28;
const PG_BLOCK: usize = 32;

#[derive(Clone, Copy, PartialEq, Debug)]
enum PgKern {
    Tiled,
    Fa2,
    Fa2x(XParams),
    Pfg(GParams),
}

impl PgKern {
    fn name(self) -> String {
        match self {
            PgKern::Tiled => "tiled".into(),
            PgKern::Fa2 => "fa2".into(),
            PgKern::Fa2x(x) => format!("fa2x f{} c{}", x.fold, x.ch),
            PgKern::Pfg(g) => format!("pfg {}", g.tag()),
        }
    }

    fn unit_tokens(self) -> usize {
        match self {
            PgKern::Pfg(g) => g.tokens_per_unit(GROUP),
            _ => TT,
        }
    }

    fn phases(self) -> &'static [&'static str] {
        match self {
            PgKern::Tiled => &["qk", "softmax", "pv"],
            PgKern::Fa2 => &["qk", "softmax+pv (fused chunk)", "merge"],
            PgKern::Fa2x(_) => &["total (one fused kernel)"],
            PgKern::Pfg(_) => &["packing", "qk", "softmax", "pv", "out"],
        }
    }
}

/// fa2 replica buffers (Tile::run's, which are private to the scratch).
struct Fa2Buf {
    s: Vec<f32>,
    p: Vec<u16>,
    part: Vec<f32>,
    ml: Vec<f32>,
    acc: Vec<f32>,
    m: Vec<f32>,
    l: Vec<f32>,
    nv: Vec<i32>,
}

struct PgThr {
    fa: Scratch,
    pg: Scratch,
    fb: Fa2Buf,
    akq: Vec<f32>,
    ap16: Vec<u16>,
    wp: Vec<f32>,
    dp: Vec<f32>,
    aot: Vec<f32>,
    /// profiled phase seconds (replicas) or ticks (pfg)
    ph: [f64; 5],
    tk: [u64; 5],
}

struct PgLayer {
    n_ctx: usize,
    v_stride: usize,
    k: Vec<u16>,
    vt: Vec<u16>,
    /// PG_BLOCK tokens x GROUP query columns
    q16: Vec<u16>,
}

impl PgLayer {
    fn new(n_ctx: usize) -> PgLayer {
        let v_stride = n_ctx + VPAD;
        let mut r = Lcg(0x9f6_b);
        let blk = f16s(&(0..(1 << 16) + 97).map(|_| r.f()).collect::<Vec<_>>());
        let fill = |n: usize, salt: usize| -> Vec<u16> {
            let mut v = vec![0u16; n];
            let mut o = salt % 97;
            for c in v.chunks_mut(1 << 16) {
                c.copy_from_slice(&blk[o..o + c.len()]);
                o = (o + 31) % 97;
            }
            v
        };
        let q16 = f16s(
            &(0..PG_BLOCK * GROUP * HD)
                .map(|_| r.f() * 4.0)
                .collect::<Vec<_>>(),
        );
        PgLayer {
            n_ctx,
            v_stride,
            k: fill(NKV * n_ctx * HD, 1),
            vt: fill(NKV * HD * v_stride, 2),
            q16,
        }
    }

    /// Unit of `nt` tokens at token offset `t` of the block at (kvh, pos0).
    fn tile(&self, kvh: usize, pos0: usize, t: usize, nt: usize) -> Tile {
        Tile {
            hd: HD,
            ncol: nt * GROUP,
            group: GROUP,
            pos0: pos0 + t,
            scale: 1.0 / (HD as f32).sqrt(),
            k: self.k[kvh * self.n_ctx * HD..].as_ptr(),
            vt: self.vt[kvh * HD * self.v_stride..].as_ptr(),
            v_stride: self.v_stride,
            n_ctx: self.n_ctx,
            q16: self.q16[t * GROUP * HD..].as_ptr(),
        }
    }

    /// One unit of kernel `k`; `prof`: accumulate the phase times into thr.ph / thr.tk.
    unsafe fn run(&self, k: PgKern, t: &Tile, thr: &mut PgThr, prof: bool) {
        // SAFETY: the caller's contract (tile within the layer, scratch sized for n_ctx).
        unsafe {
            match k {
                PgKern::Tiled => self.tiled_unit(t, thr, prof),
                PgKern::Fa2 => fa2_unit(t, &mut thr.fb, &mut thr.ph, prof),
                PgKern::Fa2x(x) => {
                    let t0 = Instant::now();
                    t.run_x(&mut thr.fa, x);
                    if prof {
                        thr.ph[0] += t0.elapsed().as_secs_f64();
                    }
                }
                PgKern::Pfg(g) => {
                    t.run_g(&mut thr.pg, g, if prof { Some(&mut thr.tk) } else { None })
                }
            }
        }
    }

    /// The tiled replica (as `Layer::old_unit`) with phase timers.
    unsafe fn tiled_unit(&self, t: &Tile, thr: &mut PgThr, prof: bool) {
        // SAFETY: the caller's contract.
        unsafe {
            let ncol = t.ncol;
            let rows = t.last() + 1;
            let n32 = rows.div_ceil(32) * 32;
            let cs = self.n_ctx + ATT_PAD;
            let nkv = padded_n_kv(t.last(), self.n_ctx);
            let nf = [nkv as i32; NCOL];
            let t0 = Instant::now();
            ffi::oxe_dot_f16_tile(
                HD as i32,
                t.k,
                HD as i64,
                rows as i32,
                t.q16,
                HD as i64,
                ncol as i32,
                thr.akq.as_mut_ptr(),
                1,
                cs as i64,
                std::ptr::null(),
            );
            let t1 = Instant::now();
            for col in 0..ncol {
                let pos = t.pos0 + col / GROUP;
                ffi::oxe_attn_softmax_row_f16(
                    thr.akq.as_ptr().add(col * cs),
                    pos as i32,
                    nkv as i32,
                    t.scale,
                    thr.wp.as_mut_ptr(),
                    thr.dp.as_mut_ptr(),
                    thr.ap16.as_mut_ptr().add(col * cs),
                    n32 as i32,
                );
            }
            let t2 = Instant::now();
            ffi::oxe_dot_f16_tile(
                n32 as i32,
                t.vt,
                t.v_stride as i64,
                HD as i32,
                thr.ap16.as_ptr(),
                cs as i64,
                ncol as i32,
                thr.aot.as_mut_ptr(),
                ncol as i64,
                1,
                nf.as_ptr(),
            );
            if prof {
                let t3 = Instant::now();
                thr.ph[0] += (t1 - t0).as_secs_f64();
                thr.ph[1] += (t2 - t1).as_secs_f64();
                thr.ph[2] += (t3 - t2).as_secs_f64();
            }
        }
    }

    /// Output [col][hd] of the last unit run on `thr` (`ncol` columns).
    fn out(&self, k: PgKern, thr: &PgThr, ncol: usize) -> Vec<f32> {
        match k {
            PgKern::Tiled => (0..ncol * HD)
                .map(|i| thr.aot[(i % HD) * ncol + i / HD])
                .collect(),
            PgKern::Fa2 => thr.fb.acc[..ncol * HD].to_vec(),
            PgKern::Fa2x(_) => (0..ncol).flat_map(|c| thr.fa.out(c).to_vec()).collect(),
            PgKern::Pfg(_) => (0..ncol).flat_map(|c| thr.pg.out(c).to_vec()).collect(),
        }
    }
}

/// Replica of `Tile::run` (fa2, PFA_CH chunks) with phase timers: [0] QK, [1] fused chunk
/// softmax + V^T . p, [2] online merge + normalisation.
unsafe fn fa2_unit(t: &Tile, b: &mut Fa2Buf, ph: &mut [f64; 5], prof: bool) {
    // SAFETY: the caller's contract (as Tile::run).
    unsafe {
        let (hd, ncol, group) = (t.hd, t.ncol, t.group);
        let rows = t.last() + 1;
        let n32 = rows.div_ceil(32) * 32;
        let acc = &mut b.acc[..ncol * hd];
        acc.fill(0.0);
        b.m[..ncol].fill(f32::NEG_INFINITY);
        b.l[..ncol].fill(0.0);
        let mut s0 = 0;
        while s0 < n32 {
            let e = (s0 + PFA_CH).min(n32);
            let len = e - s0;
            let kr = rows.min(e) - s0;
            let t0 = Instant::now();
            ffi::oxe_dot_f16_tile(
                hd as i32,
                t.k.add(s0 * hd),
                hd as i64,
                kr as i32,
                t.q16,
                hd as i64,
                ncol as i32,
                b.s.as_mut_ptr(),
                1,
                PFA_CH as i64,
                std::ptr::null(),
            );
            let t1 = Instant::now();
            for (c, v) in b.nv[..ncol].iter_mut().enumerate() {
                *v = (t.pos0 + c / group + 1).saturating_sub(s0).min(len) as i32;
            }
            let len_full = (s0 + PFA_CH).min(t.n_ctx) - s0;
            ffi::oxe_attn_fchunk_f16(
                b.s.as_mut_ptr(),
                PFA_CH as i64,
                ncol as i32,
                b.nv.as_ptr(),
                len as i32,
                len_full as i32,
                t.scale,
                t.vt.add(s0),
                t.v_stride as i64,
                hd as i32,
                b.p.as_mut_ptr(),
                b.part.as_mut_ptr(),
                b.ml.as_mut_ptr(),
            );
            let t2 = Instant::now();
            for c in 0..ncol {
                let (mc, lc) = (b.ml[c], b.ml[ncol + c]);
                if !(lc > 0.0) {
                    continue;
                }
                let mo = b.m[c];
                let o = &mut acc[c * hd..(c + 1) * hd];
                let pc = &b.part[c * hd..(c + 1) * hd];
                if mc <= mo {
                    let w = (mc - mo).exp();
                    b.l[c] += lc * w;
                    for (x, &y) in o.iter_mut().zip(pc) {
                        *x += y * w;
                    }
                } else {
                    let a = if mo == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (mo - mc).exp()
                    };
                    b.l[c] = b.l[c] * a + lc;
                    for (x, &y) in o.iter_mut().zip(pc) {
                        *x = *x * a + y;
                    }
                    b.m[c] = mc;
                }
            }
            if prof {
                let t3 = Instant::now();
                ph[0] += (t1 - t0).as_secs_f64();
                ph[1] += (t2 - t1).as_secs_f64();
                ph[2] += (t3 - t2).as_secs_f64();
            }
            s0 = e;
        }
        let t4 = Instant::now();
        for c in 0..ncol {
            let inv = 1.0 / b.l[c];
            for x in &mut acc[c * hd..(c + 1) * hd] {
                *x *= inv;
            }
        }
        if prof {
            ph[2] += t4.elapsed().as_secs_f64();
        }
    }
}

fn pg_params_from_env() -> Vec<GParams> {
    std::env::var("OXE_PFBENCH_G")
        .unwrap_or_else(|_| "16:32:128:16:ld1r,16:32:128:16:elem,32:32:128:16:ld1r,16:16:128:16:ld1r,16:32:256:16:ld1r,16:32:128:32:ld1r,16:32:64:16:ld1r".into())
        .split(',')
        .filter_map(|s| {
            let f: Vec<&str> = s.trim().split(':').collect();
            if f.len() != 5 {
                return None;
            }
            GParams::new(f[0].parse().ok()?, f[1].parse().ok()?, f[2].parse().ok()?, f[3].parse().ok()?, PgMk::parse(f[4])?)
        })
        .collect()
}

#[test]
#[ignore]
fn kbench_pfg_phases() {
    let nth = env_list("OXE_PFBENCH_THREADS", "12")[0].max(1);
    let depths = env_list("OXE_PFBENCH_DEPTH", "32768,65536");
    let appends = env_list("OXE_PFBENCH_APPEND", "2048");
    let reps = env_list("OXE_PFBENCH_REPS", "3")[0].max(1);
    let nblk = env_list("OXE_PFBENCH_BLOCKS", "24")[0].max(1);
    let on = |k: &str| std::env::var(k).map(|v| v.trim() != "0").unwrap_or(true);
    let (old_on, x_on) = (on("OXE_PFBENCH_OLD"), on("OXE_PFBENCH_FA2X"));
    let gps = pg_params_from_env();
    assert!(
        !gps.is_empty(),
        "OXE_PFBENCH_G: no valid fq:fp:ch:tb:mk entry"
    );
    let mut kerns = Vec::new();
    if old_on {
        kerns.push(PgKern::Tiled);
    }
    kerns.push(PgKern::Fa2);
    if x_on {
        kerns.push(PgKern::Fa2x(XParams::default()));
    }
    kerns.extend(gps.iter().map(|&g| PgKern::Pfg(g)));
    let n_ctx =
        (depths.iter().max().unwrap() + appends.iter().max().unwrap()).div_ceil(256) * 256 + 256;
    let cpus = std::env::var("OXE_PFBENCH_CPUS")
        .ok()
        .map(|s| sched::parse_cpu_list(&s).expect("OXE_PFBENCH_CPUS"))
        .unwrap_or_else(|| cpus(nth));
    let cpus = cpus[..nth.min(cpus.len())].to_vec();
    assert_eq!(
        cpus.len(),
        nth,
        "OXE_PFBENCH_CPUS has fewer cpus than threads"
    );
    let hz = pfg_tick_hz() as f64;
    println!("PFG kbench: head_dim {HD}, {NKV} KV heads x {GROUP}, blocks of {PG_BLOCK} tokens x {nblk}, n_ctx {n_ctx}, threads {nth} on {cpus:?}, reps {reps}, roof {ROOF_TFMA} TFMA/s, tick {hz} Hz");
    let layer = PgLayer::new(n_ctx);
    let thrs: Vec<Shared<PgThr>> = (0..nth)
        .map(|_| {
            let big = if old_on { NCOL * (n_ctx + ATT_PAD) } else { 0 };
            Shared::new(vec![PgThr {
                fa: Scratch::new(NCOL, HD),
                pg: Scratch::new_pfg(HD, GROUP),
                fb: Fa2Buf {
                    s: vec![0.0; NCOL * PFA_CH],
                    p: vec![0; NCOL * PFA_CH],
                    part: vec![0.0; NCOL * HD],
                    ml: vec![0.0; 2 * NCOL],
                    acc: vec![0.0; NCOL * HD],
                    m: vec![0.0; NCOL],
                    l: vec![0.0; NCOL],
                    nv: vec![0; NCOL],
                },
                akq: vec![0.0; big],
                ap16: vec![0; big],
                wp: vec![0.0; if old_on { n_ctx } else { 0 }],
                dp: vec![0.0; if old_on { n_ctx } else { 0 }],
                aot: vec![0.0; HD * NCOL],
                ph: [0.0; 5],
                tk: [0; 5],
            }])
        })
        .collect();
    let mut summary: Vec<String> = Vec::new();
    std::thread::scope(|s| {
        s.spawn(|| {
            let pool = sched::Pool::new(&cpus, Duration::from_micros(200)).unwrap();
            for &d in &depths {
                for &a in &appends {
                    let total = NKV * a.div_ceil(PG_BLOCK);
                    let nb = nblk.min(total);
                    let blocks: Vec<(usize, usize)> = (0..nb).map(|j| j * total / nb).map(|u| (u % NKV, d + (u / NKV) * PG_BLOCK)).collect();
                    // useful FMA of the sampled blocks: QK + PV over each column's causal rows
                    let fma: f64 = blocks.iter().map(|&(_, p0)| (0..PG_BLOCK).map(|t| (p0 + t + 1) as f64).sum::<f64>() * GROUP as f64 * HD as f64 * 2.0).sum();
                    let ctr = AtomicUsize::new(0);
                    let units_of = |k: PgKern| -> Vec<(usize, usize, usize, usize)> {
                        let nt = k.unit_tokens();
                        blocks.iter().flat_map(|&(kvh, p0)| (0..PG_BLOCK).step_by(nt).map(move |t| (kvh, p0, t, nt.min(PG_BLOCK - t)))).collect()
                    };
                    let run = |k: PgKern, prof: bool| {
                        let units = units_of(k);
                        ctr.store(0, Relaxed);
                        pool.run(|ctx| loop {
                            let i = ctr.fetch_add(1, Relaxed);
                            if i >= units.len() {
                                break;
                            }
                            let (kvh, p0, t, nt) = units[i];
                            // SAFETY: slot tid is this thread's; the layer holds every row read.
                            unsafe { layer.run(k, &layer.tile(kvh, p0, t, nt), &mut *thrs[ctx.tid].ptr(), prof) };
                        });
                    };
                    // spot check: block 0's first 8 tokens through every kernel (thread 0's slot)
                    let spot: Vec<Vec<f32>> = kerns
                        .iter()
                        .map(|&k| {
                            // SAFETY: the pool is idle; slot 0 used by this thread only.
                            let thr = unsafe { &mut *thrs[0].ptr() };
                            let (kvh, p0) = blocks[nb / 2];
                            unsafe { layer.run(k, &layer.tile(kvh, p0, 0, TT.min(k.unit_tokens())), thr, false) };
                            let nc = TT.min(k.unit_tokens()) * GROUP;
                            layer.out(k, thr, nc)[..nc * HD].to_vec()
                        })
                        .collect();
                    let base = spot[kerns.iter().position(|&k| k == PgKern::Fa2).unwrap()].clone();
                    for &k in &kerns {
                        run(k, false); // warm: page faults, scratch
                    }
                    let mut times = vec![Vec::new(); kerns.len()];
                    for r in 0..reps {
                        for j in 0..kerns.len() {
                            let ki = (j + r) % kerns.len();
                            let t0 = Instant::now();
                            run(kerns[ki], false);
                            times[ki].push(t0.elapsed().as_secs_f64());
                        }
                    }
                    println!("--- D {d} A {a}: {nb} blocks of {PG_BLOCK} tokens ({:.3e} FMA per rep)", fma);
                    let mut best_pfg = (0f64, String::new());
                    let mut rates = Vec::new();
                    for (ki, &k) in kerns.iter().enumerate() {
                        let w = median(&mut times[ki]);
                        let tf = fma / w / 1e12;
                        rates.push(tf);
                        // one profiled rep: phase split of the summed thread time
                        for t in &thrs {
                            // SAFETY: the pool is idle.
                            let thr = unsafe { &mut *t.ptr() };
                            thr.ph = [0.0; 5];
                            thr.tk = [0; 5];
                        }
                        let tp = Instant::now();
                        run(k, true);
                        let wp = tp.elapsed().as_secs_f64();
                        let mut ph = [0f64; 5];
                        for t in &thrs {
                            // SAFETY: the pool is idle.
                            let thr = unsafe { &*t.ptr() };
                            for i in 0..5 {
                                ph[i] += thr.ph[i] + thr.tk[i] as f64 / hz;
                            }
                        }
                        let names = k.phases();
                        let tot: f64 = ph[..names.len()].iter().sum();
                        let split: Vec<String> = names.iter().enumerate().map(|(i, n)| format!("{n} {:.1}%", 100.0 * ph[i] / tot.max(1e-12))).collect();
                        let us_tok = w * 1e6 * (total as f64 / nb as f64) / a as f64;
                        let dsp = rel(&spot[ki], &base);
                        let line = format!(
                            "D {d:>6} A {a:>5} {:<28} {tf:>6.3} TFMA/s = {:>5.1}% of {ROOF_TFMA} roof, {us_tok:>8.2} us/append-token/layer (wall {:.1} ms; vs fa2 rel {dsp:.1e}); profiled rep {:.1} ms, busy {:.0}%: {}",
                            k.name(),
                            100.0 * tf / ROOF_TFMA,
                            w * 1e3,
                            wp * 1e3,
                            100.0 * tot / (wp * nth as f64),
                            split.join(", ")
                        );
                        println!("{line}");
                        summary.push(line);
                        assert!(dsp < 0.05, "D {d}: {} disagrees with fa2 on the spot block (rel {dsp:.2e})", k.name());
                        if let PgKern::Pfg(g) = k {
                            if tf > best_pfg.0 {
                                best_pfg = (tf, g.tag());
                            }
                        }
                    }
                    let get = |k: PgKern| kerns.iter().position(|&x| x == k).map(|i| rates[i]);
                    let dflt = get(PgKern::Pfg(GParams::default()));
                    let vs = |x: Option<f64>| match (dflt, x) {
                        (Some(p), Some(o)) => format!("{:.2}x", p / o),
                        _ => "-".into(),
                    };
                    let verdict = match dflt {
                        Some(r) if d == 32768 => format!(
                            "GATE (D 32K, pfg default {}): {r:.3} TFMA/s -> {}",
                            GParams::default().tag(),
                            if r >= 0.60 { "PASS (>= 0.60)" } else if r < 0.50 { "KILL (< 0.50)" } else { "between kill and gate (0.50 .. 0.60)" }
                        ),
                        Some(r) => format!("pfg default {r:.3} TFMA/s (gate is registered at D 32K)"),
                        None => "pfg default parameters not in OXE_PFBENCH_G".into(),
                    };
                    let l = format!(
                        "D {d} A {a}: {verdict}; pfg default vs tiled {} vs fa2 {} vs fa2x {}; best pfg {} at {:.3} TFMA/s",
                        vs(get(PgKern::Tiled)),
                        vs(get(PgKern::Fa2)),
                        vs(get(PgKern::Fa2x(XParams::default()))),
                        best_pfg.1,
                        best_pfg.0
                    );
                    println!("{l}");
                    summary.push(l);
                }
            }
        })
        .join()
        .unwrap()
    });
    println!("=== PFG kbench summary ===");
    for l in &summary {
        println!("{l}");
    }
}
