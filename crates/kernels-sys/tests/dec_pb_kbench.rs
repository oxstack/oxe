//! v0.1.4 DEC-PB decode micro-benchmark (model-free): steps [1] (scores K^T q) and [3] (V^T . p) of
//! the engine's tiled f16 decode / verify attention at Ornith's shapes (head_dim 256, 2 KV heads x
//! 8 query heads per lane), plain decode (1 lane = 8 columns per KV head) and the MTP verify step
//! (2 lanes = 16 columns), at 16K / 32K / 48K / 64K positions; step [3] through `oxe_dot_f16_tile`
//! (legacy) against `oxe_dot_f16_tile_pb` (pb 256 / 512 / 1024). Every pb output is checked bit for
//! bit against legacy before it is timed.
//!
//! Run on the laptop in a quiet window, holding ALL build locks (BUILD-RULES 10), <= 5 min:
//! `OXE_DECPB_THREADS=8 taskset -c 0-7 cargo test --release -p kernels-sys --test dec_pb_kbench -- --ignored --nocapture`
//! Optional: OXE_DECPB_REPS (default 8), OXE_DECPB_THREADS (default "8"), OXE_DECPB_NCTX (column /
//! row stride, default 65536), OXE_DECPB_LAYERS (KV copies rotated per rep so the K / V reads come
//! from DRAM as in a real token, default 4), OXE_DECPB_NATT (attention layers per token for the
//! ms/token column, default 10 = Ornith), OXE_DECPB_POS (default "16384,32768,49152,65536"),
//! OXE_DECPB_PB (default "256,512,1024").
//!
//! Threads split the work as the engine does (scores: rows (kvh, p) evenly, 64-row tiles; outputs:
//! V^T rows (kvh, dim) evenly, 32-row tiles, tail rule at padded n_kv) and each repeats its share
//! `reps` times without a barrier between reps; time per rep = the slowest thread / reps. The softmax
//! (step [2]) is the same work in both arms and not timed. Bytes = K and V bytes of the positions.

use kernels_sys::ffi;
use std::time::Instant;

const HD: usize = 256;
const NKV: usize = 2;
const GROUP: usize = 8;
const VPAD: usize = 32;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
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

fn padded_n_kv(pos: usize, n_ctx: usize) -> usize {
    n_ctx.min(256usize.max((pos + 1).div_ceil(256) * 256))
}

#[derive(Clone, Copy)]
struct P(*const u8);
unsafe impl Send for P {}
unsafe impl Sync for P {}
impl P {
    fn p(&self) -> *const u8 {
        self.0
    }
}

/// Wall seconds per rep: `nt` threads run `f(tid, rep)` `reps` times; the slowest thread counts.
fn timed(nt: usize, reps: usize, f: &(dyn Fn(usize, usize) + Sync)) -> f64 {
    for t in 0..nt {
        f(t, 0); // warm (page faults) outside the timer
    }
    let mut worst = 0f64;
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..nt)
            .map(|t| {
                s.spawn(move || {
                    let t0 = Instant::now();
                    for r in 0..reps {
                        f(t, r);
                    }
                    t0.elapsed().as_secs_f64()
                })
            })
            .collect();
        for h in hs {
            worst = worst.max(h.join().unwrap());
        }
    });
    worst / reps as f64
}

fn env_list(k: &str, d: &str) -> Vec<usize> {
    std::env::var(k)
        .unwrap_or(d.into())
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| x.trim().parse().unwrap())
        .collect()
}

fn env_num(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// One full sweep; returns (positions, lanes, pb, legacy us, pb us) rows of step [3].
fn sweep(
    n_ctx: usize,
    layers: usize,
    positions: &[usize],
    threads: &[usize],
    pbs: &[usize],
    reps: usize,
    n_att: usize,
) -> Vec<(usize, usize, usize, f64, f64)> {
    let vs = n_ctx + VPAD;
    let mut r = Lcg(0xdec9b);
    // per layer copy: K [kvh][n_ctx][HD], V^T [kvh][HD][vs]
    let ks: Vec<Vec<u16>> = (0..layers)
        .map(|_| {
            f16s(
                &(0..NKV * n_ctx * HD)
                    .map(|_| r.f() * 0.5)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let vts: Vec<Vec<u16>> = (0..layers)
        .map(|_| f16s(&(0..NKV * HD * vs).map(|_| r.f() * 0.5).collect::<Vec<_>>()))
        .collect();
    let kp: Vec<P> = ks.iter().map(|k| P(k.as_ptr() as _)).collect();
    let vp: Vec<P> = vts.iter().map(|v| P(v.as_ptr() as _)).collect();
    let mut rows_out = Vec::new();
    println!(
        "dec-pb kbench: head_dim {HD}, {NKV} KV heads x {GROUP} q heads per lane, n_ctx stride {n_ctx}, V^T stride {vs}, {layers} KV copies rotated, reps {reps}, n_att {n_att}"
    );
    println!(
        "{:>6} {:>5} {:>3} {:>7} | {:>9} {:>9} {:>9} | {:>8} {:>8} | {:>8} {:>8} | {:>7}",
        "pos",
        "lanes",
        "thr",
        "pv",
        "sc us",
        "pv us",
        "layer us",
        "ms/tok",
        "attnGB/s",
        "V GB/s",
        "Lpv GB/s",
        "pv x"
    );
    for &pos_n in positions {
        if pos_n > n_ctx {
            continue;
        }
        for lanes in [1usize, 2] {
            let ncol = lanes * GROUP;
            let last = pos_n - 1; // kv_len = pos_n
            let pos0 = last + 1 - lanes;
            let n32 = pos_n.div_ceil(32) * 32;
            let rows = last + 1;
            let q = f16s(&(0..NKV * ncol * HD).map(|_| r.f()).collect::<Vec<_>>());
            // probabilities [kvh][col][n_ctx] (+0 beyond the column's position): a smooth random row
            let mut pv = vec![0f32; NKV * ncol * n_ctx];
            for c in 0..NKV * ncol {
                let pos = pos0 + (c % ncol) / GROUP;
                for p in 0..=pos {
                    pv[c * n_ctx + p] = (r.next() % 1000) as f32 / (500.0 * (pos + 1) as f32);
                }
            }
            let p16 = f16s(&pv);
            drop(pv);
            let nf: Vec<i32> = (0..ncol)
                .map(|c| padded_n_kv(pos0 + c / GROUP, n_ctx) as i32)
                .collect();
            let (qp, pp, nfp) = (
                P(q.as_ptr() as _),
                P(p16.as_ptr() as _),
                P(nf.as_ptr() as _),
            );
            for &nt in threads {
                let scores: Vec<std::sync::Mutex<Vec<f32>>> = (0..nt)
                    .map(|_| std::sync::Mutex::new(vec![0f32; NKV * ncol * n_ctx]))
                    .collect();
                let outs: Vec<std::sync::Mutex<(Vec<f32>, Vec<u128>, Vec<u32>)>> = (0..nt)
                    .map(|_| {
                        std::sync::Mutex::new((
                            vec![0f32; 32 * ncol],
                            vec![0u128; 32 * ncol * 4],
                            vec![0u32; NKV * HD * ncol],
                        ))
                    })
                    .collect();
                let sc = |t: usize, rep: usize| {
                    let (a0, b0) = (rows * NKV * t / nt, rows * NKV * (t + 1) / nt);
                    let kb = kp[rep % layers];
                    let mut s = scores[t].lock().unwrap();
                    for kvh in 0..NKV {
                        let a = a0.max(kvh * rows).min((kvh + 1) * rows) - kvh * rows;
                        let b = b0.max(kvh * rows).min((kvh + 1) * rows) - kvh * rows;
                        let mut p = a;
                        while p < b {
                            let pe = (p + 64).min(b);
                            // SAFETY: rows within the buffers; outputs column-strided n_ctx.
                            unsafe {
                                ffi::oxe_dot_f16_tile(
                                    HD as i32,
                                    (kb.p() as *const u16).add((kvh * n_ctx + p) * HD),
                                    HD as i64,
                                    (pe - p) as i32,
                                    (qp.p() as *const u16).add(kvh * ncol * HD),
                                    HD as i64,
                                    ncol as i32,
                                    s.as_mut_ptr().add(kvh * ncol * n_ctx + p),
                                    1,
                                    n_ctx as i64,
                                    std::ptr::null(),
                                );
                            }
                            p = pe;
                        }
                    }
                };
                // step [3]; pb 0 = legacy; outputs kept as bits in o.2 [kvh][dim][col]
                let pvk = |t: usize, rep: usize, pb: usize| {
                    let all = NKV * HD;
                    let (a0, b0) = (all * t / nt, all * (t + 1) / nt);
                    let vb = vp[rep % layers];
                    let mut g = outs[t].lock().unwrap();
                    let (tmp, st, bits) = &mut *g;
                    for kvh in 0..NKV {
                        let a = a0.max(kvh * HD).min((kvh + 1) * HD) - kvh * HD;
                        let b = b0.max(kvh * HD).min((kvh + 1) * HD) - kvh * HD;
                        let mut d = a;
                        while d < b {
                            let de = (d + 32).min(b);
                            let (x, y) = (
                                (vb.p() as *const u16).wrapping_add((kvh * HD + d) * vs),
                                (pp.p() as *const u16).wrapping_add(kvh * ncol * n_ctx),
                            );
                            // SAFETY: V^T rows within the copy (n32 <= n_ctx < vs, finite beyond pos);
                            // probabilities column-strided n_ctx; scratch is this thread's.
                            unsafe {
                                if pb == 0 {
                                    ffi::oxe_dot_f16_tile(
                                        n32 as i32,
                                        x,
                                        vs as i64,
                                        (de - d) as i32,
                                        y,
                                        n_ctx as i64,
                                        ncol as i32,
                                        tmp.as_mut_ptr(),
                                        ncol as i64,
                                        1,
                                        nfp.p() as *const i32,
                                    );
                                } else {
                                    ffi::oxe_dot_f16_tile_pb(
                                        n32 as i32,
                                        x,
                                        vs as i64,
                                        (de - d) as i32,
                                        y,
                                        n_ctx as i64,
                                        ncol as i32,
                                        tmp.as_mut_ptr(),
                                        ncol as i64,
                                        1,
                                        nfp.p() as *const i32,
                                        pb as i32,
                                        st.as_mut_ptr() as *mut u8,
                                    );
                                }
                            }
                            for dd in d..de {
                                for c in 0..ncol {
                                    bits[((kvh * HD + dd) * ncol) + c] =
                                        tmp[(dd - d) * ncol + c].to_bits();
                                }
                            }
                            d = de;
                        }
                    }
                };
                let snapshot = || -> Vec<Vec<u32>> {
                    outs.iter().map(|o| o.lock().unwrap().2.clone()).collect()
                };
                let t_sc = timed(nt, reps, &sc);
                let t_leg = timed(nt, reps, &|t, rep| pvk(t, rep, 0));
                let want = snapshot();
                let bytes_kv = (NKV * pos_n * HD * 2 * 2) as f64;
                let bytes_v = (NKV * pos_n * HD * 2) as f64;
                let line = |name: &str, t_pv: f64| {
                    let layer = t_sc + t_pv;
                    println!(
                        "{:>6} {:>5} {:>3} {:>7} | {:>9.1} {:>9.1} {:>9.1} | {:>8.3} {:>8.1} | {:>8.1} {:>8.1} | {:>7.3}",
                        pos_n,
                        lanes,
                        nt,
                        name,
                        t_sc * 1e6,
                        t_pv * 1e6,
                        layer * 1e6,
                        layer * n_att as f64 * 1e3,
                        bytes_kv / layer / 1e9,
                        bytes_v / t_pv / 1e9,
                        bytes_v / t_leg / 1e9,
                        t_leg / t_pv
                    );
                };
                line("legacy", t_leg);
                for &pb in pbs {
                    let t_pb = timed(nt, reps, &|t, rep| pvk(t, rep, pb));
                    assert!(
                        snapshot() == want,
                        "pb {pb} differs from legacy at pos {pos_n} lanes {lanes} threads {nt}"
                    );
                    line(&format!("pb{pb}"), t_pb);
                    rows_out.push((pos_n, lanes, pb, t_leg * 1e6, t_pb * 1e6));
                }
            }
        }
    }
    println!("sc = step [1] scores, pv = step [3] V^T.p, layer = sc + pv (softmax not timed, equal in both arms); ms/tok = layer x n_att");
    println!("attnGB/s = K+V bytes / layer time; V GB/s = V bytes / pv time (Lpv = legacy); pv x = legacy pv time / this pv time (>1 = faster)");
    println!("every pb row was checked bit for bit against legacy (all V^T rows x columns) before timing");
    rows_out
}

#[test]
#[ignore]
fn dec_pb_kbench() {
    let reps = env_num("OXE_DECPB_REPS", 8);
    let threads = env_list("OXE_DECPB_THREADS", "8");
    let n_ctx = env_num("OXE_DECPB_NCTX", 65536);
    let layers = env_num("OXE_DECPB_LAYERS", 4).max(1);
    let n_att = env_num("OXE_DECPB_NATT", 10);
    let positions = env_list("OXE_DECPB_POS", "16384,32768,49152,65536");
    let pbs = env_list("OXE_DECPB_PB", "256,512,1024");
    sweep(n_ctx, layers, &positions, &threads, &pbs, reps, n_att);
}

/// CI smoke (functional, no timing claim): the bench runs and its pb arms match legacy.
#[test]
fn dec_pb_kbench_smoke() {
    let rows = sweep(4096, 2, &[1000, 4096], &[1, 3], &[256, 512], 1, 10);
    assert_eq!(rows.len(), 2 * 2 * 2 * 2);
}
