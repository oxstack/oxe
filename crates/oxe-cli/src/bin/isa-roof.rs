//! isa-roof: model-free ISA roofline probe (v0.1.4 V014-DIRECTION item 1, "ISA-ROOF").
//!
//! Measures, per core (one pinned thread on each `--single` core) and all-core (one pinned thread
//! on every `--cores` core, started together):
//! - register-only loops with 24 independent accumulators (no dependency chain): FMLA.4s (f32
//!   baseline), FMLA.8h, FMLA.8h by element, LD1R + FMLA.8h, FMLAL, BFDOT, BFMMLA, BFMLALB,
//!   SDOT, SMMLA, the f16 / bf16 / f32 conversions, one FMLA.8h dependency chain (latency);
//! - ggml_v_expf (the softmax exp of every prefill path);
//! - the candidate GEMM tiles of the direction (f16 8x16 by element with fold 0 / 8 / 32, f16
//!   8x16 LD1R pre-broadcast, bf16 8x8 BFMMLA) L1-resident and with the B panel streamed from L2,
//!   each self-tested against f64; plus today's kernels: `oxe_dot_f16_tile` (L1) and one fa2 /
//!   fa2x prefill unit (64 columns, head_dim 256, 512 positions: L2-resident).
//!
//! Each kernel: calibration, warm-up, then >= 5 timed samples. Rates are ops per second (FMA,
//! int8 MAC, converted elements or exps) from the monotonic clock (CNTVCT, a fixed-frequency
//! timer). Per-cycle numbers come from the PMU cycle counter of the thread (perf_event_open,
//! user mode) when the kernel allows it, and from the OS-reported core frequency
//! (cpufreq scaling_cur_freq, read after every sample) always; both are in the receipt.
//! Features are detected at run time: a kernel whose feature is missing is reported
//! `unsupported`, never run. Bounded by `--max-secs` (default 290). No model, no NPU.
//!
//! Receipt: one JSON file (`--out`); a human table on stderr. On anything but a Qualcomm
//! (Oryon) CPU the receipt says the numbers are NOT Oryon numbers.

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use kernels_sys::ffi;
use model::engine::pfattn::{Scratch, Tile, XParams};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// oxe_isa_loop kind; ops per instruction; instructions per iteration
    Asm(i32),
    Exp,
    /// oxe_isa_tile kind, panels
    CTile(i32, i32),
    DotTile,
    Fa2,
    Fa2x,
}

struct KDef {
    name: &'static str,
    group: &'static str,
    /// what one "op" is
    unit: &'static str,
    /// ops per instruction (register loops; 0 = not an instruction count)
    per_inst: f64,
    needs: &'static [&'static str],
    kind: Kind,
    note: &'static str,
}

const HD: usize = 256;
const TILE_K: i32 = 256;
const L2_PANELS: i32 = 64;

fn kernels() -> Vec<KDef> {
    let k = |name, group, unit, per_inst, needs, kind, note| KDef {
        name,
        group,
        unit,
        per_inst,
        needs,
        kind,
        note,
    };
    vec![
        k(
            "f32_fmla_4s",
            "register",
            "FMA",
            4.0,
            &[][..],
            Kind::Asm(0),
            "f32 baseline: FMLA Vd.4S",
        ),
        k(
            "f16_fmla_8h",
            "register",
            "FMA",
            8.0,
            &["fp16"][..],
            Kind::Asm(1),
            "FMLA Vd.8H, Vn.8H, Vm.8H",
        ),
        k(
            "f16_fmla_8h_elem",
            "register",
            "FMA",
            8.0,
            &["fp16"][..],
            Kind::Asm(2),
            "FMLA Vd.8H, Vn.8H, Vm.H[i] (fa2x's form)",
        ),
        k(
            "f16_ld1r_fmla_8h",
            "register",
            "FMA",
            8.0,
            &["fp16"][..],
            Kind::Asm(3),
            "1 LD1R per 2 FMLA.8H (pre-broadcast operand)",
        ),
        k(
            "f16_fmlal",
            "register",
            "FMA",
            4.0,
            &["fp16", "fhm"][..],
            Kind::Asm(4),
            "FMLAL Vd.4S, Vn.4H, Vm.4H (the fa2x fold)",
        ),
        k(
            "bf16_bfdot",
            "register",
            "FMA",
            8.0,
            &["bf16"][..],
            Kind::Asm(5),
            "BFDOT Vd.4S: 8 bf16 MAC",
        ),
        k(
            "bf16_bfmmla",
            "register",
            "FMA",
            16.0,
            &["bf16"][..],
            Kind::Asm(6),
            "BFMMLA Vd.4S: 2x4 . 4x2, 16 bf16 MAC",
        ),
        k(
            "bf16_bfmlalb",
            "register",
            "FMA",
            4.0,
            &["bf16"][..],
            Kind::Asm(7),
            "BFMLALB Vd.4S: 4 bf16 MAC",
        ),
        k(
            "i8_sdot",
            "register",
            "MAC",
            16.0,
            &["dotprod"][..],
            Kind::Asm(8),
            "SDOT Vd.4S: 16 int8 MAC",
        ),
        k(
            "i8_smmla",
            "register",
            "MAC",
            32.0,
            &["i8mm"][..],
            Kind::Asm(9),
            "SMMLA Vd.4S: 32 int8 MAC",
        ),
        k(
            "cvt_f16_to_f32",
            "convert",
            "elem",
            4.0,
            &["fp16"][..],
            Kind::Asm(10),
            "FCVTL 4H -> 4S",
        ),
        k(
            "cvt_f32_to_f16",
            "convert",
            "elem",
            4.0,
            &["fp16"][..],
            Kind::Asm(11),
            "FCVTN 4S -> 4H",
        ),
        k(
            "cvt_f32_to_bf16",
            "convert",
            "elem",
            4.0,
            &["bf16"][..],
            Kind::Asm(12),
            "BFCVTN 4S -> 4H",
        ),
        k(
            "cvt_bf16_to_f32",
            "convert",
            "elem",
            4.0,
            &[][..],
            Kind::Asm(13),
            "SHLL #16 4H -> 4S",
        ),
        k(
            "f16_fmla_8h_chain",
            "latency",
            "FMA",
            8.0,
            &["fp16"][..],
            Kind::Asm(14),
            "ONE dependency chain: 1 / (inst per cycle) = FMLA.8H latency in cycles",
        ),
        k(
            "f32_exp_ggml",
            "softmax",
            "exp",
            0.0,
            &[][..],
            Kind::Exp,
            "ggml_v_expf, 4 vectors in flight, L1",
        ),
        k(
            "tile_f16_8x16_elem_nofold_l1",
            "tile",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::CTile(0, 1),
            "8x16 by element, pure f16 chain over k=256 (upper bound of the shape)",
        ),
        k(
            "tile_f16_8x16_elem_fold8_l1",
            "tile",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::CTile(1, 1),
            "fa2x QK shape: fold 8 into f32",
        ),
        k(
            "tile_f16_8x16_elem_fold32_l1",
            "tile",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::CTile(2, 1),
            "pfg-f16 candidate: fold 32",
        ),
        k(
            "tile_f16_8x16_ld1r_fold32_l1",
            "tile",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::CTile(3, 1),
            "pfg-f16 candidate: LD1R pre-broadcast, fold 32",
        ),
        k(
            "tile_bf16_8x8_bfmmla_l1",
            "tile",
            "FMA",
            0.0,
            &["bf16"][..],
            Kind::CTile(4, 1),
            "pfg-bf candidate: BFMMLA, f32 accumulation",
        ),
        k(
            "tile_f16_8x16_elem_fold8_l2",
            "tile",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::CTile(1, L2_PANELS),
            "fold 8, 64 B panels (256 KiB) streamed from L2",
        ),
        k(
            "tile_f16_8x16_elem_fold32_l2",
            "tile",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::CTile(2, L2_PANELS),
            "fold 32, B from L2",
        ),
        k(
            "tile_f16_8x16_ld1r_fold32_l2",
            "tile",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::CTile(3, L2_PANELS),
            "LD1R fold 32, B from L2",
        ),
        k(
            "tile_bf16_8x8_bfmmla_l2",
            "tile",
            "FMA",
            0.0,
            &["bf16"][..],
            Kind::CTile(4, L2_PANELS),
            "BFMMLA, A panels (256 KiB) from L2",
        ),
        k(
            "today_dot_f16_tile_l1",
            "today",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::DotTile,
            "oxe_dot_f16_tile: 64 K rows x 64 columns x 256 (fa2 / tiled scores)",
        ),
        k(
            "today_fa2_unit_l2",
            "today",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::Fa2,
            "fa2 unit: 64 columns, hd 256, 512 positions (useful QK+PV FMA, softmax included)",
        ),
        k(
            "today_fa2x_unit_l2",
            "today",
            "FMA",
            0.0,
            &["fp16"][..],
            Kind::Fa2x,
            "fa2x unit (fold 8, ch 128), same shape",
        ),
    ]
}

fn feature(name: &str) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        match name {
            "fp16" => std::arch::is_aarch64_feature_detected!("fp16"),
            "fhm" => std::arch::is_aarch64_feature_detected!("fhm"),
            "bf16" => std::arch::is_aarch64_feature_detected!("bf16"),
            "i8mm" => std::arch::is_aarch64_feature_detected!("i8mm"),
            "dotprod" => std::arch::is_aarch64_feature_detected!("dotprod"),
            _ => false,
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = name;
        false
    }
}

// ------------------------------------------------------------------------------------------
// benches

trait Bench {
    /// runs `iters` units of work; returns the ops done
    fn run(&mut self, iters: u64) -> f64;
}

struct AsmB {
    id: i32,
    ops_iter: f64,
}
impl Bench for AsmB {
    fn run(&mut self, iters: u64) -> f64 {
        // SAFETY: the kind's features were detected before this bench was built.
        let rc = unsafe { ffi::oxe_isa_loop(self.id, iters) };
        assert_eq!(rc, 0, "oxe_isa_loop kind {}", self.id);
        iters as f64 * self.ops_iter
    }
}

struct ExpB(f32);
impl Bench for ExpB {
    fn run(&mut self, iters: u64) -> f64 {
        // SAFETY: plain computation on stack buffers.
        self.0 += unsafe { ffi::oxe_isa_exp(iters) };
        iters as f64 * 256.0
    }
}

struct CTileB {
    h: *mut core::ffi::c_void,
    fma: f64,
}
impl CTileB {
    fn new(kind: i32, panels: i32) -> CTileB {
        // SAFETY: valid arguments; checked for NULL.
        let h = unsafe { ffi::oxe_isa_tile_new(kind, TILE_K, panels) };
        assert!(!h.is_null(), "oxe_isa_tile_new {kind} {panels}");
        // SAFETY: a live tile.
        let fma = unsafe { ffi::oxe_isa_tile_fma(h) };
        CTileB { h, fma }
    }
}
impl Bench for CTileB {
    fn run(&mut self, iters: u64) -> f64 {
        // SAFETY: a live tile owned by this thread.
        unsafe { ffi::oxe_isa_tile_run(self.h, iters) };
        iters as f64 * self.fma
    }
}
impl Drop for CTileB {
    fn drop(&mut self) {
        // SAFETY: allocated by oxe_isa_tile_new, freed once.
        unsafe { ffi::oxe_isa_tile_free(self.h) }
    }
}

fn lcg_f16(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut s = seed;
    let v: Vec<f32> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) % 20001) as f32 / 10000.0 - 1.0
        })
        .map(|x| x * scale)
        .collect();
    let mut o = vec![0u16; n];
    // SAFETY: equal lengths.
    unsafe { ffi::oxe_fp32_to_fp16_row(v.as_ptr(), o.as_mut_ptr(), n as i64) };
    o
}

struct DotB {
    k: Vec<u16>,
    q: Vec<u16>,
    out: Vec<f32>,
}
const DOT_ROWS: usize = 64;
const DOT_COLS: usize = 64;
impl Bench for DotB {
    fn run(&mut self, iters: u64) -> f64 {
        for _ in 0..iters {
            // SAFETY: k holds DOT_ROWS x HD, q DOT_COLS x HD, out DOT_COLS x DOT_ROWS.
            unsafe {
                ffi::oxe_dot_f16_tile(
                    HD as i32,
                    self.k.as_ptr(),
                    HD as i64,
                    DOT_ROWS as i32,
                    self.q.as_ptr(),
                    HD as i64,
                    DOT_COLS as i32,
                    self.out.as_mut_ptr(),
                    1,
                    DOT_ROWS as i64,
                    std::ptr::null(),
                )
            };
        }
        std::hint::black_box(&self.out);
        iters as f64 * (HD * DOT_ROWS * DOT_COLS) as f64
    }
}

/// One prefill unit as the engine runs it: 8 tokens x 8 query heads of one KV head, hd 256,
/// last token at the end of a 512-position context (K + V^T: 512 KiB, L2-resident).
struct UnitB {
    k: Vec<u16>,
    vt: Vec<u16>,
    q: Vec<u16>,
    scr: Scratch,
    x: Option<XParams>,
    n_ctx: usize,
    useful: f64,
}
const UNIT_GROUP: usize = 8;
const UNIT_COLS: usize = 64;
const UNIT_CTX: usize = 512;
impl UnitB {
    fn new(x: Option<XParams>) -> UnitB {
        let n_ctx = UNIT_CTX;
        let pos0 = n_ctx - UNIT_COLS / UNIT_GROUP;
        let useful: f64 = (0..UNIT_COLS)
            .map(|c| 2.0 * HD as f64 * (pos0 + c / UNIT_GROUP + 1) as f64)
            .sum();
        UnitB {
            k: lcg_f16(n_ctx * HD, 11, 1.0),
            vt: lcg_f16(HD * n_ctx, 12, 1.0),
            q: lcg_f16(UNIT_COLS * HD, 13, 4.0),
            scr: Scratch::new(UNIT_COLS, HD),
            x,
            n_ctx,
            useful,
        }
    }
    fn tile(&self) -> Tile {
        Tile {
            hd: HD,
            ncol: UNIT_COLS,
            group: UNIT_GROUP,
            pos0: self.n_ctx - UNIT_COLS / UNIT_GROUP,
            scale: 1.0 / (HD as f32).sqrt(),
            k: self.k.as_ptr(),
            vt: self.vt.as_ptr(),
            v_stride: self.n_ctx,
            n_ctx: self.n_ctx,
            q16: self.q.as_ptr(),
        }
    }
}
impl Bench for UnitB {
    fn run(&mut self, iters: u64) -> f64 {
        let t = self.tile();
        for _ in 0..iters {
            // SAFETY: K / V^T hold every row up to n_ctx (finite), q 64 columns; scratch for 64 x 256.
            unsafe {
                match self.x {
                    None => t.run(&mut self.scr),
                    Some(xp) => t.run_x(&mut self.scr, xp),
                }
            }
        }
        std::hint::black_box(self.scr.out(0));
        iters as f64 * self.useful
    }
}

fn make(k: &KDef) -> Box<dyn Bench> {
    match k.kind {
        Kind::Asm(id) => Box::new(AsmB {
            id,
            ops_iter: 24.0 * k.per_inst,
        }),
        Kind::Exp => Box::new(ExpB(0.0)),
        Kind::CTile(t, p) => Box::new(CTileB::new(t, p)),
        Kind::DotTile => Box::new(DotB {
            k: lcg_f16(DOT_ROWS * HD, 21, 1.0),
            q: lcg_f16(DOT_COLS * HD, 22, 1.0),
            out: vec![0.0; DOT_ROWS * DOT_COLS],
        }),
        Kind::Fa2 => Box::new(UnitB::new(None)),
        Kind::Fa2x => Box::new(UnitB::new(Some(XParams::default()))),
    }
}

// ------------------------------------------------------------------------------------------
// system

fn read_trim(p: &str) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
}

fn cur_khz(cpu: usize) -> Option<u64> {
    read_trim(&format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
    ))
    .and_then(|s| s.parse().ok())
}

fn cluster_of(cpu: usize) -> String {
    read_trim(&format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/related_cpus"
    ))
    .unwrap_or_else(|| "?".into())
}

fn online_cpus() -> Vec<usize> {
    read_trim("/sys/devices/system/cpu/online")
        .and_then(|s| sched::parse_cpu_list(&s).ok())
        .unwrap_or_else(|| {
            (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect()
        })
}

/// (implementer, parts, model name) from /proc/cpuinfo
fn cpu_ident() -> (String, Vec<String>, Option<String>) {
    let txt = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let mut imp = String::new();
    let mut parts: Vec<String> = Vec::new();
    let mut model = None;
    for l in txt.lines() {
        let (k, v) = match l.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => continue,
        };
        match k {
            "CPU implementer" if imp.is_empty() => imp = v.to_string(),
            "CPU part" if !parts.iter().any(|p| p == v) => parts.push(v.to_string()),
            "model name" | "Model" if model.is_none() => model = Some(v.to_string()),
            _ => {}
        }
    }
    (imp, parts, model)
}

struct Cycles(i32);
impl Cycles {
    fn open() -> Cycles {
        // SAFETY: plain syscall wrapper.
        Cycles(unsafe { ffi::oxe_isa_cycles_open() })
    }
    fn read(&self) -> Option<u64> {
        if self.0 < 0 {
            return None;
        }
        // SAFETY: an fd this thread opened.
        let v = unsafe { ffi::oxe_isa_cycles_read(self.0) };
        (v >= 0).then_some(v as u64)
    }
}
impl Drop for Cycles {
    fn drop(&mut self) {
        // SAFETY: closes our own fd (no-op when negative).
        unsafe { ffi::oxe_isa_cycles_close(self.0) }
    }
}

// ------------------------------------------------------------------------------------------
// measurement

#[derive(Clone, Copy, Debug)]
struct Sample {
    secs: f64,
    ops: f64,
    cycles: Option<u64>,
    khz: Option<u64>,
}

impl Sample {
    fn rate(&self) -> f64 {
        self.ops / self.secs
    }
}

#[derive(Clone, Copy)]
struct Cfg {
    samples: usize,
    sample: Duration,
    warmup: Duration,
}

/// iterations per chunk so that a chunk takes about `target`
fn calibrate(b: &mut dyn Bench, target: Duration) -> u64 {
    let mut iters = 1u64;
    loop {
        let t = Instant::now();
        b.run(iters);
        let el = t.elapsed();
        if el >= Duration::from_micros(500) || iters >= 1 << 40 {
            let per = el.as_secs_f64() / iters as f64;
            return ((target.as_secs_f64() / per.max(1e-12)) as u64).max(1);
        }
        iters *= 4;
    }
}

/// chunks of `chunk` until `dur` has passed
fn run_for(b: &mut dyn Bench, chunk: u64, dur: Duration, cyc: &Cycles, cpu: usize) -> Sample {
    let c0 = cyc.read();
    let t = Instant::now();
    let mut ops = 0.0;
    loop {
        ops += b.run(chunk);
        if t.elapsed() >= dur {
            break;
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let c1 = cyc.read();
    let cycles = match (c0, c1) {
        (Some(x0), Some(x1)) if x1 > x0 => Some(x1 - x0),
        _ => None,
    };
    Sample {
        secs,
        ops,
        cycles,
        khz: cur_khz(cpu),
    }
}

struct ThreadResult {
    cpu: usize,
    samples: Vec<Sample>,
    cycles_err: Option<i32>,
}

/// One pinned thread per cpu, started together (barrier) for warm-up and every sample.
fn measure(k: &'static KDef, cpus: &[usize], cfg: Cfg) -> Result<Vec<ThreadResult>, String> {
    let bar = Arc::new(Barrier::new(cpus.len()));
    let hs: Vec<_> = cpus
        .iter()
        .map(|&cpu| {
            let bar = bar.clone();
            std::thread::spawn(move || -> Result<ThreadResult, String> {
                let pinned =
                    sched::pin_current_thread(cpu).map_err(|e| format!("pin cpu {cpu}: {e}"));
                let cyc = Cycles::open();
                let mut b = make(k);
                let chunk = calibrate(b.as_mut(), cfg.sample / 10);
                bar.wait();
                run_for(b.as_mut(), chunk, cfg.warmup, &cyc, cpu);
                let mut samples = Vec::with_capacity(cfg.samples);
                for _ in 0..cfg.samples {
                    bar.wait();
                    samples.push(run_for(b.as_mut(), chunk, cfg.sample, &cyc, cpu));
                }
                // a pin failure is reported only after every barrier (the other threads wait there)
                pinned?;
                Ok(ThreadResult {
                    cpu,
                    samples,
                    cycles_err: (cyc.0 < 0).then_some(-cyc.0),
                })
            })
        })
        .collect();
    let mut out = Vec::new();
    let mut err = None;
    for h in hs {
        match h.join() {
            Ok(Ok(r)) => out.push(r),
            Ok(Err(e)) => err = Some(e),
            Err(_) => err = Some("measurement thread panicked".into()),
        }
    }
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn opt(x: Option<f64>) -> Value {
    x.map_or(Value::Null, |v| json!(r3(v)))
}

/// Per-thread summary: rate (G ops/s), per cycle (PMU) and per cycle at the OS frequency.
fn thread_json(k: &KDef, r: &ThreadResult) -> (Value, f64, Option<f64>, Option<f64>) {
    let rate = median(r.samples.iter().map(|s| s.rate()).collect()).unwrap_or(0.0);
    let pmu = median(
        r.samples
            .iter()
            .filter_map(|s| s.cycles.map(|c| s.ops / c as f64))
            .collect(),
    );
    let osf = median(
        r.samples
            .iter()
            .filter_map(|s| s.khz.map(|f| s.rate() / (f as f64 * 1e3)))
            .collect(),
    );
    let khz = median(
        r.samples
            .iter()
            .filter_map(|s| s.khz.map(|f| f as f64))
            .collect(),
    );
    let ipc = |x: Option<f64>| {
        if k.per_inst > 0.0 {
            x.map(|v| v / k.per_inst)
        } else {
            None
        }
    };
    let v = json!({
        "cpu": r.cpu,
        "cluster": cluster_of(r.cpu),
        "gops_median": r3(rate / 1e9),
        "gops_min": r3(r.samples.iter().map(|s| s.rate()).fold(f64::INFINITY, f64::min) / 1e9),
        "gops_max": r3(r.samples.iter().map(|s| s.rate()).fold(0.0, f64::max) / 1e9),
        "ops_per_cycle_pmu": opt(pmu),
        "inst_per_cycle_pmu": opt(ipc(pmu)),
        "ops_per_cycle_at_os_freq": opt(osf),
        "inst_per_cycle_at_os_freq": opt(ipc(osf)),
        "os_khz_median": opt(khz),
        "pmu_cycles": if r.cycles_err.is_none() { json!("perf_event cycles (user)") } else { json!(format!("unavailable (errno {})", r.cycles_err.unwrap())) },
        "samples": r.samples.iter().map(|s| json!({
            "secs": r3(s.secs * 1e3) / 1e3, "gops": r3(s.rate() / 1e9), "cycles": s.cycles, "os_khz": s.khz,
        })).collect::<Vec<_>>(),
    });
    (v, rate, pmu, osf)
}

// ------------------------------------------------------------------------------------------
// CLI

struct Opts {
    cores: Vec<usize>,
    single: Vec<usize>,
    cfg: Cfg,
    sustain_secs: u64,
    sustain_kernel: String,
    only: Vec<String>,
    max_secs: u64,
    out: String,
    all_core: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: isa-roof [--cores LIST|all] [--single LIST|auto|none] [--samples N>=5] [--sample-ms 100]\n\
         \x20               [--warmup-ms 300] [--no-all-core] [--sustain-secs 0] [--sustain-kernel f16_fmla_8h]\n\
         \x20               [--only NAME,SUBSTR,..] [--max-secs 290] [--quick] [--out isa-roof.json] [--list]\n\
         --single auto = the first core of each cpufreq cluster within --cores. Model-free."
    );
    std::process::exit(2);
}

fn parse_args(args: Vec<String>) -> Opts {
    let mut o = Opts {
        cores: online_cpus(),
        single: Vec::new(),
        cfg: Cfg {
            samples: 5,
            sample: Duration::from_millis(100),
            warmup: Duration::from_millis(300),
        },
        sustain_secs: 0,
        sustain_kernel: "f16_fmla_8h".into(),
        only: Vec::new(),
        max_secs: 290,
        out: "isa-roof.json".into(),
        all_core: true,
    };
    let mut single: Option<String> = None;
    let mut it = args.into_iter();
    let num = |v: Option<String>, what: &str| -> u64 {
        v.and_then(|s| s.trim().parse().ok()).unwrap_or_else(|| {
            eprintln!("isa-roof: {what} needs a number");
            usage()
        })
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cores" => {
                let v = it.next().unwrap_or_else(|| usage());
                if v != "all" {
                    o.cores = sched::parse_cpu_list(&v).unwrap_or_else(|e| {
                        eprintln!("isa-roof: --cores: {e}");
                        usage()
                    });
                }
            }
            "--single" => single = Some(it.next().unwrap_or_else(|| usage())),
            "--samples" => o.cfg.samples = num(it.next(), "--samples") as usize,
            "--sample-ms" => {
                o.cfg.sample = Duration::from_millis(num(it.next(), "--sample-ms").max(1))
            }
            "--warmup-ms" => o.cfg.warmup = Duration::from_millis(num(it.next(), "--warmup-ms")),
            "--sustain-secs" => o.sustain_secs = num(it.next(), "--sustain-secs"),
            "--sustain-kernel" => o.sustain_kernel = it.next().unwrap_or_else(|| usage()),
            "--only" => {
                o.only = it
                    .next()
                    .unwrap_or_else(|| usage())
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--max-secs" => o.max_secs = num(it.next(), "--max-secs").clamp(10, 300),
            "--out" => o.out = it.next().unwrap_or_else(|| usage()),
            "--no-all-core" => o.all_core = false,
            "--quick" => {
                o.cfg.sample = Duration::from_millis(10);
                o.cfg.warmup = Duration::from_millis(10);
            }
            "--list" => {
                for k in kernels() {
                    println!(
                        "{:<32} {:<9} needs {:?}: {}",
                        k.name, k.group, k.needs, k.note
                    );
                }
                std::process::exit(0);
            }
            _ => usage(),
        }
    }
    if o.cfg.samples < 5 {
        eprintln!("isa-roof: --samples must be >= 5");
        usage();
    }
    if o.cores.is_empty() {
        eprintln!("isa-roof: empty --cores");
        usage();
    }
    o.single = match single.as_deref().unwrap_or("auto") {
        "none" => Vec::new(),
        "auto" => {
            if cluster_of(o.cores[0]) == "?" {
                vec![o.cores[0]] // no cpufreq topology: the first core only
            } else {
                let mut seen: Vec<String> = Vec::new();
                let mut v = Vec::new();
                for &c in &o.cores {
                    let cl = cluster_of(c);
                    if !seen.contains(&cl) {
                        seen.push(cl);
                        v.push(c);
                    }
                }
                v
            }
        }
        s => sched::parse_cpu_list(s).unwrap_or_else(|e| {
            eprintln!("isa-roof: --single: {e}");
            usage()
        }),
    };
    o
}

fn selected(o: &Opts, k: &KDef) -> bool {
    o.only.is_empty() || o.only.iter().any(|s| k.name.contains(s.as_str()))
}

fn main() {
    sched::cli::exit_zero_on_epipe();
    let o = parse_args(std::env::args().skip(1).collect());
    let t_start = Instant::now();
    let budget = Duration::from_secs(o.max_secs);
    let (imp, parts, model) = cpu_ident();
    let qualcomm = imp.eq_ignore_ascii_case("0x51");
    let apple = imp.eq_ignore_ascii_case("0x61");
    let label = if qualcomm {
        "Qualcomm CPU (Oryon if CPU part 0x001): these are the numbers the direction asks for"
    } else if apple {
        "Apple M-series (lab-arm64 helper): NOT Oryon numbers; functional test only"
    } else {
        "not a Qualcomm CPU: NOT Oryon numbers; functional test only"
    };
    let feats: Vec<(&str, bool)> = ["fp16", "fhm", "dotprod", "bf16", "i8mm"]
        .iter()
        .map(|&f| (f, feature(f)))
        .collect();
    let has = |f: &str| feats.iter().any(|(n, h)| *n == f && *h);
    eprintln!("isa-roof: {label}; implementer {imp} parts {parts:?}; features {feats:?}");
    eprintln!(
        "isa-roof: cores {:?}, single {:?}, {} samples x {:?} after {:?} warm-up, budget {:?}",
        o.cores, o.single, o.cfg.samples, o.cfg.sample, o.cfg.warmup, budget
    );

    // tile self-tests (functional; fast)
    let mut selftests = serde_json::Map::new();
    let mut self_ok = true;
    for (kind, name, need, tol) in [
        (0, "f16_8x16_elem_nofold", "fp16", 3e-2),
        (1, "f16_8x16_elem_fold8", "fp16", 5e-3),
        (2, "f16_8x16_elem_fold32", "fp16", 1e-2),
        (3, "f16_8x16_ld1r_fold32", "fp16", 1e-2),
        (4, "bf16_8x8_bfmmla", "bf16", 1e-4),
    ] {
        if !has(need) {
            selftests.insert(
                name.into(),
                json!({"status": format!("unsupported (no {need})")}),
            );
            continue;
        }
        // SAFETY: feature detected; plain computation.
        let e = unsafe { ffi::oxe_isa_tile_selftest(kind, TILE_K) };
        let ok = (0.0..=tol).contains(&e);
        self_ok &= ok;
        selftests.insert(
            name.into(),
            json!({"rel_err_vs_f64": e, "tol": tol, "ok": ok}),
        );
        eprintln!(
            "selftest {name:<24} rel err {e:.2e} (tol {tol:.0e}) {}",
            if ok { "ok" } else { "FAIL" }
        );
    }

    let all: &'static [KDef] = Box::leak(kernels().into_boxed_slice());
    let mut rows = Vec::new();
    let mut table: Vec<String> = Vec::new();
    let mut by_name: std::collections::HashMap<&str, (Option<f64>, Option<f64>, f64)> =
        Default::default();
    for k in all.iter().filter(|k| selected(&o, k)) {
        let missing: Vec<&str> = k.needs.iter().copied().filter(|f| !has(f)).collect();
        let mut row = json!({"name": k.name, "group": k.group, "unit": k.unit, "ops_per_inst": k.per_inst, "needs": k.needs, "note": k.note});
        if !missing.is_empty() {
            row["status"] = json!(format!("unsupported (CPU lacks {})", missing.join(", ")));
            table.push(format!(
                "{:<32} unsupported (no {})",
                k.name,
                missing.join(", ")
            ));
            rows.push(row);
            continue;
        }
        let per_kernel = o.cfg.warmup + o.cfg.sample * (o.cfg.samples as u32 + 1);
        let phases = o.single.len() as u32 + o.all_core as u32;
        if t_start.elapsed() + per_kernel * phases + Duration::from_secs(o.sustain_secs) > budget {
            row["status"] = json!("skipped: time budget (--max-secs)");
            table.push(format!("{:<32} skipped (time budget)", k.name));
            rows.push(row);
            continue;
        }
        let mut singles = Vec::new();
        let mut line = format!("{:<32}", k.name);
        let mut status = "ok".to_string();
        for &cpu in &o.single {
            match measure(k, &[cpu], o.cfg) {
                Ok(rs) => {
                    let (v, rate, pmu, osf) = thread_json(k, &rs[0]);
                    let best = by_name.entry(k.name).or_insert((None, None, 0.0));
                    if rate > best.2 {
                        *best = (pmu, osf, rate);
                    }
                    line += &format!(
                        " cpu{cpu}: {:8.2} G{}/s pmu {:>6} os-f {:>6} |",
                        rate / 1e9,
                        k.unit,
                        pmu.map_or("-".into(), |x| format!("{x:.2}")),
                        osf.map_or("-".into(), |x| format!("{x:.2}"))
                    );
                    singles.push(v);
                }
                Err(e) => status = format!("error: {e}"),
            }
        }
        let mut allc = Value::Null;
        if o.all_core && status == "ok" {
            match measure(k, &o.cores, o.cfg) {
                Ok(rs) => {
                    let nsamp = o.cfg.samples;
                    let totals: Vec<f64> = (0..nsamp)
                        .map(|s| rs.iter().map(|r| r.samples[s].rate()).sum())
                        .collect();
                    let tot = median(totals.clone()).unwrap_or(0.0);
                    let threads: Vec<Value> = rs.iter().map(|r| thread_json(k, r).0).collect();
                    allc = json!({
                        "cpus": o.cores, "tops_median": r3(tot / 1e12 * 1e3) / 1e3, "tops_samples": totals.iter().map(|t| r3(t / 1e9) / 1e3).collect::<Vec<_>>(),
                        "threads": threads,
                    });
                    line += &format!(" all {}: {:8.3} T{}/s", o.cores.len(), tot / 1e12, k.unit);
                }
                Err(e) => status = format!("error: {e}"),
            }
        }
        row["status"] = json!(status);
        row["single"] = json!(singles);
        row["all_core"] = allc;
        table.push(line);
        rows.push(row);
    }

    // sustained all-core run of one kernel (windows of 5 s)
    let mut sustain = Value::Null;
    if o.sustain_secs > 0 {
        let k = all.iter().find(|k| k.name == o.sustain_kernel);
        let left = budget
            .saturating_sub(t_start.elapsed())
            .saturating_sub(Duration::from_secs(5));
        let secs = Duration::from_secs(o.sustain_secs).min(left);
        match k {
            Some(k) if k.needs.iter().all(|f| has(f)) && secs >= Duration::from_secs(5) => {
                let win = Duration::from_secs(5);
                let n = (secs.as_secs() / 5).max(1) as usize;
                let cfg = Cfg {
                    samples: n,
                    sample: win,
                    warmup: Duration::from_millis(500),
                };
                // measure() demands no minimum; the >= 5 rule is for the kernel table
                match measure(k, &o.cores, cfg) {
                    Ok(rs) => {
                        let wins: Vec<Value> = (0..n)
                            .map(|s| {
                                let tot: f64 = rs.iter().map(|r| r.samples[s].rate()).sum();
                                let mut by_cl: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
                                for r in &rs {
                                    if let Some(f) = r.samples[s].khz {
                                        by_cl.entry(cluster_of(r.cpu)).or_default().push(f as f64);
                                    }
                                }
                                let cl: serde_json::Map<String, Value> = by_cl.into_iter().map(|(c, v)| (c, opt(median(v)))).collect();
                                json!({"t_end_s": (s + 1) * 5, "tops": r3(tot / 1e9) / 1e3, "os_khz_median_by_cluster": cl})
                            })
                            .collect();
                        for w in &wins {
                            eprintln!("sustain {} {w}", k.name);
                        }
                        sustain = json!({"kernel": k.name, "secs": n * 5, "cpus": o.cores, "windows": wins});
                    }
                    Err(e) => sustain = json!({"kernel": k.name, "error": e}),
                }
            }
            Some(k) => {
                sustain =
                    json!({"kernel": k.name, "status": "skipped (unsupported or no time left)"})
            }
            None => sustain = json!({"kernel": o.sustain_kernel, "status": "unknown kernel"}),
        }
    }

    // the direction's outcome table (section 5 item 1), from the best single core
    let ipc = |name: &str, per: f64| -> (Option<f64>, &'static str) {
        match by_name.get(name) {
            Some((Some(p), _, _)) => (Some(p / per), "pmu"),
            Some((None, Some(f), _)) => (Some(f / per), "os_freq"),
            _ => (None, "n/a"),
        }
    };
    let (fmla, src_f) = ipc("f16_fmla_8h", 8.0);
    let (elem, _) = ipc("f16_fmla_8h_elem", 8.0);
    let (bfm, src_b) = ipc("bf16_bfmmla", 16.0);
    let elem_ratio = match (by_name.get("f16_fmla_8h_elem"), by_name.get("f16_fmla_8h")) {
        (Some(e), Some(f)) if f.2 > 0.0 => Some(e.2 / f.2),
        _ => None,
    };
    let mut hints = Vec::new();
    if let (Some(f), Some(r)) = (fmla, elem_ratio) {
        if f >= 3.6 && r >= 0.9 {
            hints.push(
                "FMLA.8h >= 3.6/cycle and by-element at full rate: path A (pfg-f16)".to_string(),
            );
        }
    }
    if let Some(b) = bfm {
        if b >= 2.0 {
            hints.push("BFMMLA >= 2/cycle: path B (pfg-bf) is the candidate".to_string());
        }
    }
    if let (Some(f), Some(b)) = (fmla, bfm) {
        if f <= 3.0 && b < 2.0 {
            hints.push("FMLA.8h <= 3/cycle and BFMMLA < 2/cycle: CPU roofline ~0.75 TFMA/s; stop kernel work, consider NPU Step-2".to_string());
        }
    }
    if hints.is_empty() {
        hints.push("no rule of the outcome table fires cleanly: read the numbers".to_string());
    }
    let verdict = json!({
        "applies": qualcomm,
        "note": if qualcomm { "single-core, best cluster" } else { "NOT Oryon: the outcome table does not apply to these numbers" },
        "fmla_8h_inst_per_cycle": opt(fmla), "fmla_8h_elem_inst_per_cycle": opt(elem), "by_element_rate_ratio": opt(elem_ratio),
        "bfmmla_inst_per_cycle": opt(bfm), "per_cycle_source": {"fmla": src_f, "bfmmla": src_b},
        "hints": hints,
    });

    // SAFETY: reads CNTFRQ_EL0 (EL0-readable on Linux).
    let cntfrq = unsafe { ffi::oxe_isa_cntfrq() };
    let receipt = json!({
        "tool": "isa-roof",
        "schema": 1,
        "engine_commit": option_env!("OXE_GIT_COMMIT"),
        "machine": {"label": label, "oryon_numbers": qualcomm, "implementer": imp, "parts": parts, "model": model,
                    "features": feats.iter().map(|(n, h)| (n.to_string(), json!(h))).collect::<serde_json::Map<_, _>>(),
                    "online": online_cpus(), "kernel": read_trim("/proc/sys/kernel/osrelease")},
        "timer": {"clock": "monotonic (CNTVCT, fixed frequency)", "cntfrq_hz": cntfrq},
        "freq_source": "cpufreq scaling_cur_freq after each sample (OS-reported, may be the request, not the clock)",
        "config": {"cores": o.cores, "single": o.single, "samples": o.cfg.samples, "sample_ms": o.cfg.sample.as_millis() as u64,
                   "warmup_ms": o.cfg.warmup.as_millis() as u64, "accumulators": 24, "tile_k": TILE_K, "l2_panels": L2_PANELS,
                   "unit_shape": {"cols": UNIT_COLS, "group": UNIT_GROUP, "hd": HD, "n_ctx": UNIT_CTX}},
        "selftests": selftests,
        "selftests_ok": self_ok,
        "kernels": rows,
        "sustain": sustain,
        "verdict": verdict,
        "elapsed_s": r3(t_start.elapsed().as_secs_f64()),
    });
    for l in &table {
        eprintln!("{l}");
    }
    eprintln!("verdict: {}", receipt["verdict"]);
    let txt = serde_json::to_string_pretty(&receipt).unwrap();
    if let Err(e) = std::fs::write(&o.out, txt + "\n") {
        eprintln!("isa-roof: write {}: {e}", o.out);
        std::process::exit(1);
    }
    eprintln!(
        "isa-roof: receipt {} ({:.1} s)",
        o.out,
        t_start.elapsed().as_secs_f64()
    );
    if !self_ok {
        eprintln!("isa-roof: a tile self-test FAILED");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Functional: every supported kernel runs, does work and counts ops; the tile self-tests
    /// pass; unsupported kernels are never called. Tiny durations (CI).
    #[test]
    fn every_supported_kernel_runs_and_tiles_match_f64() {
        let cfg = Cfg {
            samples: 5,
            sample: Duration::from_millis(2),
            warmup: Duration::from_millis(1),
        };
        let cpu = online_cpus()[0];
        let ks: &'static [KDef] = Box::leak(kernels().into_boxed_slice());
        let mut ran = 0;
        for k in ks {
            if !k.needs.iter().all(|f| feature(f)) {
                eprintln!("{}: unsupported here", k.name);
                continue;
            }
            let rs = measure(k, &[cpu], cfg).expect("measure");
            assert_eq!(rs[0].samples.len(), 5);
            assert!(
                rs[0].samples.iter().all(|s| s.ops > 0.0 && s.secs > 0.0),
                "{}",
                k.name
            );
            ran += 1;
        }
        assert!(ran >= 10, "only {ran} kernels ran");
        for (kind, need, tol) in [
            (0, "fp16", 3e-2),
            (1, "fp16", 5e-3),
            (2, "fp16", 1e-2),
            (3, "fp16", 1e-2),
            (4, "bf16", 1e-4),
        ] {
            if feature(need) {
                // SAFETY: feature detected.
                let e = unsafe { ffi::oxe_isa_tile_selftest(kind, TILE_K) };
                assert!((0.0..=tol).contains(&e), "tile {kind}: rel err {e}");
            }
        }
        // unknown kinds are refused, not run
        // SAFETY: refused before any instruction.
        assert_eq!(unsafe { ffi::oxe_isa_loop(99, 1) }, -1);
    }

    #[test]
    fn args_defaults_and_bounds() {
        let o = parse_args(vec![
            "--cores".into(),
            "0".into(),
            "--single".into(),
            "none".into(),
        ]);
        assert_eq!(o.cores, vec![0]);
        assert!(o.single.is_empty() && o.cfg.samples == 5 && o.max_secs <= 300);
        let o = parse_args(vec!["--max-secs".into(), "9999".into(), "--quick".into()]);
        assert_eq!(o.max_secs, 300);
        assert_eq!(o.cfg.sample, Duration::from_millis(10));
    }
}
