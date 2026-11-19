# LC-8 V-region THP-off: qualification plan (Qwen first)

2026-11-17. **Plan only.** This note proposes the measurement that decides whether the V^T region
of the state arena should stay THP-advised. No engine was run, no receipt was produced, and no
profile, default, flag, threshold or admission bound is changed here. Source read at
`69a81515ad3ce39dbf048112fd0cfad476f4b73f` (branch head). Task: issue #882 (parent #864, owner
O-383), label `after-0.2.1`, milestone v0.2.2.

The one code change in this PR makes a receipt name the arm it was measured with (section 1, last
bullet). Everything else is the plan, its acceptance fields and its decision rules.

## 1. What is already in the tree (read, not measured)

| Fact | Where |
|---|---|
| `OXE_KV_V_THP=0` (or `set_kv_v_thp(false)`) advises **only the V^T region** of the state arena `MADV_NOHUGEPAGE`; default stays THP-advised. Advice only: no layout, result or state change | [engine.rs](../../crates/model/src/engine.rs#L806) `kv_v_thp` / `set_kv_v_thp`, [advise_no_thp](../../crates/model/src/engine.rs#L827), call site [engine.rs:1264](../../crates/model/src/engine.rs#L1264) |
| The switch is read once, when the engine is created, so one process = one arm | same call site |
| The exactness control exists: the off arm must show VmFlags `nh` and no `hg` on the V region, and both arms must be bit-identical in logits and dumped state | [lc8_v_region_thp_switch](../../crates/model/src/tests/longctx/pfattn.rs#L423) |
| The measurement route exists and already writes a JSON receipt: engine commit, effective switch line, model path, cpus/threads, `n_ctx`, ms/token (median/mean/min/p90), tok/s, barriers/token, generated ids **and their hash**, load seconds/backing/cache hit, `memory_kb`, kernel cflags | `oxe bench --out FILE` ([oxe.rs](../../crates/oxe-cli/src/bin/oxe.rs#L2476)); wrapper [run_bench.sh](../../tools/run_bench.sh) |
| `memory_kb` already carries VmHWM, VmRSS, RssAnon, RssFile and `smaps.AnonHugePages` / `smaps.Rss` / `smaps.FilePmdMapped` | [memory_report](../../crates/model/src/load.rs#L51) |
| Admission uses **qualified bounds**, not observed RSS: `kv_bytes_per_position`, `scratch_bytes`, `mtp_state_bytes`, `margin_bytes` against context padded to 256; `oxe admission --json` emits them for a model and launch profile ([SERVER.md](SERVER.md), `oxe-cli/src/admission_cmd.rs`) | [admission.rs](../../../oxa/oxc-local/src/admission.rs#L11) |
| `--thp-collapse` is a different knob: it applies `MADV_HUGEPAGE` + `MADV_COLLAPSE` to the **weights** backing, never to the V region | [weights.rs](../../crates/model/src/weights.rs#L817), [M2.md](M2.md) |
| The prior THP receipt in the tree is also a different arm: `flash-next-thp.json` disables THP for a **whole process** on the Flash engine, records `production_admitted: false`, and notes that whole-process disable also removes useful dense THP | [flash-next-thp.json](../../../oxc/benchmarks/results/flash-next-thp.json) |
| This PR: the effective switch line now names the V-region arm (`kv_v_thp=0|1`) | [engine.rs:796](../../crates/model/src/engine.rs#L796) |

The line matters because the switch is process-global and can be set by a caller: the `env=[...]`
tail only records it when it arrived through the environment, and a bench row or a server start-up
log that does not name the arm cannot be compared with a row from the other arm.

The 1.24 GiB / 1.41 GiB RSS figures quoted in #864 are reported history. The cited
`row5T-final.json` is **not present in any tracked benchmark path in this checkout**, so those
numbers are not evidence and are not repeated as a result here.

## 2. Missing before any run (owner handoff) — NEED-METADATA

1. The sanitized M3 5T receipt and the **immutable identity** of the run it came from: artifact
   SHA and engine commit, model file SHA, kernel version, `/sys/kernel/mm/transparent_hugepage/enabled`,
   the V region's VmFlags and `smaps.AnonHugePages` in both arms.
2. The exact arm definitions used there: context, cpu lists, ubatch, MTP policy, caps, warmness
   (warm cache? weights `file` or `anon`?), and the number of repeats each number is a median of.
3. Explicit owner release for the run itself, and the lease/cpu window on the native runner.

Without 1-3 this note is a protocol, not a result.

## 3. Protocol (bounded, Qwen first)

Only `OXE_KV_V_THP` differs between arms. The binary, model SHA, weights backing, `n_ctx`, cpu
lists, ubatch, MTP policy and caps stay identical; every arm is a fresh process and keeps its
`oxe bench --out` row plus its effective switch line.

| Step | Runs | What it establishes |
|---|---|---|
| 0. A/A startup | 2 fresh processes, both default (THP on), warm weights, same warm-up | the **noise floor** of startup wall and peak RSS. If the noise band is at least as large as the effect looked for, stop and report that |
| 1. Qwen A/B | Qwen3.6, default vs `OXE_KV_V_THP=0`, interleaved A-B-B-A, repeats >= 3, fresh process each | the Qwen startup wall and peak-memory delta; decode tok/s as the cost side |
| 2. Exactness | both arms of step 1 on a fixed prompt | `generated_hash` equality, one batched prefill, and the [lc8 test](../../crates/model/src/tests/longctx/pfattn.rs#L423) green on the qualified runner (VmFlags `nh` visible on the off arm). A gain without this is not a qualified gain |
| 3. Ornith A/B | the same pair, decode only | whether the Qwen result transfers. It is expected **not** to be assumed: Ornith is the deployed default and its decode is the latency-sensitive side |

Ordering: startup wall is the primary axis (that is where the V^T residency claim comes from:
with THP a V^T row spans many 2 MiB pages and the first positions of every row touch them all, so
the region is resident at the cap rather than at the context in use — [engine.rs:806](../../crates/model/src/engine.rs#L806)).
Decode throughput is the cost axis. Neither is declared a win on one run.

## 4. Receipt fields (acceptance)

The owner's receipt must be one row per arm, recording every field below; "already" means the
existing route writes it and the plan only asks that it be kept, not re-derived.

| Field | Source |
|---|---|
| Artifact identity: engine commit, binary SHA, GPUs/kernel | already (`engine_commit`, `kernel_cflags`) plus the owner's binary SHA |
| Kernel and THP state: THP mode, V region VmFlags (`hg`/`nh`), `smaps.AnonHugePages` | VmFlags from the lc8 control; `AnonHugePages` already in `memory_kb` |
| Model / context / threads / MTP policy | already (`model`, `n_ctx`, `cpus`, `threads`, `effective`); MTP policy to be stated per arm |
| Warmness: warm-up tokens, weights backing, cache hit | already (`warmup`, `load.backing`, `load.cache_hit`) |
| Startup wall: process spawn to ready | **external observation**; the Rust route does not time readiness (`load.seconds` is weight load only) |
| Memory: VmHWM / VmRSS / RssAnon / smaps peaks | already (`memory_kb`) |
| Token throughput | already (`ms_per_token`, `tok_s`) |
| Output/checksum equivalence | already (`generated_ids`, `generated_hash`) plus state identity from the lc8 control |
| Noise and limits: repeat count, interleaving, what differed | to be stated; one number per arm is not a result |

## 5. Decision rules

- **No default change from this note.** A flip needs B better than the A/A noise band on the
  primary axis and no decode regression beyond the band on either model — and it needs its own PR
  with the receipt attached.
- **No admission change.** `allocation_budget` stays on the qualified bounds; a sampled RSS saving
  is not a bound. Requalifying bounds is a separate task with its own measurements.
- **Qwen first, but never extrapolated to Ornith.** Both models are measured or the note says why
  not.
- **Do not rerun disqualified variants.** MTP depths already qualified/refused in
  [M3-MTP.md](M3-MTP.md) and the legacy per-head attention path in [M3-ATTN.md](M3-ATTN.md) stay
  as they are; this plan touches one madvise only.
- **Exactness first.** A throughput or memory difference with unequal `generated_hash` is
  reported as an exactness problem, not as a gain.

## 6. Recommendation

Measure the V-region switch on Qwen first, with the A/A startup pair as step 0, and take the
result as a hypothesis until it is repeated.

- **Qwen startup uncertainty.** Qwen's process startup is the largest and least stable part of the
  earlier handoff (that route reported a startup wall several times larger for Qwen than for
  Ornith — reported history, [RUST-SERVER-QUALIFICATION.md](../../../oxc/docs/RUST-SERVER-QUALIFICATION.md)),
  and it is exactly the axis the V^T residency argument predicts. That makes the Qwen startup
  number the one most likely to be dominated by noise, which is why the A/A pair gates the rest of
  the plan rather than being a formality.
- **Ornith decode trade-off.** Ornith is the model the earlier handoff recommends as the default
  with MTP off, and its decode is the sensitive side:
  turning THP off for V^T can trade resident-set reduction for TLB misses on the rows now read in
  4 KiB pages. A Qwen startup win alone would not justify the switch for Ornith, and the
  whole-process THP arm already in the tree (`production_admitted: false`) is evidence that a
  broader disable is not a free win.
- No all-axis win is claimed, and no safer admission claim follows from memory numbers alone.

## 7. Limitations of this note

No hardware was used; no engine, benchmark, provider call or owner receipt was involved. The
kernel/THP state of the native runner is unknown here, so "THP on" is only the engine's own
default. `memory_report` is a coarse snapshot (whole process, not the V region), and page cache and
power state are uncontrolled across sequential arms. Where this note quotes a number it is marked
as reported history, and history is not a measurement.
