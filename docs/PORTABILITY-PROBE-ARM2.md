# Portability probe, step 1: second ARM machine (Apple M2, Asahi-style Linux)

Issue #293. No product code was changed. Build output stayed in `engine/target` (`CARGO_TARGET_DIR=$PWD/target`).

## 1. The CPU
- `lscpu`: aarch64, Apple M2 (big.LITTLE): 4 x Blizzard-M2 (cpu0-3, max 2424 MHz) + 4 x Avalanche-M2 (cpu4-7, max 3504 MHz), 1 thread/core, 8 CPUs. L2 20 MiB (2 instances). Kernel 7.2.11 Arch.
- Memory: 15 GiB (MemTotal), about 10 GiB available during the run. Rust 1.99.1, GCC 16.1.1 (README asks for GCC 16).
- Flags present: asimd, asimdhp/fphp (fp16), asimddp (dotprod), asimdfhm (fhm), **i8mm, bf16**, asimdrdm, sha3, sha512, aes, atomics, lrcpc, fcma, jscvt, frint, bti, paca/pacg, ecv.
- Flags absent: **sve, sve2, sme** (the engine builds with `+nosve+nosme`, so it needs none). No hardware RNG flag (`rng`) visible.
- `rustc --print target-cpus` knows `apple-m4/m5` but only reports `native = apple-m2`; there is no `apple-m2` name in this rustc.

## 2. Build of the real engine (not stub)
Command (as in engine/README "Build from source"): `cd engine; CARGO_BUILD_JOBS=4 cargo build --release -p server -p oxe-cli` -> finished in 44 s (no Flash build: needs the ggml libs from the lab repo; not tried).
- It built **unchanged**, with the repo's `.cargo/config.toml` `target-cpu=oryon-1` (not the CPU of this machine). `kernels-sys/build.rs` uses `-march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme`, which this core implements, so nothing refused to build.
- Runtime guard: `crates/server/src/startup_check.rs` (`REQUIRED_CPU_FEATURES`, `cpu_has`) checks i8mm, bf16, dotprod, fp16 at start and exits with a message when one is missing. `crates/model/src/engine.rs:78-83` reports dotprod/i8mm/fp16/bf16/sve. All four are present here, so the guard passes. Fallback when a feature is missing: none; the build is compile-time ARMv8.6 (the README says other cores stop with "Illegal instruction"; the guard now turns that into a message). Not testable here: this CPU has all four.
- Risk: `target-cpu=oryon-1` lets rustc emit instructions of Oryon's feature set (e.g. `rng`, `crc`; Oryon is ARMv8.7-ish). Apple M2 lacks some Oryon-only features, and nothing hit one in the tests below, but this is luck of codegen and not guaranteed. `qwen4exp/build.rs:75` (Flash) hardcodes `-mcpu=oryon-1+rng+crc+nopauth+...`; the `rng` part is not in this CPU's flags, so the Flash build on this machine is a likely failure or SIGILL; it was not built.
- `oxe-server --engine oxe --model /nonexistent.gguf` starts and stops with `tokenizer: No such file or directory`, as expected without a model.

## 3. Model-free tests and probe
`cd engine; cargo test --release -p kernels-sys -p model -p sched -p tokenizer -p chat -p server` (8 min 19 s, RUST_TEST_THREADS=1 from config):
- chat 29, kernels-sys 25 (6 ignored), model 92 (2 ignored) plus fattn 4, pfattn_kernel 10, pfg_kernel 7, sched 19, tokenizer 6, server 233 (41 ignored, they need the Ornith GGUF) plus startup_check 11, flash_template 5. **0 failed.** The long-context and prefill-attention accuracy tests (vs f64) also pass; their comments already name this Apple M2 as a second target ("lab-arm64 helper").

ISA roofline probe: `./target/release/isa-roof --quick --out target/isa-roof.json` (5.9 s; receipt says "NOT Oryon numbers; functional test only"; all 5 GEMM tile self-tests ok). Per core, cpu4 = Avalanche (P core, ~3.2 GHz, all-8-cores figure in parentheses):

| kernel | cpu0 (E) | cpu4 (P) | all 8 |
|---|---|---|---|
| f32 FMLA.4s GFMA/s | 19.2 | 50.7 | 256 |
| f16 FMLA.8h GFMA/s | 35.9 | 101.2 | 511 |
| f16 FMLA.8h by element | 38.3 | 101.3 | 513 |
| bf16 BFMMLA GFMA/s | 9.6 | 25.3 | 128 |
| int8 SDOT GMAC/s | 76.3 | 202.3 | 1019 |
| int8 SMMLA GMAC/s | 76.5 | 172.3 | 928 |
| tile f16 8x16 elem nofold L1 | 23.7 | 65.8 | 328 |
| tile f16 8x16 LD1R fold32 L1 | 22.5 | 62.6 | 310 |
| tile bf16 8x8 BFMMLA L1 | 9.4 | 24.6 | 125 |
| today fa2 unit L2 | 17.1 | 42.8 | 219 |
| today fa2x unit L2 | 16.1 | 41.8 | 211 |

PMU cycle counts work: FMLA.8h = 4.0 inst/cycle, BFMMLA 0.5 inst/cycle (the probe's "path A: pfg-f16" hint: f16 FMLA, not BFMMLA, is the fast path here). The probe's Oryon verdict table does not apply. For the Snapdragon target the code comments quote 0.770 TFMA/s for the L1 f16 tile on 12 Oryon cores; this machine's all-core 0.33 TFMA/s on 8 mixed cores is about 0.43 of that. Note: "all 8" includes 4 slow E cores, started together, and the 5 x 10 ms `--quick` samples are short (no sustained/thermal data; fanless MacBook Air). Memory bandwidth was not measured (no model-free bandwidth probe was run; `barrier-bench` was not run either).

## 4. Model
No weights fetched. The supported models are 35B MoE GGUFs (Ornith 17.4 GB, Qwen3.6 22 GB; `oxc/dependencies/model-artifacts.json`); nothing in that file is under 2 GB except a 10.9 MB item. The 15 GiB RAM cannot hold Ornith 17.4 GB resident (mmap would stream from SSD; only the Flash engine is built for that). No sub-2 GB model that the engine supports is named in the docs. Next step: either use the Flash engine's SSD streaming (needs the Flash build, see above) or have the maintainer name a small qwen35moe-architecture GGUF for a smoke test. Model paths in the tests expect `~/.local/share/ai-stack/models/llm`.

## 5. Comparison with the Snapdragon target
- **Ran unchanged:** full `oxe-server`/`oxe`/`isa-roof` release build with the repo's own flags; all model-free tests; startup CPU-feature guard; isa-roof with PMU.
- **Had to be switched off:** nothing, for the non-Flash build. Not tried: Flash build (needs external ggml libs; hardcoded `-mcpu=oryon-1+rng...`), NPU/oxn paths (Qualcomm-only), real-model runs.
- **Slower or missing:** about 0.4x of the Oryon L1-tile roof; heterogeneous cores (4 E + 4 P) where the engine's static plan with thread barriers (170-210 per token) and per-core placement assumes 12 equal cores, so E cores will likely stall every barrier; default `--oxe-cpus` should be P cores 4-7 only. 15 GiB vs the laptop's memory. No SVE/SME (not used).
- **Estimate to make it a supported target (about 1-2 weeks, mostly with a model present):** 1 day, make `target-cpu` per-machine (`native` or an `apple-m2` flag set) instead of oryon-1, in `.cargo/config.toml` and `qwen4exp/build.rs`, plus a CI/run check for SIGILL; 2-3 days, thread placement and plan for 4P+4E (P-core-only default, tune barriers/partition); 1-2 days, a small-model smoke/exactness test (needs a small GGUF); 2-4 days, a bandwidth probe, tuning tile choice (fp16 by-element path wins here), and receipts; Flash on this machine, +2-3 days if wanted. Numerics: the byte-exact gates were defined on the Snapdragon build and need re-baselining.

## 6. Where is the win? (issue comment)
On this CPU, the existing kernels already run at their hardware roof (f16 FMLA 4/cycle, SDOT/SMMLA near peak), so hand tuning is not the open item. The open item is the GPU/accelerator, which this probe cannot measure: the Apple GPU is not reachable through the CPU kernels, and the omarchy-mlx approach was not evaluated here. On this machine the likely wins are the serving layer (oxc: API, cache/echo, cloud routing) over an existing backend, then a GPU backend for decode; CPU kernels need only retuning for placement. x86 GPU was out of scope for this probe.
