# Portability probe, step 2: x86_64 (issue #293)

Date 2026-11-15. No product code changed, no weights downloaded, no drivers installed.
Scope limit: nothing here was measured against an external llama.cpp server; the serving-layer
benchmark from the issue (prefix reuse, session restore, TTFT vs llama.cpp) is **not done** and is listed as a follow-up.

## 1. Machine
- CPU: Intel Core i9-9880H (Coffee Lake), 8 cores / 16 threads, 2.3 GHz base. SIMD: AVX, AVX2, FMA. **No AVX-512, no AVX-VNNI/AMX, no native bf16/fp16 arithmetic.**
- Memory: 16 GiB (about 4.7 GiB available at the time).
- GPU: Intel UHD 630 (iGPU) and AMD Radeon Pro 5300M/5500M (Navi 14, 4 GiB class). `nvidia-smi` and `vulkaninfo` are not installed, so no compute stack was confirmed. The AMD part is RDNA1: Vulkan works with Mesa RADV, ROCm does not support it officially. This is a typical (modest) desktop-class case, not a proof for others.

## 2. What is architecture-specific
Everything that computes is AArch64 (NEON, dotprod, i8mm, fp16, bf16). Build flags are `-march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme` and `.cargo/config.toml` sets `target-cpu=oryon-1` for aarch64.

| Area | Files (approx. lines) | Portable fallback? |
|---|---|---|
| `kernels-sys` hand-written ops | `csrc/oxe_ops.cpp` 1420, `oxe_pfg.cpp` 545, `oxe_pfa2x.cpp` 323, `oxe_isa_roof.cpp` 491, `oxe_pf_tiles.inc` 867, `oxe_vl_q5k.inc` 154 | No. `#include <arm_neon.h>`; `vendor/ggml-subset/compat.h` has `#error` on non-AArch64 |
| Vendored ggml subset (llama.cpp CPU backend) | `vendor/ggml-subset/generated/repack_subset.cpp` 3954, `quants_subset.c` 372, upstream `vec.cpp` 713, `vec.h` 1709, `simd-mappings.h` 1379 | Upstream ggml has AVX2/AVX-512 paths in `simd-mappings.h`/`vec.h`, but the extracted subset (repack, quants) is the ARM `arm_quants` path only. Reusing upstream's x86 files is possible, not turnkey |
| Flash vendored kernels | `vendor/ggml-flash/generated/arm_quants_subset.c` 1704, `expert-r4.c` 487, `oxf_kernels.c` | No (ARM quant dot products, R4 repack) |
| `int8-gemm` | `csrc/i8g_kernels.c` 320, `i8g_fattn.c` 275 | No (NEON/i8mm) |
| `qwen4exp` (Flash graph) | `csrc/flash_graph.cpp` 2732, `flash_fuse_*.inc` ~2250, `pf_attn.cpp` 523, `pf_kernels.c` 237, `pf_expert*.c` ~370, `flash_mtp.inc` 605 | No |
| `oxn` tiles/NPU | `src/tiles.rs` 1328, `oxn-cpu-shared-bw.rs` | Snapdragon NPU/shared-memory path; not applicable |
| `model` crate (Rust) | `src/engine.rs`, `load.rs` | Partial. `engine.rs:72-91` feature detection and `:201-225` `cntvct_el0`/`cntfrq_el0` timer have `cfg(not(aarch64))` fallbacks (monotonic clock). `load.rs:24` **refuses to load a model** unless aarch64 |
| `kernels-sys` build/FFI | `build.rs` returns early on non-aarch64; `src/lib.rs` `ffi_decls!` makes every entry point a refusing stub | Yes, by design (O-174) |

**Thread placement and memory tuning** assume the Snapdragon X Elite (Oryon, 12 homogeneous cores, 48 GiB-class unified memory, NPU):
- `crates/sched/src/lib.rs` (about 400 lines of code plus tests): `sched_setaffinity` pinning, `parse_cpu_list`. The mechanism is generic Linux and works here; the defaults are not (`--oxe-cpus 0-7`, `--flash-threads 4`, `--flash-threads-batch 6`, `--flash-io-cpu*` in `server/src/main.rs`). No SMT-sibling or hybrid-core awareness (the i9 has 2 threads per core).
- `model/src/engine.rs`: L1 set-aliasing padding (around line 505), `MADV_NOHUGEPAGE` (line 835), phase thread splits (`OXE_PF_ATTN_THREADS`, THREADS-BY-PHASE ~1633). Tuned to Oryon cache geometry.
- Flash cache tiers 20/18/16/14 GiB (`--flash-cache-gib auto`) assume a 32-64 GiB machine and refuse (exit 74) below that; 16 GiB here would refuse. PSI admission and the desktop floor are generic Linux.

**What the stub replaces:** `server/src/engine.rs` `StubEngine` (line 209, `impl Engine` at 302) in a 483-line file: tokenizer plus chat template and scripted tokens with optional per-token delay; no weights, no real KV state. Selected with `--engine stub` (default).

## 3. `cargo check` of the "real" engine on x86_64
- **Passes:** `cargo check -p server --all-targets` (about 25 s warm; re-run on this head). `kernels-sys`, `model` and `server` pass only because `kernels-sys/build.rs` compiles nothing off-aarch64, so this is not a real build.
- **Fails (link/build category: C compiler flags):** `cargo check --workspace` and `cargo check --workspace --all-targets` both stop at the `int8-gemm` and `qwen4exp` build scripts (`--all-targets` does not exclude them, and cargo stops at the first failure unless `--keep-going` is given). They have no architecture guard and pass `-march=armv8.6-a+...` to host GCC: `cc1: error: bad value 'armv8.6-a+dotprod+i8mm+fp16+nosve+nosme' for '-march=' switch` (qwen4exp: `...+bf16+nosve+nosme`). `-p server` does not depend on them. This is a hygiene gap compared with `kernels-sys`, not fixed here.
- Forcing the sources through host g++/gcc with `-fsyntax-only` (to see what would come next, not via cargo):
  - Missing intrinsics header: `fatal error: arm_neon.h: No such file or directory` in `oxe_ops.cpp:516`, `pf_kernels.c:30`, `i8g_kernels.c:11`.
  - cfg gate: `vendor/ggml-subset/compat.h:10: #error "kernels-sys targets AArch64 with NEON (Snapdragon X Elite)"`.
  - Link errors were not reached (no object compiles). Expect unresolved `ggml_vec_dot_*`/repack symbols and the Rust `model` engine calling every `ffi_decls!` entry.
- Cross-check: Rust target `aarch64-unknown-linux-gnu` was not tried (no cross toolchain here).

## 4. What is portable and runs here today
- **Server layer with stub engine:** `cargo test -p server`: 218 passed, 0 failed, 38 ignored (plus 16 integration tests), **5 min 6 s wall** on this machine, of which test run time about 17 s; the rest is compilation (`jobs = 1` in `.cargo/config.toml`). Covers HTTP/OpenAI API (`api.rs` 697, `http.rs` 407, `service.rs` 2314), generation loop (`generate.rs` 3307), cache/prefix logic (`cache.rs` 821, `prefix_store.rs` 884), state store/session restore (`state_store.rs` 2033), echo cache (`echo.rs` 1424). About 12k lines, all architecture-neutral (Rust, libc).
- State cache and prefix reuse logic are exercised through the stub (snapshot = token prefix; `StubSnapshot`). Their **correctness against real KV tensors is not exercised** on x86: the stub carries no tensors, so snapshot size, restore cost and speed are not representative.
- **Controller (`oxc`, Python) in front of an external OpenAI-compatible server:** expected to work since it speaks OpenAI v1 over loopback, but I did not run it or an external server in this probe. Follow-up.
- Sandbox/permission tests of oxa are unrelated to the engine.

## 5. Options

### (a) Kernel interface plus generic backend delegating to an existing library
Define a narrow trait/C ABI for what `model` calls (quantized matvec/matmul per block type, q8 quantize, RMS norm, rope, softmax, attention score/PV, f16 vec dot) and put the current NEON code behind it as the `aarch64-tuned` backend; add a `generic` backend.
- What Flash already links: **vendored llama.cpp/ggml CPU backend** (`vendor/ggml-subset` and `vendor/ggml-flash`, pinned commit `8009864`, `vec.cpp` verbatim, GCC required). Upstream ggml already contains AVX2/AVX-512/AMX and scalar paths for the same quant types, so the generic backend can compile upstream ggml-cpu for the host (not the ARM-only extracts) and call `ggml_vec_dot_*`, `ggml_gemv/gemm`.
- Catch: numerics. The repo claims bit-exactness against llama.cpp with GCC FMA contraction; generic backend must be tested for tolerance, not identity. The Snapdragon gate (no slowdown of the tuned target) must run.
- Estimate (one engineer): interface extraction and aarch64 backend behind it with identical benchmark results 2-3 weeks; generic ggml backend for the dense path (`oxe` engine, i.e. `kernels-sys`+`int8-gemm`) 2-3 weeks; Flash/qwen4exp fused graph (about 8k lines of NEON glue) is a separate 4-8 weeks or stays aarch64-only. Resulting x86 speed: roughly llama.cpp CPU speed, with no advantage. Gain is mainly "runs everywhere" plus our serving layer.

### (b) GPU backend for the common desktop case
Per the owner note, this is where an x86 user benefits. Options: Vulkan (works on AMD, Intel, NVIDIA, no vendor stack beyond Mesa/driver) or CUDA (NVIDIA only, fastest). Cheapest route: do not write kernels; reuse ggml's Vulkan/CUDA backends through the same interface as (a), keeping our server/cache/state on top, or front an external llama.cpp server (zero kernel work).
- Needs (a)'s interface first, plus a device-resident KV/state design: our state store snapshots and prefix cache assume host-memory KV, so GPU KV save/restore needs a copy path.
- Estimate: via ggml Vulkan/CUDA behind the interface: 4-6 weeks after (a); own hand-written GPU kernels: 3+ months and a hardware matrix we do not have. On this machine specifically: a 4 GiB RDNA1 GPU limits it to small models and Vulkan only; needs real hardware validation (not done).
- Cheapest first experiment: run the oxe server layer and oxc in front of llama.cpp (Vulkan build) and measure the serving-layer value alone, as the issue step 2 describes. That needs an engine adapter (server `Engine` trait implemented as an HTTP client), roughly 1-2 weeks.

### Answer to "where is the win?"
- x86_64: GPU backend or the serving layer in front of an existing backend; CPU kernels will not beat llama.cpp.
- Second ARM: CPU kernels (existing code, needs a measurement on a second ARM core).
- Apple Silicon/Asahi: GPU/accelerator access (see issue note); out of scope for this machine.

## Hygiene findings (no fix made)
- `int8-gemm/build.rs` and `qwen4exp/build.rs` should return early on non-aarch64 like `kernels-sys/build.rs`, so `cargo check --workspace` works on x86_64.
