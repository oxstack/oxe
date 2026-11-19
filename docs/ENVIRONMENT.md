# Environment variables of the engine workspace

The engine reads its environment once, at start (or at engine construction), unless a row says otherwise. Rows of class `experiment` are A/B switches between code paths and are candidates for removal once lab numbers pick a winner.

This page is checked by `crates/gguf/tests/env_documented.rs`: every `OXE_*`, `OXF_*` and `OXI_*` name that non-test source in `crates/*/src` or a `build.rs` reads as a string literal needs a row here, and every row (except class `test`) needs such a reader. Class `test` rows must still appear in some `.rs` file.

Classes: `option` (an owner would set it: paths, ports, key file), `tuning` (performance knob with a safe default; diagnostic switches that only add log output are listed here too), `experiment` (A/B switch between code paths), `test` (only tests, benches or fixtures read it), `build` (read or set by `build.rs`). "Read in" paths are relative to `engine/crates/`. Most server rows are also set by a command line flag of the same meaning: the flag writes the variable before the engine starts, so a flag wins over an inherited variable.

`server/src/startup_check.rs` validates the form of most `OXE_*` values at start and refuses a malformed one.

## Server (`oxe-server`)

| Name | Read in | Values and default | What it does | Class |
|---|---|---|---|---|
| `OXE_API_KEY_FILE` | server/src/main.rs | path; unset or empty: none | API key file used when `--api-key-file` is not given | option |
| `OXE_WEIGHTS` | server/src/oxe_engine.rs | `file` (default) or `anon` | weight backing: file-backed cache mapping, or a THP-advised anonymous arena (`--oxe-weights`) | option |
| `OXE_MLOCK` | server/src/oxe_engine.rs | any value (even `0`) enables; unset: off | lock the weights in RAM, or fail to start (`--mlock`) | option |
| `OXE_STATE_BUDGET_SCOPE` | server/src/state_store.rs | `own` = per-store budget; anything else or unset: global | scope of the state cache byte budget (`--state-budget-scope`) | option |
| `OXE_DEFAULT_MAX_TOKENS` | server/src/service.rs | integer; `0` = unlimited (-1); unset, unparsable or negative: n_ctx/4 clamped to 1..16384 | max_tokens used when a request gives none (`--default-max-tokens`) | option |
| `OXE_SSE_KEEPALIVE_SECS` | server/src/service.rs | seconds as float, `0` = off; unset or unparsable: 15 | SSE comment interval while a request waits in the queue or prefills (`--sse-keepalive`) | option |
| `OXE_STREAM_STALL_SECS` | server/src/service.rs | positive seconds as float; unset or invalid: 30 | stream stall timeout (`--stream-stall-timeout`) | option |
| `OXE_FAIR_QUANTUM_MS` | server/src/fair.rs | 0..=600000 ms; unset or unparsable: 0 (off) | time slice for fair slicing between generations (`--fair-quantum-ms`) | option |
| `OXE_ECHO` | server/src/echo.rs | `0`, `off`, `false`, `no` disable; anything else enables; unset: on | cache-perfect echo of the exact generated token ids (`--echo`) | option |
| `OXE_ECHO_MAX_TOKENS` | server/src/echo.rs | integer; unset or unparsable: 4194304 | echo store size limit in tokens (`--echo-max-tokens`) | option |
| `OXE_ECHO_TTL_SECS` | server/src/echo.rs | seconds; unset or unparsable: 604800 (7 days) | echo record lifetime (`--echo-ttl-secs`) | option |
| `OXE_ECHO_COMPACT_BYTES` | server/src/echo.rs | bytes; unset or unparsable: 4194304 | compaction threshold of the echo store (`--echo-compact-bytes`) | option |
| `OXE_FLASH_PF_PROTECT_HOT` | server/src/flash_engine.rs | `1` enables (after trim); unset or other: off | Flash engine: hot-resident protection in batched prefill ops (`--flash-pf-protect-hot`) | experiment |
| `OXE_MTP_GUARD` | server/src/mtp_guard.rs | `off`, `0`, `RATIO` or `RATIO:WINDOW:COOLDOWN`; defaults 256 and 512; unset: no guard; read at each call | turns speculative drafting off while the acceptance ratio is below RATIO (`--mtp-guard`) | tuning |
| `OXE_MTP_FROM` | server/src/oxe_engine.rs | path to a GGUF; unset: sidecar head or the model itself | load the MTP block from another GGUF of the same shape | experiment |
| `OXE_MTP_DRAFT_VOCAB` | server/src/oxe_engine.rs | path to a ranking file; unset: the sidecar's stored vocabulary | restricted-vocabulary draft head from this ranking | experiment |
| `OXE_MTP_DRAFT_K` | server/src/oxe_engine.rs | integer; unset or unparsable: none | row count passed with `OXE_MTP_DRAFT_VOCAB` to the draft-vocabulary loader | experiment |
| `OXE_IDLE_SPIN_US` | server/src/oxe_engine.rs | microseconds; unset or unparsable: 200 | how long idle decode workers spin before sleeping | tuning |
| `OXE_SMALL_PREFILL` | server/src/oxe_engine.rs | integer N; unset or unparsable: 0 (off) | prompt segments of at most N tokens run on the decode team instead of the prefill team | tuning |
| `OXE_PREFILL_QUIET` | server/src/oxe_engine.rs | `1` enables; unset or other: off | silences the prefill progress output | tuning |
| `OXE_LOOKUP_LANES` | server/src/service.rs | integer; unset or unparsable: off | a value above 1 makes `/props.speculative` report `lookup` when no MTP depth is active | tuning |
| `OXE_SERVER_LEGACY` | server/src/generate.rs | any value enables; unset: off | A/B diagnostic: the pre-fast-path decode loop with a full re-parse of the reply | experiment |
| `OXE_TIMING_DIAG` | server/src/generate.rs | any value enables; unset: off | per-request timing split on stderr | tuning |
| `OXE_TIMING_DIAG_WARMUP` | server/src/generate.rs | integer; unset or unparsable: 16 | warm-up requests skipped by `OXE_TIMING_DIAG` | tuning |
| `OXE_MODEL_SHA256` | server/src/prefix_store.rs, model/src/weights.rs | 64 hex digits; unset: computed with `sha256sum` and cached | model identity for the state and prefix caches; a malformed value is an error in `weights.rs` and ignored by `prefix_store.rs` | option |

## Model engine (`oxe` path: `model`, `server`, `oxe-cli`)

Switches that select an arithmetic path are part of the state cache identity (`state_store.rs` `ARITH_KNOBS`, `OxeEngine::arith_id`). `OXE_ATTN_OUT` is deliberately outside it (bit-identical kernels).

| Name | Read in | Values and default | What it does | Class |
|---|---|---|---|---|
| `OXE_KV_TYPE` | model/src/engine/kv8.rs | `f16` (default), `k8`, `v8`, `kv8`; `--cache-type-k/-v q8_0` sets it | KV cache element type; a bad value is an error | experiment |
| `OXE_KV_VPAD` | model/src/engine.rs | integer, at most 4096; unset or unparsable: 32 | padding of V^T rows off the n_ctx stride | experiment |
| `OXE_KV_V_THP` | model/src/engine.rs | `0` off; anything else or unset: on | transparent huge page advice on the V cache; no layout or result change | tuning |
| `OXE_KV8_FUSED` | model/src/engine/kv8.rs | `0` = off; anything else or unset: on | int8 KV: fused reads, or the first-version dequantise-to-scratch path | experiment |
| `OXE_KV8_PV` | model/src/engine/kv8.rs | `fast` (default), `ref`, `i16`, `i8`; unknown: `fast` with a warning | kernel of the fused V . p for int8 V; `i16` and `i8` are lossy | experiment |
| `OXE_ATTN` | model/src/engine.rs | `fused` enables fused decode attention; `legacy` selects the per-head path; unset: tiled | decode attention path | experiment |
| `OXE_ATTN_OUT` | model/src/engine.rs | `legacy` (default), `pb`, `pb<N>` (N multiple of 32 in 32..=4096); unparsable: `legacy` with a warning | V^T . p kernel of the tiled decode and verify attention (`--attn-out`) | experiment |
| `OXE_ATTN_TILED_MIN_KV` | model/src/engine.rs | integer; unset or unparsable: 4096 | decode cutoff (padded n_kv) from which the tiled attention is used | experiment |
| `OXE_ATTN_FUSED_MIN_KV` | model/src/engine.rs | integer; unset or unparsable: 4096 | multi-lane cutoff of the fused attention | experiment |
| `OXE_ATTN_FUSED_1LANE_MIN_KV` | model/src/engine.rs | integer; unset or unparsable: 16384 | single-lane cutoff of the fused attention | experiment |
| `OXE_ATTN_FUSED_CH` | model/src/engine/fattn.rs | integer, clamped to 128..=4096 and rounded down to a multiple of 32; default 256 | positions per fused attention work item | experiment |
| `OXE_ATTN_STATS` | model/src/engine.rs | any value enables; unset: off | counts 32-position probability blocks of the tiled decode attention | tuning |
| `OXE_DEC_ATTN_PAD` | model/src/engine.rs | `0` = off; anything else or unset: on | column padding of the decode attention scratch | experiment |
| `OXE_DEC_FUSED_DN` | model/src/engine.rs | `0` = off; anything else or unset: on | fused 4-row delta-net step in decode and verify | experiment |
| `OXE_VL` | model/src/engine.rs | decimal bit mask; unset or unparsable: 0 | verify-lane GEMV variants (bits in `engine.rs`, `VL_*`) | experiment |
| `OXE_PF_TILES` | model/src/engine/prefill.rs | decimal bit mask; unset or unparsable: 1021 | kernel mask of the MoE prefill; `0` = reference kernels | experiment |
| `OXE_PF_GROUP` | model/src/engine/prefill.rs | 0..=16; `0`, unset or unparsable: by size | llama.cpp ubatches run through each layer together | experiment |
| `OXE_PF_GROUP_T1` | model/src/engine/prefill.rs | integer; unset or unparsable: 100 | by-size grouping: from this many new tokens group 2 (`--pf-group-t1`) | experiment |
| `OXE_PF_GROUP_T2` | model/src/engine/prefill.rs | integer; unset or unparsable: 0 (never) | by-size grouping: from this many new tokens group 4 (`--pf-group-t2`) | experiment |
| `OXE_PF_GROUP_UNALIGNED` | model/src/engine/prefill.rs | `0` = off; any other value on; unset: off | group prefill calls that start off a multiple of 256 too | experiment |
| `OXE_PF_MMID_MIN` | model/src/engine/prefill.rs | integer; unset or unparsable: 0 (built-in rule) | forces the pair tiles for every expert with at least this many rows | experiment |
| `OXE_PF_ATTN_PB` | model/src/engine/prefill.rs | integer, rounded down to a multiple of 32, at least 32; default 512 | positions per block of the position-blocked prefill attention | experiment |
| `OXE_PF_ATTN` | model/src/engine/pfattn.rs | `auto` (default), `tiled`, `fa2`, `fa2x`, `pfg`; unparsable: `auto` with a warning (`--pf-attn`) | prefill attention kernel | experiment |
| `OXE_PF_ATTN_AUTO_MIN_KV` | model/src/engine/pfattn.rs | positions or `legacy`; empty or `default`: engine constant `PFA_AUTO_MIN_KV` (`--pf-attn-auto-min-kv`) | cutoff from which `auto` leaves the tiled kernel | experiment |
| `OXE_PF_ATTN_AUTO_KERNEL` | model/src/engine/pfattn.rs | `fa2` (default), `fa2x`, `pfg` (`--pf-attn-auto-kernel`) | kernel `auto` runs beyond its cutoff | experiment |
| `OXE_PF_ATTN_THREADS` | model/src/engine/pfattn.rs | integer; unset, unparsable or 0: all prefill threads | prefill threads that take attention work units; same bits at any value | tuning |
| `OXE_PF_FA2X_FOLD` | model/src/engine/pfattn.rs | 8, 16 or 32; default 8 | f16 chain length of the fa2x kernel; read once, an invalid pair keeps the defaults | experiment |
| `OXE_PF_FA2X_CH` | model/src/engine/pfattn.rs | multiple of 32 up to 512; default 128 | chunk positions of the fa2x kernel; read once | experiment |
| `OXE_PF_PFG_FOLD` | model/src/engine/pfattn.rs | `16`, `32`, `16:16`, `16:32`, `32:16`, `32:32`; default 16:32 (`--pf-pfg-fold`) | pfg kernel f16 chain lengths (QK:PV); read once | experiment |
| `OXE_PF_PFG_CH` | model/src/engine/pfattn.rs | 32..=256, multiple of 32; default 128 (`--pf-pfg-ch`) | pfg kernel chunk; read once | experiment |
| `OXE_PF_PFG_TB` | model/src/engine/pfattn.rs | 8, 16 or 32; default 16 (`--pf-pfg-tb`) | pfg kernel unit size; read once | experiment |
| `OXE_PF_PFG_MK` | model/src/engine/pfattn.rs | `ld1r` (default) or `elem` (`--pf-pfg-mk`) | pfg micro-kernel form; read once | experiment |
| `OXE_MTP_TRACE` | model/src/mtp.rs | any value enables; unset: off; read once | one stderr line per speculative step | tuning |
| `OXE_MTP_LOG` | model/src/mtp.rs | any value enables; unset: off | one stderr line per MTP prefill seeding | tuning |

## Flash engine (`qwen4exp`)

| Name | Read in | Values and default | What it does | Class |
|---|---|---|---|---|
| `OXF_IO` | qwen4exp/src/io.rs | `IoTuning` spec; `default` = `IoTuning::default()`; unset: `spins=1000000` (Flash) | IO thread tuning of the expert streamer; a malformed value is an error | tuning |
| `OXF_EXPERT_TRACE` | qwen4exp/src/bin/flash-m0.rs | path to an expert-id CSV; unset: the cached fixed353 trace | trace file of the `flash-m0` tool (`--trace` overrides) | option |
| `FLASH_ARENA_ALIGN2M` | qwen4exp/src/uring.rs | `1` arms; unset or other: off | aligns the IO arena to 2 MiB (A/B arm); not an `OX*` name, so not scanned | experiment |

## Image engine (`image-dit`, `image-flow`)

| Name | Read in | Values and default | What it does | Class |
|---|---|---|---|---|
| `OXI_IDLE_SPIN_US` | image-flow/src/bin/oxi.rs | microseconds; unset or unparsable: 200 | idle spin of the `oxi` thread pool before sleeping | tuning |
| `OXI_NPU_STATS` | image-flow/src/bin/oxi.rs, image-dit/src/fast_model.rs | `1` enables | accelerator counts non-finite values and the largest absolute value | tuning |
| `OXI_NPU_AUDIT` | image-flow/src/bin/oxi.rs | `1` enables (with `--npu`) | accelerator keeps the accuracy audit | tuning |
| `OXI_PERTURB_ULP` | image-flow/src/bin/oxi.rs | integer N; unset or unparsable: off | moves the token embeddings by up to N ulps (self-noise control) | experiment |
| `OXI_FIXTURE_THREADS` | image-dit/src/prod_model_fixture.rs | integer; default 12 | thread count of the production-shape fixtures (ignored tests) | test |
| `OXI_DIT_FIXTURE_4608` | image-dit/src/prod_fixture.rs | set (any value) | adds the 4608-token case to the block fixture | test |
| `OXI_DIT_TOKENS` | image-dit/src/prod_model_fixture.rs | comma-separated counts; default `1536,4608` | token counts of the model-linears fixture | test |
| `OXI_DIT_DOUBLE` | image-dit/src/prod_model_fixture.rs | integer; default 1 | double blocks of the fixture model | test |
| `OXI_DIT_SINGLE` | image-dit/src/prod_model_fixture.rs | integer; default 1 | single blocks of the fixture model | test |
| `OXI_DIT_REPS` | image-dit/src/prod_model_fixture.rs | integer; default 4 | repetitions of the fixture | test |
| `OXI_DIT_CHUNK` | image-dit/src/prod_model_fixture.rs | integer; default 256 | chunk size of the fixture config | test |
| `OXI_DIT_TUNE` | image-dit/src/prod_model_fixture.rs | `;`-separated `unit_batch,plain_batch,plain_rows`; unset: one default variant | linear tuning variants compared by the fixture | test |

## Other names read outside tests

Not `OX*` names, so the scan test does not check these rows.

| Name | Read in | Values and default | What it does | Class |
|---|---|---|---|---|
| `XDG_CACHE_HOME` | model/src/weights.rs, image-flow/src/cond_cache.rs | directory; unset: `$HOME/.cache` | base of the weight and conditioning caches | option |
| `HOME` | several crates | directory | home directory for default cache and model paths | option |

## Build scripts

Names a `build.rs` reads, or sets with `cargo:rustc-env` for the crate to read with `env!` or `option_env!`.

| Name | Read in | Values and default | What it does | Class |
|---|---|---|---|---|
| `OXF_GGML_PROD` | qwen4exp/build.rs | directory; default `$HOME/.cache/local-ai/snapdragon-engine/ggml-prod` | location of the production ggml archives (build panics if `lib/libggml-cpu.a` is missing) | build |
| `OXE_GIT_COMMIT` | oxe-cli/build.rs, qwen4exp/build.rs | set by those scripts: short commit, `-dirty` suffix | engine commit recorded in reports and receipts | build |
| `OXE_KERNEL_CFLAGS` | kernels-sys/src/lib.rs | set by `kernels-sys/build.rs` | compile flags of the C/C++ kernels, recorded in reports | build |
| `OXE_INT8_GEMM_CFLAGS` | int8-gemm/src/lib.rs | set by `int8-gemm/build.rs` | compile flags of the int8 GEMM kernels, recorded in reports | build |
| `OXF_KERNEL_CFLAGS` | qwen4exp/src/lib.rs | set by `qwen4exp/build.rs` | compile flags of the Flash kernels, recorded in receipts | build |
| `OXF_GGML_PROD_CPU_SHA` | qwen4exp/src/lib.rs | set by `qwen4exp/build.rs` (aarch64 only) | SHA-256 of the production ggml CPU archive, carried by Flash receipts | build |

## Tests and benches

Read only by tests, ignored tests and benches, never by a serving process.

| Name | Read in | Values and default | What it does | Class |
|---|---|---|---|---|
| `OXE_TEST_MODEL` | server/src/tests.rs | path to a model GGUF | model for server tests that need one | test |
| `OXE_TEST_CPUS` | server/src/tests.rs | CPU list; default `0-7` | CPUs for those server tests | test |
| `OXE_TEST_MTP` | server/src/tests.rs | integer; unset or unparsable: 0 | MTP draft depth used by one server test | test |
| `OXE_MODEL_QWEN36` | gguf/src/tests.rs | path | Qwen3.6 GGUF compared with the fixture in the reader test | test |
| `OXE_MODEL_ORNITH` | gguf/src/tests.rs | path | Ornith GGUF compared with the fixture in the reader test | test |
| `OXE_TOKENIZER_GGUF` | tokenizer/src/tests.rs | path; default the production Ornith model | model GGUF whose vocabulary the tokenizer tests use | test |
| `OXE_MINI_OUT` | model/src/tests.rs | path | where the ignored test `mini_gguf_write_for_cli` writes the synthetic mini model | test |
| `OXE_NPU_TEST_GGUF` | server/src/oxe_engine.rs | path to a synthetic mini GGUF | NPU emulation state-identity test (feature `npu-prefill`, ignored) | test |
| `OXE_FUZZ_RECORDING` | chat/src/tests.rs | path to a `session_replay` JSONL | replays a recorded session through the parser | test |
| `OXE_NUMERICS_GOLDEN_UNLISTED` | model/src/tests/longctx.rs | `print` | numerics golden test prints instead of failing on a kernel path that is not allowlisted | test |
| `OXE_SCHED_PANIC_CHILD` | sched/src/tests.rs | `caller` or `worker` | child process of the pool abort test: which thread panics | test |
| `OXE_KBENCH_THREADS` | kernels-sys/tests/kv_kbench.rs | integer or list (default 8 / `1,8` per bench) | thread count of the kernel benches | test |
| `OXE_KBENCH_REPS` | kernels-sys/tests/kv_kbench.rs | integer (default 5 / 40 / 30 per bench) | repetitions of the kernel benches | test |
| `OXE_KBENCH_CPUS` | model/tests/fattn_kbench.rs | CPU list | CPUs of the kernel benches | test |
| `OXE_KBENCH_CASES` | kernels-sys/tests/vl_kbench.rs | case list | cases of the kernel benches | test |
| `OXE_KBENCH_LAYERS` | model/tests/fattn_kbench.rs | integer | layer count of the attention bench | test |
| `OXE_KBENCH_CTX` | model/tests/fattn_kbench.rs | integer | context length of the attention bench | test |
| `OXE_KBENCH_CH` | model/tests/fattn_kbench.rs | integer | chunk size of the attention bench | test |
