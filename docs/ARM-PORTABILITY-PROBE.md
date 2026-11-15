# Second-ARM engine probe (#293, step 1)

**Status: runbook only. No second-machine build or measurement has been performed.**
This measures whether the real CPU engine runs usefully on one other Linux ARM64
machine. It does not add a backend, change the tuned Snapdragon path, or cover
#293's x86/GPU/Apple-accelerator work. A CPU result alone cannot decide the best
accelerator backend for a platform.

## Bind before the operator starts

Record an immutable ox-llm commit, the LLVM/Rust and C/C++ toolchains, an immutable
llama.cpp commit, a small supported GGUF's source/licence/SHA-256, and every build
and launch argument. Both engines must load the **same GGUF bytes**, tokenizer,
chat template and context size. If using the tiny-real converted fixture, include
the original hash, conversion script/source commit and converted hash; establish
that both engines accept that same file. Do not silently substitute a different
quantization or model architecture. Store synthetic prompts and hashes with the
binding; never export local prompts, home paths, machine identifiers or keys.

Choose two thread counts that actually fit the process's allowed CPU set, the
context size (4096 is enough for this screen), prompt sizes (512 and 2048 tokens),
256 generated tokens, the CPU affinity, KV type and an observation deadline.
Select the cooling/start rule from that machine's idle temperature before any
results. Preserve the system's governor and power profile. Specify the desired
comparison and Snapdragon regression limits before any follow-up implementation;
this exploratory probe has no post-hoc promotion threshold.

## 0. Inventory and compatibility (no model load)

Record `uname -sr`, `uname -m`, `lscpu -J`, `/proc/self/status`'s `Cpus_allowed_list`,
RAM/swap, the governor and power profile, and the available read-only thermal
sensors. Record a software/kernel inventory rather than the machine's serial or
hostname. Save versions, hashes and build logs locally.

The current real engine is ARM64-only. Its startup check requires **i8mm, bf16,
dotprod and fp16** (`engine/crates/server/src/startup_check.rs`); the shipped
release targets Oryon-1, so do not assume a Snapdragon binary is portable just
because the second machine is ARM64. Compare the four runtime feature checks
with the inventory. Missing features are a valid **UNSUPPORTED** finding, not a
reason to bypass the check, use `--engine stub`, or start a port during the probe.
An unexpected SIGILL is a stop and a recorded defect.

## 1. Build the real engine on that host

Use a clean checkout and an isolated target directory. Override only Rust's
Oryon code-generation target for this exploratory build, preserving the source
and release profile. For example, from `engine/`:

```sh
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS \
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS='-C target-cpu=native' \
  CARGO_TARGET_DIR="$PROBE_ROOT/oxe-target" CARGO_BUILD_JOBS=2 \
  cargo build --locked --release -p server --bin oxe-server
```

`PROBE_ROOT` is a private operator-chosen directory. This is a **host-native
probe build**, not the published Oryon artifact. It does not make explicitly
architecture-specific kernels portable. Record build/feature failures without
patching around them. Run `oxe-server --check-args` from that build: it performs
the real CPU-feature check before any model load, exits 3 for missing features,
and reports effective engine environment. Clear inherited `OXE_*`/`OXF_*`/
`GGML_*`/`LLAMA_*` knobs first. `--check-args` validates flag names, not all flag
values; verify the loaded server metadata too. `--help` alone is not a CPU test.

Build pinned llama.cpp CPU-only on the same host with its native target and no
GPU/accelerator backend. Save the full CMake configure/build commands, compiled
feature list and hashes. Do not label the comparison generic-vs-generic unless
both actual build targets support that claim.

## 2. Correctness before speed

Start one engine at a time on unused loopback ports, with a private HOME/state
root, owner-only receipts and no production stores or keys. Use the host's normal
memory safety controls; retain the engine's no-swap cgroup and record limits,
MemAvailable, PSI, swap and OOM events. Refusal or inadequate memory ends this
point rather than lowering a guard. Never install or modify system tuning.

Use the real `--engine oxe` route. Pin MTP/speculation **off**, thinking off,
temperature zero, one request/slot, the same KV precision, context limit and CPU
sets on both servers. Keep local prefill policy explicit; capture effective
settings rather than trusting a flag accepted for compatibility. Use the generic
OpenAI chat endpoint for each, the same request bytes, and `X-Local-AI-Origin: lab`
when a controller participates. Do not involve oxa/client compaction in this
kernel screen.

First prove template/tokenizer parity: equal rendered prompt token IDs, not just
equal token counts. Run fixed synthetic code, prose, chat and tool-JSON cases.
Compare generated token IDs against the pinned llama.cpp reference and the
pre-bound Snapdragon output on the same GGUF. Keep the first divergent position,
tokens and available logit margin if identity differs. Do not silently accept a
mismatch, call a near-tie a confirmed kernel bug, or claim quality qualification
from speed. Stop timed comparisons on a crash, hang, non-finite values, malformed
tool JSON or unexplained output divergence; report the raw evidence and scope.

## 3. Paired measurements

On the same boot, with no other heavy work, perform an unscored warm-up per engine
and one **A/A** pair for the incumbent to expose run noise. Then use balanced
**A B B A**, repeated three times per point (A=llama.cpp, B=oxe). Apply the
pre-bound thermal start rule to both arms, and log each start/end temperature,
governor, power profile and actual CPU set. Stop a timed arm if the governor,
power profile or competing load changes. No system-setting changes in the run.

Measure separately:

- **Fresh prefill:** 512 and 2048 input tokens, new private cache identity or a
  verified cold context, then a one-token reply. Report evaluated/cached tokens,
  prefill seconds and tok/s; reject an alleged fresh sample with cache reuse.
- **Warm decode:** a fixed saved prompt with 256 output tokens (or equal actual
  token counts if an EOS ends it). Report engine decode ms/token and tok/s.
- **User-visible latency:** whole-request wall and time to the first *content or
  tool* token, excluding role chunks, headers, comments and reasoning. Also
  record model-load time separately; do not hide it in prefill or TTFT.

At 1 Hz log frequency, temperatures, memory, PSI and process major faults. Record
peak RSS/PSS and cache footprint. Compare medians with all per-arm samples and the
A/A spread; show prompt/output counts and tokenizer/template proof beside the
ratios. Different output, cache warmth or affinity makes the speed point
**INCOMPARABLE**, not a faster engine.

## 4. Receipt and next decision

One table should state: host CPU class/features (without private identifiers),
real engine build/run outcome, requirements missing, flags changed/disabled,
output identity, TTFT/prefill/decode with A/A spread, RSS/PSS, and stop reasons.
Include code/asset hashes and toolchain/build settings. Mark untouched points
NOT RUN; never fill the table with estimates presented as measurements.

Conclude separately what worked unchanged, what is Oryon/ISA/affinity-specific,
and whether there is evidence for a CPU-kernel win, only a serving-layer win, or
no useful result. Accelerator choices require another measurement/design and
are not inferred from this CPU screen.

Before merging any portability implementation, rerun the unchanged Snapdragon
baseline and candidate on the same boot/model/settings, with paired warm
prefill/decode and correctness checks. Seal its tolerance before the run. Keep
specialized paths opt-in until that gate passes; do not alter the tuned default
to make a second host boot. No such implementation or device run is authorized
by this runbook itself.
