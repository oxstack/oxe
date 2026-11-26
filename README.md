# oxe

CPU inference for large mixture-of-experts GGUF models on Snapdragon X laptops under Linux
(aarch64), including a Flash engine that streams experts from an SSD.

This repository is the engine, split out of [oxstack/ox-llm](https://github.com/oxstack/ox-llm)
at tag `final-monorepo`. History before the split is in oxstack/ox-llm and, earlier,
oxstack/snapdragon-engine.

## What is in the repository

| Binary / crate | What it is |
|---|---|
| `oxe` (`crates/oxe-cli`) | The command-line tool: `oxe serve` and `oxe models add` (see [docs/SERVE.md](docs/SERVE.md)), plus `gen`, `trace`, `compare`, `verify`, `bench` and `cache`. |
| `oxe-server` (`crates/server`) | An OpenAI-compatible HTTP server in llama-server shapes, over an `Engine` trait. |
| `local-ai-memlat-boot` (`crates/memlat`) | The MEMLAT boot-default tool; see [crates/memlat/README.md](crates/memlat/README.md). |

Two engines run behind `oxe-server`:

- **`--engine oxe`** (`crates/model`, with `crates/kernels-sys` and `crates/sched`): Ornith-1.5-35B-A3B
  and Qwen3.6-35B-A3B. A static per-model plan, repacked int8 `i8mm` kernels, tiled attention,
  batched prefill, MTP speculation depth 1 and prefix/session reuse.
- **`--engine flash`** (`crates/qwen4exp`): Qwen3.8-Flash-Next (125B MoE). Dense weights in RAM,
  experts streamed from NVMe through a cache, `O_DIRECT` reads on a dedicated IO core, memory
  admission measured before start.

`--engine stub` serves the tokenizer and chat template only, with no weights; the model-free tests
and `tools/openai-contract` use it.

## Requirements

- Linux on aarch64. The packaged binaries target ARMv8.6-A cores with dotprod, i8mm, fp16 and
  bf16, and were measured on the Snapdragon X Elite (X1E80100) only; the X Plus has not been
  tested. A core without i8mm or bf16 is expected to stop with "Illegal instruction".
- CPU only. The release binaries contain no Adreno GPU or NPU path.
- Model weights are not part of this repository: bring your own GGUF files.

## Install from a release tarball

Each [release](https://github.com/oxstack/oxe/releases) has one Linux aarch64 tarball, built by
GitHub Actions from the tagged source (`.github/workflows/release.yml`). It unpacks to a directory
named after the tarball, holding `usr/bin/oxe`, `usr/bin/oxe-server` and
`usr/bin/local-ai-memlat-boot`, the licences and a `README.txt` that records the source commit,
the workflow run, the toolchain and the minimum glibc.

```sh
sha256sum -c SHA256SUMS
gh attestation verify oxe-vX.Y.Z-linux-aarch64.tar.gz --repo oxstack/oxe \
  --signer-workflow oxstack/oxe/.github/workflows/release.yml
tar xzf oxe-vX.Y.Z-linux-aarch64.tar.gz
cd oxe-vX.Y.Z-linux-aarch64
./usr/bin/oxe --version
```

The attestation states that this repository's release workflow built the tarball and from which
commit and tag; it needs GitHub CLI 2.49 or newer. `SHA256SUMS` lists the tarball's SHA-256, and
`README.txt` inside it lists the SHA-256 of each binary.

**Rolling dev builds.** Every push to main publishes a `dev` prerelease
(`.github/workflows/dev-build.yml`), signed and sum-checked the same way.
`tools/lab/dev-update.sh` installs the newest one under `~/.local/state/oxe/dev-builds/default/`
and switches its `current` symlink; it does not touch `PATH` or any stable install.

## Build from source

Linux aarch64, Rust (CI pins 1.99.1 in `tools/ci/setup_toolchain.sh`) and GCC 16 for the vendored C
kernels. `.cargo/config.toml` sets `target-cpu=oryon-1`; change it for another core (the per-target
setting and the owner gate are in [docs/CPU-TARGET-AND-PLACEMENT.md](docs/CPU-TARGET-AND-PLACEMENT.md)).

```sh
cargo build --release --locked -p server --bin oxe-server
cargo build --release --locked -p oxe-cli --bin oxe
```

The model-free tests are the workspace minus the Flash crate:

```sh
cargo test --release --locked --workspace --exclude qwen4exp
```

**The Flash engine is not built by these commands.** It links a production ggml archive that is
not in this repository, and the vendored Flash sources do not build from a fresh clone:
[issue #7](https://github.com/oxstack/oxe/issues/7).

## Quick start

Register a model, then serve it:

```sh
oxe models add            # register a GGUF file; see docs/SERVE.md
oxe serve                 # OpenAI-compatible server on 127.0.0.1:8090
```

Model arguments, cache and API-key handling are in [docs/SERVE.md](docs/SERVE.md). Use it from
`ox` as a local provider:

```sh
ox provider add oxe --base-url http://127.0.0.1:8090/v1 --wire chat_completions --env-key OXE_API_KEY
```

`oxe-server` binds `127.0.0.1` by default, accepts only loopback `Host` names and refuses a
non-loopback bind without an API key file (`--api-key-file` or `OXE_API_KEY_FILE`).

## MEMLAT boot tool

`local-ai-memlat-boot` selects and falls back between a MEMLAT device-tree overlay and the stock
boot default, and rebuilds the overlay across kernel updates. It is a native binary with no Python
runtime dependency; installation, the policy file and the hooks are described in
[crates/memlat/README.md](crates/memlat/README.md). Selecting MEMLAT changes how the laptop boots,
so install it only with the policy you mean to run.

## NPU crates and the Hexagon SDK

The NPU work (`crates/oxn`, `crates/image-npu`, the `npu-prefill` / `npu-device` features) is
opt-in and off in the default build. The device backend compiles against FastRPC headers that you
supply: point `FASTRPC_INC` at that include directory (`crates/oxn/build.rs`) and check the
licence of your copy. No Qualcomm Hexagon SDK, QAIRT headers, IDL, stubs or shared objects are
part of this repository.

## Layout

```
crates/
  kernels-sys/        cc build of the vendored ggml kernel subset, FFI and reference tests
  gguf/               mmap GGUF reader
  sched/              pinned thread pool, spin barrier, LPT balancer
  model/              MoE descriptor, load-time repack, static plan, MTP, attention
  tokenizer/ chat/    GGUF BPE tokenizer; chat templates and tool-call parser
  server/             oxe-server (engines: stub, oxe, flash)
  oxe-cli/            oxe: serve, models, gen, trace, compare, verify, bench, cache
  qwen4exp/           Flash graph, expert cache, IO queue (feature flash)
  memlat/             local-ai-memlat-boot
  int8-gemm/ image-*/ oxi image engine work (experimental)
  oxn/                NPU host API and CPU emulation of the HMX GEMM contract
vendor/               verbatim extracts, each with its own PROVENANCE.md
docs/                 environment table, portability probes, engine notes
```

The environment variables of the engine are documented in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
The local HTTP completion backend design is [docs/EXTERNAL-ENGINE-CONTRACT.md](docs/EXTERNAL-ENGINE-CONTRACT.md)
(design only). Historical engine notes are in [docs/lab-notes/](docs/lab-notes/).

## Correctness

Every change declares its class before it is measured: **lossless** (bit-identical logits or state
against the reference or the previous path), **reorder** (different float order, mean KLD within
twice the reference's own batch noise) or **lossy** (opt-in only). Bit identity is the bug
detector. [CONTRIBUTING.md](CONTRIBUTING.md) states the measurement rules and the checks a pull
request runs.

## Issues

Open work is tracked as issues: #27 for local model handling and #28 for engine performance and
hardware. Report security problems as described in [SECURITY.md](SECURITY.md).

## Licence

MIT; see [LICENSE](LICENSE). Copyright (c) 2026 Oscar Nordström (n0rpan).

Vendored code keeps the licence of its source: `vendor/ggml-subset/` and `vendor/ggml-flash/` are
MIT, and the image crates port code from stable-diffusion.cpp (MIT). Code ported from llama.cpp is
named in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Model weights, vendor runtimes and the Hexagon
SDK are not included and have their own terms.

Snapdragon and Hexagon are trademarks of Qualcomm Incorporated. This is an independent project,
not affiliated with, sponsored by, or endorsed by Qualcomm.
