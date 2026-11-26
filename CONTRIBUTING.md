# Contributing

Open work is tracked as issues in this repository (#27 local models, #28 engine performance and
hardware). This page holds the rules a change follows; they are not optional.

## Classify a change before you measure it

1. **Declare the class** first: *lossless* (bit-identical logits or state against the reference or
   the previous path), *reorder* (different floating-point order, mean KLD within twice the
   reference's own batch noise) or *lossy* (opt-in only).
2. **Interleave arms** (A B B A), drop the warm-up, compare medians; a change under 3% is noise for
   a decision. This laptop class throttles within about 30 s of all-core load.
3. **Production shapes**: test at n_ctx 16,384/32,768 with the real head and thread counts.
4. **Keep changes behind a flag**, default off, until measured. A new kernel ships with a
   model-free bitwise or reference test.
5. **Builds fail loudly**: `set -o pipefail`, check the binary's SHA-256 after building, give every
   test a timeout.
6. **Privacy**: no weights, transcripts, home paths, user or host names, serials or keys in
   commits or issues.
7. **No Qualcomm SDK material**: no Hexagon/QAIRT headers, IDL, stubs, skeletons, shared objects
   or ELF objects. The NPU device backend is opt-in and compiles against FastRPC headers the
   builder supplies (`FASTRPC_INC`).

## Before a pull request

```sh
cargo fmt --all
cargo clippy --locked --workspace --exclude qwen4exp --all-targets
cargo test --release --locked --workspace --exclude qwen4exp
```

The Flash crate is excluded because it links a production ggml archive that is not in this
repository ([issue #7](https://github.com/oxstack/oxe/issues/7)). CI
(`.github/workflows/ci.yml`) runs the same commands on GitHub-hosted runners, plus the sharded
nextest run, the test inventory guard and the stub-engine server on x86_64.

For a second ARM host, the [portability probe runbook](docs/ARM-PORTABILITY-PROBE.md) defines a
CPU-feature check and a paired real-engine comparison. It is a plan; it does not claim that
another machine has been validated.

## Licence and upstream notices

Contributions are accepted under the MIT licence. Keep the upstream notices in `vendor/` intact
and update `vendor/*/PROVENANCE.md` when re-extracting kernels.
