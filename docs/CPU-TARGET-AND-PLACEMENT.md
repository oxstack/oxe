# Per-machine CPU target and hybrid-aware placement (engine N2, issue #913)

**Status: bounded contract, proposal and fixture plan. No default, flag, pool, numerics or scheduler
change is made here.** This document and the isolated selector it describes exist so the policy can be
reviewed and fixture-tested before any owner-qualified production change. Parent #910, owner O389,
tracking milestone v0.2.2, label `after-0.2.1`; no v0.2.1 implementation.

The source specification is a comment on #795. That comment is not part of this checkout, so this
document works from the issue text and the local sources it names, inspected at main
`69a81515ad3ce39dbf048112fd0cfad476f4b73f`. Nothing here was compiled or measured on hardware.

Related but out of scope: the portability umbrella #293 and probe PR #385 (other machines, not this
bounded fixture-based placement task), the Flash C flags in `crates/qwen4exp/build.rs`, and every
NPU/device path.

## 1. Per-target CPU code generation (documented, unchanged)

Today, exactly as on `main`:

- `../.cargo/config.toml` sets `jobs = 1`, `RUST_TEST_THREADS = 1`, and
  `[target.aarch64-unknown-linux-gnu] rustflags = ["-C", "target-cpu=oryon-1"]`.
- There is no `[target.x86_64-unknown-linux-gnu]` entry: x86_64 gets rustc's default baseline, which
  is what the CI stub server builds with.

The setting is already explicit *per target triple*, and it is part of the release identity: the ARM
job in `../.github/workflows/build.yml` builds the shipped `aarch64` binaries with these flags. A
generic-CPU or x86_64 build is not proof of production identity and must not be substituted for it.

The documented per-machine form, **not applied here**, is either the same table with another
`target-cpu` value for the triple, or an environment override of it, for example
`CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-cpu=native"` (the environment wins over
the table). `native` is machine-specific, so whatever value a release was built with has to be recorded
with the binary.

Why this is a deliberate, owner-qualified change and not a drive-by: `PORTABILITY-PROBE-ARM2.md`
records that `target-cpu=oryon-1` lets rustc emit Oryon-only features (`rng`, `crc`) and that the M2
build worked "by luck of codegen", not by guarantee. A per-machine target is therefore a supported-target
decision with a SIGILL check, not a comment edit.

What CI checks for this task: `rustfmt` and clippy on the engine workspace, the `x86-64-stub` job
building and testing `oxe-server` on hosted x86_64, and `build.yml` building the real aarch64 binary and
running the model-free workspace tests with the production flags. Since this change touches no flag,
those existing jobs are the supported-target check for it; no workflow change is needed or made.

## 2. Placement selector contract (implemented, deliberately unwired)

`crates/sched/src/placement.rs` holds a pure selector plus a read-only sysfs reader. It is public
library API with no call site: `crates/server/src/main.rs` still parses `--oxe-cpus` (default `0-7`) and
`--oxe-prefill-cpus`, drops CPUs with `sched::clamp_to_allowed`, and builds the pools exactly as before.
Nothing here reads `/sys`, changes an affinity mask, or touches a pool.

```text
select(data: &[CpuData], requested: Option<&[usize]>, allowed: Option<&[usize]>, threads: usize)
    -> Result<Placement, PlacementError>
read_cpu_dir(id, dir) -> CpuData          // one cpuN directory, or a fixture tree
read_cpu_tree(root) -> io::Result<Vec<CpuData>>
```

`data` is injected, so the selector itself is pure. `CpuData { id, online, capacity }` is one CPU's
platform data; `allowed` is the process affinity mask as a CPU list, `None` when it could not be read;
`threads` is the caller's current thread count. The result is the chosen `cpus` in pool order plus an
ordered list of `PlacementNote`s, or one of `PlacementError::{NoUsableCpu, NoThreads,
RequestedUnavailable}`. Notes and errors have a stable `Display`, so a caller can print one line each.

Rules:

1. **An explicit user list is the user's decision.** It is never reordered, extended, truncated or
   re-classified by capacity; it defines its own size (`threads` is ignored for it). CPUs in it that are
   not online, or outside the affinity mask, are dropped with a note — today's `clamp_to_allowed`
   behaviour — and a list with nothing left is `RequestedUnavailable` (the existing `--oxe-cpus` exit-2
   case).
2. **Capacity decides only when every candidate reports one.** The highest-capacity class is selected,
   at most `threads` of it, ties in ascending CPU id. Equal known capacities therefore reproduce the
   current ascending-id order exactly: `equal-12-threads8` is today's `--oxe-cpus 0-7` on the 12-core
   Oryon target.
3. **The count is preserved, the cores are re-picked.** Auto selection never pads up to `threads` with
   slower cores and never exceeds it: on a 4E+4P machine with `threads = 8` it returns the four fastest
   cores, and with `threads = 4` the four fastest of a 2E+8P machine.
4. **Capacity is never inferred from CPU numbering.** Only the injected `capacity` is used; the
   `mixed-inverted-threads3` golden has the fastest core at the lowest id.
5. **Missing data degrades conservatively.** A candidate with absent, empty, unparsable or `0` capacity
   (all of which the reader reports as unknown) turns the whole machine into the equal-capacity order
   with a `CapacityMissing` note, rather than a guess. Offline CPUs and CPUs outside the affinity mask
   are never returned, and are reported once each. Conflicting entries for one id make that id a
   non-candidate with a `DataInconsistent` note. An unreadable affinity mask (`allowed = None`) keeps
   every online CPU, matching the existing `clamp_to_allowed` fallback.
6. **Nothing is invented.** No CPU is ever added, no numbering heuristic is applied, and an empty
   candidate set or a zero thread budget is an error the caller resolves with its current default.

`--oxe-prefill-cpus` keeps its own list today, so it is not derived here. Wiring the selector into
`main.rs` — and thereby changing the default placement of either pool — is exactly the owner-gated step
in §4.

## 3. Fixture plan and goldens

`crates/sched/tests/fixtures/cpu/<case>/` is a synthetic, read-only sysfs-like tree
(`cpuN/online`, `cpuN/cpu_capacity`) next to the case's `golden.txt`. The test
`fixture_trees_match_goldens` in `crates/sched/src/tests_placement.rs` reads the tree, runs `select`
with the case's injected affinity mask, requested list and thread count, and compares the rendered
outcome (chosen list plus one line per note, or the error line) with the golden. No host affinity is
changed, no sysfs is written and no pool is created.

| case | tree | affinity | requested | threads | expected |
|---|---|---|---|---|---|
| `equal-12-threads8` | 12 × 1024, no `online` file | — | — | 8 | `[0..7]` (today's default preserved) |
| `equal-8` | 8 × 1024 | — | — | 8 | `[0..7]` |
| `mixed-4e4p` | 0-3 × 512, 4-7 × 1024 | — | — | 8 | `[4,5,6,7]` (P cores only) |
| `mixed-4e4p-threads2` | same | — | — | 2 | `[4,5]` |
| `mixed-2e8p-threads4` | 0-1 × 512, 2-9 × 1024 | — | — | 4 | `[2,3,4,5]` |
| `mixed-inverted-threads3` | 0 × 2048, 1-2 × 1024 | — | — | 3 | `[0]` (no padding, no numbering guess) |
| `id-gaps` | 0,1,4,5,6 × 1024 | — | — | 8 | `[0,1,4,5,6]` |
| `cpuset` | 8 × 1024 | 2,3,6,7 | — | 8 | `[2,3,6,7]` + not-allowed note |
| `missing-capacity` | 0,1 × 1024, 2 unknown, 3 × 512 | — | — | 4 | `[0,1,2,3]` + missing note |
| `malformed` | `abc`, empty, `0`, 1024 | — | — | 4 | `[0,1,2,3]` + missing note |
| `offline` | 0-5 online, 6-7 offline | — | — | 8 | `[0..5]` + offline note |
| `explicit-override` | 0-3 × 512, 4-7 × 1024 | — | 0,1 | 8 | `[0,1]` (slow cores kept) |
| `explicit-partly-unavailable` | 0-3 × 1024 | 0,3 | 2,0,8,3 | 8 | `[0,3]` + offline and not-allowed notes |
| `explicit-none-available` | 0-3 × 1024 | 0,3 | 8,9 | 8 | error: nothing available |
| `inconsistent-tree` | `cpu3` and `cpu03` disagree | — | — | 8 | `[0,1,2]` + inconsistent note |
| `empty-tree` | no `cpuN` entries | — | — | 8 | error: no usable CPU |
| `all-offline` | 0-1 offline | — | — | 8 | error: no usable CPU |

`loader_reads_online_and_capacity_conservatively` pins the reader's own rules (missing `online` file
means online; `0`, empty and unparsable capacity mean unknown; a tree with no CPUs is not an error),
`auto_selection_refuses_zero_threads` pins the zero budget, and `explicit_list_defines_its_own_size`
pins rule 1.

## 4. Gates before any production change

- **Owner review first** for any change to the default `--oxe-cpus`/placement, to either pool, or to the
  `target-cpu` value or triple in `../.cargo/config.toml`. This PR changes none of them.
- **Hard scheduler or numerics changes** additionally need a bit-identity proof and a separately
  pre-registered owner hardware batch. Placement changes are exactly the kind of change that moves
  numerics on a barrier-synchronised pool, so the goldens above are a specification, not a licence.
- **No measurement claims.** Nothing here was run on the Snapdragon laptop, on a second machine, or with
  a model; the portability evidence quoted in §1 is from the earlier probe runbook and its recorded
  result, not from this task.
- Preserved unchanged: the memory admission policy, numerical compile flags, release identity, the
  sandbox/permission/cancel/cache/durability guards and every existing CI policy.
