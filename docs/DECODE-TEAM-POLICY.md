# Decode-team policy from the active context length (issue #917)

**Status: design plus the pure selector, deliberately unwired.** The default is unchanged: the
engine still decodes on the whole pool. Nothing calls [`sched::decode_team`](../crates/sched/src/decode_team.rs)
from `server` or `model` yet, so this document and its fixtures are the whole delivery for v0.2.2.
Implementation is proposed for **v0.2.3** and starts only after the owner has approved the scheduler
seam in §9 and pre-registered the table in §8. This document records no measurements and proposes no
crossover threshold: the table edges are the owner's to fill in.

## 1. What the policy is, and what it is not

The decode team is a fixed pool today: `OxeEngine` builds one `sched::DetachedPool` from `--oxe-cpus`
(`engine/crates/server/src/oxe_engine.rs`) and every decode step runs on all of it. The policy adds an
*optional* rule that chooses a smaller decode team as the active context grows, where a smaller team
can be cheaper for long contexts than a larger one.

It does **not** touch:

- the default: no table and no override means the whole pool, byte for byte the current behaviour;
- prefill: `team_for` / `OXE_SMALL_PREFILL` and `--oxe-prefill-cpus` keep their own policy;
- cancellation, timeouts, admission or memory limits: no new limit, no changed limit;
- MTP: speculation keeps its own team and driver state;
- the pool itself: no worker is added or removed while the process runs, and no thread is created
  per token.

## 2. Pure selection, then execution

The rule is split in two:

- **Selection** (`sched::decode_team`) is a pure function of `(active_context, allowed CPUs, table,
  explicit override)` returning a `TeamPlan { team, cpus, source, bucket_min_context }`. It holds no
  pool, spawns nothing, blocks on nothing and reads no environment. That is what the fixtures in
  `engine/crates/sched/src/tests_decode_team.rs` exercise.
- **Execution** is where a plan becomes a running team. That is the part that needs the owner seam
  (§9), because `sched::Pool` fixes its barrier membership (`Barrier::new(n)`) at `Pool::new` time.

## 3. Active-context semantics

The policy is consulted at a decode-step boundary with

> `active_context` = the number of positions the job will have written **after** this step, i.e.
> `n_past + 1` for a one-token decode step.

Rationale: the team must be chosen before the step's barrier work starts, and the step's own token
becomes position `n_past`. Batched steps that write more than one position use `n_past + n_new`.
There is no other reading of "context length": not the KV capacity (`n_ctx`), not the allocation
granularity (`padded_n_kv` / `validate_n_ctx`), not the request's max tokens. Fixtures must state the
same number the caller would pass, and a step that crosses a boundary mid-job does not resize (§7).

## 4. Bucket edges

A table is an ordered list of rows `MIN=TEAM`:

- row *i* covers the half-open range `[min_context_i, min_context_{i+1})`;
- the last row is open ended: `[min_context_last, ∞)`;
- the first row must start at `0`, so the table is a total function with no uncovered range;
- edges are integers, compared with `<` / `>=` on `active_context`; `min_context` is inclusive, the
  next row's `min_context` is exclusive.

`TeamTable::parse` rejects a first row that does not start at 0, a duplicate or out-of-order edge, a
zero team, a team above the pool size, and any malformed row. Those are the "invalid/overlapping
tables" fixtures.

## 5. Team sizes and allowed CPU masks

- `TeamSize::Pool` means every worker; `TeamSize::Threads(n)` means exactly `n`, `1 <= n <= pool
  threads`. A team larger than the pool is refused at construction: the policy never creates threads.
- A plan's CPU mask is the first `team` entries of the pool's own worker->CPU list (`Pool::cpus`), in
  pool order, so worker *i* keeps CPU *i*.
- Availability is checked against the CPUs the process may actually run on (the same notion as
  `sched::clamp_to_allowed`). If fewer than `team` of them are usable here, `select` returns
  `PolicyError::CpusShort` instead of silently shrinking the team. The caller's documented fallback
  is the unchanged default (the whole pool); a shrink would be a silent policy change.

## 6. Explicit overrides

An explicit user override (`Some(TeamSize)`) wins over the table at every context:

- `Threads(n)`: that team, at any active context;
- `Pool`: the whole pool, at any active context — the way to turn a pre-registered table off for one
  run without editing it.

`PlanSource` records which rule fired (`Default`, `Table`, `Override`), and `bucket_min_context` is
`None` for `Default` and `Override` so a report can say exactly why a team was chosen.

## 7. Never resize a pool inside a job

`admit(active_team, plan)` encodes the rule as data: while a job runs on `active_team`, only a plan
with the same team applies; any other plan returns `HoldUntilJobBoundary`. A context crossing an edge
inside a decode job therefore does not resize anything — the new team takes effect at the next job
boundary. This is the pure statement of the rule; the execution half must not implement it by
rebuilding a pool or changing barrier membership while workers are inside a barrier.

## 8. Owner pre-registration (no threshold proposed here)

The table is configuration the owner writes before any default change. Format:

```
MIN=TEAM,MIN=TEAM,...        TEAM = pool | positive decimal
```

- The empty table (the default) means the policy is off.
- No edge or team value is proposed by this document or by the fixtures; the fixture values
  (`0=pool,4096=6,16384=4`) are test data only.
- A table is pre-registered per model (or per model class), together with the CPU masks it may use,
  and it is not changed by a later PR without a new owner decision.

## 9. Scheduler seam (approval required before implementation)

A team smaller than the pool cannot run on the current `sched::Pool`: `Barrier::new(n)` fixes the
membership at `Pool::new`, and workers that are not in the team must not arrive at the barrier. The
options to be reviewed by the owner (source review, not implemented here):

1. one pool per distinct team size, created at startup from pre-registered masks — no runtime
   resizing, but more pinned threads and more idle spin unless idle waking (#878) is in place;
2. a job-scoped team selection inside the existing pool with the barrier resized only at a job
   boundary and every non-member parked before the epoch — a real change to pool membership
   semantics, so it needs the owner's explicit approval and its own identity gate;
3. no seam at all: the table may only select `pool`, and the policy stays documentation until 1 or 2
   is approved.

The default and this PR are unaffected by the choice. Machine placement is N2's; idle waking is
#878's; neither is decided here.

## 10. Identity gates

The team size must not change any result. The contract, per team size in the allowed set:

- **token ids**: the same greedy ids for the same prompt, prefix and sampling parameters;
- **full state**: `dump_state` / prefix-store bytes bit-identical, and `state_layout` unchanged;
- **continuation identity**: decode rows after a state restore or a prefix-store resume bit-identical;
- **MTP**: committed logit rows bit-identical to MTP-off rows at the same position (the existing M3
  gate) at every team size, and cancel mid-decode leaves the same visible state;
- **numerics epoch**: the team size is *not* added to `NUMERICS_EPOCH` or to the state identity —
  instead the invariance above is proven, the way `mini_gguf_decode_is_thread_count_invariant` and
  `mini_gguf_prefill_is_thread_count_invariant` already prove it for 1..4 threads.

If any of these fails for a team size, that size is not allowed in the table.

## 11. Deterministic fixture and identity plan

Hosted CI (GitHub-hosted runners, isolated `HOME`/`XDG`, synthetic model-free fixtures):

- the pure fixtures already in this PR: exact edges below/at/above each boundary, empty table,
  invalid and overlapping tables, explicit overrides, unavailable CPUs, and the mid-job admission
  rule;
- once the seam is approved: the existing mini-GGUF invariance tests extended over the allowed team
  sizes (token ids, state, continuation, MTP and cancel), reusing `mini_engine_parts` / `same_bits` /
  `same_state` so the comparison is deterministic and model-free; a refusal fixture for a team that
  cannot be staffed; and a fixture that a mid-job edge crossing does not resize.

Owner-authorized route only (never from a helper, never on this machine): any Oryon-dependent or
NPU-dependent validation, and any warm HTTP batch that would justify a default change. Those runs are
separate batches; their numbers are not accepted as measurements here.

## 12. Related work

- **N2** owns machine placement; this policy only chooses a team size inside a placement.
- **#878** owns idle waking; a startup pool per team size (§9.1) depends on it.
- **#910** tracks the parent work; this design is filed under **v0.2.2** with implementation proposed
  for **v0.2.3**.
