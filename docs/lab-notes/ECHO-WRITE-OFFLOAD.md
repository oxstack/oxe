# Echo write offload: qualification packet (issue #883)

**Status: design and qualification only. No engine behaviour is changed by this document, and no
implementation is authorised by it.** Issue #883 is a bounded task for an assigned helper; the
engine/durability owner must release the implementation before any engine behaviour changes. This
packet is the input to that release decision, not the release.

Parent: #864 (owner O-383). Milestone v0.2.2, label `after-0.2.1`. Nothing here lands in v0.2.1.

The canonical context is #583 and the unfinished #614. The path this packet specifies is the one
already in `main` — the N4 deferred append (`PendingAppend`) — completed, not replaced. It adds no
second mechanism, no daemon, no queue framework and no new knob.

## 1. Verified source facts

Read from the source in this checkout. The line numbers cited in #883 were taken at
`69a81515ad3ce39dbf048112fd0cfad476f4b73f`; they were re-checked on `70370d79` and are unchanged.

| Fact | Where |
|---|---|
| `append_line` opens the group file (0600, `O_NOFOLLOW`), writes the line and calls `sync_data` | `engine/crates/server/src/echo.rs:463` |
| `PendingAppend::write` takes the `EchoIo` mutex, checks the group/global epoch, appends, refreshes disk usage | `engine/crates/server/src/echo.rs:534` |
| `record_deferred` returns the line to the caller and does **not** write it; the store mutex is released before the caller writes | `engine/crates/server/src/echo.rs:689` |
| `echo_record` calls `p.write()` synchronously, in the same function, after the store mutex is released | `engine/crates/server/src/service.rs:486`, write at `:518` |
| `echo_adopt` does the same for adopted history turns | `engine/crates/server/src/service.rs:524`, write at `:539` |
| JSON chat path: `echo_record` runs before the response body is built | `engine/crates/server/src/service.rs:2214` |
| Streaming chat path: `echo_record` runs before the final frames are sent (frames at `:2365`, `[DONE]` at `:2371`) | `engine/crates/server/src/service.rs:2348` |
| SIGTERM path writes the state cache, then `std::process::exit(0)`; it does not wait for echo appends | `engine/crates/server/src/main.rs:1517`–`:1578` |
| The in-repo pattern for an offloaded durable write is a spawned writer plus a `flush()` that joins it | `engine/crates/server/src/prefix_store.rs:554`, `:679` |

So the claim in #883 is correct: the method is named *deferred*, but the fsync still runs on the
reply path, before the client's last frame or body. The store mutex is not held across it (N4 did
that part); the *request* still waits for the disk.

The 5–20 ms figure in #883 is **unmeasured**. It stays a hypothesis until a separate immutable
measurement receipt exists (section 6).

## 2. What must be reconciled with #614 (NEED-METADATA)

The #614 diff is not in this checkout and model tools do not read GitHub. Before implementing, the
owner or the outside runner must supply, or confirm against, the following. Until then this packet
specifies the invariants and the test plan, not a patch:

1. Does #614 offload the write with a dedicated writer (thread or task) or with
   `tokio::task::spawn_blocking` per append? The choice decides the shutdown contract (section 4.4).
2. Does #614 keep `PendingAppend::write` as the single write path, or does it move the epoch check
   to enqueue time? The epoch check must stay at write time (section 4.3).
3. Does #614 bound its queue, and what does it do when the bound is reached?
4. Does #614 touch `append_line`'s flags, `sync_data`, compaction serialisation or the disk-usage
   accounting? None of those may change.
5. Does #614 add a knob, a config field or a default? This packet adds none.

If #614 already answers 1–5, the implementation PR is #614 rebased onto `main` plus the tests in
section 5 — not a new patch. If it does not, the smallest completion of the N4 path is section 4.

## 3. Invariants the offload must preserve

These are the properties the current code has and the tests in `tests_echo.rs` and
`state_store.rs` assert. Every one of them must hold after the change, with the existing assertions
kept as they are.

- **Splice visibility.** The record is in RAM when `record_deferred` returns, so the next request's
  splice sees it whether or not the line has reached the disk
  (`store_unit_deferred_echo_append_is_exact_and_never_resurrects_a_cleared_session`).
- **Per-session ordering.** File line order is not RAM order (ECHO-WRITE-ORDER); reload orders by
  `seq` across files, and a late line after a compaction is tolerated
  (`store_unit_reverse_deferred_writes_reload_the_newest_record`,
  `store_unit_late_deferred_append_after_a_compaction_keeps_the_newest_record`,
  `store_unit_sequence_survives_restarts_and_new_records_outrank_the_old`). An offload must not
  make a *newer* record's line land before an older one in a way reload cannot order — a FIFO
  writer keeps the current order or improves it.
- **Epochs.** `clear`, `drop_group` (eviction), `drop_hint_hash` (session clear), `set_disk(None)`
  (budget 0) and `attach` bump the group or global epoch under the `EchoIo` mutex; a pending append
  whose epoch no longer matches is dropped, so a cleared session never comes back on disk. The
  check must stay in `PendingAppend::write`, at write time, under the same mutex.
- **Compaction serialisation.** `maybe_compact` holds the `EchoIo` mutex across the rewrite and the
  rename, so no append lands between them. The offload must take the same mutex, not a new one.
- **Disk accounting.** `PendingAppend::write` calls `io.refresh` so unhinted groups and partial
  appends are counted (`echo_accounting_tracks_deferred_appends_compaction_and_removal`).
- **Privacy cleanup.** No resurrection after a wipe, and the file flags (0600, `O_NOFOLLOW`) and
  `sync_data` stay exactly as they are.
- **Cancellation.** A cancelled turn is `TurnEnd::Other` and is never recorded; the offload must not
  turn a cancelled turn into a line.

## 4. Bounded design (the smallest completion of the N4 path)

### 4.1 Seam

`PendingAppend` already carries everything a writer needs: `path`, `line`, `group`, `epoch` and a
clone of the `EchoIo` handle. The only change is *who calls `write()`*: today the request thread,
after the offload the writer. `PendingAppend::write` itself is unchanged.

### 4.2 Writer and bound

One writer per `EchoStore`, started lazily on the first append, fed by a bounded FIFO channel
(`std::sync::mpsc::sync_channel`, capacity in the tens — a small constant, not a knob). The writer
calls `PendingAppend::write()` in order. FIFO preserves per-session order; the `EchoIo` mutex keeps
the epoch check and the compaction serialisation exactly as they are.

Backpressure, not loss and not growth: on a full channel the caller writes the line inline, as it
does today. The queue is therefore bounded by construction, the worst case is the current
behaviour, and no line is dropped silently. No per-token thread: one append per finished turn, one
writer per store.

### 4.3 Epochs and ordering

Unchanged. The epoch is captured at enqueue time (as today) and checked at write time under the
`EchoIo` mutex (as today). A queued line whose group was cleared, evicted or taken off disk is
refused by the existing check.

### 4.4 Shutdown and crash-loss semantics

This is the part that needs the owner's explicit decision, because it is a **weaker durability
contract than today's**:

- Today the line is on disk (fsynced) before the reply is delivered. A crash after the reply cannot
  lose it.
- With the offload, a line still in the queue when the process dies is lost. The RAM record dies
  with the process too, so the next turn is a splice **miss** and falls back to the render — never a
  wrong splice, and never a resurrected cleared session. The cost of a lost line is a lost
  optimisation, not a wrong answer.
- SIGTERM must drain: the shutdown path in `main.rs` closes the channel and joins the writer with a
  bounded deadline (the existing shutdown budget is 2.5 s, worker wait 1 s) before
  `std::process::exit(0)`, and logs what it could not write. SIGKILL and power loss keep the
  documented loss above.
- The drain must be reachable from tests, like `PrefixStore::flush` (`prefix_store.rs:679`).

### 4.5 Explicitly out of scope

No new daemon, no new framework, no unbounded queue, no per-token thread, no change to
`append_line`'s flags or `sync_data`, no change to epochs, compaction, eviction, TTL, budget or
identity, no new config field, env knob or default, no change to the state cache or the manifest
writer, no filesystem bypass, and no timing-limit weakening.

## 5. Acceptance test plan (for the implementation PR, not this one)

The tests below are the acceptance criteria of #883. They are specified here so the implementation
PR can be reviewed against them; none of them exists yet, and none is added by this packet.

- **T-A (the headline test).** A synthetic blocked fsync: a `#[cfg(test)]` hook in `append_line`
  (or a test-only delay injected at the same point) that blocks the writer until the test releases
  it. Drive the streaming path with the stub engine and assert that the final frame and
  `data: [DONE]` arrive while the write is still blocked; then release and assert the line lands.
  The same for the JSON path: the body must be returned while the write is blocked. This test must
  fail on `main` (the reply waits for the fsync) and pass with the offload.
- **T-B** clear, session clear and eviction while an append is pending: the line is refused, the
  file stays gone, no resurrection after a restart.
- **T-C** restart: a line written by the offload reads back with the same splice.
- **T-D** failure: a write error is logged and the reply is unaffected.
- **T-E** cancellation: a cancelled turn produces no line.
- **T-F** queue bound: more appends than the channel holds still write every line (inline
  backpressure), and the queue does not grow.
- **T-G** shutdown drain: a line queued before the drain is on disk after it; a line the drain
  cannot write is logged.
- **T-H** the existing `tests_echo.rs` and `state_store.rs` assertions are retained unchanged.

No test may weaken a timing limit or bypass the filesystem to go green.

## 6. Measurement protocol (no numbers here)

The 5–20 ms claim is a hypothesis. A receipt is only valid if it is produced separately, on the
owner's hardware, and stored immutably. The protocol to pre-register:

- **Latency:** last token to `[DONE]` (streaming) and last token to body (JSON), with and without
  the offload, under a controlled disk delay (the same synthetic hook, or a slow device), reported
  as a distribution, not a single run.
- **Next-turn splice latency:** the time the next request waits for the store, which is what N4
  already removed and the offload must not reintroduce.
- **Queue bound:** the maximum queue depth observed, and the number of inline (backpressure) writes.
- **RSS and threads:** before and after, to show the writer costs one thread and a bounded buffer.
- **Limitations:** the delay is synthetic; the numbers say nothing about a real device's fsync
  latency, and nothing about the NPU or the Snapdragon laptop.

## 7. Gates

- **Owner release:** the engine/durability owner must release the implementation before any engine
  behaviour changes. This packet is not that release.
- **Hardware:** any timing, RSS or thread measurement needs the owner's explicit handoff.
- **Merge:** the implementation PR is opened as a draft, is not merged by the helper, and keeps
  every required check.
- **Tier:** T2 (product code) for the implementation; T1 (docs) for this packet.
