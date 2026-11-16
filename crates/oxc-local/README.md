# Local engine ownership

Design for #331's local-engine series: one owner of one `oxe-server` process, with
no controller daemon or worker thread of its own. The engine remains a separate
process. The library is not linked by oxa until the final default-off wiring slice.

1. This slice: synchronous, caller-driven supervision. `start` launches in a new
   process group; `poll` checks exit and calls a bounded health probe supplied by
   the transport owner. Startup and consecutive health failures stop the child;
   automatic restarts have a finite budget and backoff. Stop sends TERM, then KILL
   after a grace period, retaining the child until wait proves exit. Explicit
   stop cancels restart. No request is replayed by the supervisor.
2. A bounded fair request queue grants one RAII lease; cancellation removes a
   waiter without consuming a turn. The lease lasts until upstream cancellation
   is complete. Load and unload use that same exclusion.
3. Model selection owns the supervisor, idle deadline and memory admission/stop
   policy. Unload cannot report success while the old process remains alive.
   Memory pressure stops the engine rather than throttling admission indefinitely.
4. A typed chat request passes directly from the gateway into a bounded loopback
   transport. Streams hold the lease through completion/cancellation. No state
   cache, primers, harness registry, slimming, image pages or legacy completions.
5. Default-off wiring alongside the embedded cloud gateway; the caller drives
   timers only while a local process or queued work exists.

The Python lifecycle, lease, queue, native runtime and memory policy informed this
shape. Their many backend-specific hooks, systemd scopes, polling threads, legacy
protocol routes and diagnostic machinery are not copied. Later memory policy
must preserve admission/stop guarantees before any real model is supported.

`poll` is deliberately nonblocking apart from spawn, process syscalls and the
supplied health probe. The probe must impose its own short I/O deadline and must
identify the owned engine endpoint; a generic successful TCP connection is not
readiness. Production transport and launch authentication are later slices.
Caller timestamps must be monotonic. Health is advisory between polls: callers
must tolerate connection failure after a Ready snapshot. No supervision happens
unless the owner polls; this is an integration obligation, not a hidden thread.

Footprint: no extra threads, timers, sockets or allocations before construction;
construction stores configuration only. No background work when stopped. The
production binary is unchanged because it has no dependency on this crate.
