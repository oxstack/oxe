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

## Request lease (slice 2)

`queue::RequestQueue` owns one engine semaphore and bounded waiting capacity.
Acquisition is FIFO from first poll; overflow returns `Full` immediately. Dropping
an acquisition future removes its ticket, frees capacity and transfers any grant
to the next waiter. A retry joins the back. Clones share the same lease/queue.
`close` permanently refuses new work and wakes waiters while the active owner
retains its lease for cleanup. Zero waiting capacity permits only immediate work.

The lease covers generation, upstream cancellation cleanup, load and unload.
Cancellation of the downstream consumer must not release it before upstream has
stopped; the transport slice will enforce this lifetime. `waiting()` is an
approximate status count, including grants not yet resumed, never an idle/unload
admission check. Idle unload must first acquire the same lease without waiting.

Only Tokio's synchronization feature is used by this library. There is no timer,
executor, background task or thread. Waiting futures carry their own semaphore
wait nodes; the bounded queue does not allocate a request record for each ticket.
No production target references this crate until the final wiring slice.

## Model lifecycle (slice 3)

`lifecycle::Lifecycle` adds caller-driven load, switch, explicit unload and idle
unload. The owner retains the queue lease through startup and stop; a switch
cannot spawn until the old process has been reaped. `try_request` grants a lease
only when ready. Retain it through upstream cancellation cleanup; its drop starts
the idle interval. Idle unload acquires that same lease without waiting. The
default idle interval is two hours; disabling idle unload does not disable memory
stops. No request is replayed after a crash.

Launch profiles supply an explicit, qualified required available-memory budget.
Admission refuses missing/invalid memory, insufficient availability, nonzero
engine swap or a missing zero-swap limit. The caller supplies fresh samples from
the engine's cgroup and host; no production memory reader or cgroup setup is
introduced in this slice. Do not use fabricated samples for real models.

The active memory guard preserves GUARD-2b's behaviour: no-swap violations,
cgroup/global OOM and pids-limit events stop immediately relative to the
pre-launch counters. Host available plus free swap below 2 GiB, available below
256 MiB, or ready PSI full above 20 percent stops after three continuous seconds.
Loading PSI full above 60 percent has a separate five-second timer. A healthy
sample resets its timer; read/parse errors stop rather than disabling the guard.
Memory stops apply during active work too, cancel automatic restart and pending
switches, and retain process ownership until exit. An explicit subsequent load
must pass fresh admission. Flash-specific swap-tolerant policy is not supported.

`poll` must run while a process exists, including while generating and while
stopping; there is no watchdog thread. Health probes must be bounded and identify
the owned endpoint. The production sampler, authenticated loopback transport,
launch-profile resolution and oxa timer driver are later slices. The standalone
`stub_lifecycle` example proves load/idle-unload/reload/memory-stop against the
real stub executable and synthetic GGUF using explicitly fake memory samples.

## Typed chat transport (slice 4)

`transport::Transport` accepts the same borrowed `oxa_proto::ChatRequest` used by
the native gateway. It reuses oxa-engine's pure request/header builder, bounded
SSE reader, decoded `GatewayEvent` chunks, reply assembler and HTTP error
classification. There is one serialization at the actual engine HTTP boundary;
no intermediate JSON request is reconstructed. Endpoints are numeric loopback
addresses, with no proxy, redirect, DNS configuration or controller fallback.
Only `/v1/chat/completions` is sent. Status comes from the lifecycle, and the
caller supplies bounded health/memory probes for the owned process.

One scoped I/O worker and one queued event exist during a request; delivery stays
on the calling thread. The caller drives lifecycle checks every 250 ms during
streaming, including silent prefill. Cancellation is observed within the 50 ms
receive wait plus the existing socket cancellation bound. No generation is
retried. Buffered events and HTTP error bodies are capped at 1 MiB, and encoded
requests at 16 MiB; the default HTTP client's limits are unchanged.

The transport consumes the request lease. Success releases it only after worker
completion and full reply assembly. Failure or callback panic closes the socket
and transfers the lease into lifecycle stop/reap ownership. Continue polling
until Stopped; returning a failure does not let a new request overlap unfinished
generation. This conservative first path stops the child after any sent request
fails. Endpoint authentication and production construction remain slice 5 work.
The stub example now also executes a typed chat and receives streamed output;
its fake memory still makes it a model-free protocol/lifecycle proof only.
