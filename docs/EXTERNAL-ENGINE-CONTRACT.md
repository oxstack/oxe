# ExternalEngine: price and contract for a local HTTP adapter (engine N4)

Status: **design packet only**. Parent #910, owner O389, source specification #795 Step 3.
Tracking milestone v0.2.2 (label `after-0.2.1`); implementation is proposed for **v0.2.3**.
Native MLX is v0.3 and out of scope. This document authorizes no implementation, no default or
admission change, no device access and no measurement. Source inspection used main
`69a81515ad3ce39dbf048112fd0cfad476f4b73f`; no compilation or runtime claim is made here.

Tags: [D] design decision, [E] estimate (arithmetic, not a measurement); source citations are
`file:line` at the revision above.

## 1. What the current `Engine` trait actually requires

`engine/crates/server/src/engine.rs` is a **token-level, logits-level** interface. The server owns
sampling, stop handling, token accounting and all state; the engine owns the forward pass.

| Trait item | Source | What the server does with it |
|---|---|---|
| `prefill(&[u32]) -> Result<&[f32]>` | engine.rs:41 | prompt evaluation; the returned row is sampled |
| `decode(u32) -> Result<&[f32]>` | engine.rs:57 | one token per step; the row is sampled |
| `decode_argmax` / `decode_spec*` | engine.rs:86-99 | greedy fast path and MTP speculation |
| `snapshot` / `restore` | engine.rs:61-63 | prefix cache rewind, parked sessions |
| `state_id` / `save_state` / `load_state` | engine.rs:130-146 | persistent prefix store |
| `kv_row_bytes` / `kv_rows_*` / `snap_*` | engine.rs:154-183 | segmented state store (CACHE-PERFECT P2) |
| `set_cancel` | engine.rs:51 | F4 in-flight cancel; `CANCELLED_MARK` rollback |
| `n_ctx` / `n_vocab` / `n_past` | engine.rs:33-36 | context limit, tokenizer check, `/props` |
| `warm_hint` / `apply_warm_hint` | engine.rs:193 | `--flash-hotset` cache refill |

The generation loop (`generate.rs:2898` decode loop, `2950` EOG, `3007` length stop, `3073`
speculation) is written against exactly this: **it samples from a logits row the engine returns**.
`usage_json` (`api.rs:369`) is built from the server's own `n_prompt` / `n_decoded` counters.

An OpenAI-compatible completion server exposes none of that. It takes text, samples internally and
returns text (plus, optionally, usage). It has no logits row, no token-prefill entry point, no
snapshot export and no KV-row export.

## 2. The seam: `impl Engine` cannot be honest [D]

**Finding (the escalation the issue asks for):** the `Engine` trait cannot represent an
OpenAI-compatible HTTP completion server without fabricating at least one of logits, token-prefill
or snapshot support. The three ways to force it are all dishonest:

1. **Fake logits.** Return a one-hot row for the text the upstream produced. This silently disables
   the sampler, `logprobs`, `logit_bias`, `seed`, `n > 1` and MTP, and makes the server's own
   sampling parameters look honoured when they are not.
2. **Replay the prompt through `prefill`.** The upstream would re-evaluate the prompt on every
   request; the server would report `prompt_tokens_details.cached_tokens` from a local prefix cache
   that does not exist. That is a false cache claim.
3. **Snapshot/state stubs that succeed.** Any success here lets `cache.rs`, `state_store.rs` and
   `prefix_store.rs` write or read state that the external server never produced.

**Recommended seam [D]:** a sibling trait next to `Engine`, not a new `Engine` implementation and
not a generic backend framework. Sketch (names for the owner to settle):

```rust
/// A text-level completion backend: the upstream samples, the server does not.
pub trait Completion: Send + 'static {
    fn n_ctx(&self) -> usize;
    /// One request. `sink` receives text deltas in order; `cancel` is the same flag
    /// `Engine::set_cancel` carries. Returns the validated terminal outcome.
    fn complete(
        &mut self,
        req: &CompletionRequest,
        cancel: &std::sync::atomic::AtomicBool,
        sink: &mut dyn FnMut(Delta) -> Result<()>,
    ) -> Result<CompletionOutcome>;
}

pub struct CompletionOutcome {
    pub finish: FinishReason,         // stop | length | tool_calls | content_filter
    pub usage: Option<UpstreamUsage>, // None = the upstream did not report it
    pub tool_calls: Vec<ToolCall>,    // assembled from streamed deltas
}
```

`service.rs` picks the path once per request: a local `Engine` keeps the existing
`generate.rs` loop unchanged; a `Completion` backend goes through a new, small text path that
reuses the existing SSE writer (`http.rs:414`), the existing request parsing (`api.rs`) and the
existing error shape (`api.rs:16`). The `Engine` trait, `generate.rs` and every local engine are
untouched.

**Rejected alternatives [D]:** (a) `impl Engine` with refusals everywhere — the server then cannot
serve a single request, so it is not a seam, it is a dead end; (b) a `dyn Backend` enum threaded
through `generate.rs` — a generic backend framework, explicitly out of scope; (c) a second server
binary — duplicates the route layer and the access checks.

**Owner decision needed before any code:** accept the sibling `Completion` seam, or name a
different one. This packet stops here.

## 3. Capability / ownership / cancel / cache matrix [D]

Disposition of every server feature for an external backend. "Refuse" = a clean OpenAI error
(`api.rs:16`), never a silent success.

| Feature | Disposition | Reason / where enforced |
|---|---|---|
| chat/text completion, streaming | pass-through | the upstream's own SSE, re-framed by the existing writer |
| tool calls | pass-through | assembled from streamed deltas; validated like the local path |
| `max_tokens` / `n_predict` | pass-through | sent upstream; the server also enforces its own cap |
| `stop` strings | pass-through | sent upstream; the server keeps its own stop scan as a backstop |
| `temperature` / `top_p` / `top_k` / `min_p` | pass-through | upstream sampler |
| `seed` | pass-through | upstream; the server does not claim determinism |
| `logprobs` / `top_logprobs` | **refuse** | no logits row exists; a fabricated one is a lie |
| `logit_bias` | **refuse** | same |
| `n > 1` | **refuse** | the local path fans out over one engine state; the upstream contract differs |
| `thinking_budget` / reasoning effort | pass-through if the upstream accepts it, else refuse | never silently dropped |
| MTP / speculation | unavailable | `decode_spec*` is not on this seam |
| prefix cache (`cache.rs`) | unavailable | no snapshot; `state_id` is `None` |
| state store (`state_store.rs`) | unavailable | `kv_row_bytes` is 0; nothing is written |
| prefix store (`prefix_store.rs`) | unavailable | no `save_state`; nothing is written |
| parked sessions / turn checkpoints | unavailable | they are snapshots |
| `cached_tokens` in usage | **never claimed** | the upstream's own cache is not ours to report |
| echo cache | unavailable | it is a local KV feature |
| memory admission (`--flash-cache-gib`, `--flash-pf-admit`) | not applicable, **not bypassed** | the external backend holds no local weights; local engines keep their admission unchanged |
| `--offline` / `OXA_OFFLINE` | not an engine flag today; the caller (oxa/oxc) owns the refusal | the engine has no offline switch (`grep offline engine/crates/server/src` is empty); the adapter must not add an outbound path that bypasses the caller's offline mode, and refuses under one if the engine ever gains it |
| `/props` capabilities | must not advertise state-preload, thinking-budget or echo | `service.rs:233`, `main.rs:40-42` |
| `/v1/models` `n_ctx` / `max_output_tokens` | from the configured value only | never invented from the upstream |

**Ownership [D]:** the adapter owns the HTTP client, the request/response translation, the cancel
wiring and the terminal validation. The server keeps the tokenizer, the chat template, the route
layer, the access checks and the usage ledger. The upstream owns sampling, its own KV cache and its
own admission. No local prefix reuse is claimed anywhere.

**Cancel [D]:** `set_cancel`'s flag is the same `Arc<AtomicBool>`; the adapter drops the upstream
response future on the flag and returns `CANCELLED_MARK`-prefixed error text so `generate.rs:1817`
and the cache stay in step. A cancel must leave no session state behind.

**Security [D]:** the endpoint is an explicitly configured **loopback** URL. This is the first
outbound request `oxe-server` would ever make (the server is inbound-only today), so the guard is
part of the seam, not an afterthought. The adapter reuses
`http::bind_is_loopback` (`http.rs:275`) to reject a non-loopback host, follows no redirects,
ignores proxy environment variables, and sends no credentials — no provider or auth change. The
existing loopback `Host`/`Origin` check and the optional API key on the inbound side are unchanged.

## 4. Terminal validation and token accounting [D]

The upstream stream is untrusted input. Rules:

1. A stream is complete only when a chunk carries a non-null finish_reason (or the legacy
   sentinel after one). A stream that ends without it is a **truncated terminal**: a distinct
   error, never a success with finish_reason stop.
2. A malformed frame (bad JSON, missing choices, a delta that is neither text nor a tool call) is
   an error at the point it is seen; already-sent text stays sent, and the error is reported like
   the existing mid-stream error (`contract.py:337`).
3. `usage` is reported **only** when the upstream reports it. If it is absent, the usage object is
   omitted; the server does not count the upstream's tokens with its own tokenizer and present that
   as the upstream's usage.
4. `prompt_tokens_details.cached_tokens` is always 0 for this backend, and the field is only
   emitted where the local path already emits it.
5. A refusal (any row in section 3) must leave no cache, session or parked state behind: the
   adapter holds none, and the server must not have written any for this backend.

## 5. Fake-server acceptance plan [D]

All of it runs in GitHub-hosted CI, isolated HOME/XDG, synthetic and model-free, loopback only.

**Fixture.** A scripted OpenAI-compatible SSE server in Rust, in the server crate's test module,
built on the `hyper` server and `tokio` already in `server/Cargo.toml` — no new crate, no third
decoder or client. It binds `127.0.0.1:0` and serves a table of scripted responses. For the
SDK-level run it is exposed as a test-only binary target (`required-features = ["test-fixture"]`,
never built in production); the owner may prefer to keep it test-module-only and skip the SDK run.

**Scenarios and assertions** (`cargo test -p server --lib`, new `tests_external.rs`):

| Scenario | Assertion |
|---|---|
| split SSE chunks | a frame split across TCP reads is reassembled; the text is byte-identical to the unsplit case |
| normal completion | finish_reason and text match the script; usage is the upstream's |
| malformed frame | a clean error, not a panic and not a silent stop |
| truncated terminal | a distinct error; no finish_reason stop is invented |
| disconnect mid-stream | the client sees the error; the adapter holds no state afterwards |
| cancel in flight | the upstream request is dropped; the error starts with `CANCELLED_MARK` |
| usage absent | the usage object is omitted, not fabricated |
| refusal rows | `logprobs`, `logit_bias`, `n > 1` return the OpenAI error shape and write no state |
| capability honesty | `/props` and `/v1/models` advertise nothing the backend cannot do |
| no stale state | after every refusal and every cancel, the cache/session/state stores are empty for this backend |

**SDK-level contract.** `tools/openai-contract/run.sh` (`checks.yml:342`) already drives the real
`openai-python` SDK against the real binary. Add a third mode `external` with its own pinned list
in `tests.txt`: the tests that need logits, `logprobs`, `n > 1`, `seed` determinism or a local
prefix cache are pinned to **refusal** assertions, not to the local-engine expectations. The
existing `engine` and `controller` modes and their pinned lists are unchanged.

**Unchanged:** every existing native `Engine` test, the stub smoke, the loopback OpenAI contract
for the local engines, and all scanners and required CI policy.

## 6. Price [E]

Estimates, arithmetic only; no measurement was taken and none is claimed.

| Item | Estimate |
|---|---|
| `Completion` seam + text path in `service.rs` (reusing the SSE writer and error shape) | 2-3 days |
| HTTP client on `hyper`/`hyper-util` (already dependencies; `client` + `client-legacy` features) + loopback guard | 1-2 days |
| terminal validation, cancel wiring, usage honesty | 1-2 days |
| fixture + the ten scenarios above | 2 days |
| `external` contract mode, pinned list and refusal list | 1 day |
| docs, `--engine external` / `--external-url` flags, `/props` capability honesty | 1 day |
| review rounds (T2, one round expected) | 1-2 days |
| **total** | **9-13 engineer-days** |

CI cost: the new tests are model-free and loopback-only; the added wall time is the fixture's
scripted sleeps, well under a minute, on the existing hosted runners. No new runner, no new cache.

Risks [D]: the seam decision (section 2) is the only blocking one; the fixture's binary target is a
packaging choice for the owner; the upstream's SSE dialect is the main source of edge cases, which
is why the fixture is scripted rather than a real server.

## 7. What this packet does not authorize

No implementation, no `--engine` default change, no admission or numerical policy, no device or
SDK access, no provider or auth change, no new crate, no measurement. Implementation is proposed
for v0.2.3 and needs the owner's seam review first. #795's sandbox and packaging work stays
independently owned.
