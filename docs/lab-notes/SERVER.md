# oxe-server

> **Historical.** This is the early oxe-server integration plan and status note. It is kept for the record. Its opening status ("the model is still a stub") is out of date: the production engine is wired. For how to run the server, see the [README](../../README.md).

An OpenAI-compatible HTTP server shaped like llama-server, so the controller, OpenCode and
dsh can talk to the Rust engine without changes. It is the M4b "standalone" route from
`M3-M4-PLAN.md`, built early so it is ready when the engine is.

Status (2026-11-09): tokenizer, chat template, tool-call parsing, HTTP API and the prefix
cache are done and tested against llama.cpp itself. The model is still a stub
(`StubEngine`); wiring the M2 engine is the remaining step (section 4).

## 1. Layout

| Crate | What | Ported from (llama.cpp @ 954de24, the production source) |
|---|---|---|
| `crates/tokenizer` | GGUF byte-level BPE, qwen2/qwen35 pre-tokenizers, special-token partitioning, detokenize, streaming UTF-8 hold-back | `src/llama-vocab.cpp`, `src/unicode.cpp`, tables generated from `src/unicode-data.cpp` (`tools/gen_unicode.py`) |
| `crates/chat` | template rendering (minijinja + llama.cpp jinja semantics), OpenAI request normalization, Qwen3-Coder tool-call parser with streaming diffs | `common/chat.cpp`, `common/jinja/value.cpp`, `common/parsers/qwen3-coder.cpp`, `common/chat-peg-parser.cpp`, `tools/server/server-common.cpp` (`oaicompat_chat_params_parse`) |
| `crates/server` | `Engine` trait + stub, prefix cache, sampler, generation loop, endpoints, hyper front end, `oxe-server` binary | `tools/server/server-context.cpp`, `server-task.cpp`, `server-chat.cpp` |
| `crates/server/oracle` | `llama-oracle`: a small C++ program linked against llama.cpp's own shared libs (vocab-only load, no weights) that answers tokenize / apply-template / parse-stream questions with llama.cpp's code. Generates the test fixtures. | |

## 2. Running it

`$MODELS` is the directory that holds the GGUF files.

```sh
CARGO_BUILD_JOBS=1 nice -n 19 cargo build --release -p server
target/release/oxe-server --model $MODELS/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf \
  --alias ornith --host 127.0.0.1 --port 8093 --ctx-size 16384 \
  --chat-template-kwargs '{"enable_thinking":false}' --engine stub [--stub-reply TEXT]
```

The stub replies with `--stub-reply` (tokenized), then EOS, and it uses no model memory, so
it is safe to run without the inference lease. llama-server flags that the controller
passes (`--threads`, `--jinja`, `-b/-ub`, `--spec-type`, `--reasoning-budget`, ...) are
accepted and ignored (`IGNORED` in `src/main.rs`); unknown flags are an error.

### Access

The port is for local clients. `oxe-server` sends no CORS headers, refuses (403) a request whose
`Host` is not `127.0.0.1`, `localhost` or `[::1]` with the server's port (add names with
`--allow-host NAME`; a specific non-loopback `--host` address is accepted as a Host), and refuses
any request whose `Origin` is not loopback or that carries `Sec-Fetch-Site: cross-site`. That
stops web pages (including DNS rebinding) from using the engine. Against other local users, set
a key: `--api-key-file PATH` or `OXE_API_KEY_FILE=PATH` (one key per line, file mode 0600); every
route, `/health` included, then needs `Authorization: Bearer KEY` (or `X-Api-Key: KEY`), else 401.
The lab controller creates a fresh key per launch. A `--host` that is not loopback (a LAN address,
`0.0.0.0`, `::` or a host name) without a key is refused at start (exit 2): the Host check stops
DNS rebinding, it does not authenticate. Sampling values out of range (`min_p` or
`top_p` outside 0..1, `top_k` beyond 32 bits, non-finite numbers, penalties outside their range,
`n_probs` / `top_logprobs` above 100) are a 400. A streaming client that takes no data for
`--stream-stall-timeout` seconds (30) is dropped and its generation cancelled.

Long prefills (v0.1.3 LONG-APPEND): a streamed request sends its HTTP head at once and then, while
it waits for the worker or prefills, one SSE comment every `--sse-keepalive` seconds (15; env
`OXE_SSE_KEEPALIVE_SECS`; 0 = off): `: queued wait_s=12`, then `: prefill 12288/28500 361s` (prompt
tokens in the engine state / prompt length, seconds since the request). Comment grammar (v0.2.0,
F10; one format for the engine and the lab controller's queue): `: queued [pos=<n>] wait_s=<s>`,
key=value fields after `queued` (a client ignores keys it does not know); `wait_s` = whole seconds
waited, rounded down; `pos` = place in an ordered queue, 1 = next, sent only by a component that
keeps one (the controller; the engine's worker is a lock, so the engine sends no `pos`).
`: prefill <done>/<total> <s>s`: a client reads the first two numbers and ignores later fields.
Comments stop at the first
decoded token. SSE parsers ignore comment lines (WHATWG EventSource, `eventsource-parser` in the AI
SDK, OpenCode's Effect parser, the OpenAI SDKs), so a client sees the same data frames; the bytes
keep byte-level idle-read timeouts of clients and proxies (the lab controller's included) from
firing during a prefill of many minutes (a 28.5K append at ~83K depth: > 600 s on CPU). They do
not reset event-level timeouts: OpenCode 2.2.7's `chunkTimeout` (default 900 s) counts only data
events, so set it (and `headerTimeout`, default 300 s) generously for local models. A non-streamed request sends nothing until it ends: a client
needs a whole-request timeout above the longest prefill.

Prefill attention (v0.1.3 PREFILL-POLICY): `--pf-attn auto|tiled|fa2|fa2x|pfg` (env `OXE_PF_ATTN`,
default auto). `auto` resolves per engine: `--pf-attn-auto-min-kv N` (env `OXE_PF_ATTN_AUTO_MIN_KV`)
runs `--pf-attn-auto-kernel fa2|fa2x` (env `OXE_PF_ATTN_AUTO_KERNEL`, default fa2) for prefill units
whose padded n_kv exceeds N (clamped to 8192..131072) and the tiled kernel, bit for bit, below it.
The default min-kv is `legacy` (a placeholder until the AP-1 crossover measurement): fa2 beyond 8192
only when `--ctx-size` exceeds 131072, i.e. tiled everywhere else (the v0.1.3 lane-A behaviour).
The start-up log names the resolved path (`oxe-server: prefill attention fa2x beyond n_kv 16384
(...)`); the state identity keys on it (`pf_attn=fa2`, `pf_attn=fa2x.f8.c128`, plus `.k<N>` when the
cutoff is not 8192), so a state is never resumed under another prefill numerics class.

v0.1.4 PFG (opt-in, default unchanged): `--pf-attn pfg` runs the units beyond the cutoff through the
packed GEMM-shaped kernel (blocks of `--pf-pfg-tb` 16 tokens x 8 query heads, f16 chains of
`--pf-pfg-fold` 16:32 (QK:PV) folded into f32, chunks of `--pf-pfg-ch` 128 positions, `--pf-pfg-mk ld1r|elem`;
env `OXE_PF_PFG_FOLD` / `_CH` / `_TB` / `_MK`). It is its own numerics class
(`pf_attn=pfg.q16.p32.c128.t16.ld1r`), qualified only through O-117 (`oxe fused-ref --candidate prefill-pfg`).

Endpoints (llama-server shapes, key order included):

| Endpoint | Status |
|---|---|
| `POST /v1/chat/completions`, `/chat/completions` | stream and non-stream; `messages` (system/developer/user/assistant/tool, `tool_calls`, `tool_call_id`, typed content parts), `tools`, `tool_choice` (auto/none/required, string form), `parallel_tool_calls`, `chat_template_kwargs`, `reasoning_format`, `reasoning_effort` (the thinking switch: `none`/`minimal` off, `low`/`medium`/`high`/`xhigh` on, absent or null = the server default; an explicit `chat_template_kwargs.enable_thinking` wins; an unknown word switches nothing; a non-string is a 400), `max_tokens`/`max_completion_tokens`/`n_predict`, `temperature`/`top_k`/`top_p`/`min_p`/`seed`, `stop`, `cache_prompt`, `stream_options.include_usage`, `logprobs`/`top_logprobs` (non-stream, and streamed on the delta chunks since v0.2.0), `presence_penalty`/`frequency_penalty`/`logit_bias`. SSE: role chunk, one chunk per diff (`reasoning_content` / `content` / `tool_calls[i]` fragments), finish chunk with `finish_reason` stop/tool_calls/length, optional usage chunk with `choices: []`, `timings` on the last chunk, `data: [DONE]`. |
| `POST /completion`, `/completions` | native shape; prompt as string / token ids / mixed; `n_predict`, `n_probs` (+`post_sampling_probs` key names), `return_tokens`, `stop`, stream (no `[DONE]`, like llama-server) |
| `GET /health`, `/v1/health` | `{"status":"ok"}` |
| `GET /v1/models`, `/models`, `GET /props` | shape-compatible |
| `POST /tokenize` (`add_special`, `parse_special`, `with_pieces`), `/detokenize`, `/apply-template` | yes |
| `GET /slots` | with `--slots` |
| `POST /slots/0?action=save/restore` | 501 for now (see gaps) |
| errors | OpenAI object `{"error":{"code","message","param","type"}}` (string `code`, see "OpenAI v1 conformance"), 400 `exceed_context_size_error` with `n_prompt_tokens`/`n_ctx`, 404 `not_found_error` |

### Speculative decoding report (v0.2.0)

`/props.speculative` = `{"active": "draft-mtp"|"lookup"|"none", "depth", "requested", "fallback"}`.
`--spec-type draft-mtp` (or a per-model MTP default) on a model WITHOUT a usable MTP block (no
`blk.<n_layer>` nextn tensors and no sidecar head) is not a load failure: the server logs a
`WARNING`, runs without speculative decoding (`--spec-type none` behaviour; the output is the same,
only slower) and reports `requested: "draft-mtp", active: "none"` with the reason in `fallback`. The
lab controller reads it after a launch: the loaded model's `model_limits[id].capabilities` then show
`mtp: false` and `mtp_fallback`.

### OpenAI v1 conformance (v0.2.0)

- **Default output limit.** A request that names no limit (`max_completion_tokens`, `max_tokens`,
  `n_predict`; null = absent) stops at `--default-max-tokens N` (env `OXE_DEFAULT_MAX_TOKENS`;
  default min(16384, ctx / 4); 0 = none, the reply then runs to the context end) with
  `finish_reason: "length"`. An explicit value always wins (`max_completion_tokens` before
  `max_tokens`; -1 = no limit; 0 = the H6 prefill). Published as `max_output_tokens` (and
  `context_length`) in each `/v1/models` row, in its `meta`, and in `/props.default_generation_settings`.
- **`GET /v1/models/{id}`**: the model object of the served alias; any other id is 404
  `model_not_found` (param `model`). `/completion(s)` keep llama-server's native shape.
- **`POST /v1/completions`** (legacy text completions, `text_completion`): the native generation
  path, streamed (`data:` chunks, `include_usage`, `[DONE]`) or not. One prompt: a string, an array
  of one string, or token ids (several prompts: 400 `unsupported_value`, param `prompt`). `suffix`
  (insert mode) is refused (400, param `suffix`). `logprobs` is 0..=5 and gives the legacy object
  (`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`; offsets continue after an `echo`ed
  prompt, and across stream chunks). `best_of` and `user` are ignored. The reply id is `cmpl-` plus
  24 hex digits.
- **`n`** on `/v1/chat/completions` and `/v1/completions`: absent, null or 1 is the one choice; an
  integer above 1 is 400 `unsupported_value` (param `n`); any other value is 400 `invalid_value`.
- **`x-request-id`**: every HTTP reply (JSON, stream, 204, access refusals) carries `req_` plus 32
  hex digits, new per request; a client-sent id is not read.
- **Errors**: `{"error": {"code", "message", "param", "type"}}` with a string `code`
  (`invalid_request`, `not_found`, `model_not_found`, `unsupported_endpoint`, `unsupported_content_type`,
  `invalid_api_key`, `forbidden`, `unavailable`, `server_error`, ...), `param` the field at fault or
  null. Context overflow: code `context_length_exceeded`, param `messages`, with llama-server's type
  `exceed_context_size_error`, message text and `n_prompt_tokens` / `n_ctx` kept.
- **Image / audio / file content parts** on this text-only server: 400 `unsupported_content_type`
  (param `messages`), as OpenAI answers a non-vision model.
- **Usage**: `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`
  (the generated tokens up to and including `</think>`).
- **Sampler**: `presence_penalty` / `frequency_penalty` (OpenAI semantics over this reply's tokens)
  and `logit_bias` (token id -> -100..100, added to the logit; zero entries are dropped). Absent or
  zero: the logits are read as they are (bit-identical output, the greedy fast path and speculation
  unchanged). `logprobs` report the distribution the token was drawn from (the adjusted row), in the
  non-streamed reply and on the streamed delta chunks (`choices[0].logprobs.content`). `/props`
  states it: `oxe_hints.sampling` = `["presence_penalty", "frequency_penalty", "logit_bias"]`.

### Request hints (v0.2.0, client-neutral)

Optional, for any client (no client detection). A server without a hint ignores it. A request
without a hint behaves exactly as before, and a hint never changes the generated bytes of the request
that carries it.

**H6 `max_tokens: 0` (also `max_completion_tokens` / `n_predict`): prefill only.** The prompt is
evaluated into the cache and nothing is generated: no sampling, no decode step, no ECHO record (there
is no assistant turn), no idle re-prefill. The post-reply state write runs as for any request, so the
prefilled state is durable like a reply's. On `/v1/chat/completions` the prompt is the conversation
WITHOUT the assistant opener a generating request appends (`add_generation_prompt` defaults to
false here; an explicit value from the client is kept), so the next request (the same messages, or
the same messages plus any further message) continues it exactly: its `cache_n` equals this
request's `prompt_tokens`, and it returns what it would have returned without the prefill. The
conversation may end with an assistant message here (prefill after a reply): it is rendered as a
finished turn (with `<|im_end|>`). The conversation may also hold no user message at all (a
client preloading its system + tools prefix before the first prompt): a template that requires a
user query (Qwen3.5, Ornith: `No user query found in messages.`) is then rendered with a
placeholder user turn and the prompt is cut right before that turn's `<|im_start|>user`, so it is
exactly the prefix of the first real request (only with `add_generation_prompt` false).

**Trailing assistant message (v0.2.0 F9, OpenAI v1 behaviour).** A generating request whose last
message is an assistant message treats it as history: it is rendered as a finished turn and the
reply is a NEW assistant turn after it (the generation prompt is appended as for any request).
ECHO handles it like any earlier turn (after a reply, sending the conversation back with that reply
last reuses every fed token of it). Only an explicit in-place continuation, `add_generation_prompt:
false` on a generating request (llama-server's assistant prefill), is a 400: it is not supported.

Reply: `choices[0].message` = `{"role": "assistant", "content": ""}`, `finish_reason: "length"` (the
limit of 0 tokens was reached); `usage.completion_tokens: 0`, `usage.prompt_tokens` = the prefilled
tokens, `usage.prompt_tokens_details.cached_tokens` = what was reused; `timings.predicted_n: 0`.
Streamed: the role chunk, the finish chunk, the usage chunk (`stream_options.include_usage`) with
`timings`, `data: [DONE]`; no content or reasoning delta. `/completion`: `content: ""`,
`tokens_predicted: 0`, `stop_type: "limit"`. A `max_tokens` of 1 or more is unchanged.

**H4 `X-Oxe-Snapshot: keep` (header; any other value = no hint): keep a checkpoint at the end of the
last message, as near as the request's own schedule allows.** A hint never changes generated bytes,
so it never adds a split to the prefill. Definitions, as token positions of the prompt the engine fed
(after the chat template and the ECHO splice):

- `end` = the END OF THE LAST MESSAGE: chat: the prompt without the generation prompt the template
  appends after the last message, i.e. right after that message's `<|im_end|>\n`, at the start of
  `<|im_start|>assistant\n` (+ `<think>\n` with thinking on, `<think>\n\n</think>\n\n` with it
  off); the same for every last role (user, tool, assistant) and both thinking settings. With H6 (or
  `add_generation_prompt: false`) `end` is the prompt end. `/completion`: the prompt end. It is where
  any continuation (one more message of any role) first differs from this prompt.
- `start` = the start (`<|im_start|>`) of the last message.
- `snap_pos` = the DEEPEST BOUNDARY THE REQUEST PROCESSES ON ANYWAY in `[start, end]`: `end` itself
  when it is the prompt end or the position before the last prompt token (the engine stops there in
  any case: always so with H6); else the deepest of the ubatch-grid points of this request's prefill
  (`restore point + k * ubatch`, the floor rule of the turn checkpoints; on the message schedule of
  Flash, its message starts) and the reused-prefix end; inside the reused prefix, an existing
  checkpoint (it takes the class; no copy).

So `snap_pos <= end`. The reply reports `timings.snap_pos` (only when the hint was sent) = the exact
position of a checkpoint the engine holds when the reply is sent; a client that cuts back to a
prefix of the fed prompt of at least `snap_pos` tokens resumes there. `null` with `timings.snap_reason`:
`no_boundary_in_last_message` (the last message is shorter than the distance to the next boundary
below `end`), `inside_reused_prefix` (no checkpoint in `[start, end]` of the part reused from the
cache), `end_of_message_not_found` (the prompt does not end with exactly the tokens of the template's
generation prompt), `no_slot` (no checkpoint slot for the class: `--ctx-checkpoints` full of
higher-ranked points). The intended use is H4 with H6 (prefill the head the client will cut back
to): then `snap_pos = end = usage.prompt_tokens`, exactly.

Bound and eviction: the 2 most recently placed or used snap points per session are kept; a third
evicts the least recently used one. They rank with anchors (evicted after tail, turn and query
points). A parked session (another conversation took the engine) carries its own points. A request
that resumes the live session below a point drops the points above its resume position (their
state is overwritten), unless the session was parked first. On disk (state cache) a snap point is
stored as class `cut` (kept by thinning). Nothing is an error: a client that cuts back to an evicted
or never-kept point gets the deepest checkpoint below it, which shows as `cached_tokens < snap_pos`.
Keeping a point uses a checkpoint slot, so it can change WHICH earlier checkpoint a later request
resumes from (retention), never the arithmetic of the request that carries the hint.

Arithmetic: the request with the hint evaluates exactly the ubatches and decode steps of the same
request without it (tests `tests_v020_hints.rs`: engine ubatches and decode steps on the stub; tokens,
full state bytes incl. MTP and drafts on the tiny synthetic model, MTP off and on).

H6 numerics: the next request continues from the prefilled state, so its prefill starts at the H6
prompt end instead of on the cold ubatch grid: the same class as every agentic turn, which continues
from the previous reply's end. The stub engine and the tiny synthetic model generate the same tokens
with and without the prefill (tests `tests_v020_hints.rs`).

**H5 `X-Oxe-Turn-Kind` (header): a free label, ledger only.** Any value (`chat`, `compaction`,
`branch_summary`, `prefill`, `subagent`, ...) is copied to the request ledger as `turn_kind` (at most
32 characters of `[A-Za-z0-9_.-]`, others dropped). It places no checkpoint and selects nothing: the
v0.1.3 compaction cut is gone (the no-op `--compaction-cut` flag was removed in v0.2.0; its disk class `cut`
is still read and is now written for H4 points). A client that wants a point to cut back to sends H4.

**Session hint `X-Oxe-Session` and hint-scoped cleanup (v0.2.0 EG5).** The engine records the
request's session hint on every state-cache manifest it creates for that conversation (as
`hint` = the first 32 hex of SHA-256(hint); the echo file `echo/<hint hash>.jsonl` uses the same
stem). Each manifest also records its `owners`: the hint hash of every request hint whose conversation it
holds, the creator first (a stored lineage continued or absorbed by a request under another hint
keeps its earlier owners and gains the new one; a request without a hint owns as the anonymous
owner `-`, so a hinted continuation of an unhinted conversation is never its sole owner; a write
without a request of its own, at shutdown or when the live session is left, records the owner of
the request that last advanced it; older manifests without the field are owned by their `hint`, or
anonymously without one). `POST /state-cache/clear {"scope": "hint", "hint": "<hint>"}` visits the sessions
that hint owns. A session another hint ALSO owns is kept: only this hint's ownership goes (the
manifest is rewritten; its `hint` moves to the earliest remaining owner; reply `released`). A
session the hint owns alone is removed (`sessions`) as a `scope: session` clear: its manifest, its snapshots, the
blocks NO other session names, its own live and parked RAM state and legacy store entries, and the
hint's echo records. Blocks another session still names (a parent's prefix, a shared anchor such as
the system prompt) stay, and so do the other sessions. The hint is 1-128 characters of
`[A-Za-z0-9._-]` (a child id like `parent.s1` is valid); anything else is 400, and a missing hint is
400 (never a wider clear: there is no fallback to `model` or `all`). A hint with no stored session
is a 200 with `sessions: 0` (idempotent). Reply: `{freed_bytes, used_bytes, sessions, released,
ram_states, legacy_entries, echo_records}`. Refused with 409 while a request runs, like every clear. Durable before the reply: a rewritten manifest is written
atomically (`.pending`, fsync, rename, directory fsync) and a removed manifest or echo file is
followed by a directory fsync, so a crash cannot bring a cleared session back or lose an owners
rewrite (an orphan block or snapshot that comes back is named by no manifest; the next sweep removes
it). The controller's offline clear does the same. A session
id (`scope: session`) remains the 32-hex manifest id derived from the tokens; the hint is the
client-facing handle. A session that continued a conversation started under another hint keeps
its earlier owners, so clearing either hint never removes what the other still holds.

Through the controller: `POST /_local/cache/clear {"scope": "hint", "hint": "<X-Session-Id the
requests sent>"}` maps the client's id to the keyed session hint the controller sends as
`X-Oxe-Session` (request_ledger.session_key) and forwards the hint clear; with no engine running it
applies the same rule to the files itself (blocks still named by another manifest stay).

Discovery: `GET /props` carries `"oxe_hints": {"snapshot_keep": 2, "prefill_only": true, "hint_clear": true}`
(the snap bound per session, H6 support, the `scope: hint` clear). A client sends H4 / H6 (and
uses the hint clear) only when they are listed; an engine
without the field may sample a token for `max_tokens: 0`, so a warm-up must be skipped there.

`/apply-template` with `"count_only": true` answers `{"n_prompt_tokens": N, "n_ctx": C}` (plus
`"n_echo_tokens"` with `"echo": true`) and nothing else. N is the length of the same render that
`return_tokens` reports, so a client can size a prompt without receiving its text. Listed as
`oxe_hints.count_only: true` in `/props`. Without the field the reply is unchanged.

MTP acceptance guard (`--mtp-guard off|RATIO[:WINDOW:COOLDOWN]`, env `OXE_MTP_GUARD`, default off):
`mtp_guard.rs` is a pure policy (acceptance over the last WINDOW drafted tokens, default 256; below
RATIO it refuses speculation for COOLDOWN decoded tokens, default 512, then probes again). It changes
speed only, never output. The decode loop does not call it yet.

Through the controller (local chat requests to the oxe engine; a LAN-share request carries none):
`X-Oxe-Snapshot: keep` is passed on as exactly `keep`; a client's own `X-Oxe-Turn-Kind` (a valid
label as above) replaces the controller's guess and reaches the engine and the controller ledger
unchanged; `max_tokens: 0` passes as sent. Other client `X-Oxe-*` headers are dropped.

Prefix cache: the engine holds one sequence. Before each request the server reuses the
longest reachable prefix of the new prompt: the live state if it is a prefix, else the
deepest checkpoint inside the common prefix (attention KV is kept in place; the recurrent
state comes from the checkpoint). This is what makes agentic turns cheap: turn N+1 re-sends
turn N's prompt + reply + tool result, so only the new tail is prefilled
(`usage.prompt_tokens_details.cached_tokens` and `timings.cache_n` report it).

Checkpoints (`--ctx-checkpoints`, default 4; ~63 MiB each), see `cache.rs`:
- `prompt_len - 1` and the end of every generation (a repeated prompt, or one whose re-sent
  assistant turn differs from the generated one, resumes there);
- chat requests: the **anchor**, the deepest ubatch-grid position <= the first user message
  (system prompt + tools; a new conversation with the same agent resumes there), and, when the
  request ends with a user message, the **query** checkpoint, the deepest grid position
  <= `prompt_len - 4`. Why: the Qwen3.x / Ornith templates render an assistant turn with
  `<think>\n...\n</think>\n\n` only while it follows the last user message. When the next
  user message arrives, every assistant turn of the previous query is re-rendered without
  it, so the new prompt differs from the cached tokens right after `<|im_start|>assistant\n`
  of the query's first reply: 4 tokens before the end of that query's first prompt with
  `enable_thinking:false` (`<think>`, `\n\n`, `</think>`, `\n\n`), 2 with thinking on. No
  checkpoint existed below that point, so such requests fell back to 0 cached tokens (the
  `cache_n 0` of the toy transcript's 4th turn). Tool-result turns are unaffected (the
  history keeps the think block and the reply matches token for token).
- Eviction: lowest priority first (anchor > query > the rest), then least recently used.

Grid positions are multiples of `--ubatch-size` counted from where this request's prefill
started (from 0 when the state is "clean", i.e. was itself prefilled from 0 on the grid).
Splitting a prefill at a grid position changes no ubatch boundary, so a request computes
exactly what an unsplit prefill computes; output for a given request sequence is
deterministic. Resuming from any checkpoint changes later ubatch boundaries relative to a
cold prefill of the whole prompt, as in llama-server (its checkpoints sit at user-message
starts and `n_tokens - 4 - n_ubatch` / `n_tokens - 4`); resuming from a clean grid
checkpoint (anchor, query of a cold request, prefix-store entry) is bit-identical to a cold
prefill.

llama-server comparison: it checkpoints at the start of the last user message (tool results
are user messages in these templates), at `n - 4 - n_ubatch` and `n - 4` (`--batch-size 256`
makes the first `n - 256`), and with `--ctx-checkpoints 4` keeps in practice only its very
first checkpoint (the first user message start) plus the current request's ones. So on a new
user message after a tool chain it resumes from the first user message of the conversation and
re-prefills the whole chain; real OpenCode tasks (one user message, then tool turns) reuse
the cache on every request in both servers.

Persistent prefix store (`--prefix-cache DIR|default`, off unless given; default directory
`~/.cache/local-ai/snapdragon-engine/prefix`, `--prefix-cache-max-bytes` (4 GiB, LRU by mtime),
`--prefix-cache-min-tokens` (1024)), `prefix_store.rs`: when a chat request creates its anchor
from a clean state, the whole engine state at the anchor (KV rows of every attention layer
incl. the MTP layer, recurrent state, MTP driver state) is written in the background to
`<sha256>.bin`. Only system prompt + tool tokens are stored, never user turns or tool output.
A later request, also after a restart, whose prompt starts with a stored prefix loads it if
it is longer than what the in-memory cache offers (log line `prefix store: loaded N tokens in
M ms`). Identity = model (`OXE_MODEL_SHA256`, as the controller exports it, else path + size +
mtime), SHA-256 of the server binary, the engine state layout (MTP on/off) and the ubatch; the
file repeats identity and token ids, both compared in full. Bad or truncated entries are
deleted and the request continues cold. Size: ~22 KiB per token (11 attention layers x K+V
f16) + 63 MiB recurrent state, ~180 MiB for a 5.5K-token OpenCode prefix. The controller's
`/slots` save/restore protocol is not needed: pass `--prefix-cache default` in the oxe args.

### Fair time-slicing between generations (v0.2.0, off by default)

`--fair-quantum-ms N` (env `OXE_FAIR_QUANTUM_MS`, 0..600000; 0 or unset = off: every request takes
the unchanged path, first come first served on the one engine slot). With N > 0 concurrent
generations take turns: every generation takes a ticket in one FIFO queue; the front ticket holds
the engine. A generation gives the engine away at a token boundary once its slice lasted
max(N ms, 8 x its last measured switch) and another generation waits. Its whole session (KV,
recurrent and MTP state, checkpoints) is held in RAM outside the engine, the next generation runs,
and when its turn comes back the live session is put away as a session switch does (unwritten
state to the state store, the whole session to the RAM park) and the held one is loaded back bit
for bit (never re-prefilled). The output is identical to an unsliced run: tokens, text, stop,
logprobs, cache state. Code: `fair.rs` (queue, driver, lock order), `Worker::generate_start` /
`generate_slice` / `generate_abort` in `generate.rs`.

- RAM: ONE budget (`--session-ram-bytes`; 0 = nothing can be held, so nothing yields). The sessions of
  suspended generations (state copy and their checkpoints) and the parked sessions share it: a run yields
  only when what suspended runs already hold plus its own state and checkpoints fit; parked sessions are
  then evicted (least recently used first) to make room, and a park that does not fit beside the held
  sessions is skipped. Parked + held never exceed the budget.
- Not sliced: prefill (a request waits for a running prefill), LAN-share (ephemeral) and
  prefill-only (`max_tokens: 0`) requests; they still queue in the same FIFO.
- `timings.fair_yields` / `timings.fair_wait_ms` (and the ledger fields of the same names) appear
  only when a generation yielded; `predicted_ms` counts decode time only (not the wait).
- A suspended stream gets `: queued wait_s=<s>` SSE comments every `--sse-keepalive` seconds.
- A client that leaves while its generation is suspended: the generation ends cancelled, its
  session joins the RAM parked sessions (the next turn reuses it). SIGTERM cancels queued and
  suspended generations too, so the bounded shutdown write gets the worker; the shutdown write
  covers the live session only (a suspended session's RAM copy is not written to disk).
- `/state-cache/clear` and `/state-cache/budget` (session and hint purges included) answer 409
  while a generation is suspended, as while one runs; `/state-cache/preload` answers busy.
- Post-reply jobs (state write, idle re-prefill) are skipped when another generation is queued at
  a generation's end (the next one's switch path writes that state with its own session hint);
  a sliced generation that ends with nobody queued keeps them.
- Test aid: `--stub-decode-us N` (stub engine) makes each decoded token take N us.

## 3. Tests (all with the stub engine; no model weights are read)

```sh
flock -n ${OXC_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/oxc/inference.lock} true &&   # don't build while Codex holds the lease
CARGO_BUILD_JOBS=1 nice -n 19 cargo test --release -p tokenizer -p chat -p server
```

| Test | What it proves |
|---|---|
| tokenizer `llama_vocab_test_vectors_qwen35` | llama.cpp's own `ggml-vocab-qwen35` vectors: 50/50 |
| tokenizer `llama_oracle_fixture_ornith` | 279 strings (code, unicode, combining marks, whitespace runs, special tokens, slices of real source/docs) x {parse_special+add_special, plain} = 558 cases identical to llama.cpp `common_tokenize` on the Ornith vocab, plus detokenize round trip |
| chat `llama_fixtures_ornith`, `llama_fixtures_qwen36` | 88 + 88 fixtures produced by llama.cpp's `oaicompat_chat_params_parse` + `common_chat_session`: rendered prompts and generation prompts byte-equal (system/no system, developer, typed parts, multi-turn tool calls with nested/float/unicode arguments, reasoning history, tools with/without params, thinking on/off); model outputs fed token by token give identical per-token diffs and final messages (text, tool call + text, parallel calls, every schema type incl. untyped/anyOf/enum/nullable, reordered required args, missing args, unknown function, truncation, text after calls, reasoning split, tool_choice none, parallel off, reasoning_format none) |
| server `chat_nonstream_shape`, `stream_usage_chunk_and_done` | response/chunk key order, usage/timings, SSE framing, `[DONE]` |
| server `chat_stream_tool_call_roundtrip` | streamed tool call accumulates to the exact arguments string; sending it back as an assistant `tool_calls` + `tool` message renders, and the second turn hits the prefix cache (only the new tail is evaluated) |
| server `completion_n_probs_and_limits` | `/completion` n_probs shape (router), `post_sampling_probs`, `n_predict` limit, context overflow error, native stream |
| server `multiturn_tool_calls_reuse_cache` | OpenCode-like chat (system + tools, tool call, tool result, answer + new user message): turn 2 reuses prompt + reply, the new user message diverges exactly at the `<think>` 4 tokens before the end of the first prompt and resumes from the query checkpoint (was 0), a new conversation resumes from the anchor |
| server `prefix_store_survives_restart` | the anchor is written to disk, a new server over the same directory resumes from it with the same reply; another identity does not see it; a truncated entry is discarded and the request runs cold |
| server `prefix_store_real_model` (`#[ignore]`, `OXE_TEST_MODEL`) | real model: grid splits change no token, a stored-prefix resume after "restart" gives the cold run's token ids, every turn after the first reuses the cache, MTP on = off |
| server `prefix_cache_logic` | repeated prompt, continuation, divergence, unrelated prompt, `cache_prompt:false`; the stub rejects any restore whose KV prefix was overwritten |
| server `stop_strings_and_limits`, `helper_endpoints` | stop words (truncation, hold-back), cancellation, tokenize/detokenize/apply-template/models/404/400 |

Regenerating fixtures (needs only the llama.cpp build tree; vocab-only loads, seconds):

```sh
crates/server/oracle/build.sh <llama.cpp checkout> /tmp/llama-oracle
python3 -I crates/tokenizer/tools/make_corpus.py <llama>/models/ggml-vocab-qwen35.gguf.inp <text files...> > corpus.jsonl
python3 -I crates/tokenizer/tools/oracle_fixture.py /tmp/llama-oracle <model.gguf> corpus.jsonl crates/tokenizer/tests/fixtures/qwen35_llama_tokenize.jsonl
python3 -I crates/chat/tools/make_fixtures.py /tmp/llama-oracle <model.gguf> crates/chat/tests/fixtures/ornith.jsonl
```

## 4. Connecting the real engine (after M2)

Implement `server::engine::Engine` for the model crate and add an `--engine oxe` arm in
`src/main.rs` (next to `stub`). Sketch:

```rust
pub struct OxeEngine {
    eng: model::engine::Engine<'static>,   // plan + weights leaked: the server owns them for life
    pool: sched::Pool,
    logits: Vec<f32>,
}
impl server::engine::Engine for OxeEngine {
    type Snapshot = model::RecurrentSnapshot;     // NEW in model: conv_state + ssm_state + pos
    fn n_ctx(&self) -> usize { self.eng.n_ctx }
    fn n_vocab(&self) -> usize { self.eng.plan.desc.n_vocab }
    fn n_past(&self) -> usize { self.eng.pos() }
    fn reset(&mut self) { self.eng.reset() }
    fn prefill(&mut self, toks: &[u32]) -> Result<&[f32]> {
        for &t in toks { self.eng.step(&self.pool, t); }   // until M3.2 batched prefill
        Ok(self.eng.logits())                            // EngineConfig { logits: true, .. }
    }
    fn snapshot(&mut self) -> Result<Self::Snapshot> { Ok(self.eng.save_recurrent()) }
    fn restore(&mut self, s: &Self::Snapshot) -> Result<()> { self.eng.restore_recurrent(s); Ok(()) }
}
```

What the model crate needs for this (not in M1 yet):
1. `save_recurrent()` / `restore_recurrent()`: copy `conv_state` + `ssm_state` (62.8 MiB for
   Ornith) and set `pos`. KV stays in place: attention only reads positions `< pos`, so
   rewinding `pos` is enough for the KV part. 4 checkpoints = ~250 MiB.
2. Owning construction (`Engine<'a>` borrows `Plan`/`WeightStore`; leaking them is fine for a
   server process, or an owning wrapper).
3. Logits every step (`EngineConfig.logits = true`) for sampling and `n_probs`. Greedy-only
   requests could use `step()`'s argmax directly; add `fn decode_argmax` to the trait if the
   full-logits copy shows up in the profile.
4. Batched prefill (M3.2) behind the same `prefill(&[u32])` call. Token-by-token prefill is
   correct but ~30 tok/s, too slow for 7K-token OpenCode/dsh system prompts on a cold cache.
5. MTP (M3.1) fits behind `decode` later (draft + verify inside the engine, same outputs).

## 5. Verification once the engine is wired (the few tests the owner asked for)

All three run under the inference lease (Codex lane), one model in memory at a time.

(a) **Identical greedy output vs llama-server on 3 fixed prompts.** Start llama-server and
oxe-server on the same GGUF and ctx, then:
`python3 crates/server/tools/compare_greedy.py --ref http://127.0.0.1:8093 --oxe http://127.0.0.1:8094 --n 256`.
It checks the rendered prompt bytes, `/completion` greedy token ids (`return_tokens`) and the
chat message / tool calls for 3 prompts (code, prose, tool call). Use the native-layout
llama.cpp build as the reference (M1.md): the production build uses KleidiAI for Q6_K/Q8_0,
which is a different computation, so token identity against production is not guaranteed;
against production, report first-divergence position and use the M2 KLD gate instead.

(b) **One tool-call round trip in OpenCode.** Point the controller's backend at oxe-server
(or run OpenCode against it directly on a scratch dir), ask for a task that needs one file
read; check that the tool call executes, the tool result goes back, and the second request
shows `cached_tokens > 0` (prefix cache) in the server log/usage.

(c) **The protected coding task once** (the 13-case run is not needed): one run through
the controller with the production profile, compare pass/fail and wall time with the
llama-server baseline.

## 6. Known gaps (none affects greedy, well-formed outputs)

- No grammar-constrained sampling. llama-server activates a lazy tool-call grammar after
  `<tool_call>`; with greedy decoding on a good model the argmax already follows the format,
  but a malformed call that llama.cpp's grammar would have prevented is passed through
  (and parsed the same way llama.cpp parses it). `tool_choice: required` and
  `response_format`/`json_schema` are not enforced either.
- Reasoning budget (`--reasoning-budget`) is not enforced (irrelevant with
  `enable_thinking:false`, the production setting).
- Sampling with temperature > 0 follows llama.cpp's chain order but not its RNG; only greedy
  is bit-comparable.
- `POST /slots/0?action=save|restore` (controller prefix store) returns 501; the server's
  own `--prefix-cache` replaces it (section 2).
- Not implemented: trailing-assistant prefill (`continue_final_message`), `/infill`, embeddings, multimodal content,
  `/v1/responses`, Anthropic `/v1/messages`.
- `serde_json` is built with `preserve_order` in these crates (llama-server key order). A
  whole-workspace build unifies the feature, so other crates' JSON output then keeps
  insertion order instead of sorted keys.
