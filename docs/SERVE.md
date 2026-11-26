# Running oxe standalone

`oxe serve` is one OpenAI-compatible HTTP listener in front of one supervised `oxe-server`
child. Any client that speaks the OpenAI chat-completions wire can use it: the `ox` relay
(`--wire chat_completions`), Codex, Claude Code through `ox`, or plain `curl`.

```
oxe serve [--config PATH] [--listen 127.0.0.1:8090] [--allow-remote]
```

The default config path is `${XDG_CONFIG_HOME:-$HOME/.config}/oxe/models.toml`. The listener
is loopback-only; a non-loopback `--listen` needs `--allow-remote`. Any `Authorization` bearer
token is accepted and ignored (clients like `ox` require one), so keep the listener on loopback
or put your own authenticating proxy in front of it.

## Endpoints

| Route | Meaning |
| --- | --- |
| `GET /health` | `{"status":"ok","models":N,"loaded":"<id>","state":"..."}` |
| `GET /v1/models` | every configured model, OpenAI list shape |
| `POST /v1/chat/completions` | routed by the request's `model` field |
| `POST /v1/completions` | same routing, legacy completions path |

One model is resident at a time. A request for another model waits for in-flight requests,
stops the old child (TERM, then KILL after the grace period) and admits the new one only after
memory admission passes. SSE bytes are passed through unchanged: `:` comment lines (prefill
and queue progress) and in-stream `event: error` bodies reach the client verbatim. A client
disconnect closes the upstream request. An unknown model is a 404 OpenAI-style error; a full
queue is a 503 with `Retry-After: 1`. The child is unloaded after `idle_unload_seconds`
(default 600) without a request.

## Config

```toml
version = 1
# The engine binary. Default: oxe-server beside the oxe executable.
binary = "/usr/local/bin/oxe-server"
listen = "127.0.0.1:8090"
allow_remote = false
idle_unload_seconds = 600
startup_timeout_seconds = 180
# Requests that may wait for the single engine slot. 0 = no queue (503 when busy).
max_waiters = 8

[environment]
# Passed to every child, in addition to the private OXE_API_KEY_FILE oxe serve creates.
HF_HUB_OFFLINE = "1"

[models.ornith]
model = "/home/me/models/Ornith-1.0-35B-Q4_K_M.gguf"
arguments = ["--engine", "oxe", "--ctx-size", "131072", "--oxe-cpus", "0-7"]
context_window = 131072
max_output = 4096

[models.ornith.allocation_budget]
kv_bytes_per_position = 22528
scratch_bytes = 1048576
mtp_state_bytes = 1048576
margin_bytes = 1048576
```

`oxe serve` creates a fresh child API key in a private owner-only directory, passes it as
`OXE_API_KEY_FILE`, and binds each model to its own private loopback port. Do not put
`--api-key` or `OXE_API_KEY*` in the config; that is refused.

The four `allocation_budget` fields describe allocations, not the weights file: the supervisor
adds the full resident weights size and refuses the launch when the sum does not fit the
available memory (`available >= kv_bytes_per_position * round_up(ctx, 256) + scratch_bytes +
mtp_state_bytes + margin_bytes + weights_bytes`). `margin_bytes` must be positive.

## Adding a model

```sh
oxe models add ~/models/Ornith-1.0-35B-Q4_K_M.gguf --id ornith --ctx 131072
oxe models list
oxe models rm ornith
```

`models add` runs the engine's own header reader (`oxe admission`, no weights are mapped) and
stores the launch arguments and limits it reports. Narrow `max_output` by hand if you want a
smaller reply bound.

## Importing an oxa settings file

```sh
oxe import-oxa --from ~/.oxa/agent/settings.json
oxe import-oxa --from ~/.oxa/agent/settings.json --dry-run
oxe import-oxa --from ~/.oxa/agent/settings.json --force
```

`oxa.local.models.<id>.{model,arguments}`,
`oxa.local.limits.<id>.{context_window,max_output,allocation_budget}` and
`oxa.local.environment` are converted into `models.toml`. The source file is never modified.
Existing ids refuse to be overwritten without `--force`.

## Using it from ox

```sh
OXE_API_KEY=local-only ox provider add oxe \
  --base-url http://127.0.0.1:8090/v1 \
  --wire chat_completions \
  --env-key OXE_API_KEY
```

The key is a placeholder: `oxe serve` accepts any bearer token. Keep it set so `ox` sends the
header it expects.

## Running as a user service

`packaging/systemd/oxe.service` is a `systemd --user` unit with `MemorySwapMax=0`, which is what
the supervisor's no-swap admission checks. Adjust `MemoryMax` for your models, then:

```sh
install -Dm644 packaging/systemd/oxe.service ~/.config/systemd/user/oxe.service
systemctl --user daemon-reload
systemctl --user enable --now oxe
```

The service tests use `OXE_SERVE_FAKE_MEMORY=1` with a fake child; that hook is internal and
must not be set in a real service.
