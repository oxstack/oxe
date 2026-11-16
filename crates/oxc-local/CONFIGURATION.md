# Native local selection

The CLI factory hook uses the user-only `oxa.engine.inProcessLocal` setting,
default `false`, and calls `configured_engine` with the user `oxa.local` subtree,
the private agent directory and the selected canonical stream timeouts. Project
settings must never select the executable, model paths, arguments or environment.
Enabled configuration errors are returned; they do not fall back to a controller.
The local engine remains available in offline mode because it uses only an owned
numeric loopback endpoint. The cloud selector and provider implementation remain
separate.

The factory's explicit installed-layout input is `binary` plus the `models` launch
entries. It uses the pure `launch_profile` resolver for literal native arguments
and environment. No PATH, home, Python trampoline or old runtime file is inferred.
The following is synthetic test configuration, not a qualified real-model budget:

```json
{
  "oxa": {
    "engine": { "inProcessLocal": true },
    "local": {
      "binary": "/opt/ox-llm/bin/oxe-server",
      "models": {
        "synthetic": {
          "model": "/models/synthetic.gguf",
          "arguments": ["--engine", "stub", "--ctx-size", "4096"]
        }
      },
      "limits": {
        "synthetic": {
          "context_window": 4096,
          "max_output": 64,
          "required_available_bytes": 1048576
        }
      }
    }
  }
}
```

For real use, supply the installed model's actual context/output limits and a
qualified required-availability budget including the host reserve. Startup defaults
to 180 seconds; `startup_timeout_seconds` can select another positive duration.
Idle unload defaults to two hours; `idle_unload_seconds: null` disables only idle
unload. Missing or unsafe memory metadata still refuses or stops the child. The
caller and child must already be in an inherited no-swap cgroup; this library
does not change system configuration or weaken the policy to make loading work.
Flash-specific swap-tolerant policy is not supported.

Per-model `mtp_max_context` (positive u32) and `sampled_mtp` (boolean) select the
native MTP launch controls. By the v0.2.1 owner decision (#763), canonical alias
`ornith-35b-a3b-local` defaults to cutoff 32768 with sampled MTP on, and
`qwen3.6-35b-a3b-local` to cutoff 16384 with sampled MTP on. These retain the
published profile crossovers; they do not change engine numerics or kernel defaults.
Other aliases, including Flash, acquire neither default. Custom aliases can set
both controls explicitly. Set `sampled_mtp: false` to omit the sampled flag.
Explicit literal `--mtp-max-ctx`/`--mtp-sampled` arguments remain usable and are
not duplicated; do not combine a literal flag with its structured counterpart.
Model context/output limits and literal `--ctx-size` remain separately explicit.
The existing one-owner load/unload/idle lifecycle is unchanged.

Construction and typed discovery open no model, authentication file or socket and
start no thread. Load creates fresh private child authentication under the agent
directory, writes it without reading a credential store, and retains an anchored
directory descriptor until the child is reaped. The token never appears in argv,
status or Debug output. Only its anchored file path enters the child environment.
User arguments/environment cannot override the owned authentication input. Health
checks verify that the listening socket belongs to the child before sending data.

The same bounded FIFO ticket covers cold load, model switch and generation. A
caller-driven request streams on its own thread; a local owner thread polls only
while a process exists and exits after stop. Discovery/status reads do not wait
for a generation. Interrupted streams keep exclusion until process teardown, and
model switching never overlaps two children. `clear_hint` takes that same lease
and calls `/state-cache/clear` directly; no controller route is sent. This path
does not provide controller state caches, primers, a harness registry, request
slimming, Studio, legacy completions or a separate controller CLI.

`native_local_smoke` exercises the real factory against the real stub and
synthetic GGUF. Tests mount explicit fake kernel metadata in a private namespace;
that is a protocol/lifecycle proof, not hardware qualification or real-model
acceptance. The orchestrator performs the real local task acceptance separately.
