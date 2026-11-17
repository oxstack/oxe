# Native local selection

The CLI factory hook uses the user-only `oxa.engine.inProcessLocal` setting,
default `false`, and calls `configured_engine` with the user `oxa.local` subtree,
the private agent directory and the selected canonical stream timeouts. Project
settings must never select the executable, model paths, arguments or environment.
Enabled configuration errors are returned; they do not fall back to a controller.
The local engine remains available in offline mode because it uses only an owned
numeric loopback endpoint. The cloud selector and provider implementation remain
separate.

The factory's installed-layout input is `models` plus an optional absolute `binary`.
Without that override, oxa locates `oxe-server` beside its own OS-reported
executable path; there is no PATH/current-directory search. It uses the pure `launch_profile` resolver for literal native arguments
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
          "allocation_budget": {
            "kv_bytes_per_position": 1,
            "scratch_bytes": 0,
            "mtp_state_bytes": 0,
            "margin_bytes": 1044480
          }
        }
      }
    }
  }
}
```

For real use, supply the installed model's actual context/output limits and qualified
allocation bounds for its context, KV format, thread teams, lanes and MTP settings.
The example above is synthetic stub data, not usable real-model admission data.
The production factory requires every `allocation_budget` component (unsigned bytes)
and a positive margin; full weights residency is added by the factory, not user-discounted.
The former opaque `required_available_bytes` setting refuses
with migration guidance. Missing/unknown fields, invalid values and anonymous-budget
overflow fail closed during discovery. Full-size addition is checked at cold launch
before authentication or process creation.

Admission requires `MemAvailable >= weights file size + KV + scratch + MTP state + margin`:

- `C = ceil(context_window / 256) * 256` matches native context allocation rounding.
- `KV = C * kv_bytes_per_position`. Include the MTP attention layer in this coefficient
  once, not again in `mtp_state_bytes`. Owner-audited Qwen f16 K+V is 22,528 B/position:
  131,072 positions cost 2,952,790,016 B = 2.75 GiB; 16,384 cost 369,098,752 B = 0.34375 GiB.
- `scratch_bytes` bounds the non-KV peak for this launch: decode/prefill team scratch,
  lanes, recurrent state, pads, retained buffers and other non-MTP allocations.
  `mtp_state_bytes` covers additional non-KV MTP state. Requalify these bounds when
  context, threads, lanes, KV format or arguments change; zero is only an explicit
  qualified absence, never a fallback for missing data.
- Weights are fully resident: the current regular GGUF file's metadata length is
  added at cold load/switch. Qwen's file-backed mmap is not treated as a reclaimable
  working-set discount: evicting experts causes disk reads per token. This path never
  uses the legacy fixed file-size-plus-3-GiB formula. Discovery stats no model.
- `margin_bytes` is the explicitly stated remaining safety/host margin, not a hidden
  constant. No real-model scratch/MTP/margin defaults are guessed here.

The engine already describes its allocations in `CtxMem`/`ctx_mem_with_auto` and
`CtxMem::total` (`engine/crates/model/src/engine.rs`): its recurrent slots, lanes,
per-decode-thread and both retained prefill teams must be included in the supplied
bounds. This crate does not import or duplicate the model engine's allocation planner.
Flash streaming admission is separate and is not inferred by this resident factory;
an explicit `--engine flash` refuses here rather than receiving the resident formula.
The memory budget and actual `--ctx-size N`/`-c N` must match `context_window`;
missing context argv is supplied explicitly, duplicates/conflicts refuse. Low available
memory reports available/required bytes and advises reducing context or freeing memory.
Existing no-swap, PSI/OOM/pid and runtime-pressure stops remain enabled.

Startup defaults
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

At cold load or model switch, the factory checks that the configured GGUF path
names a regular file and that the engine path names an executable file, before
creating child authentication. Missing weights return `not_found` with the
`oxa.local.models.<alias>.model` setting to fix; no download is attempted. Supply
an explicit absolute GGUF path as in the example above. Missing/nonexecutable
engine errors explain the sibling install or `oxa.local.binary` override.

Before admitting generation, the owned/authenticated child's `/props` must carry
`build_info: "oxe-<oxa version>"`. Missing/mismatched build identity refuses with
an engine-version diagnostic and retains teardown ownership. The probe is bounded
and runs on startup/restart, not each resident health poll; no engine CLI or source
change is required. A resident model reuses its frozen launch snapshot if files
change; a later cold load validates the paths and current file size again.

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
