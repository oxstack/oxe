#!/usr/bin/env bash
# OpenAI v1 contract job (v0.2.0 M3): the real openai-python SDK (pinned in requirements.txt,
# installed into a private venv, never system-wide) against the REAL `oxe-server --engine stub`
# (synthetic GGUF + the committed Qwen3.6 tokenizer fixture; no weights) and against the real lab
# controller in front of it with the queue ON (LOCAL_AI_QUEUE=1, LOCAL_AI_REPLAY=1).
#
# usage: tools/openai-contract/run.sh --mode engine|controller|both [--server BIN] [--work DIR]
#                                     [--venv DIR] [--strict] [--keep] [--fake]
#   --server BIN  a built oxe-server (default: cargo build --release --locked -p server in engine/)
#   --venv DIR    a venv to (re)use for the SDK (default: DIR under --work); requirements.txt is installed
#   --strict      no expected failures (ignore xfail.<mode>.txt)
#   --fake        controller mode over the lab's Python fake engine instead of oxe-server (a local
#                 smoke run without a Rust build; slow-prefill tests then fail as skipped)
# Servers: engine E1 (stub reply text, slow prefill 2 ms/token so a request can be cut in flight),
# engine E2 (stub reply = two tool calls), controller C1 -> E1 (the qwen profile's output default
# is 4 tokens), C2 -> E2, C0 -> E1 (only ever asked for unsupported routes: must stay unloaded).
# Prints 'OPENAI CONTRACT RESULT: PASS|FAIL ...' per mode; exit 0 only when every mode passed.
# Model-free; no network beyond 127.0.0.1 except pip (PyPI) for the pinned SDK.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
mode= server= work= venv= strict=0 keep=0 fake=0 started= e1= e2= c0= c1= c2=
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) mode=$2; shift 2 ;;
    --server) server=$2; shift 2 ;;
    --work) work=$2; shift 2 ;;
    --venv) venv=$2; shift 2 ;;
    --strict) strict=1; shift ;;
    --keep) keep=1; shift ;;
    --fake) fake=1; shift ;;
    -h|--help) sed -n 2,21p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$mode" in engine|controller|both) ;; *) echo "--mode engine|controller|both is required" >&2; exit 2 ;; esac
[ "$fake" = 0 ] || [ "$mode" = controller ] || { echo "--fake works in controller mode only" >&2; exit 2; }

if [ -z "$work" ]; then work=$(mktemp -d "${TMPDIR:-/tmp}/openai-contract.XXXXXX"); fi
mkdir -p "$work"; work=$(cd "$work" && pwd)
pids=()
cleanup() {
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
  if [ "$keep" = 0 ]; then rm -rf "$work"; else echo "kept $work"; fi
}
trap cleanup EXIT
trap 'exit 143' INT TERM

free_port() { python3 -I -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
wait_http() { # url, seconds
  local i
  for ((i = 0; i < $2 * 10; i++)); do
    if python3 -I -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' "$1" 2>/dev/null; then return 0; fi
    sleep 0.1
  done
  echo "not ready: $1" >&2; return 1
}

# the pinned SDK, in a private venv
venv=${venv:-$work/venv}
[ -x "$venv/bin/python" ] || python3 -m venv "$venv"
"$venv/bin/python" -m pip install --quiet --disable-pip-version-check --no-input -r "$here/requirements.txt"
"$venv/bin/python" -c 'import openai, sys; print("openai-python", openai.__version__)'

N_CTX=32768
ALIAS=ornith-35b-a3b-local     # the controller stack's selected alias (lab/tests/openai_api_stack.py)
LIMITED=qwen3.6-35b-a3b-local  # its output default is 4 tokens on C1
TOOL_REPLY=$'Checking both.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=zone>\nUTC\n</parameter>\n</function>\n</tool_call>'
mkdir -p "$work/home" "$work/logs"

# start_engine / start_controller set $started to the new port (no subshell: the pids must stay here)
start_engine() { # name, extra args...
  local name=$1 port; shift
  port=$(free_port)
  HOME="$work/home" XDG_CACHE_HOME="$work/home/.cache" OXE_SSE_KEEPALIVE_SECS=0.2 \
    "$server" --engine stub --model "$work/synth-qwen36.gguf" --alias "$ALIAS" --ctx-size "$N_CTX" --reasoning off \
    --port "$port" "$@" >"$work/logs/engine-$name.log" 2>&1 &
  pids+=($!)
  wait_http "http://127.0.0.1:$port/health" 60
  started=$port
}
start_controller() { # name, backend port or "fake", extra stack args...
  local name=$1 backend=$2 port; shift 2
  port=$(free_port)
  local args=(--port "$port")
  if [ "$backend" != fake ]; then args+=(--backend-port "$backend"); fi
  HOME="$work/home" LOCAL_AI_QUEUE=1 LOCAL_AI_REPLAY=1 LOCAL_AI_LEDGER_DIR="$work/ledger-$name" \
    PYTHONPATH="$repo/lab/src:$repo/lab/tests" python3 "$repo/lab/tests/openai_api_stack.py" "${args[@]}" "$@" \
    >"$work/logs/controller-$name.log" 2>&1 &
  pids+=($!)
  wait_http "http://127.0.0.1:$port/v1/models" 60
  started=$port
}

if [ "$fake" = 1 ]; then
  start_controller c1 fake --output-default "$LIMITED=4" --fake-think; c1=$started
  start_controller c2 fake --fake-tools; c2=$started
  start_controller c0 fake; c0=$started
  slow=0
else
  if [ -z "$server" ]; then
    (cd "$repo/engine" && cargo build --release --locked -p server --bin oxe-server)
    server="${CARGO_TARGET_DIR:-$repo/engine/target}/release/oxe-server"
  fi
  [ -x "$server" ] || { echo "no oxe-server at $server" >&2; exit 2; }
  python3 -I "$repo/engine/tools/ci/synth_qwen36_model.py" --fixture "$repo/engine/tests/fixtures/qwen36-tokenizer.json.gz" \
    --out "$work/synth-qwen36.gguf" >"$work/logs/synth.log" 2>&1
  start_engine e1 --stub-prefill-us 2000; e1=$started
  start_engine e2 --stub-reply "$TOOL_REPLY"; e2=$started
  slow=1
  if [ "$mode" != engine ]; then
    start_controller c1 "$e1" --output-default "$LIMITED=4"; c1=$started
    start_controller c2 "$e2"; c2=$started
    start_controller c0 "$e1"; c0=$started
  fi
fi

rc=0
run_mode() { # mode, url, tools url, idle url, engine url
  local m=$1 args=(--mode "$1")
  if [ "$strict" = 1 ]; then args+=(--strict); fi
  echo "== OpenAI contract, mode $m"
  set +e
  OAC_URL=$2 OAC_TOOLS_URL=$3 OAC_IDLE_URL=$4 OAC_ENGINE_URL=$5 OAC_SLOW=$slow OAC_DEFAULT_MODEL=$LIMITED OAC_N_CTX=$N_CTX \
    "$venv/bin/python" -I "$here/run_contract.py" "${args[@]}" 2>&1 | tee "$work/logs/contract-$m.log"
  local status=${PIPESTATUS[0]}
  set -e
  [ "$status" = 0 ] || rc=1
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then grep -h 'OPENAI CONTRACT RESULT' "$work/logs/contract-$m.log" >>"$GITHUB_STEP_SUMMARY" || true; fi
}
if [ "$mode" = engine ] || [ "$mode" = both ]; then
  run_mode engine "http://127.0.0.1:$e1" "http://127.0.0.1:$e2" "" "http://127.0.0.1:$e1"
fi
if [ "$mode" = controller ] || [ "$mode" = both ]; then
  run_mode controller "http://127.0.0.1:$c1" "http://127.0.0.1:$c2" "http://127.0.0.1:$c0" "${e1:+http://127.0.0.1:$e1}"
fi
if [ "$rc" != 0 ]; then
  echo "--- server logs (tail)"; for f in "$work"/logs/engine-*.log "$work"/logs/controller-*.log; do
    [ -f "$f" ] && { echo "## $f"; tail -n 30 "$f"; }; done
fi
exit "$rc"
