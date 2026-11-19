#!/usr/bin/env python3
"""End-to-end check of the ox suite: oxa drives the real oxe engine over HTTP (tiny synthetic model).

  e2e_tiny.py --oxe-server BIN --oxa BIN --model FILE --config tools/ci/e2e-tiny.toml --out DIR

Asserts (a) it works: oxa -p completes real turns through oxe-server's oxe engine and the engine
reports tokens, (b) the cache stays: the second run of the same session reuses the cached prefix
(cached prompt tokens > 0 in oxa's --usage-log), (c) time to first token and turn time are inside the
budgets of the config file (warn above, fail only above the hard ceiling). Writes report.json and
report.md into --out and the markdown into the GitHub step summary. Standard library only.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

failures = []
warnings = []


def fail(msg):
    failures.append(msg)
    print(f'::error::{msg}', flush=True)


def warn(msg):
    warnings.append(msg)
    print(f'::warning::{msg}', flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def wait_health(port, timeout, proc):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
            c.request('GET', '/health')
            ok = c.getresponse().status == 200
            c.close()
            if ok:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def probe(port, messages, max_tokens):
    """One streaming chat request. Returns time to first content chunk, total time, usage, timings."""
    body = json.dumps({
        'messages': messages,
        'max_tokens': max_tokens, 'temperature': 0, 'stream': True,
        'stream_options': {'include_usage': True},
    })
    t0 = time.monotonic()
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    c.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f'probe: HTTP {r.status}: {r.read()[:300]!r}')
    first = first_any = None
    text = ''
    usage = timings = None
    for raw in r:
        line = raw.decode('utf-8', 'replace').strip()
        if not line.startswith('data:') or line == 'data: [DONE]':
            continue
        now = time.monotonic()
        frame = json.loads(line[5:])
        first_any = first_any if first_any is not None else now
        for ch in frame.get('choices') or []:
            d = ch.get('delta') or {}
            text += d.get('content') or ''
            if first is None and (d.get('content') or d.get('reasoning_content')):
                first = now
        usage = frame.get('usage') or usage
        timings = frame.get('timings') or timings
    total = time.monotonic() - t0
    c.close()
    t_first = (first if first is not None else first_any)
    if t_first is None:
        raise RuntimeError('probe: the stream held no data frame')
    return {'ttft_ms': (t_first - t0) * 1e3, 'total_ms': total * 1e3, 'usage': usage, 'timings': timings, 'text': text}


def run_oxa(oxa, prompt, cwd, env, usage_log, timeout, max_time, max_tokens, name, out):
    cmd = [oxa, '-p', '--thinking', 'off', '--no-mcp', '--no-web-search', '--session-id', 'e2e-tiny',
           '--usage-log', str(usage_log), '--max-time', str(max_time), '--max-tokens', str(max_tokens), prompt]
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    wall = (time.monotonic() - t0) * 1e3
    (out / f'{name}.stdout.txt').write_text(p.stdout)
    (out / f'{name}.stderr.txt').write_text(p.stderr)
    result = {}
    for line in p.stderr.splitlines():
        if line.startswith('{"type":"oxa_result"'):
            result = json.loads(line)
    return p.returncode, wall, result


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_oxa_with_usage(oxa, prompt, cwd, env, usage_log, timeout, max_time, max_tokens, name, out):
    """Run one invocation and return the records from its freshly replaced usage log."""
    code, wall, result = run_oxa(
        oxa, prompt, cwd, env, usage_log, timeout, max_time, max_tokens, name, out
    )
    return code, wall, result, records(usage_log)


def check_ratio(name, ratio, cfg, report):
    """Reuse ratio = cached tokens / previous prompt tokens. Floor and per-model baseline from the config."""
    c = cfg['cache']
    report.setdefault('reuse_ratio', {})[name] = ratio
    if ratio < c['min_reuse_ratio']:
        fail(f"{name}: reused {ratio:.1%} of the previous context, below {c['min_reuse_ratio']:.0%}")
    base = (c.get('baseline') or {}).get(report['model_file'])
    if base is not None:
        drop = base - ratio
        if drop > c['baseline_fail_drop']:
            fail(f"{name}: reuse {ratio:.1%} is {drop * 100:.1f} points below the baseline {base:.1%}")
        elif drop > c['baseline_warn_drop']:
            warn(f"{name}: reuse {ratio:.1%} is {drop * 100:.1f} points below the baseline {base:.1%}")


def check_budget(name, value, warn_ms, fail_ms):
    if value > fail_ms:
        fail(f'{name} {value:.0f} ms is above the hard ceiling {fail_ms} ms')
    elif value > warn_ms:
        warn(f'{name} {value:.0f} ms is above the budget {warn_ms} ms')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oxe-server', required=True)
    ap.add_argument('--oxa', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    a.oxa, a.oxe_server, a.model = (os.path.abspath(x) for x in (a.oxa, a.oxe_server, a.model))
    cfg = tomllib.loads(Path(a.config).read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    home = out / 'home'
    work.mkdir(exist_ok=True)
    home.mkdir(exist_ok=True)
    report = {'model_file': os.path.basename(a.model), 'model_bytes': os.path.getsize(a.model)}

    sha = sha256_of(a.model)
    report['model_sha256'] = sha
    pin = cfg['model']['sha256']
    if not pin:
        warn(f'model sha256 is not pinned in the config; the file is {sha}')
    elif pin != sha:
        fail(f'model sha256 {sha} differs from the pin {pin}')
        return finish(report, out)

    port = cfg['server']['port']
    env = dict(os.environ, HOME=str(home), OXE_MODEL_SHA256=sha)
    log = open(out / 'oxe-server.log', 'w')
    server = subprocess.Popen(
        [a.oxe_server, '--engine', 'oxe', '--model', a.model, '--host', '127.0.0.1', '--port', str(port),
         '--ctx-size', str(cfg['server']['ctx_size'])],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_health(port, cfg['server']['startup_timeout_s'], server):
            fail('oxe-server did not become healthy (see oxe-server.log)')
            return finish(report, out)
        b = cfg['budget']
        pc = cfg['probe']
        prompt = 'hello world ' * pc['prompt_repeat']
        user = lambda t: {'role': 'user', 'content': t}
        cold = probe(port, [user(prompt)], pc['max_tokens'])
        warm = probe(port, [user(prompt)], pc['max_tokens'])
        # a conversation that grows: a follow-up turn, then a tool call + tool result + follow-up
        m2 = [user(prompt), {'role': 'assistant', 'content': cold['text'] or 'ok'}, user('and once more')]
        t2 = probe(port, m2, pc['max_tokens'])
        m3 = m2 + [{'role': 'assistant', 'content': t2['text'] or None, 'tool_calls': [
                      {'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"a.txt"}'}}]},
                   {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'file contents ' * 40}, user('thanks')]
        t3 = probe(port, m3, pc['max_tokens'])
        report['probe'] = {'cold': cold, 'warm': warm, 'follow_up': t2, 'after_tool_result': t3}
        check_budget('cold time to first token', cold['ttft_ms'], b['ttft_cold_warn_ms'], b['ttft_cold_fail_ms'])
        check_budget('warm time to first token', warm['ttft_ms'], b['ttft_warm_warn_ms'], b['ttft_warm_fail_ms'])
        # prefill time: the warm (cached) request must be a small fraction of the cold one
        def prefill_ms(r):
            t = r.get('timings') or {}
            return t.get('prompt_ms') if isinstance(t.get('prompt_ms'), (int, float)) else r['ttft_ms']
        frac = prefill_ms(warm) / max(prefill_ms(cold), 1e-9)
        report['warm_over_cold_prefill'] = frac
        if frac > b['warm_over_cold_prefill_fail']:
            fail(f"warm prefill is {frac:.2f} of cold, above {b['warm_over_cold_prefill_fail']}")
        elif frac > b['warm_over_cold_prefill_warn']:
            warn(f"warm prefill is {frac:.2f} of cold, above {b['warm_over_cold_prefill_warn']}")
        # growing conversation: cached must cover the whole previous prompt (the new turn is the rest)
        def cached_of(r):
            return ((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
        for name, cur, prev in (('follow-up turn', t2, cold), ('turn after a tool result', t3, t2)):
            prev_prompt = (prev['usage'] or {}).get('prompt_tokens')
            c = cached_of(cur)
            if c is None or not prev_prompt:
                fail(f'{name}: the server reported no cached/prompt token counts')
                continue
            check_ratio(f'probe {name}', c / prev_prompt, cfg, report)

        oenv = dict(os.environ, HOME=str(home), OXA_AGENT_DIR=str(home / 'agent'),
                    OXA_ENGINE_URL=f'http://127.0.0.1:{port}')
        usage_log = out / 'usage.jsonl'
        o = cfg['oxa']
        runs = []
        for name, text in (('turn1', o['first_prompt']), ('turn2', o['second_prompt'])):
            code, wall, result, recs = run_oxa_with_usage(
                a.oxa, text, work, oenv, usage_log, o['run_timeout_s'], o['max_time_s'],
                o['max_tokens'], name, out
            )
            runs.append({'name': name, 'exit': code, 'error_class': result.get('error_class'), 'wall_ms': wall, 'requests': recs})
            # the synthetic model has filler weights and never stops: a turn that ends on the output
            # limit or on the token budget still went through the whole oxa -> oxe path
            if code != 0 and result.get('error_class') not in o['accept_error_classes']:
                fail(f"{name}: oxa -p exited {code} ({result.get('error_class')}; see {name}.stderr.txt)")
        report['oxa'] = runs

        # (a) it works: both runs reached the engine and got tokens back
        for r in runs:
            ok = [q for q in r['requests'] if q.get('usage_reported') and (q.get('output') or 0) > 0]
            if not ok:
                fail(f"{r['name']}: no model call with reported usage and output tokens")
            elif (ok[0].get('prompt') or 0) <= 0:
                fail(f"{r['name']}: the engine reported no prompt tokens")
        # (b) the cache stays: the second run reuses the prefix of the first
        second = runs[1]['requests']
        cached = second[0].get('cached') if second else None
        prev_prompt = next((q.get('prompt') for q in reversed(runs[0]['requests']) if q.get('prompt')), None)
        report['second_run_first_request_cached'] = cached
        if not cached or cached <= 0:
            fail(f'turn2: the first request of the second run reused no cached prompt tokens (cached={cached})')
        elif second[0].get('prompt') and cached > second[0]['prompt']:
            fail(f"turn2: cached {cached} exceeds the prompt {second[0]['prompt']}")
        elif prev_prompt:
            # the cached prefix must cover the whole context the previous run ended with
            check_ratio('oxa turn2', cached / prev_prompt, cfg, report)
        # (c) latency
        for r in runs:
            check_budget(f"{r['name']} total time", r['wall_ms'], b['turn_warn_ms'], b['turn_fail_ms'])
    except subprocess.TimeoutExpired as e:
        fail(f'oxa run timed out: {e}')
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    return finish(report, out)


def finish(report, out):
    report['warnings'] = warnings
    report['failures'] = failures
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    md = ['## e2e (oxa + oxe, tiny synthetic model)', '',
          f"model `{report.get('model_file')}`, {report.get('model_bytes')} bytes, sha256 `{report.get('model_sha256')}`", '']
    pr = report.get('probe')
    if pr:
        md += ['| probe (direct, streaming) | TTFT ms | total ms | cached |', '|---|---|---|---|']
        for k in ('cold', 'warm'):
            u = (pr[k].get('usage') or {}).get('prompt_tokens_details') or {}
            md.append(f"| {k} | {pr[k]['ttft_ms']:.0f} | {pr[k]['total_ms']:.0f} | {u.get('cached_tokens', 'n/a')} |")
        md.append('')
    for r in report.get('oxa', []):
        md.append(f"- `oxa -p` {r['name']}: exit {r['exit']}, {r['wall_ms']:.0f} ms, requests "
                  + json.dumps([{k: q.get(k) for k in ('prompt', 'cached', 'output', 'duration_ms')} for q in r['requests']]))
    md += [''] + [f'- WARN {w}' for w in warnings] + [f'- FAIL {f}' for f in failures]
    text = '\n'.join(md) + '\n'
    (out / 'report.md').write_text(text)
    print(text)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as f:
            f.write(text)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
