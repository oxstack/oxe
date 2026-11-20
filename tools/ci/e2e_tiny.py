#!/usr/bin/env python3
"""End-to-end check of the ox suite: oxa drives the real oxe engine over HTTP (a small real model).

  e2e_tiny.py --oxe-server BIN --oxa BIN --model FILE --config tools/ci/e2e-tiny.toml --out DIR

Asserts (a) it works: oxa -p completes real turns through oxe-server's oxe engine, the engine
reports tokens, and the first turn's tool task leaves its file with the asked word, (b) the cache
stays: the second run of the same session reuses the cached prefix (cached prompt tokens > 0 in
oxa's --usage-log), (c) time to first token and turn time are inside the budgets of the config file
(warn above, fail only above the hard ceiling). Writes report.json and
report.md into --out and the markdown into the GitHub step summary. Standard library only.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
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


def probe(port, messages, max_tokens, headers=None):
    """One streaming chat request. Returns time to first content chunk, total time, usage, timings."""
    body = json.dumps({
        'messages': messages,
        'max_tokens': max_tokens, 'temperature': 0, 'stream': True,
        'stream_options': {'include_usage': True},
    })
    t0 = time.monotonic()
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    c.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json', **(headers or {})})
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


def run_oxa(oxa, prompt, cwd, env, usage_log, timeout, max_time, max_tokens, name, out, session_id='e2e-tiny'):
    # The request prefix (system prompt + tool declaration) is what a user's interactive session
    # gets: no --thinking, --no-mcp or --no-web-search, so a change to the real prompt or tool set
    # shows up in the prefix hash and the cache checks. Kept, and why:
    #  * -p: a non-interactive run is the only way to drive it from a script.
    #  * --yolo (owner decision, 2026-11-19): the container has no usable bwrap, so a sandboxed run has
    #    no bash at all, and the 0.8B model then retries `cat hello.txt` until --max-time. The container
    #    is throwaway. Yolo adds no tools and changes no prompt text.
    #  * --session-id / --usage-log: isolation, and the evidence (tokens, prefix hash).
    #  * --max-time / --max-tokens: the hang guards.
    cmd = [oxa, '-p', '--yolo', '--session-id', session_id,
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


def product_launch(oxe, model, ctx):
    """The arguments the native launch gives oxe-server: `oxe admission` is what `oxa local add` saves."""
    p = subprocess.run([oxe, 'admission', '--model', model, '--ctx', str(ctx), '--json'],
                       capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f'oxe admission failed: {p.stderr.strip()[:500]}')
    return json.loads(p.stdout)['arguments']


def start_server(server_bin, args, port, env, log_path):
    log = open(log_path, 'w')
    proc = subprocess.Popen([server_bin, *args, '--host', '127.0.0.1', '--port', str(port)],
                            env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def stop_server(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def cached_of(r):
    return ((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')


def subagent_probe(port, cfg):
    """parent request -> a child-shaped request (other system prompt, session `<parent>.s1`, background
    priority) -> parent follow-up. Returns the parent follow-up's cached tokens over the previous parent
    prompt: the share of the parent's context the engine still had after the child ran."""
    pc = cfg['probe']
    user = lambda t: {'role': 'user', 'content': t}
    system = lambda t: {'role': 'system', 'content': t}
    parent_h = {'X-Oxe-Session': 'guard-parent'}
    child_h = {'X-Oxe-Session': 'guard-parent.s1', 'X-Oxe-Priority': 'background'}
    body = 'the quick brown fox ' * pc['guard_repeat']
    m1 = [system('You are the parent coding agent. Use tools when needed.'), user(body)]
    r1 = probe(port, m1, pc['max_tokens'], parent_h)
    m2 = m1 + [{'role': 'assistant', 'content': r1['text'] or 'ok'}, user('and once more')]
    r2 = probe(port, m2, pc['max_tokens'], parent_h)
    child = [system('You are a read-only search subagent. Report file paths only.'),
             user('Task: find the word fox. ' + 'jumps over the lazy dog ' * (pc['guard_repeat'] // 2))]
    rc = probe(port, child, pc['max_tokens'], child_h)
    m3 = m2 + [{'role': 'assistant', 'content': r2['text'] or 'ok'}, user('thanks')]
    r3 = probe(port, m3, pc['max_tokens'], parent_h)
    prev = (r2['usage'] or {}).get('prompt_tokens')
    c = cached_of(r3)
    if c is None or not prev:
        raise RuntimeError('subagent probe: the server reported no cached/prompt token counts')
    return {'parent_prompt': prev, 'child_prompt': (rc['usage'] or {}).get('prompt_tokens'),
            'parent_cached_after_child': c, 'ratio': c / prev,
            'parent_ttft_ms': r3['ttft_ms'], 'parent_prefill_ms': (r3.get('timings') or {}).get('prompt_ms')}


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
    ap.add_argument('--oxe', help='the oxe CLI: its `admission` report gives the native launch arguments')
    ap.add_argument('--model', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    a.oxa, a.oxe_server, a.model = (os.path.abspath(x) for x in (a.oxa, a.oxe_server, a.model))
    a.oxe = os.path.abspath(a.oxe) if a.oxe else None
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
    pc = cfg['probe']
    ctx = cfg['server']['ctx_size']
    bare = ['--engine', 'oxe', '--model', a.model, '--ctx-size', str(ctx)]
    if a.oxe:
        # the same arguments `oxa local add` saves for the native launch, not bare flags
        product = ['--model', a.model, *product_launch(a.oxe, a.model, ctx)]
        report['launch_arguments'] = product
        # evidence that the guard bites: the old bare launch, probed once, reported, not asserted
        proc, blog = start_server(a.oxe_server, bare, port, env, out / 'oxe-server-bare.log')
        try:
            if wait_health(port, cfg['server']['startup_timeout_s'], proc):
                report['subagent_probe_bare'] = subagent_probe(port, cfg)
            else:
                warn('the bare launch did not become healthy; its subagent probe is skipped')
        finally:
            stop_server(proc, blog)
    else:
        product = bare
        warn('no --oxe given: the server runs with bare flags and the subagent cache probe is skipped')
    server, log = start_server(a.oxe_server, product, port, env, out / 'oxe-server.log')
    try:
        if not wait_health(port, cfg['server']['startup_timeout_s'], server):
            fail('oxe-server did not become healthy (see oxe-server.log)')
            return finish(report, out)
        if a.oxe:
            g = subagent_probe(port, cfg)
            report['subagent_probe'] = g
            if g['ratio'] < cfg['cache']['min_reuse_ratio']:
                fail(f"subagent probe: the parent reused {g['ratio']:.1%} of its context after a child-shaped "
                     f"request, below {cfg['cache']['min_reuse_ratio']:.0%} (cached {g['parent_cached_after_child']} of {g['parent_prompt']})")
        b = cfg['budget']
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
        for name, cur, prev in (('follow-up turn', t2, cold), ('turn after a tool result', t3, t2)):
            prev_prompt = (prev['usage'] or {}).get('prompt_tokens')
            c = cached_of(cur)
            if c is None or not prev_prompt:
                fail(f'{name}: the server reported no cached/prompt token counts')
                continue
            check_ratio(f'probe {name}', c / prev_prompt, cfg, report)

        oenv = dict(os.environ, HOME=str(home), OXA_AGENT_DIR=str(home / 'agent'),
                    OXA_ENGINE_URL=f'http://127.0.0.1:{port}')
        o = cfg['oxa']
        runs = []
        for name, text in (('turn1', o['first_prompt']), ('turn2', o['second_prompt']), ('turn3', o['third_prompt'])):
            if name == 'turn3':
                # a fact only the file holds, so the model has to call the read tool to answer
                (work / o['read_file']).write_text(o['read_word'] + '\n')
            # one log per run: oxa writes --usage-log for its own process (a later run replaces it)
            usage_log = out / f'usage-{name}.jsonl'
            code, wall, result = run_oxa(a.oxa, text, work, oenv, usage_log, o['run_timeout_s'], o['max_time_s'], o['max_tokens'], name, out)
            recs = records(usage_log)
            runs.append({'name': name, 'exit': code, 'error_class': result.get('error_class'), 'wall_ms': wall, 'requests': recs})
            if code != 0 and result.get('error_class') not in o['accept_error_classes']:
                fail(f"{name}: oxa -p exited {code} ({result.get('error_class')}; see {name}.stderr.txt)")
        report['oxa'] = runs

        # (a) it works: the tool task left its file, and every run reached the engine and got tokens
        task = work / o['task_file']
        content = task.read_text(errors='replace') if task.is_file() else None
        report['task'] = {'file': o['task_file'], 'exists': content is not None, 'content': (content or '')[:200]}
        if content is None:
            fail(f"turn1: oxa did not create {o['task_file']} (see turn1.stdout.txt)")
        elif not re.search(rf"\b{re.escape(o['task_word'])}\b", content, re.IGNORECASE):
            fail(f"turn1: {o['task_file']} does not contain {o['task_word']!r}: {content[:80]!r}")
        for r in runs:
            ok = [q for q in r['requests'] if q.get('usage_reported') and (q.get('output') or 0) > 0]
            if not ok:
                fail(f"{r['name']}: no model call with reported usage and output tokens")
            elif (ok[0].get('prompt') or 0) <= 0:
                fail(f"{r['name']}: the engine reported no prompt tokens")
        # (b) the cache stays across the whole session (3 oxa processes, 1 session): every chat request
        # reuses the cached prefix of the previous one, i.e. at least the floor of its whole context.
        # This covers turn 2, turn 3, and the requests that follow a real tool result (turn 1 creates
        # the file with a tool; turn 3 reads it back).
        chat = [(r['name'], i, q) for r in runs for i, q in enumerate(r['requests'])
                if q.get('kind') == 'chat' and q.get('usage_reported') and q.get('prompt')]
        report['cache_requests'] = []
        for (_, _, prev), (rname, i, cur) in zip(chat, chat[1:]):
            label = f"oxa {rname} request {i + 1}" + (' (first of the run)' if i == 0 else ' (after a tool result)')
            c = cur.get('cached')
            report['cache_requests'].append({'at': label, 'prompt': cur['prompt'], 'cached': c, 'prev_prompt': prev['prompt']})
            if not c or c <= 0:
                fail(f'{label}: reused no cached prompt tokens (cached={c})')
            elif c > cur['prompt']:
                fail(f"{label}: cached {c} exceeds the prompt {cur['prompt']}")
            else:
                check_ratio(label, c / prev['prompt'], cfg, report)
        t3 = runs[2]['requests']
        if len([q for q in t3 if q.get('kind') == 'chat']) < 2:
            fail('turn3: no request followed a tool result (the model did not call a tool; see turn3.stdout.txt)')
        # (d) the prefix (system prompt + tool declaration) is byte-identical in every request of
        # the session: oxa logs the sha256 of the frozen prefix it sends, per request
        reqs = [q for r in runs for q in r['requests']]
        hashes = {q.get('prefix_sha256') for q in reqs}
        sizes = {q.get('prefix_bytes') for q in reqs}
        report['prefix'] = {'sha256': sorted(h for h in hashes if h), 'bytes': sorted(x for x in sizes if x), 'requests': len(reqs)}
        if None in hashes or None in sizes:
            fail('usage log: a request carries no prefix_sha256/prefix_bytes')
        elif len(hashes) != 1 or len(sizes) != 1:
            fail(f'the system prompt + tool declaration changed within the session: {len(hashes)} distinct hashes over {len(reqs)} requests')
        # temperature-0 determinism (warn only): the first prompt again, a fresh session, same tokens?
        # same-length directory name: the cwd is a line of the system prompt, so a longer path alone shifts the prompt tokens
        det_work = out / 'dets'
        det_work.mkdir(exist_ok=True)
        d_log = out / 'usage-det.jsonl'
        d_code, d_wall, _ = run_oxa(a.oxa, o['first_prompt'], det_work, oenv, d_log, o['run_timeout_s'], o['max_time_s'], o['max_tokens'], 'det', out, session_id='e2e-det')
        path = lambda rs: [(q.get('kind'), q.get('prompt'), q.get('output')) for q in rs if q.get('kind') == 'chat']
        p1, p2 = path(runs[0]['requests']), path(records(d_log))
        same = d_code == 0 and p1 == p2
        report['determinism'] = {'match': same, 'first_run': p1, 'repeat': p2}
        if not same:
            warn(f'temperature-0 determinism: the repeated first prompt took another token path: {p1} vs {p2}')
        # (c) latency
        for r in runs:
            check_budget(f"{r['name']} total time", r['wall_ms'], b['turn_warn_ms'], b['turn_fail_ms'])
    except subprocess.TimeoutExpired as e:
        fail(f'oxa run timed out: {e}')
    finally:
        stop_server(server, log)
    return finish(report, out)


def finish(report, out):
    report['warnings'] = warnings
    report['failures'] = failures
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    md = ['## e2e (oxa + oxe, real model)', '',
          f"model `{report.get('model_file')}`, {report.get('model_bytes')} bytes, sha256 `{report.get('model_sha256')}`", '']
    pr = report.get('probe')
    if pr:
        md += ['| probe (direct, streaming) | TTFT ms | total ms | cached |', '|---|---|---|---|']
        for k in ('cold', 'warm'):
            u = (pr[k].get('usage') or {}).get('prompt_tokens_details') or {}
            md.append(f"| {k} | {pr[k]['ttft_ms']:.0f} | {pr[k]['total_ms']:.0f} | {u.get('cached_tokens', 'n/a')} |")
        md.append('')
    for key, label in (('subagent_probe_bare', 'old bare launch'), ('subagent_probe', 'native launch (oxe admission arguments)')):
        g = report.get(key)
        if g:
            md.append(f"- parent -> child -> parent probe, {label}: parent cached {g['parent_cached_after_child']} of "
                      f"{g['parent_prompt']} tokens ({g['ratio']:.1%}), time to first token {g['parent_ttft_ms']:.0f} ms")
    px = report.get('prefix')
    if px:
        md += [f"- prefix (system prompt + tool declaration): **{', '.join(map(str, px['bytes'])) or 'n/a'} bytes**, "
               f"sha256 `{', '.join(px['sha256']) or 'n/a'}`, over {px['requests']} requests"]
    for c in report.get('cache_requests', []):
        md.append(f"- cache {c['at']}: cached {c['cached']} of prompt {c['prompt']} (previous prompt {c['prev_prompt']})")
    d = report.get('determinism')
    if d:
        md.append(f"- temperature-0 determinism (first prompt run twice): {'token paths match' if d['match'] else 'token paths DIFFER'}")
    t = report.get('task')
    if t:
        md += [f"- tool task: {t['file']} {'created' if t['exists'] else 'MISSING'}, content {t['content']!r}"]
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
