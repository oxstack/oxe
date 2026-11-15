#!/usr/bin/env python3
"""Compare the engine tests cargo-nextest lists with the ones `cargo test` lists, so none silently disappears.

usage: engine_test_count.py LIBTEST_LIST NEXTEST_JSON
  LIBTEST_LIST  stdout of `cargo test <same args> -- --list --format terse` (unit, integration and doc tests)
  NEXTEST_JSON  stdout of `cargo nextest list <same args> --message-format json`

Doc tests (names `path - item (line N)`) are not nextest's; they run separately with `cargo test --doc`
and are counted apart. Every other name must appear in nextest exactly as often as in libtest's list
(names repeat across binaries, so this compares multisets, ignored tests included).
"""
from collections import Counter
import json
import re
import sys

DOCTEST = re.compile(r' - .* \(line \d+\)$')


def compare(libtest_text, nextest_json):
    """Return (libtest unit+integration count, doc test count, nextest count); raise ValueError on a mismatch."""
    names = [line[:-len(': test')] for line in libtest_text.splitlines() if line.endswith(': test')]
    doc = [n for n in names if DOCTEST.search(n)]
    unit = Counter(n for n in names if not DOCTEST.search(n))
    nextest = Counter(name for suite in json.loads(nextest_json)['rust-suites'].values()
                      for name in suite.get('testcases', {}))
    if not unit:
        raise ValueError('cargo test listed no tests')
    missing, extra = unit - nextest, nextest - unit
    if missing or extra:
        raise ValueError(f'nextest misses {sorted(missing)[:20]}; nextest extra {sorted(extra)[:20]}')
    return sum(unit.values()), len(doc), sum(nextest.values())


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with open(sys.argv[1], encoding='utf-8') as a, open(sys.argv[2], encoding='utf-8') as b:
        try:
            unit, doc, nextest = compare(a.read(), b.read())
        except ValueError as e:
            raise SystemExit(f'engine test count: {e}')
    print(f'engine tests before (cargo test --list): {unit} unit/integration + {doc} doc tests')
    print(f'engine tests after (cargo nextest list): {nextest} unit/integration; doc tests via cargo test --doc: {doc}')


if __name__ == '__main__':
    main()
