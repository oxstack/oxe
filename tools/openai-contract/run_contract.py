"""Run contract.py for one mode and judge it fail-closed against tests.txt and the xfail list.

usage: python run_contract.py --mode engine|controller [--xfail FILE | --strict]

PASS only when: every test pinned for the mode ran (none skipped) and passed, except the tests in
the xfail list, which must fail; no test outside the pin list exists; the tests not pinned for the
mode skipped themselves. An expected failure that passes is a FAIL (update the list). Prints one
line 'OPENAI CONTRACT RESULT: PASS n/n (mode)' or 'OPENAI CONTRACT RESULT: FAIL ...'.
"""
import argparse
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def names(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            parts = line.split()
            out[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 else set()
    return out


def judge(mode, pinned, xfail, outcomes):
    """(ok, problems) for {test: 'pass'|'fail'|'skip'} against the pins."""
    problems = []
    expected = {name for name, modes in pinned.items() if mode in modes}
    for name in sorted(set(outcomes) - set(pinned)):
        problems.append('%s: not in tests.txt' % name)
    for name in sorted(set(pinned) - set(outcomes)):
        problems.append('%s: pinned but not executed' % name)
    for name in sorted(xfail - expected):
        problems.append('%s: in the xfail list but not pinned for mode %s' % (name, mode))
    for name in sorted(expected & set(outcomes)):
        got = outcomes[name]
        if got == 'skip':
            problems.append('%s: skipped' % name)
        elif name in xfail and got == 'pass':
            problems.append('%s: expected to fail, now passes (update the xfail list)' % name)
        elif name not in xfail and got != 'pass':
            problems.append('%s: %s' % (name, got))
    for name in sorted(set(outcomes) - expected):
        if name in pinned and outcomes[name] != 'skip':
            problems.append('%s: ran outside its modes' % name)
    return not problems, problems


class Collect(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.outcomes = {}

    def _name(self, test):
        return test.id().rsplit('.', 1)[-1]

    def addSuccess(self, test):
        super().addSuccess(test); self.outcomes[self._name(test)] = 'pass'

    def addFailure(self, test, err):
        super().addFailure(test, err); self.outcomes[self._name(test)] = 'fail'

    def addError(self, test, err):
        super().addError(test, err)
        name = self._name(test)
        self.outcomes[name if name.startswith('test_') else 'setup:' + test.id()] = 'error'

    def addSkip(self, test, reason):
        super().addSkip(test, reason); self.outcomes[self._name(test)] = 'skip'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('engine', 'controller'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--xfail')
    group.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    os.environ['OAC_MODE'] = args.mode
    pinned = names(HERE / 'tests.txt')
    xfail_file = None if args.strict else Path(args.xfail) if args.xfail else HERE / ('xfail.%s.txt' % args.mode)
    xfail = set(names(xfail_file)) if xfail_file is not None and xfail_file.exists() else set()
    sys.path.insert(0, str(HERE))
    import contract
    suite = unittest.defaultTestLoader.loadTestsFromModule(contract)
    result = unittest.TextTestRunner(resultclass=Collect, verbosity=2, stream=sys.stdout).run(suite)
    ok, problems = judge(args.mode, pinned, xfail, result.outcomes)
    expected = sum(1 for modes in pinned.values() if args.mode in modes)
    if ok:
        print('OPENAI CONTRACT RESULT: PASS %d/%d (%s, %d expected failures)' % (expected, expected, args.mode, len(xfail)))
        return 0
    print('OPENAI CONTRACT RESULT: FAIL (%s): %s' % (args.mode, '; '.join(problems)))
    return 1


if __name__ == '__main__':
    sys.exit(main())
