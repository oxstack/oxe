"""Verdict controls for run_contract.judge (stdlib only; no SDK, no server)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_contract  # noqa: E402

PINNED = {'a': {'engine', 'controller'}, 'b': {'controller'}, 'c': {'engine', 'controller'}}


class Judge(unittest.TestCase):
    def test_all_pass(self):
        self.assertEqual(run_contract.judge('controller', PINNED, set(), {'a': 'pass', 'b': 'pass', 'c': 'pass'}), (True, []))
        self.assertTrue(run_contract.judge('engine', PINNED, set(), {'a': 'pass', 'b': 'skip', 'c': 'pass'})[0])

    def test_expected_failure(self):
        self.assertTrue(run_contract.judge('engine', PINNED, {'c'}, {'a': 'pass', 'b': 'skip', 'c': 'fail'})[0])
        ok, problems = run_contract.judge('engine', PINNED, {'c'}, {'a': 'pass', 'b': 'skip', 'c': 'pass'})
        self.assertFalse(ok)
        self.assertIn('now passes', problems[0])

    def test_fail_closed(self):
        for outcomes in ({'a': 'pass', 'b': 'pass'},                        # c never ran
                         {'a': 'pass', 'b': 'pass', 'c': 'pass', 'd': 'pass'},  # d is not pinned
                         {'a': 'pass', 'b': 'skip', 'c': 'pass'},           # a pinned test skipped
                         {'a': 'error', 'b': 'pass', 'c': 'pass'},
                         {'a': 'pass', 'b': 'pass', 'c': 'pass', 'setup:x': 'error'}):
            self.assertFalse(run_contract.judge('controller', PINNED, set(), outcomes)[0], outcomes)
        self.assertFalse(run_contract.judge('engine', PINNED, set(), {'a': 'pass', 'b': 'pass', 'c': 'pass'})[0])  # b ran in engine
        self.assertFalse(run_contract.judge('engine', PINNED, {'b'}, {'a': 'pass', 'b': 'skip', 'c': 'pass'})[0])  # xfail not pinned

    def test_pinned_files_agree_with_the_suite(self):
        here = Path(run_contract.__file__).resolve().parent
        pinned = run_contract.names(here / 'tests.txt')
        source = (here / 'contract.py').read_text()
        defined = {line.split('def ', 1)[1].split('(', 1)[0] for line in source.splitlines() if line.strip().startswith('def test_')}
        self.assertEqual(set(pinned), defined)
        for mode in ('engine', 'controller'):
            xfail = set(run_contract.names(here / ('xfail.%s.txt' % mode)))
            self.assertLessEqual(xfail, {n for n, m in pinned.items() if mode in m}, mode)


if __name__ == '__main__':
    unittest.main()
