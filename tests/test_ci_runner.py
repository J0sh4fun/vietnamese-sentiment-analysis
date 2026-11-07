import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.run_tests import BASELINE_ALLOWED_SKIPS, run_suite


class CIRunnerTests(unittest.TestCase):
    def execute(self, suite, profile="full"):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "report.json"
            code = run_suite(suite, profile=profile, report=report, stream=io.StringIO())
            return code, json.loads(report.read_text(encoding="utf-8"))

    def test_empty_discovery_is_a_failure_instead_of_green(self):
        code, report = self.execute(unittest.TestSuite())
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")

    def test_real_assertion_failure_propagates_nonzero_status(self):
        class Failing(unittest.TestCase):
            def runTest(self):
                self.assertEqual(1, 2)
        code, report = self.execute(unittest.TestSuite([Failing()]))
        self.assertEqual(code, 1)
        self.assertEqual(len(report["failures"]), 1)

    def test_only_exact_optional_skips_are_allowed_in_baseline(self):
        class Passing(unittest.TestCase):
            def runTest(self):
                pass
        class Skipped(unittest.TestCase):
            def runTest(self):
                self.skipTest("missing optional asset")
        allowed = next(iter(BASELINE_ALLOWED_SKIPS))
        for profile, test_id, expected in (("baseline", allowed, 0), ("full", allowed, 1),
                                            ("baseline", "unknown.optional.test", 1)):
            case = Skipped()
            case.id = lambda test_id=test_id: test_id
            code, _ = self.execute(unittest.TestSuite([Passing(), case]), profile=profile)
            self.assertEqual(code, expected)
        only_skipped = Skipped()
        only_skipped.id = lambda: allowed
        self.assertEqual(self.execute(unittest.TestSuite([only_skipped]), profile="baseline")[0], 1)
