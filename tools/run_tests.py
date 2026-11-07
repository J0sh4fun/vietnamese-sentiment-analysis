"""Run real unittest discovery, reject empty/failed suites and enforce skip policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Only these asset/dependency-dependent tests may skip in the baseline-only job.
BASELINE_ALLOWED_SKIPS = {
    "test_benchmarking.BenchmarkTests.test_transformer_adapter_loads_local_weights_and_obeys_saved_class_order",
    *{"test_model_inputs.ModelFamilyRouteTests." + name for name in (
        "test_transformer_receives_original_case_punctuation_emoji_once_without_tfidf_processor",
        "test_exact_boundary_and_overflow_audit_preserve_one_row_per_review",
        "test_empty_punctuation_oov_and_dynamic_padding_keep_samples",
        "test_batch_token_ids_match_individual_calls_and_boundaries",
        "test_changed_profile_is_rejected",
        "test_real_token_ids_fit_xlmr_architecture_at_512_without_downloading_weights")},
}


def run_suite(suite, *, profile, report, stream=sys.stderr):
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    skipped = [{"test": test.id(), "reason": reason} for test, reason in result.skipped]
    allowed = BASELINE_ALLOWED_SKIPS if profile == "baseline" else set()
    unexpected = [row for row in skipped if row["test"] not in allowed]
    executed = result.testsRun - len(skipped)
    passed = executed > 0 and result.wasSuccessful() and not unexpected and not result.unexpectedSuccesses
    receipt = {"profile": profile, "status": "PASS" if passed else "FAIL", "tests_run": result.testsRun,
               "tests_executed": executed,
               "failures": [{"test": test.id(), "traceback": trace} for test, trace in result.failures],
               "errors": [{"test": test.id(), "traceback": trace} for test, trace in result.errors],
               "skipped": skipped, "unexpected_skips": unexpected,
               "unexpected_successes": [test.id() for test in result.unexpectedSuccesses],
               "expected_failures": [test.id() for test, _ in result.expectedFailures],
               "seconds": time.perf_counter() - started,
               "python": platform.python_version(), "platform": platform.platform(),
               "isolated_venv": sys.prefix != sys.base_prefix}
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    if result.testsRun == 0:
        print("CI ERROR: no tests discovered", file=stream)
    elif executed == 0:
        print("CI ERROR: every discovered test was skipped", file=stream)
    if unexpected:
        print("CI ERROR: unexpected skipped tests: " + ", ".join(row["test"] for row in unexpected), file=stream)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("baseline", "full"), required=True)
    parser.add_argument("--report", type=Path, required=True, help="Fresh JSON receipt path; never overwrite")
    args = parser.parse_args()
    # Running a script under tools/ does not otherwise put repository root on sys.path.
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    return run_suite(suite, profile=args.profile, report=args.report)


if __name__ == "__main__":
    raise SystemExit(main())
