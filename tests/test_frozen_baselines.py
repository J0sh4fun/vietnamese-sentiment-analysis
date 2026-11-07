from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import f1_score

from src.experiments import file_fingerprint
from src.frozen_baselines import BASELINE_ID, NEW_COHORT, new_output, text_hash, verify_bundle, write_json
from src.evaluate_baseline_v2 import (evaluate, paired_product_bootstrap, validate_inputs, validate_release)


class FrozenBaselineTests(unittest.TestCase):
    def test_manifest_detects_modified_file_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "model.bin"
            payload.write_bytes(b"frozen weights")
            manifest = {"schema_version": 1, "baseline_id": BASELINE_ID,
                        "files": {"model.bin": file_fingerprint(payload)}}
            write_json(root / "manifest.json", manifest)
            self.assertEqual(verify_bundle(root)["baseline_id"], BASELINE_ID)
            payload.write_bytes(b"modified weights")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                verify_bundle(root)
            from json import dumps
            manifest["files"] = {"../outside": {}}
            (root / "manifest.json").write_text(dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes bundle"):
                verify_bundle(root)

    def test_evaluation_cannot_write_inside_frozen_bundle_or_overwrite_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            with self.assertRaisesRegex(ValueError, "immutable"):
                new_output(bundle / "new_results", bundle)
            output = new_output(root / "evaluation", bundle)
            with self.assertRaises(FileExistsError):
                new_output(output, bundle)

    def test_not_ready_refuses_before_opening_test_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            release = root / "release.json"
            write_json(release, {"status": "NOT_READY"})
            with patch("src.evaluate_baseline_v2.verify_bundle", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "NOT_READY"):
                    evaluate(root / "no_bundle", release, root / "no_output")
            self.assertFalse((root / "no_output").exists())

    def test_v2_time_roles_and_all_gates_are_required(self):
        release = {"status": "SEALED_READY", "baseline_id": BASELINE_ID, "cohort": NEW_COHORT,
                   "opened_at": "2026-12-06T00:00:00+07:00", "custodian": "custodian",
                   "evaluation_operator": "operator", "annotation_receipt": "audit/annotations",
                   "contamination_audit_receipt": "audit/overlap", "gates": {f"G{i}": True for i in range(1, 8)}}
        with self.assertRaisesRegex(ValueError, "cannot open before"):
            validate_release(release, now=datetime.fromisoformat("2026-11-07T12:00:00+07:00"))
        now = datetime.fromisoformat("2026-12-07T00:00:00+07:00")
        validate_release(release, now=now)
        release["gates"]["G3"] = False
        with self.assertRaisesRegex(ValueError, "G1-G7"):
            validate_release(release, now=now)
        release["gates"]["G3"] = True
        release["evaluation_operator"] = release["custodian"]
        with self.assertRaisesRegex(ValueError, "separate roles"):
            validate_release(release, now=now)

    def test_v2_input_checks_products_dates_and_old_data_overlap(self):
        rows = [{"source_dataset": "artificial-fixture", "review_id": f"new-{i}",
                 "product_key": f"product-{i//5}", "review_created_at": "2026-11-09T12:00:00+07:00",
                 "collected_at": "2026-12-06T00:00:00+07:00", "source_reference": "fixture-only",
                 "review": f"artificial never collected sample {i}"} for i in range(1000)]
        registry = {"raw_text_hashes": [], "legacy_review_ids": []}
        validate_inputs(rows, registry)
        registry["raw_text_hashes"] = [text_hash(rows[0]["review"])]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_inputs(rows, registry)
        registry["raw_text_hashes"] = []
        rows[0]["review_created_at"] = "2026-03-01T00:00:00+07:00"
        with self.assertRaisesRegex(ValueError, "outside prospective"):
            validate_inputs(rows, registry)
        rows[0]["review_created_at"] = "2026-11-09T12:00:00+07:00"
        rows[0]["product_key"] = "product-1"
        with self.assertRaisesRegex(ValueError, "products"):
            validate_inputs(rows, registry)

    def test_paired_cluster_bootstrap_matches_explicit_review_resampling(self):
        truth = ["negative", "negative", "positive", "negative", "positive", "positive"]
        pred = {"ComplementNB": ["negative"] * 6, "MLP": truth.copy()}
        products = ["a", "a", "b", "c", "c", "c"]
        result = paired_product_bootstrap(truth, pred, products, repeats=40, seed=42)
        self.assertEqual(result, paired_product_bootstrap(truth, pred, products, repeats=40, seed=42))
        rng = np.random.default_rng(42)
        expected = []
        for _ in range(40):
            keys = ["a", "b", "c"]
            indices = [i for g in rng.integers(0, 3, size=3) for i, p in enumerate(products) if p == keys[g]]
            scores = [f1_score([truth[i] for i in indices], [pred[name][i] for i in indices],
                              labels=["negative", "positive"], average="macro", zero_division=0)
                      for name in ("ComplementNB", "MLP")]
            expected.append([*scores, scores[1] - scores[0]])
        np.testing.assert_allclose(result["resamples"], expected, atol=1e-14)
        np.testing.assert_allclose(result["ci95"]["delta_mlp_minus_nb"],
                                   np.percentile(np.asarray(expected)[:, 2], [2.5, 97.5], method="linear"))


if __name__ == "__main__":
    unittest.main()
