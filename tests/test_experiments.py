import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import evaluate, train
from src.data_pipeline import read_jsonl
from src.experiments import (code_snapshot, file_fingerprint, seed_everything,
                             validate_run_name, validate_training_args)
from src.inference import load_model


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "original.jsonl"
        rows = [{"review": f"sản phẩm {'tốt' if i % 2 else 'hỏng'} {i}",
                 "label": "positive" if i % 2 else "negative"} for i in range(60)]
        self.data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def args(self, **overrides):
        with patch("sys.argv", ["train.py"]):
            args = train.parse_args()
        args.train_data, args.output_dir = self.data, self.root
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def run_training(self, name, *extra):
        argv = ["train.py", "--train-data", str(self.data), "--output-dir", str(self.root),
                "--run-name", name, "--algorithm", "multinomial_nb", "--min-df", "1",
                "--random-state", "123", *extra]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        run = self.root / name
        return run, json.loads((run / "train_metadata.json").read_text(encoding="utf-8"))

    def test_complete_metadata_and_relocated_artifact(self):
        run, metadata = self.run_training("first", "--generate-aug")
        self.assertEqual(metadata["metadata_schema_version"], 2)
        self.assertEqual(metadata["effective_config"]["algorithms"], ["multinomial_nb"])
        self.assertEqual(metadata["random_seed"], 123)
        self.assertEqual(metadata["environment"]["python"], platform.python_version())
        for package in ("numpy", "pandas", "scikit-learn", "underthesea", "joblib"):
            self.assertEqual(metadata["environment"]["packages"][package], importlib.metadata.version(package))
        self.assertIn("revision", metadata["code"])
        self.assertIn("src/train.py", metadata["code"]["file_fingerprints"])
        self.assertEqual(metadata["data_sources"][0]["sha256"], file_fingerprint(self.data)["sha256"])
        self.assertEqual(metadata["data_sources"][0]["role"], "original")
        self.assertTrue(metadata["data_sources"][0]["relocation_required"])
        self.assertNotIn(str(self.root), json.dumps(metadata))
        self.assertEqual(metadata["preprocessing"]["settings"]["unicode_normalization"], "NFC")
        self.assertEqual(metadata["model_path"], "sentiment_pipeline.joblib")
        self.assertEqual(metadata["augmentation_counts"]["generated_candidates"], 48)
        self.assertEqual(metadata["augmentation_counts"]["accepted_training_rows"], 48)
        for name, filename in metadata["split_paths"].items():
            split = pd.read_csv(run / filename)
            self.assertEqual(metadata["split_label_distributions"][name], split.label.value_counts().to_dict())
        for filename, fingerprint in metadata["artifact_fingerprints"].items():
            self.assertEqual(fingerprint, file_fingerprint(run / filename))
        pins = (run / "requirements-resolved.txt").read_text(encoding="utf-8")
        self.assertIn(f"pandas=={importlib.metadata.version('pandas')}", pins)
        moved = self.root / "relocated"
        shutil.copytree(run, moved)
        original = load_model(run / metadata["model_path"])
        restored = load_model(moved / metadata["model_path"])
        self.assertEqual(original.infer("tốt"), restored.infer("tốt"))
        with patch("sys.argv", ["evaluate.py", "--run-dir", str(moved)]), contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
        evaluation = json.loads((moved / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(evaluation["metrics"]["labels"], metadata["labels"])
        self.assertEqual(evaluation["model_path"]["path"], "sentiment_pipeline.joblib")

    def test_effective_disabled_flags_and_same_seed_reproduce_splits(self):
        first, a = self.run_training("a", "--disable-aug", "--generate-aug")
        second, b = self.run_training("b", "--disable-aug", "--generate-aug")
        self.assertTrue(a["requested_config"]["generate_aug"])
        self.assertFalse(a["effective_config"]["generate_aug"])
        self.assertEqual(a["effective_config"]["aug_data"], [])
        self.assertEqual(a["augmentation_counts"]["accepted_training_rows"], 0)
        for filename in a["split_paths"].values():
            self.assertEqual(file_fingerprint(first / filename), file_fingerprint(second / filename))
        self.assertEqual(load_model(first / a["model_path"]).infer(["tốt", "hỏng"]),
                         load_model(second / b["model_path"]).infer(["tốt", "hỏng"]))

    def test_overwrite_protection_precedes_training_and_preserves_all_files(self):
        run, _ = self.run_training("existing", "--disable-aug")
        before = {p.name: file_fingerprint(p) for p in run.iterdir()}
        with patch("src.train.prepare_and_split_dataset") as prepare:
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                self.run_training("existing")
            prepare.assert_not_called()
        self.assertEqual(before, {p.name: file_fingerprint(p) for p in run.iterdir()})
        argv = ["evaluate.py", "--run-dir", str(run)]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
        before = {p.name: file_fingerprint(p) for p in run.iterdir()}
        with patch("sys.argv", argv), self.assertRaisesRegex(FileExistsError, "fresh --output-dir"):
            evaluate.main()
        self.assertEqual(before, {p.name: file_fingerprint(p) for p in run.iterdir()})
        with patch("sys.argv", argv + ["--output-dir", str(self.root / "new-evaluation")]), \
                contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
        self.assertEqual(before, {p.name: file_fingerprint(p) for p in run.iterdir()})

    def test_hash_is_of_exact_bytes_consumed_including_bom_and_order(self):
        content = b'\xef\xbb\xbf{"review":"a","label":"positive"}\r\n\n{"review":"b","label":"negative"}\n'
        self.data.write_bytes(content)
        frame = read_jsonl(self.data)
        self.assertEqual(frame.attrs["input_fingerprint"]["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(frame.attrs["input_fingerprint"]["size_bytes"], len(content))
        self.data.write_bytes(content + b"\n")
        self.assertNotEqual(frame.attrs["input_fingerprint"]["sha256"], file_fingerprint(self.data)["sha256"])

    def test_label_order_and_matrix_axes_are_saved_together(self):
        metrics = evaluate.evaluate_predictions(pd.Series(["positive", "positive", "negative"]),
            pd.Series(["positive", "negative", "negative"]), ["positive", "negative"])
        self.assertEqual(metrics["labels"], ["positive", "negative"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertEqual(metrics["confusion_matrix_axes"]["rows"], "true labels")

    def test_git_unavailable_is_explicit(self):
        with patch("src.experiments.subprocess.run", side_effect=FileNotFoundError):
            snapshot = code_snapshot(self.root)
        self.assertIsNone(snapshot["revision"])
        self.assertIsNone(snapshot["dirty"])

    def test_seeds_are_repeated_and_forwarded_to_logistic_regression(self):
        seed_everything(123)
        before = (random.random(), np.random.random())
        seed_everything(123)
        self.assertEqual(before, (random.random(), np.random.random()))
        self.assertEqual(train.build_classifier(self.args(random_state=123), 2, "logreg").random_state, 123)

    def test_invalid_parameters_have_readable_errors(self):
        for overrides, message in [
            ({"test_size": float("nan")}, "test-size"), ({"val_size": float("inf")}, "val-size"),
            ({"test_size": .5, "val_size": .5}, "less than 1"),
            ({"random_state": -1}, "random-state"), ({"random_state": 2**32}, "random-state"),
            ({"min_df": 0}, "min-df"), ({"max_features": -1}, "max-features"),
            ({"ngram_min": 3, "ngram_max": 1}, "ngram-max"),
            ({"regularization": float("nan")}, "regularization"), ({"nb_alpha": 0}, "nb-alpha"),
            ({"max_samples": 0}, "max-samples"), ({"max_iter": 0}, "max-iter"),
            ({"allowed_labels": ["positive"]}, "allowed-labels"), ({"text_column": " "}, "blank"),
        ]:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                validate_training_args(self.args(**overrides))
        for name in ("../escape", "a/b", "a\\b", "CON", "NUL.txt", "bad:", "trailing.", " "):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "portable directory name"):
                validate_run_name(name)


if __name__ == "__main__":
    unittest.main()
