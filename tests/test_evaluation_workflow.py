import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import run_evaluation as workflow
from src.data_pipeline import assert_no_overlap, prepare_dataset
from src.inference import SentimentModel, load_model
from src.preprocessor import VietnameseTextProcessor


class EvaluationWorkflowTests(unittest.TestCase):
    def fixture(self, root):
        path = root / "data.jsonl"
        rows = [{"review": f"{'máy đẹp tốt' if i % 2 else 'hỏng tệ lỗi'} mãriêng{i}",
                 "label": "positive" if i % 2 else "negative"} for i in range(80)]
        rows += [{"review": "!!!", "label": "negative"}, {"review": "???", "label": "positive"}]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return workflow.parse_args(["--train-data", str(path), "--output-dir", str(root),
            "--run-name", "new", "--min-df", "1", "--timing-repeats", "1"])

    def test_segmentation_switch_and_explicit_v2_compatibility(self):
        processor = VietnameseTextProcessor(word_segmentation=False, remove_stopwords=False)
        with patch("src.preprocessor.word_tokenize", side_effect=AssertionError("must not segment")):
            self.assertEqual(processor.transform(["SẢN PHẨM không tốt", "khuấy_đều"]),
                             ["sản phẩm không tốt", "khuấy_đều"])
        restored = VietnameseTextProcessor.from_config(processor.to_config())
        self.assertFalse(restored.word_segmentation)
        legacy_config = VietnameseTextProcessor().to_config()
        legacy_config["implementation_version"] = 2
        del legacy_config["settings"]["word_segmentation"]
        legacy = VietnameseTextProcessor.from_config(legacy_config)
        self.assertTrue(legacy.word_segmentation)
        texts = ["khuấy đều", "không tốt", "hoà", "https://example.com"]
        self.assertEqual(legacy.transform(texts), VietnameseTextProcessor().transform(texts))
        tampered = copy.deepcopy(legacy_config)
        tampered["settings"]["word_segmentation"] = False
        with self.assertRaises(ValueError):
            VietnameseTextProcessor.from_config(tampered)

    def test_ablation_cohort_and_augmentation_source_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            splits = prepare_dataset(args, VietnameseTextProcessor())
            for config in workflow.CONFIGURATIONS:
                processor = VietnameseTextProcessor(word_segmentation=config["word_segmentation"],
                    remove_stopwords=config["remove_stopwords"])
                changed = workflow.representation(splits, processor, "review")
                for before, after in zip(splits, changed):
                    self.assertEqual(before.source_id.tolist(), after.source_id.tolist())
                    self.assertEqual(before.label.tolist(), after.label.tolist())
                self.assertFalse(processor.tone_repositioning)
            args.generate_aug = True
            augmented = prepare_dataset(args, VietnameseTextProcessor())
            self.assertTrue(augmented[0].is_augmented.any())
            self.assertTrue(set(augmented[0].loc[augmented[0].is_augmented, "source_id"]) <= set(splits[0].source_id))
            for before, after in zip(splits, augmented):
                self.assertEqual(before.source_id.tolist(), after.loc[~after.is_augmented].source_id.tolist())
            assert_no_overlap(*augmented)

    def test_overlap_in_alternate_representation_fails_closed(self):
        frames = [pd.DataFrame({"review": [word], "source_id": [str(i)], "normalized_text": [word]})
                  for i, word in enumerate(["tốt!", "tốt?", "xấu"])]
        with self.assertRaisesRegex(ValueError, "Leakage"):
            workflow.representation(frames, VietnameseTextProcessor(remove_stopwords=False), "review")

    def test_selection_uses_only_validation_and_stable_tie_order(self):
        rows = [{"metrics": {"f1_macro": .7, "accuracy": .9}, "test_metrics": {"f1_macro": 1}},
                {"metrics": {"f1_macro": .8, "accuracy": .8}, "test_metrics": {"f1_macro": 0}},
                {"metrics": {"f1_macro": .8, "accuracy": .8}}]
        self.assertIs(workflow.choose_validation_result(rows), rows[1])

    def test_error_sample_is_real_reproducible_and_covers_both_directions(self):
        predictions = pd.DataFrame({"source_id": [str(i) for i in range(80)],
            "true_label": ["negative"] * 65 + ["positive"] * 15,
            "predicted_label": ["positive"] * 65 + ["negative"] * 15,
            "raw_text": [f"observed review {i}" for i in range(80)]})
        errors, sample = workflow.sample_errors(predictions, 40, 42)
        self.assertEqual(len(errors), 80)
        self.assertEqual(len(sample), 40)
        self.assertEqual(sample.true_label.value_counts().to_dict(), {"negative": 25, "positive": 15})
        self.assertTrue(set(sample.source_id) <= set(predictions.source_id))
        self.assertEqual(sample.source_id.tolist(), workflow.sample_errors(predictions, 40, 42)[1].source_id.tolist())
        _, empty = workflow.sample_errors(predictions.assign(predicted_label=predictions.true_label), 40, 42)
        self.assertTrue(empty.empty)

    def test_full_small_workflow_metadata_baseline_persistence_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            original_infer = SentimentModel.infer
            test_calls = []
            def tracked_infer(model, texts, **kwargs):
                # The workflow writes the test split only after locked selection.
                if (root / "new/test_split.csv").exists():
                    self.assertTrue((root / "new/selection.json").exists())
                    self.assertTrue((root / "new/sentiment_pipeline.joblib").exists())
                    test_calls.append(list(texts))
                return original_infer(model, texts, **kwargs)
            with patch.object(SentimentModel, "infer", tracked_infer), contextlib.redirect_stdout(io.StringIO()):
                run_dir = workflow.run(args)
            self.assertEqual(len(test_calls), 1)
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            results = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8"))
            final = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results), 16)
            self.assertEqual(len(metadata["configurations"]), 4)
            self.assertIn("sha256", metadata["selection_fingerprint_before_test"])
            for config in metadata["configurations"]:
                self.assertTrue(all(split["excluded_rows"] == 0 for split in config["splits"].values()))
            for row in results:
                self.assertGreaterEqual(row["fit_seconds"], 0)
                self.assertEqual(row["timing"]["repeats"], 1)
                if row["algorithm"] == "majority":
                    matrix = row["metrics"]["confusion_matrix"]
                    self.assertTrue(sum(column != 0 for column in matrix[0]) <= 1)
                    self.assertTrue(sum(sum(c) > 0 for c in zip(*matrix)) == 1)
            model = load_model(run_dir / "sentiment_pipeline.joblib")
            test = pd.read_csv(run_dir / "test_split.csv", keep_default_na=False)
            self.assertEqual(final["metrics"]["labels"], model.classes_.tolist())
            self.assertEqual(final["preprocessing"]["input_rows"], len(test))
            # Unique held-out tokens never enter the training TF-IDF vocabulary.
            for text in test.clean_text:
                for token in text.split():
                    if token.startswith("mãriêng"):
                        self.assertNotIn(token, model._pipeline.named_steps["tfidf"].vocabulary_)
            before = (run_dir / "metadata.json").read_bytes()
            with self.assertRaises(FileExistsError):
                workflow.run(args)
            self.assertEqual((run_dir / "metadata.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
