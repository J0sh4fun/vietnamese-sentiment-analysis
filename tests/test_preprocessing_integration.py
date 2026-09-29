import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from src import evaluate, train
from src.preprocessor import VietnameseTextProcessor


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_training_artifact_records_restorable_preprocessing_and_split_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"review": f"{'khuấy máy tốt' if i % 2 else 'không tốt'} {i}",
                     "label": "positive" if i % 2 else "negative"} for i in range(60)]
            rows += [{"review": "!!!", "label": "positive"}, {"review": "???", "label": "negative"}]
            source = root / "original.jsonl"
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            argv = ["train.py", "--train-data", str(source), "--disable-aug", "--min-df", "1",
                    "--algorithm", "multinomial_nb", "--output-dir", str(root), "--run-name", "new"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                train.main()
            metadata = json.loads((root / "new" / "train_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["artifact_version"], 2)
            self.assertEqual(metadata["inference_input"], "raw_text")
            processor = VietnameseTextProcessor.from_config(metadata["preprocessing"])
            self.assertEqual(processor.transform(["khuấy"]), ["khuấy"])
            self.assertFalse(metadata["preprocessing"]["settings"]["tokenizer_token_normalization"])
            reports = metadata["preprocessing_by_split"]
            self.assertEqual(sum(item["input_rows"] for item in reports.values()), 62)
            self.assertEqual(sum(item["empty_rows"] for item in reports.values()), 2)
            self.assertTrue(all(item["exclusion_rate"] == 0 for item in reports.values()))
            for split in ("train", "validation", "test"):
                saved = pd.read_csv(root / "new" / f"{split}_split.csv", keep_default_na=False)
                self.assertEqual(int(saved.clean_text.eq("").sum()), reports[split]["empty_rows"])

    def test_evaluation_keeps_empty_features_and_reports_full_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sentiment_pipeline.joblib").touch()
            pd.DataFrame({"review": ["!!!", "TỐT"], "clean_text": ["stale", "stale"],
                          "label": ["negative", "positive"]}).to_csv(
                root / "test_split.csv", index=False, encoding="utf-8-sig")
            model = Mock()
            model.raw_text_column = "review"
            model.label_column = "label"
            model.artifact_version = 2
            model.classes_ = np.array(["negative", "positive"])
            model.infer.return_value = [
                {"label": "negative", "empty_after_preprocessing": True},
                {"label": "positive", "empty_after_preprocessing": False},
            ]
            with patch("sys.argv", ["evaluate.py", "--run-dir", str(root)]), \
                    patch("src.evaluate.load_model", return_value=model), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(model.infer.call_args.args[0].tolist(), ["!!!", "TỐT"])
            model.infer.assert_called_once()
            result = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
            report = result["preprocessing"]
            self.assertEqual(report["input_rows"], 2)
            self.assertEqual(report["empty_rows"], 1)
            self.assertEqual(report["empty_rate"], .5)
            self.assertEqual(report["excluded_rows"], 0)
            predictions = pd.read_csv(root / "test_predictions.csv", keep_default_na=False)
            self.assertEqual(len(predictions), 2)
            self.assertTrue(predictions.empty_after_preprocessing.iloc[0])


if __name__ == "__main__":
    unittest.main()
