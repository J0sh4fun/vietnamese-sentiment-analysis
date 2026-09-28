import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src import evaluate
from src.inference import ARTIFACT_VERSION, SentimentModel, load_model
from src.preprocessor import VietnameseTextProcessor
from src.train import build_model, parse_args


ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "sentiment_pipeline.joblib"
        self.processor = VietnameseTextProcessor()
        self.raw = ["Sản phẩm tốt", "Máy khuấy tốt", "Không tốt", "Sản phẩm hỏng",
                    "Rất hài lòng", "Thất vọng", "Máy đẹp", "Máy hỏng"]
        self.labels = ["positive", "positive", "negative", "negative",
                       "positive", "negative", "positive", "negative"]
        self.probes = ["MÁY KHUẤY TỐT!!!", unicodedata.normalize("NFD", "không tốt"),
                       "https://example.com", "", "!!!", "và"]
        with patch("sys.argv", ["train.py", "--min-df", "1"]):
            self.args = parse_args()
        self.model = self.build("multinomial_nb")

    def build(self, algorithm):
        pipeline = build_model(self.args, 2, algorithm)
        pipeline.fit(self.processor.transform(self.raw), self.labels)
        return SentimentModel(pipeline, self.processor.to_config())

    def test_raw_predictions_and_probabilities_survive_save_load_for_all_algorithms(self):
        for algorithm in ("logreg", "multinomial_nb", "complement_nb"):
            with self.subTest(algorithm=algorithm):
                model = self.build(algorithm)
                cleaned = self.processor.transform(self.probes)
                expected = model._pipeline.predict(cleaned)
                scores = model._pipeline.predict_proba(cleaned)
                np.testing.assert_array_equal(model.predict(self.probes), expected)
                np.testing.assert_allclose(model.predict_proba(self.probes), scores)
                path = self.root / f"{algorithm}.joblib"
                joblib.dump(model, path)
                for restored in (load_model(path), joblib.load(path)):
                    np.testing.assert_array_equal(restored.predict(self.probes), expected)
                    np.testing.assert_allclose(restored.predict_proba(self.probes), scores)
                    self.assertEqual(restored.infer(self.probes, include_probabilities=True),
                                     model.infer(self.probes, include_probabilities=True))

    def test_each_public_call_preprocesses_exactly_once(self):
        for method, kwargs in [("predict", {}), ("predict_proba", {}),
                               ("infer", {"include_probabilities": True})]:
            with self.subTest(method=method), patch.object(self.model._processor, "transform",
                    wraps=self.model._processor.transform) as transform:
                getattr(self.model, method)(self.probes, **kwargs)
                transform.assert_called_once_with(self.probes)

    def test_invalid_inputs_fail_before_any_preprocessing(self):
        for inputs in (None, 123, b"bytes", {"review": "text"}, {"unordered"},
                       ["valid", None], [float("nan")], [False], [["nested"]],
                       pd.DataFrame({"review": ["text"]}), np.array([["text"]])):
            with self.subTest(inputs=inputs), patch.object(self.model._processor, "transform") as transform:
                with self.assertRaises(TypeError):
                    self.model.infer(inputs)
                transform.assert_not_called()

    def test_single_string_empty_batch_and_empty_reviews_are_explicit(self):
        np.testing.assert_array_equal(self.model.predict("TỐT"), self.model.predict(["TỐT"]))
        self.assertEqual(self.model.infer([]), [])
        self.assertEqual(self.model.predict([]).shape, (0,))
        self.assertEqual(self.model.predict_proba([]).shape, (0, 2))
        results = self.model.infer(["", " ", "!!!", "và"], include_probabilities=True)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(row["empty_after_preprocessing"] for row in results))
        self.assertTrue(all(row["probability_kind"] == "uncalibrated" for row in results))

    def test_probability_mapping_uses_classes_and_label_uses_predict(self):
        pipeline = self.model._pipeline
        with patch.object(type(pipeline), "classes_", new_callable=PropertyMock,
                          return_value=np.array(["positive", "negative"])), \
             patch.object(pipeline, "predict", return_value=np.array(["positive"])), \
             patch.object(pipeline, "predict_proba", return_value=np.array([[.2, .8]])):
            row = self.model.infer("review", include_probabilities=True)[0]
        self.assertEqual(row["probabilities"], {"positive": .2, "negative": .8})
        self.assertEqual(row["label"], "positive")  # UI must not invent a second decision rule.

    def test_old_and_unknown_version_artifacts_are_rejected(self):
        joblib.dump(self.model._pipeline, self.path)
        with self.assertRaisesRegex(ValueError, "Retrain"):
            load_model(self.path)
        self.model.artifact_version = 999
        future = self.root / "future.joblib"
        joblib.dump(self.model, future)
        with self.assertRaisesRegex(ValueError, "Retrain"):
            joblib.load(future)

    def test_saved_settings_are_used_even_when_default_stopwords_change(self):
        before = self.model.infer(self.probes)
        joblib.dump(self.model, self.path)
        with patch("src.preprocessor.VIETNAMESE_STOPWORDS", {"khuấy", "tốt", "hỏng"}):
            restored = load_model(self.path)
            self.assertEqual(restored.infer(self.probes), before)
        self.assertEqual(restored.preprocessing_config, self.model.preprocessing_config)

    def test_runtime_or_preprocessing_version_mismatch_is_rejected(self):
        self.model._preprocessing_config["implementation_version"] = -1
        joblib.dump(self.model, self.path)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_model(self.path)

    def test_evaluation_matches_raw_inference_and_ignores_stale_clean_text(self):
        self.model.raw_text_column = "raw_review"
        self.model.label_column = "sentiment"
        joblib.dump(self.model, self.path)
        expected = self.model.infer(self.probes)
        pd.DataFrame({"raw_review": self.probes, "clean_text": ["WRONG"] * len(self.probes),
                      "sentiment": [row["label"] for row in expected]}).to_csv(
            self.root / "test_split.csv", index=False, encoding="utf-8-sig")
        with patch("sys.argv", ["evaluate.py", "--run-dir", str(self.root)]), \
             patch("src.evaluate.load_model", return_value=self.model), \
             patch.object(self.model._processor, "transform", wraps=self.model._processor.transform) as transform, \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
        transform.assert_called_once_with(self.probes)
        output = pd.read_csv(self.root / "test_predictions.csv", keep_default_na=False)
        self.assertEqual(output.predicted_label.tolist(), [row["label"] for row in expected])
        self.assertEqual(output.empty_after_preprocessing.tolist(), [row["empty_after_preprocessing"] for row in expected])
        with patch("sys.argv", ["evaluate.py", "--run-dir", str(self.root), "--text-column", "clean_text"]):
            with self.assertRaisesRegex(ValueError, "raw reviews"):
                evaluate.main()

    def test_cli_loads_and_predicts_in_a_fresh_process(self):
        joblib.dump(self.model, self.path)
        result = subprocess.run([sys.executable, str(ROOT / "src" / "predict.py"),
            "--model-path", str(self.path), "--text", "KHUẤY TỐT", "!!!", "--probabilities"],
            cwd=ROOT, capture_output=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["artifact_version"], ARTIFACT_VERSION)
        self.assertEqual(payload["predictions"], self.model.infer(["KHUẤY TỐT", "!!!"], include_probabilities=True))

    def test_streamlit_uses_same_inference_and_displays_mapped_scores(self):
        joblib.dump(self.model, self.path)
        raw = "KHUẤY TỐT"
        expected = self.model.infer(raw, include_probabilities=True)[0]
        transform_function = VietnameseTextProcessor.transform
        with patch.dict(os.environ, {"SENTIMENT_MODEL_PATH": str(self.path)}):
            app = AppTest.from_file(str(ROOT / "app.py")).run()
            self.assertFalse(app.exception)
            with patch.object(VietnameseTextProcessor, "transform", autospec=True,
                              side_effect=transform_function) as transform:
                app.text_area[0].input(raw)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(transform.call_count, 1)
                self.assertEqual(transform.call_args.args[1], [raw])
            self.assertEqual({item.label: item.value for item in app.metric},
                             {label: f"{score:.2%}" for label, score in expected["probabilities"].items()})
            self.assertIn("chưa hiệu chỉnh", app.caption[0].value)
            app.text_area[0].input("!!!")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)


if __name__ == "__main__":
    unittest.main()
