import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.benchmarking import (TrainingMeasurement, latency_summary, process_memory,
                             quality_metrics, read_training_receipt)
from src.benchmark_models import TransformerAdapter, artifact_identity, measure_latency


class BenchmarkTests(unittest.TestCase):
    def test_quality_keeps_label_order_and_missing_class_in_macro(self):
        scores = quality_metrics(["positive", "positive"], ["positive", "positive"])
        self.assertEqual(scores["macro_f1"], 0.5)
        self.assertEqual(scores["confusion_matrix"], [[0, 0], [0, 2]])
        self.assertEqual(scores["confusion_matrix_axes"]["labels"], ["negative", "positive"])
        self.assertEqual(scores["classification_report"]["negative"]["support"], 0)
        for truth, pred in [([], []), (["negative"], []), (["LABEL_0"], ["negative"])]:
            with self.assertRaises(ValueError):
                quality_metrics(truth, pred)

    def test_latency_weights_partial_batch_by_actual_review_count(self):
        stats = latency_summary([0.010, 0.030], [4, 1])
        self.assertAlmostEqual(stats["amortized_ms_per_review"], 8)
        self.assertAlmostEqual(stats["throughput_reviews_per_second"], 125)
        self.assertEqual(stats["actual_batch_sizes"], [1, 4])
        self.assertAlmostEqual(stats["batch_p95_ms"], 29)
        for values, counts in [([], []), ([0], [1]), ([float("nan")], [1]), ([1], [0])]:
            with self.assertRaises(ValueError):
                latency_summary(values, counts)

    def test_timer_includes_preparation_only_in_end_to_end(self):
        clock = [0.0]
        class Adapter:
            device = "cpu"
            def prepare(self, texts):
                clock[0] += 0.005
                return texts
            def classify(self, texts):
                clock[0] += 0.002
                return ["positive"] * len(texts)
        with patch("src.benchmark_models.time.perf_counter", side_effect=lambda: clock[0]):
            end = measure_latency(Adapter(), ["a", "b", "c"], batch_size=2, warmup=2, repeats=2, mode="end_to_end")
            classifier = measure_latency(Adapter(), ["a", "b", "c"], batch_size=2, warmup=2, repeats=2, mode="classifier_only")
        self.assertEqual(end["timed_batches"], 4)  # warmup excluded
        self.assertEqual(end["timed_reviews"], 6)
        self.assertAlmostEqual(end["batch_mean_ms"], 7)
        self.assertAlmostEqual(classifier["batch_mean_ms"], 2)

    def test_artifact_fingerprint_detects_bytes_and_includes_all_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "weights").write_bytes(b"abc")
            original = artifact_identity(root)
            (root / "tokenizer").write_bytes(b"def")
            self.assertNotEqual(original["sha256"], artifact_identity(root)["sha256"])
            self.assertEqual(artifact_identity(root)["size_bytes"], 6)
            (root / "weights").write_bytes(b"abd")
            self.assertNotEqual(original["sha256"], artifact_identity(root)["sha256"])

    def test_native_memory_counters_include_rss_and_lifetime_peak(self):
        values = process_memory()
        if values["rss_bytes"] is not None:
            self.assertGreater(values["rss_bytes"], 0)
            self.assertGreaterEqual(values["peak_rss_bytes"], values["rss_bytes"])

    def test_training_receipt_must_match_artifact_split_and_complete_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "training.json"
            identity = {"config": {"sha256": "config"}, "train_split": {"sha256": "train"},
                        "hardware": {"cpu": "test"}, "validation_split": {"sha256": "val"},
                        "inference_artifact": {"sha256": "artifact"}}
            with TrainingMeasurement(path, model="MLP", device="cpu", identity=identity):
                pass
            self.assertEqual(read_training_receipt(path, model="MLP", artifact={"sha256": "artifact"},
                                                   split={"sha256": "val"})["status"], "COMPLETE")
            with self.assertRaises(ValueError):
                read_training_receipt(path, model="MLP", artifact={"sha256": "different"}, split={"sha256": "val"})
            with self.assertRaises(FileExistsError):
                with TrainingMeasurement(path, model="MLP", device="cpu", identity=identity):
                    pass
            failed = Path(temp) / "failed.json"
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with TrainingMeasurement(failed, model="MLP", device="cpu", identity=identity):
                    raise RuntimeError("training failed")
            self.assertEqual(json.loads(failed.read_text())["status"], "FAILED")
            with self.assertRaises(ValueError):
                read_training_receipt(failed, model="MLP", artifact={"sha256": "artifact"}, split={"sha256": "val"})

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "optional Transformer stack absent")
    def test_transformer_adapter_loads_local_weights_and_obeys_saved_class_order(self):
        # Tiny synthetic weights verify adapter/serialization only, not sentiment quality.
        import torch
        from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification
        config = XLMRobertaConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, num_labels=2,
            id2label={0: "positive", 1: "negative"}, label2id={"positive": 0, "negative": 1})
        model = XLMRobertaForSequenceClassification(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier.out_proj.bias[0] = 2
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model.save_pretrained(root / "model")
            raw = ["SẢN PHẨM!!!"]
            with patch("src.transformer_inputs.XLMRReviewInputs") as inputs:
                inputs.return_value.profile = {"external_processing": "identity"}
                inputs.return_value.max_length = 512
                inputs.return_value.tokenize_raw.return_value = ({
                    "input_ids": torch.tensor([[0, 4, 2]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)},)
                adapter = TransformerAdapter(root, "cpu")
                self.assertEqual(adapter.classify(adapter.prepare(raw)), ["positive"])
                inputs.return_value.tokenize_raw.assert_called_once_with(raw)
            model.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
            model.save_pretrained(root / "ambiguous/model")
            with patch("src.transformer_inputs.XLMRReviewInputs"), self.assertRaisesRegex(ValueError, "id2label"):
                TransformerAdapter(root / "ambiguous", "cpu")


if __name__ == "__main__":
    unittest.main()
