import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

from src.mlp import (SentimentMLP, TfidfDataset, make_loader, restore_training_checkpoint,
                     seed_everything, training_checkpoint, train_epoch)
from src.mlp_pipeline import RawTextMLP, export_pipeline, load_pipeline
from src.preprocessor import VietnameseTextProcessor
from src.package_mlp import fit_baseline


class MLPPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything(17)
        self.config = VietnameseTextProcessor(word_segmentation=False, remove_stopwords=False).to_config()
        processor = VietnameseTextProcessor.from_config(self.config)
        texts = ["sản phẩm tốt", "rất tốt", "đẹp tốt", "hài lòng", "không tốt", "rất tệ", "hỏng xấu", "thất vọng"]
        self.vectorizer = TfidfVectorizer(dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
        features = self.vectorizer.fit_transform(processor.transform(texts))
        self.dataset = TfidfDataset(features, [1, 1, 1, 1, 0, 0, 0, 0])
        self.model = SentimentMLP(features.shape[1], hidden_size=5, dropout=0.3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loader = make_loader(self.dataset, 4, shuffle=True, seed=17)
        self.training_config = {"optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0,
                                "batch_size": 4, "seed": 17, "threads": 1, "epochs": 2}

    def checkpoint(self):
        return training_checkpoint(self.model, self.optimizer, self.loader, epoch=1,
                                   validation_macro_f1=0.5, config=self.training_config)

    def pipeline(self):
        return RawTextMLP(self.model, self.vectorizer, self.config, batch_size=3)

    def test_each_raw_public_method_preprocesses_once_and_inference_has_no_grad(self):
        pipeline = self.pipeline()
        seen = []
        hook = self.model.register_forward_hook(lambda m, inputs, outputs: seen.append(
            (m.training, torch.is_grad_enabled())))
        for method in (lambda: pipeline.infer(["SẢN PHẨM tốt"], include_probabilities=True, include_logits=True),
                       lambda: pipeline.predict(["SẢN PHẨM tốt"]),
                       lambda: pipeline.predict_proba(["SẢN PHẨM tốt"])):
            with patch.object(pipeline._processor, "transform", wraps=pipeline._processor.transform) as transform:
                method()
                transform.assert_called_once_with(["SẢN PHẨM tốt"])
        hook.remove()
        self.assertTrue(all(not mode and not grad for mode, grad in seen))

    def test_empty_punctuation_oov_and_empty_batch_are_explicit(self):
        pipeline = self.pipeline()
        rows = pipeline.infer(["", "!!!", "xqzvvvzzrandomtoken20261106"], include_logits=True)
        self.assertEqual([r["empty_after_preprocessing"] for r in rows], [True, True, False])
        self.assertTrue(all(r["no_known_features"] and r["warnings"] for r in rows))
        np.testing.assert_allclose(rows[0]["logits"], rows[2]["logits"], atol=1e-7)
        self.assertEqual(pipeline.infer([]), [])
        self.assertEqual(pipeline.predict_proba([]).shape, (0, 2))

    def test_invalid_values_fail_before_preprocessing(self):
        pipeline = self.pipeline()
        with patch.object(pipeline._processor, "transform", side_effect=AssertionError("must not preprocess")):
            for invalid in (None, 3, {"review": "tốt"}, {"tốt"}, b"text", [None], [["tốt"]], np.zeros((1, 1))):
                with self.subTest(value=repr(invalid)), self.assertRaises(TypeError):
                    pipeline.infer(invalid)

    def test_bundle_relocation_keeps_logits_and_mapping_without_original_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = export_pipeline(root / "source", self.checkpoint(), self.vectorizer, self.config)
            shutil.copytree(source, root / "relocated")
            shutil.rmtree(source)
            loaded = load_pipeline(root / "relocated")
            before = self.pipeline().infer(["không tốt", "rất tốt"], include_logits=True, include_probabilities=True)
            after = loaded.infer(["không tốt", "rất tốt"], include_logits=True, include_probabilities=True)
            self.assertEqual([r["label"] for r in before], [r["label"] for r in after])
            np.testing.assert_allclose([r["logits"] for r in before], [r["logits"] for r in after], atol=1e-7)
            self.assertEqual(list(after[0]["probabilities"]), ["negative", "positive"])
            self.assertEqual(loaded.preprocessing_config, self.config)

    def test_tampered_bundle_or_runtime_configuration_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            source = export_pipeline(Path(temp) / "bundle", self.checkpoint(), self.vectorizer, self.config)
            with (source / "tfidf.joblib").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_pipeline(source)
        wrong_runtime = copy.deepcopy(self.config)
        wrong_runtime["runtime"]["unicode"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "runtime mismatch"):
            RawTextMLP(self.model, self.vectorizer, wrong_runtime)

    def test_resume_restores_adam_dropout_rng_and_shuffle_exactly(self):
        train_epoch(self.model, self.loader, self.optimizer)
        snapshot = self.checkpoint()
        self.assertEqual(snapshot["label_mapping"], {"negative": 0, "positive": 1})
        self.assertEqual(snapshot["architecture"]["hidden_size"], 5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"
            torch.save(snapshot, path)
            train_epoch(self.model, self.loader, self.optimizer)
            expected = copy.deepcopy(self.model.state_dict())
            expected_adam = copy.deepcopy(self.optimizer.state_dict())
            model, optimizer, loader, checkpoint = restore_training_checkpoint(path, self.dataset)
            self.assertEqual(checkpoint["epoch"], 1)
            train_epoch(model, loader, optimizer)
            for name, value in expected.items():
                self.assertTrue(torch.equal(model.state_dict()[name], value), name)
            for key, state in expected_adam["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(optimizer.state_dict()["state"][key][name], value))


    def test_raw_baseline_keeps_named_labels_consistent_with_mlp(self):
        labels = ["positive"] * 4 + ["negative"] * 4
        raw, classifier, seconds = fit_baseline(ComplementNB(alpha=0.5), self.vectorizer,
            self.dataset.features, labels, self.config)
        predictions = raw.predict(["rất tốt", "rất tệ"]).tolist()
        self.assertEqual(predictions, ["positive", "negative"])
        self.assertEqual([row["label"] for row in raw.infer(["rất tốt", "rất tệ"])], predictions)
        with self.assertRaisesRegex(ValueError, "string labels"):
            fit_baseline(ComplementNB(), self.vectorizer, self.dataset.features,
                         self.dataset.labels, self.config)


if __name__ == "__main__":
    unittest.main()
