import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.review_inputs import RawReview, RawReviewDataset, assert_same_samples, batch_identity
from src.transformer_inputs import DEFAULT_TOKENIZER_DIR, XLMRReviewInputs

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "models/releases/baseline-v1.0.0"
if not BASELINE.is_dir():
    BASELINE = ROOT / "models/baselines/baseline_v1_20261107"
OPTIONAL_READY = bool(importlib.util.find_spec("transformers")) and (DEFAULT_TOKENIZER_DIR / "processor_config.json").is_file() and BASELINE.is_dir()


class SharedReviewInputTests(unittest.TestCase):
    def test_loader_uses_raw_review_and_never_falls_back_to_clean_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "split.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_id", "review", "clean_text", "label"])
                writer.writeheader()
                writer.writerow({"source_id": "a", "review": "SẢN PHẨM tốt!!!", "clean_text": "POISON", "label": "positive"})
            self.assertEqual(RawReviewDataset.from_csv(path)[0].review, "SẢN PHẨM tốt!!!")
            path.write_text("source_id,clean_text,label\na,POISON,positive\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "never a fallback"):
                RawReviewDataset.from_csv(path)

    def test_same_sample_ids_raw_hashes_labels_and_order_are_required(self):
        rows = (RawReview("a", "Tốt", "positive"), RawReview("b", "Tệ", "negative"))
        identity = batch_identity(rows)
        self.assertEqual(identity.labels.tolist(), [1, 0])
        self.assertEqual(identity.labels.dtype, torch.long)
        with self.assertRaisesRegex(ValueError, "samples or order"):
            assert_same_samples(identity, batch_identity(rows[::-1]))
        with self.assertRaisesRegex(ValueError, "different labels"):
            assert_same_samples(identity, batch_identity((RawReview("a", "Tốt", "negative"), rows[1])))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            RawReviewDataset([rows[0], rows[0]])
        with self.assertRaisesRegex(ValueError, "mix labeled"):
            batch_identity([rows[0], RawReview("c", "")])

    def test_clean_text_lists_missing_labels_and_invalid_values_are_not_coerced(self):
        with self.assertRaises(TypeError):
            batch_identity(["sản_phẩm tốt"])
        with self.assertRaises(TypeError):
            RawReview("a", None, "negative")
        with self.assertRaises(ValueError):
            RawReview("a", "bình thường", "neutral")
        self.assertIsNone(batch_identity([RawReview("a", "")]).labels)


@unittest.skipUnless(OPTIONAL_READY, "Install optional Transformer requirements and prepare local pinned tokenizer/baseline")
class ModelFamilyRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from src.tfidf_inputs import TfidfReviewInputs
        cls.tfidf = TfidfReviewInputs.from_baseline(BASELINE)
        cls.transformer = XLMRReviewInputs()

    def test_transformer_receives_original_case_punctuation_emoji_once_without_tfidf_processor(self):
        rows = [RawReview("a", "SẢN PHẨM này KHÔNG tốt!!! 😞", "negative"),
                RawReview("b", "Đẹp, rất hài lòng!", "positive")]
        left = self.tfidf.encode(rows)
        class Spy:
            def __init__(self, wrapped):
                self.wrapped, self.calls = wrapped, []
            def __call__(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return self.wrapped(*args, **kwargs)
            def __getattr__(self, name):
                return getattr(self.wrapped, name)
        spy = Spy(self.transformer.tokenizer)
        with patch.object(self.transformer, "tokenizer", spy), patch(
                "src.preprocessor.VietnameseTextProcessor.transform", side_effect=AssertionError("TF-IDF path forbidden")):
            right = self.transformer.encode(rows)
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0][0][0], [row.review for row in rows])
        self.assertNotEqual(left.clean_text[0], rows[0].review)
        assert_same_samples(left.identity, right.identity)
        self.assertEqual(right.model_inputs["labels"].tolist(), [0, 1])
        self.assertEqual(set(right.model_inputs), {"input_ids", "attention_mask", "labels"})
        normalized = self.transformer.tokenizer.backend_tokenizer.normalizer.normalize_str(rows[0].review)
        self.assertIn("SẢN", normalized)
        self.assertIn("!!!", normalized)
        self.assertIn("😞", normalized)

    def test_exact_boundary_and_overflow_audit_preserve_one_row_per_review(self):
        rows = [RawReview(str(n), "tốt " * n, "positive") for n in (510, 511, 1200)]
        batch = self.transformer.encode(rows)
        self.assertEqual(batch.model_inputs["input_ids"].shape, (3, 512))
        self.assertEqual(batch.full_token_lengths, (512, 513, 1202))
        self.assertEqual(batch.truncated, (False, True, True))
        self.assertEqual(batch.token_lengths, (512, 512, 512))
        self.assertTrue(all(row[-1].item() == self.transformer.tokenizer.eos_token_id for row in batch.model_inputs["input_ids"]))

    def test_empty_punctuation_oov_and_dynamic_padding_keep_samples(self):
        rows = [RawReview("a", "", "negative"), RawReview("b", "!!!", "positive"),
                RawReview("c", "xqzvvvzzrandomtoken20261106", "negative")]
        left, right = self.tfidf.encode(rows), self.transformer.encode(rows)
        assert_same_samples(left.identity, right.identity)
        self.assertEqual(left.empty_after_preprocessing[:2], (True, True))
        self.assertEqual(right.no_content_tokens[:2], (True, False))
        self.assertTrue(left.no_known_features[2])
        self.assertGreater(right.token_lengths[2], 2)
        self.assertEqual(right.model_inputs["input_ids"].shape[1], max(right.token_lengths))
        mask = right.model_inputs["attention_mask"]
        self.assertTrue(torch.all(right.model_inputs["input_ids"][mask == 0] == self.transformer.tokenizer.pad_token_id))
        empty = self.transformer.encode([])
        self.assertEqual(empty.model_inputs["input_ids"].shape, (0, 0))
        self.assertEqual(self.tfidf.encode([]).features.shape, (0, 24982))

    def test_batch_token_ids_match_individual_calls_and_boundaries(self):
        rows = [RawReview("a", "Sản phẩm tốt!"), RawReview("b", "KHÔNG hài lòng!!!")]
        batch = self.transformer.encode(rows)
        for index, row in enumerate(rows):
            single = self.transformer.encode([row])
            length = batch.token_lengths[index]
            self.assertTrue(torch.equal(batch.model_inputs["input_ids"][index, :length], single.model_inputs["input_ids"][0]))
        with self.assertRaisesRegex(ValueError, "max_length"):
            XLMRReviewInputs(max_length=513)
        with self.assertRaisesRegex(ValueError, "max_length"):
            XLMRReviewInputs(max_length=2)

    def test_changed_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "tokenizer"
            shutil.copytree(DEFAULT_TOKENIZER_DIR, directory)
            path = directory / "processor_config.json"
            profile = json.loads(path.read_text(encoding="utf-8"))
            profile["lowercase"] = True
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "flags differ"):
                XLMRReviewInputs(directory)

    def test_real_token_ids_fit_xlmr_architecture_at_512_without_downloading_weights(self):
        # Random tiny encoder tests shape/position limits only; no sentiment accuracy is implied.
        from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification
        config = XLMRobertaConfig(vocab_size=250002, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, max_position_embeddings=514,
            pad_token_id=1, bos_token_id=0, eos_token_id=2, num_labels=2,
            id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})
        model = XLMRobertaForSequenceClassification(config).eval()
        batch = self.transformer.encode([RawReview("a", "tốt " * 600, "positive"), RawReview("b", "", "negative")])
        with torch.no_grad():
            output = model(**batch.model_inputs)
        self.assertEqual(output.logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(output.loss))


if __name__ == "__main__":
    unittest.main()
