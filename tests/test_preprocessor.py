import json
import unicodedata
import unittest
from unittest.mock import patch

from assets.vietnamese_stopwords import VIETNAMESE_STOPWORDS
from src.preprocessor import (
    NEGATION_WORDS, VietnameseTextProcessor, normalize_unicode,
    normalize_word_tone, stopword_key,
)


class ToneRegressionTests(unittest.TestCase):
    def test_correct_khuay_is_not_changed(self):
        self.assertEqual(normalize_word_tone("khuấy"), "khuấy")

    def test_correct_words_and_qu_gi_cases_are_preserved(self):
        words = ["khuấy", "khuyến", "người", "tưởng", "quả", "quý", "quyển",
                 "quốc", "già", "gì", "giữ", "giỏi", "giếng"]
        processor = VietnameseTextProcessor(remove_stopwords=False)
        for word in words:
            with self.subTest(word=word):
                self.assertEqual(normalize_word_tone(word), word)
                self.assertEqual(processor.transform([word]), [word])

    def test_alternative_tone_placements_are_not_rewritten_or_equated(self):
        words = ["hòa", "hoà", "hóa", "hoá", "thủy", "thuỷ"]
        self.assertEqual([normalize_word_tone(word) for word in words], words)
        self.assertEqual(VietnameseTextProcessor(remove_stopwords=False).transform(words), words)

    def test_nfc_composes_without_repositioning(self):
        for word in ["khuấy", "hoà", "hòa", "giữ", "quả"]:
            with self.subTest(word=word):
                decomposed = unicodedata.normalize("NFD", word)
                self.assertEqual(normalize_unicode(decomposed), word)
                self.assertEqual(normalize_word_tone(decomposed), word)
                self.assertEqual(VietnameseTextProcessor(remove_stopwords=False).transform([decomposed]), [word])

    def test_legacy_opt_in_is_explicitly_unvalidated(self):
        with self.assertWarnsRegex(UserWarning, "unvalidated"):
            processor = VietnameseTextProcessor(tone_repositioning=True, remove_stopwords=False)
        # Document the known defect; do not claim the legacy algorithm is fixed.
        self.assertEqual(processor.transform(["khuấy"]), ["khúây"])
        self.assertFalse(VietnameseTextProcessor().to_config()["settings"]["tone_repositioning"])


class ProcessorTests(unittest.TestCase):
    def test_underscore_compounds_preserve_tone_and_boundaries(self):
        words = ["khuấy_đều", "sản_phẩm", "quả_quýt", "giữ_gìn"]
        self.assertEqual(VietnameseTextProcessor(remove_stopwords=False).transform(words), words)

    def test_actual_stopword_list_conventions_and_negation(self):
        self.assertTrue(any("_" in word for word in VIETNAMESE_STOPWORDS))
        self.assertFalse(any(" " in word for word in VIETNAMESE_STOPWORDS))
        self.assertTrue({"không_phải", "chưa_từng", "chớ"} <= VIETNAMESE_STOPWORDS)
        processor = VietnameseTextProcessor()
        protected = {word for word in VIETNAMESE_STOPWORDS
                     if NEGATION_WORDS.intersection(stopword_key(word).split("_"))}
        self.assertTrue(protected)
        self.assertTrue(protected.isdisjoint(processor.stopwords))
        words = ["không", "chẳng", "chưa", "chớ", "đừng", "không_phải", "chưa_từng", "chẳng_những"]
        self.assertEqual(processor.transform(words), words)
        for sentence, negation in [("không tốt", "không"), ("chưa đẹp", "chưa"), ("đừng mua", "đừng")]:
            self.assertIn(negation, processor.transform([sentence])[0].replace("_", " ").split())

    def test_space_and_underscore_stopwords_use_same_key(self):
        processor = VietnameseTextProcessor(stopwords={"bây giờ", "KHÔNG PHẢI", "chưa_từng"})
        self.assertEqual(processor.stopwords, frozenset({"bây_giờ"}))
        with patch("src.preprocessor.word_tokenize", return_value="bây_giờ không_phải chưa_từng khuấy_đều"):
            self.assertEqual(processor.transform(["fixture"]), ["không_phải chưa_từng khuấy_đều"])

    def test_urls_emails_and_phone_numbers(self):
        cases = {
            "https://example.com/path?x=1": "<url>",
            "HTTP://EXAMPLE.COM": "<url>",
            "WWW.EXAMPLE.COM": "<url>",
            "Review.User+tag@Example.COM": "<email>",
            "0912345678": "<phone>",
            "+84912345678": "<phone>",
        }
        processor = VietnameseTextProcessor()
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(processor.transform([raw]), [expected])
        self.assertEqual(processor.transform(["https://a.com user@a.com 0912345678"]),
                         ["<url> <email> <phone>"])
        self.assertNotIn("<phone>", processor.transform(["123456789012345"])[0])

    def test_literal_placeholder_substrings_are_not_replaced(self):
        self.assertEqual(VietnameseTextProcessor(remove_stopwords=False).transform(
            ["stokurlx", "TOKURL", "tokemailbox"]), ["stokurlx", "tokurl", "tokemailbox"])

    def test_nfc_and_cleaning_precede_segmentation_and_filtering_follows_it(self):
        seen = []
        def segment(text, format, use_token_normalize):
            self.assertFalse(use_token_normalize)
            seen.append(text)
            return "khuấy_đều và"
        with patch("src.preprocessor.word_tokenize", side_effect=segment):
            result = VietnameseTextProcessor().transform([unicodedata.normalize("NFD", "KHUẤY ĐỀU!!!")])
        self.assertEqual(seen, ["khuấy đều"])
        self.assertEqual(result, ["khuấy_đều"])

    def test_entities_bypass_segmentation(self):
        with patch("src.preprocessor.word_tokenize", return_value="") as segment:
            result = VietnameseTextProcessor().transform(["https://example.com user@example.com 0912345678"])
        segment.assert_not_called()
        self.assertEqual(result, ["<url> <email> <phone>"])

    def test_empty_inputs_preserve_batch_length_and_order(self):
        processor = VietnameseTextProcessor()
        self.assertEqual(processor.transform([]), [])
        self.assertEqual(processor.transform(["", "  ", "!!!", "và", "khuấy"]), ["", "", "", "", "khuấy"])
        self.assertEqual(processor.transform(["<b></b>"]), [""])

    def test_invalid_inputs_are_not_silently_stringified(self):
        for bad in ([None], [123], "one string"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                VietnameseTextProcessor().transform(bad)

    def test_configuration_json_round_trip_includes_stopwords_and_runtime(self):
        processor = VietnameseTextProcessor(stopwords={"bây giờ", "và", "không"})
        config = json.loads(json.dumps(processor.to_config(), ensure_ascii=False))
        restored = VietnameseTextProcessor.from_config(config)
        self.assertEqual(restored.to_config(), config)
        self.assertEqual(config["settings"]["empty_policy"], "keep")
        self.assertIn("underthesea", config["runtime"])
        texts = ["khuấy", "không_phải", "và", "user@example.com"]
        self.assertEqual(processor.transform(texts), restored.transform(texts))
        for key, value in [("implementation_version", -1), ("runtime", {})]:
            incompatible = dict(config, **{key: value})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                VietnameseTextProcessor.from_config(incompatible)


if __name__ == "__main__":
    unittest.main()
