import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_pipeline import (
    assert_no_overlap, generate_unaccented, normalize_text, prepare_dataset,
    preprocess_frame, read_jsonl, source_id_for, validate_frame,
)
from src.train import parse_args, save_artifacts, train_and_select_best_model


class IdentityProcessor:
    def transform(self, texts):
        return [normalize_text(text) for text in texts]


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{"review": f"sản phẩm {'tốt' if i % 2 else 'tệ'} mẫu {i}",
                      "label": "positive" if i % 2 else "negative"} for i in range(100)]
        self.original = self.write("original.jsonl", self.rows)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def args(self, **overrides):
        with patch("sys.argv", ["train.py"]):
            args = parse_args()
        args.train_data = self.original
        args.output_dir = self.root
        args.test_size = .2
        args.val_size = .2
        args.aug_data = []
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def prepare(self, **overrides):
        return prepare_dataset(self.args(**overrides), IdentityProcessor())

    def test_generated_variants_only_from_training_and_holdouts_unchanged(self):
        baseline = self.prepare()
        args = self.args(generate_aug=True)
        train, val, test = prepare_dataset(args, IdentityProcessor())
        pd.testing.assert_frame_equal(baseline[1], val)
        pd.testing.assert_frame_equal(baseline[2], test)
        aug = train.loc[train.is_augmented]
        self.assertEqual(len(aug), len(baseline[0]))
        self.assertEqual(set(aug.source_id), set(baseline[0].source_id))
        assert_no_overlap(train, val, test)
        stages = {entry["stage"] for entry in args.data_audit}
        self.assertTrue({"original:before_validation", "original:after_validation",
                         "split:train:originals", "final:train", "final:test"} <= stages)
        self.assertTrue(all("label_distribution" in entry for entry in args.data_audit))

    def test_imports_exclude_holdout_unknown_and_forged_provenance(self):
        base = self.prepare()
        all_originals = pd.concat(base)
        aug = generate_unaccented(all_originals, "review", "label")
        good = aug.iloc[0].to_dict()
        bad_unknown = dict(good, source_id="unknown", review="unknown text")
        bad_source = dict(good, source_text="wrong source", review="wrong source")
        bad_transform = dict(good, review="invented text")
        bad_label = dict(good, label="negative" if good["label"] == "positive" else "positive")
        # Separate files ensure each provenance failure is checked before cross-file deduplication.
        files = [self.write("all.jsonl", aug.to_dict("records"))]
        files += [self.write(f"bad{i}.jsonl", [row]) for i, row in enumerate(
            [bad_unknown, bad_source, bad_transform, bad_label])]
        train, val, test = self.prepare(aug_data=files)
        self.assertEqual(set(train.loc[train.is_augmented, "source_id"]), set(base[0].source_id))
        self.assertEqual(len(train), 2 * len(base[0]))
        pd.testing.assert_frame_equal(base[1], val)
        pd.testing.assert_frame_equal(base[2], test)
        assert_no_overlap(train, val, test)

    def test_untraceable_legacy_is_excluded(self):
        legacy = self.write("legacy.jsonl", [{"review": "san pham tot", "label": "positive", "id": 1}])
        with warnings.catch_warnings(record=True) as caught:
            actual = self.prepare(aug_data=[legacy])
        self.assertIn("untraceable", str(caught[0].message))
        for left, right in zip(actual, self.prepare()):
            pd.testing.assert_frame_equal(left, right)

    def test_shared_source_groups_never_cross_splits(self):
        rows = [dict(row, source_id=f"source-{i // 2}-{row['label']}") for i, row in enumerate(self.rows)]
        # Two different texts per source, with consistent labels.
        rows += [dict(row, review=row["review"] + " khác") for row in rows]
        self.original = self.write("groups.jsonl", rows)
        for seed in (1, 42, 123):
            parts = self.prepare(generate_aug=True, random_state=seed)
            assert_no_overlap(*parts)
            self.assertEqual(sum(part.loc[~part.is_augmented].shape[0] for part in parts), 200)

    def test_deduplication_cannot_silently_break_supplied_source_groups(self):
        rows = [dict(row, source_id=f"source-{i}") for i, row in enumerate(self.rows)]
        rows += [dict(rows[0], source_id="alias"),
                 dict(rows[0], source_id="alias", review="different variant of source zero")]
        self.original = self.write("aliases.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Ambiguous original source_id"):
            self.prepare(generate_aug=True)

    def test_same_validation_for_every_source(self):
        rows = [
            {"review": "  TỐT  ", "label": "positive"},
            {"review": "tốt", "label": "positive"},
            {"review": "bad", "label": "negative"},
            {"review": "BAD", "label": "positive"},
            {"review": None, "label": "positive"},
            {"review": "   ", "label": "positive"},
            {"review": 123, "label": "positive"},
            {"review": "invalid label", "label": "neutral"},
            {"review": "missing label"},
            {"review": "numeric label", "label": 1},
        ]
        for source in ("original", "imported", "generated"):
            audit = []
            result = validate_frame(pd.DataFrame(rows), "review", "label", ["positive", "negative"], audit, source)
            self.assertEqual(result.normalized_text.tolist(), ["tốt"])
            self.assertEqual(audit[0]["samples"], 10)
            self.assertEqual(audit[-1]["conflicting_rows"], 2)
            self.assertEqual(audit[-1]["duplicate_rows"], 1)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                validate_frame(pd.DataFrame({"review": ["text"]}), "review", "label", ["positive"], [], source)
        malformed = self.write("malformed.jsonl", [{"review": "missing label"}])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.prepare(aug_data=[malformed])

    def test_accents_are_not_an_identity_merge_rule(self):
        self.assertNotEqual(source_id_for("má"), source_id_for("ma"))
        result = validate_frame(pd.DataFrame([
            {"review": "má", "label": "positive"},
            {"review": "ma", "label": "negative"},
        ]), "review", "label", ["positive", "negative"], [], "original")
        self.assertEqual(len(result), 2)

    def test_preprocessed_duplicates_conflicts_and_empty_text_are_filtered(self):
        class CleaningProcessor:
            def transform(self, texts):
                return [text.replace("!", "").strip() for text in texts]

        frame = pd.DataFrame([
            {"review": "good", "label": "positive"},
            {"review": "good!", "label": "positive"},
            {"review": "bad", "label": "negative"},
            {"review": "bad!", "label": "positive"},
            {"review": "!!!", "label": "negative"},
        ])
        audit = []
        result = preprocess_frame(frame, "review", "label", CleaningProcessor(), audit, "original")
        self.assertEqual(result.clean_text.tolist(), ["good"])
        self.assertEqual(audit[0]["samples"], 4)
        self.assertEqual(audit[-1]["conflicting_rows"], 2)
        self.assertEqual(audit[-1]["duplicate_rows"], 1)

    def test_all_imported_rows_rejected_is_valid_and_counted(self):
        path = self.write("rejected.jsonl", [{
            "review": "variant", "label": "positive", "source_id": "unknown",
            "source_text": "original", "augmentation_method": "unaccented_v1",
        }])
        args = self.args(aug_data=[path])
        actual = prepare_dataset(args, IdentityProcessor())
        self.assertFalse(actual[0].is_augmented.any())
        self.assertEqual(tuple(map(len, actual)), (60, 20, 20))
        stage = next(entry for entry in args.data_audit if entry["stage"] == f"{path}:provenance")
        self.assertEqual(stage["excluded_rows"], 1)

    def test_augmentation_collision_with_original_is_filtered_without_merging(self):
        base = self.prepare()
        # Add an unaccented original that collides with a generated variant.
        source = base[0].iloc[0]
        candidate = generate_unaccented(base[0].iloc[:1], "review", "label").iloc[0]
        self.original = self.write("collision.jsonl", self.rows + [
            {"review": candidate.review, "label": source.label}])
        parts = self.prepare(generate_aug=True)
        originals = pd.concat(parts).loc[lambda df: ~df.is_augmented]
        self.assertEqual(len(originals), 101)
        self.assertIn(source.source_id, set(originals.source_id))
        self.assertIn(source_id_for(candidate.review), set(originals.source_id))
        augmented = parts[0].loc[parts[0].is_augmented]
        self.assertNotIn(normalize_text(candidate.review), set(augmented.normalized_text))
        assert_no_overlap(*parts)

    def test_overlap_checks_fail_for_each_identity_key(self):
        def frame(sid, raw, clean):
            return pd.DataFrame({"source_id": [sid], "normalized_text": [raw], "normalized_clean_text": [clean]})
        train, val, test = frame("a", "a", "a"), frame("b", "b", "b"), frame("c", "c", "c")
        for key in train:
            bad = val.copy()
            bad[key] = train[key]
            with self.assertRaisesRegex(ValueError, key):
                assert_no_overlap(train, bad, test)

    def test_disable_aug_and_empty_aug_list(self):
        expected = self.prepare()
        actual = self.prepare(disable_aug=True, generate_aug=True, aug_data=[self.root / "missing.jsonl"])
        for left, right in zip(expected, actual):
            pd.testing.assert_frame_equal(left, right)

    def test_validation_ratio_uses_full_original_group_count(self):
        parts = self.prepare(test_size=.1, val_size=.1)
        self.assertEqual(tuple(map(len, parts)), (80, 10, 10))

    def test_invalid_ratios_small_dataset_and_source_conflicts(self):
        for overrides in ({"test_size": 0}, {"val_size": .9}, {"max_samples": 0}, {"max_samples": 3}):
            with self.assertRaises(ValueError):
                self.prepare(**overrides)
        self.original = self.write("bad-id.jsonl", [dict(row, source_id="same") for row in self.rows])
        with self.assertRaisesRegex(ValueError, "source_id has conflicting labels"):
            self.prepare()

    def test_json_schema_does_not_coerce_invalid_values(self):
        path = self.write("types.jsonl", [{"review": 123, "label": 1, "source_id": "001"}])
        frame = read_jsonl(path)
        self.assertEqual(frame.source_id.iloc[0], "001")
        result = validate_frame(frame, "review", "label", ["positive"], [], "typed")
        self.assertTrue(result.empty)
        path.write_text('[]\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected a JSON object"):
            read_jsonl(path)

    def test_tfidf_vocabulary_and_idf_fit_training_only(self):
        args = self.args(algorithm="multinomial_nb", min_df=1)
        train = pd.DataFrame({"clean_text": ["good shared", "bad shared"], "label": ["positive", "negative"]})
        val = pd.DataFrame({"clean_text": ["validationonly shared", "validationonly bad"], "label": ["positive", "negative"]})
        model, *_ = train_and_select_best_model(args, train, val, "label")
        vectorizer = model.named_steps["tfidf"]
        self.assertNotIn("validationonly", vectorizer.vocabulary_)
        self.assertEqual(vectorizer.idf_[vectorizer.vocabulary_["shared"]], 1.0)
        before = dict(vectorizer.vocabulary_)
        model.predict(["testonly shared"])
        self.assertEqual(before, vectorizer.vocabulary_)

    def test_existing_artifacts_are_never_overwritten(self):
        args = self.args(run_name="existing")
        run = self.root / args.run_name
        run.mkdir()
        sentinel = run / "sentiment_pipeline.joblib"
        sentinel.write_bytes(b"existing model")
        with self.assertRaises(FileExistsError):
            save_artifacts(args, None, None, None, None, [], [], {}, [], {})
        self.assertEqual(sentinel.read_bytes(), b"existing model")


if __name__ == "__main__":
    unittest.main()
