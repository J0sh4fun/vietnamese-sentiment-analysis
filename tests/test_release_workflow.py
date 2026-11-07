import hashlib
import io
import json
from pathlib import Path
import random
import re
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from src.experiments import seed_everything
from src.prepare_data import ensure_originals, verify_splits
from src.release_config import DEFAULT_CONFIG, load_config


class ReleaseWorkflowTests(unittest.TestCase):
    def test_shared_seed_repeats_python_numpy_and_torch(self):
        def draw():
            return random.random(), float(np.random.random()), float(torch.rand(()))
        seed_everything(42, torch_enabled=True)
        first = draw()
        seed_everything(42, torch_enabled=True)
        self.assertEqual(first, draw())

    def test_pinned_regex_is_a_word_token_pattern_not_literal_backslashes(self):
        config = load_config()
        self.assertEqual(re.findall(config["tfidf"]["token_pattern"], "rất tốt"), ["rất", "tốt"])
        self.assertEqual(config["seed"], 42)

    def fixture(self, root, *, member="data.jsonl", content=b'{"review":"good","label":"positive"}\n'):
        archive = root / "source.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(member, content)
        config = {"data": {"files": {"data.jsonl": {"sha256": hashlib.sha256(content).hexdigest(), "rows": 1}}}}
        return archive, config

    def test_offline_archive_is_verified_and_existing_data_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, config = self.fixture(root)
            files = ensure_originals(config, root / "raw", archive_path=archive)
            self.assertEqual(files["data.jsonl"]["sha256"], config["data"]["files"]["data.jsonl"]["sha256"])
            self.assertEqual(ensure_originals(config, root / "raw", offline=True), files)
            (root / "raw/data.jsonl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Existing data hash mismatch"):
                ensure_originals(config, root / "raw", archive_path=archive)
            self.assertEqual((root / "raw/data.jsonl").read_bytes(), b"changed")

    def test_archive_path_traversal_and_wrong_payload_are_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, config = self.fixture(root, member="../data.jsonl")
            with self.assertRaisesRegex(ValueError, "archive members"):
                ensure_originals(config, root / "raw", archive_path=archive)
            self.assertFalse((root / "data.jsonl").exists())
            archive, config = self.fixture(root)
            config["data"]["files"]["data.jsonl"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "version/content differs"):
                ensure_originals(config, root / "raw2", archive_path=archive)
            self.assertFalse((root / "raw2/data.jsonl").exists())

    def test_missing_data_in_offline_mode_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, config = self.fixture(root)
            with self.assertRaisesRegex(FileNotFoundError, "Pinned data missing"):
                ensure_originals(config, root / "raw", offline=True)

    def test_split_seed_or_config_cannot_silently_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "split_manifest.json").write_text(json.dumps({"config": {"sha256": "wrong", "size_bytes": 0}, "seed": 43}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config/seed"):
                verify_splits(DEFAULT_CONFIG, root)


if __name__ == "__main__":
    unittest.main()
