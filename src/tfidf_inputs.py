"""TF-IDF adapter: reuse the existing saved preprocessing and fitted vectorizer unchanged."""
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np

from src.frozen_baselines import verify_bundle
from src.preprocessor import VietnameseTextProcessor
from src.review_inputs import BatchIdentity, batch_identity, raw_strings


@dataclass
class TfidfBatch:
    identity: BatchIdentity
    features: object  # scipy CSR, float32; densify only at the MLP batch boundary
    clean_text: tuple[str, ...]
    empty_after_preprocessing: tuple[bool, ...]
    no_known_features: tuple[bool, ...]


class TfidfReviewInputs:
    def __init__(self, vectorizer, preprocessing_config):
        self.vectorizer = vectorizer
        self.processor = VietnameseTextProcessor.from_config(preprocessing_config)

    @classmethod
    def from_baseline(cls, baseline_dir):
        baseline_dir = Path(baseline_dir)
        if (baseline_dir / "release_manifest.json").is_file():
            from src.release_baseline import verified_run
            verified_run(baseline_dir)
        else:
            verify_bundle(baseline_dir)
        directory = baseline_dir / "artifacts/mlp"
        return cls(joblib.load(directory / "tfidf.joblib"),
                   json.loads((directory / "preprocessing.json").read_text(encoding="utf-8")))

    def transform_raw(self, reviews):
        reviews = raw_strings(reviews)
        clean = self.processor.transform(reviews) if reviews else []
        if clean:
            features = self.vectorizer.transform(clean).tocsr().astype(np.float32)
        else:
            from scipy.sparse import csr_matrix
            features = csr_matrix((0, len(self.vectorizer.vocabulary_)), dtype=np.float32)
        return features, tuple(clean)

    def encode(self, records):
        records = tuple(records)
        identity = batch_identity(records)
        features, clean = self.transform_raw([row.review for row in records])
        return TfidfBatch(identity, features, clean, tuple(not text.strip() for text in clean),
                          tuple(bool(value == 0) for value in features.getnnz(axis=1)))
