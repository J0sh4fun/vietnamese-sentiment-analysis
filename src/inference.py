"""Versioned raw-review inference. Preprocessed text is an internal detail."""
from __future__ import annotations

from collections.abc import Mapping, Set
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.preprocessor import VietnameseTextProcessor


ARTIFACT_VERSION = 2
MIGRATION_MESSAGE = (
    "Expected a version-2 raw-text sentiment artifact. Retrain with src/train.py "
    "and use the new run's sentiment_pipeline.joblib; legacy models are not upgraded automatically."
)


class SentimentModel:
    """A fitted clean-text estimator bundled with its exact preprocessing config.

    Public methods take a raw string or an ordered iterable of raw strings.
    Empty texts are retained as zero-feature inputs; infer() flags them.
    """

    def __init__(self, pipeline: Pipeline, preprocessing_config: dict,
                 raw_text_column: str = "review", label_column: str = "label"):
        if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["tfidf", "classifier"]:
            raise ValueError("Expected the fitted TF-IDF/classifier pipeline, without another preprocessing step.")
        check_is_fitted(pipeline)
        self.artifact_version = ARTIFACT_VERSION
        self.raw_text_column = raw_text_column
        self.label_column = label_column
        self._pipeline = pipeline
        self._preprocessing_config = deepcopy(preprocessing_config)
        self._restore_processor()

    @property
    def classes_(self):
        return self._pipeline.classes_.copy()

    @property
    def preprocessing_config(self):
        return deepcopy(self._preprocessing_config)

    def _restore_processor(self):
        if getattr(self, "artifact_version", None) != ARTIFACT_VERSION:
            raise ValueError(MIGRATION_MESSAGE)
        self._processor = VietnameseTextProcessor.from_config(self._preprocessing_config)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_processor", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Also enforced on direct joblib.load(), not just through load_model().
        self._restore_processor()

    def _prepare(self, raw_texts):
        if isinstance(raw_texts, str):
            raw_texts = [raw_texts]
        elif isinstance(raw_texts, (bytes, Mapping, Set)) or getattr(raw_texts, "ndim", 1) != 1:
            raise TypeError("Expected a raw string or an ordered iterable of raw strings.")
        else:
            try:
                raw_texts = list(raw_texts)
            except TypeError as error:
                raise TypeError("Expected a raw string or an ordered iterable of raw strings.") from error
        for index, value in enumerate(raw_texts):
            if not isinstance(value, str):
                raise TypeError(f"Input at index {index} must be a string; missing/numeric inputs are invalid.")
        return self._processor.transform(raw_texts) if raw_texts else []

    def predict(self, raw_texts):
        """Predict labels from raw reviews; never pass previously cleaned text."""
        cleaned = self._prepare(raw_texts)
        return self._pipeline.predict(cleaned) if cleaned else np.array([], dtype=self.classes_.dtype)

    def predict_proba(self, raw_texts):
        """Uncalibrated model probabilities; columns follow classes_."""
        cleaned = self._prepare(raw_texts)
        return self._pipeline.predict_proba(cleaned) if cleaned else np.empty((0, len(self.classes_)))

    def infer(self, raw_texts, *, include_probabilities=False):
        """Shared evaluation/UI path, preprocessing once even when returning scores."""
        cleaned = self._prepare(raw_texts)
        if not cleaned:
            return []
        labels = self._pipeline.predict(cleaned)
        probabilities = self._pipeline.predict_proba(cleaned) if include_probabilities else None
        results = []
        for index, (label, text) in enumerate(zip(labels, cleaned)):
            result = {"label": str(label), "empty_after_preprocessing": not bool(text.strip())}
            if probabilities is not None:
                result["probabilities"] = {str(name): float(score)
                    for name, score in zip(self.classes_, probabilities[index])}
                result["probability_kind"] = "uncalibrated"
            results.append(result)
        return results


def load_model(path: str | Path) -> SentimentModel:
    model = joblib.load(path)
    if not isinstance(model, SentimentModel) or model.artifact_version != ARTIFACT_VERSION:
        raise ValueError(MIGRATION_MESSAGE)
    return model
