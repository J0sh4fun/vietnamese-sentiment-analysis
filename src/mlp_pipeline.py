"""Portable raw-review MLP bundle. python -m src.mlp_pipeline --help."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Set
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.experiments import file_fingerprint
from src.mlp import LABELS, LABEL_TO_ID, load_checkpoint
from src.preprocessor import VietnameseTextProcessor


def raw_strings(values):
    if isinstance(values, str):
        return [values]
    if isinstance(values, (bytes, Mapping, Set)) or getattr(values, "ndim", 1) != 1:
        raise TypeError("Expected a raw string or an ordered iterable of strings.")
    try:
        values = list(values)
    except TypeError as error:
        raise TypeError("Expected a raw string or an ordered iterable of strings.") from error
    if any(not isinstance(value, str) for value in values):
        raise TypeError("All reviews must be strings; missing/numeric/nested values are invalid.")
    return values


class RawTextMLP:
    """Raw strings in; preprocess exactly once per call, with explicit zero-feature flags."""
    def __init__(self, model, vectorizer, preprocessing_config, *, batch_size=128):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(vectorizer.vocabulary_) != model.hidden.in_features:
            raise ValueError("TF-IDF vocabulary and MLP dimensions do not match")
        self.model = model.eval()
        self.vectorizer = vectorizer
        self._preprocessing_config = deepcopy(preprocessing_config)
        self._processor = VietnameseTextProcessor.from_config(preprocessing_config)
        self.batch_size = batch_size

    @property
    def classes_(self):
        return np.asarray(LABELS)

    @property
    def preprocessing_config(self):
        return deepcopy(self._preprocessing_config)

    def infer(self, raw_texts, *, include_probabilities=False, include_logits=False):
        values = raw_strings(raw_texts)
        if not values:
            return []
        cleaned = self._processor.transform(values)  # The only preprocessing call.
        features = self.vectorizer.transform(cleaned).tocsr().astype(np.float32)
        logits = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = torch.from_numpy(features[start:start + self.batch_size].toarray())
                logits.append(self.model(batch))
        scores = torch.cat(logits)
        probabilities = scores.softmax(dim=1) if include_probabilities else None
        known = np.asarray(features.getnnz(axis=1)).ravel()
        results = []
        for index, text in enumerate(cleaned):
            empty = not bool(text.strip())
            warnings = []
            if empty:
                warnings.append("empty_after_preprocessing")
            if known[index] == 0:
                warnings.append("no_known_features: prediction uses learned biases; no known text features")
            result = {"label": LABELS[int(scores[index].argmax())],
                      "empty_after_preprocessing": empty,
                      "no_known_features": bool(known[index] == 0), "warnings": warnings}
            if include_probabilities:
                result["probabilities"] = dict(zip(LABELS, probabilities[index].tolist()))
                result["probability_kind"] = "uncalibrated"
            if include_logits:
                result["logits"] = scores[index].tolist()
            results.append(result)
        return results

    def predict(self, raw_texts):
        return np.asarray([row["label"] for row in self.infer(raw_texts)], dtype=str)

    def predict_proba(self, raw_texts):
        rows = self.infer(raw_texts, include_probabilities=True)
        return np.asarray([[row["probabilities"][label] for label in LABELS] for row in rows],
                          dtype=np.float32).reshape(-1, 2)


def export_pipeline(output_dir, checkpoint, vectorizer, preprocessing_config):
    """Write a standalone bundle without optimizer state, using exclusive directory creation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    inference_checkpoint = {key: deepcopy(checkpoint[key]) for key in (
        "artifact_version", "input_contract", "labels", "label_mapping", "architecture", "epoch",
        "validation_macro_f1", "model_state_dict")}
    torch.save(inference_checkpoint, output_dir / "model.pt")
    joblib.dump(vectorizer, output_dir / "tfidf.joblib")
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing_config, ensure_ascii=False, indent=2), encoding="utf-8")
    from importlib.metadata import version
    manifest = {"bundle_version": 1, "kind": "raw_text_mlp", "input_contract": "raw_review",
                "labels": LABEL_TO_ID, "batch_size": 128,
                "zero_feature_policy": "retain; predict from biases; return explicit warning flags",
                "environment": {name: version(name) for name in ("torch", "scikit-learn", "numpy", "underthesea")},
                "files": {name: file_fingerprint(output_dir / name) for name in
                          ("model.pt", "tfidf.joblib", "preprocessing.json")}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def load_pipeline(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("bundle_version") != 1 or manifest.get("kind") != "raw_text_mlp"
            or manifest.get("labels") != LABEL_TO_ID):
        raise ValueError("Unsupported raw-text MLP bundle")
    names = {"model.pt", "tfidf.joblib", "preprocessing.json"}
    if set(manifest["files"]) != names:
        raise ValueError("Incomplete bundle manifest")
    for name in names:
        if file_fingerprint(path / name) != manifest["files"][name]:
            raise ValueError(f"Bundle fingerprint mismatch: {name}")
    model, checkpoint = load_checkpoint(path / "model.pt")
    vectorizer = joblib.load(path / "tfidf.joblib")
    config = json.loads((path / "preprocessing.json").read_text(encoding="utf-8"))
    return RawTextMLP(model, vectorizer, config, batch_size=manifest["batch_size"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", nargs="+")
    inputs.add_argument("--input-json", type=Path, help="UTF-8 JSON array of raw reviews")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--logits", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    texts = args.text if args.input_json is None else json.loads(args.input_json.read_text(encoding="utf-8"))
    rows = load_pipeline(args.model_dir).infer(texts, include_probabilities=True, include_logits=args.logits)
    output = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output_json:
        with args.output_json.open("x", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
