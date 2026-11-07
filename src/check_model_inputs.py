"""Audit separate TF-IDF/XLM-R routes on the SAME old-development samples and labels.

No fitting, pretrained model weights, sentiment predictions or test access.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch

from src.experiments import PROJECT_ROOT, file_fingerprint
from src.frozen_baselines import verify_bundle
from src.review_inputs import RawReview, RawReviewDataset, assert_same_samples
from src.tfidf_inputs import TfidfReviewInputs
from src.transformer_inputs import DEFAULT_TOKENIZER_DIR, MODEL_ID, REVISION, XLMRReviewInputs


def identity_digest(records):
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def run(baseline, tokenizer_dir, output_dir, batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    baseline, output_dir = Path(baseline).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(baseline):
        raise ValueError("Do not write into the frozen baseline")
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    is_release = (baseline / "release_manifest.json").is_file()
    if is_release:
        from src.release_baseline import verified_run
        manifest = verified_run(baseline)
    else:
        manifest = verify_bundle(baseline)
    tfidf = TfidfReviewInputs.from_baseline(baseline)
    transformer = XLMRReviewInputs(tokenizer_dir)
    report = {"selected_model": MODEL_ID, "revision": REVISION, "max_length": transformer.max_length,
              "single_review_special_tokens": transformer.special_tokens,
              "baseline_id": manifest.get("baseline_id", manifest.get("release_id")),
              "baseline_manifest": file_fingerprint(baseline / ("release_manifest.json" if is_release else "manifest.json")),
              "tokenizer_profile": transformer.profile, "splits": {}, "examples": [],
              "test_accessed": False, "fit_performed": False, "model_weights_downloaded": False,
              "sentiment_model_trained": False,
              "runtime": {name: version(name) for name in ("torch", "numpy", "scikit-learn", "underthesea", "transformers", "tokenizers")}}
    for split in ("train", "validation"):
        path = baseline / ("splits" if is_release else "data") / f"{split}_split.csv"
        data = RawReviewDataset.from_csv(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            saved_clean = [row["clean_text"] for row in csv.DictReader(handle)]
        full_lengths, identities = [], []
        truncated = no_features = no_content = unknown = 0
        for start in range(0, len(data), batch_size):
            records = data[start:start + batch_size]
            left, right = tfidf.encode(records), transformer.encode(records)
            assert_same_samples(left.identity, right.identity)
            if left.clean_text != tuple(saved_clean[start:start + len(records)]):
                raise RuntimeError("TF-IDF representation changed from frozen split")
            full_lengths.extend(right.full_token_lengths)
            truncated += sum(right.truncated)
            no_features += sum(left.no_known_features)
            no_content += sum(right.no_content_tokens)
            unknown += sum(right.unknown_token_counts)
            identities.extend([[row.source_id, digest, int(label)] for row, digest, label in
                               zip(records, left.identity.raw_sha256, left.identity.labels.tolist())])
            if start == 0:
                for i, row in enumerate(records[:3]):
                    report["examples"].append({"split": split, "source_id": row.source_id,
                        "label": row.label, "label_id": int(left.identity.labels[i]), "raw_review": row.review,
                        "tfidf_clean_text": left.clean_text[i], "transformer_input_text": row.review,
                        "transformer_token_ids_first_24": right.model_inputs["input_ids"][i, :24].tolist(),
                        "transformer_full_length": right.full_token_lengths[i],
                        "transformer_truncated": right.truncated[i]})
        report["splits"][split] = {"rows": len(data), "file": file_fingerprint(path),
            "label_counts": dict(Counter(row.label for row in data)), "sample_order_raw_hashes_labels_match": True,
            "ordered_source_raw_label_sha256": identity_digest(identities),
            "tfidf_matches_saved_clean_text": True, "tfidf_no_known_features": no_features,
            "transformer_no_content_tokens": no_content, "transformer_unknown_tokens": unknown,
            "truncated_reviews": truncated, "truncated_fraction": truncated / len(data),
            "untruncated_token_length": {"median": float(np.median(full_lengths)),
                                         "p95": float(np.percentile(full_lengths, 95)), "max": max(full_lengths)}}
        print(f"{split}: {len(data)} identical samples/labels; Transformer truncated {truncated}", flush=True)
    edges = [RawReview("edge-empty", ""), RawReview("edge-punctuation", "!!!"),
             RawReview("edge-case-emoji", "SẢN PHẨM này KHÔNG tốt!!! 😞"),
             RawReview("edge-oov", "xqzvvvzzrandomtoken20261106"),
             RawReview("edge-long", "tốt " * 1200 + "không tốt")]
    left, right = tfidf.encode(edges), transformer.encode(edges)
    assert_same_samples(left.identity, right.identity)
    report["edge_cases"] = [{"source_id": row.source_id, "raw_preview": row.review[:100],
        "tfidf_clean_preview": left.clean_text[i][:100], "tfidf_empty": left.empty_after_preprocessing[i],
        "tfidf_no_known_features": left.no_known_features[i], "transformer_empty_raw": right.empty_raw[i],
        "transformer_no_content_tokens": right.no_content_tokens[i], "transformer_unknown_tokens": right.unknown_token_counts[i],
        "transformer_full_length": right.full_token_lengths[i], "transformer_encoded_length": right.token_lengths[i],
        "transformer_truncated": right.truncated[i]} for i, row in enumerate(edges)]
    if not right.truncated[-1] or right.token_lengths[-1] != 512:
        raise RuntimeError("Long-input boundary check failed")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Saved preprocessing audit: {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "models/baselines/baseline_v1_20261107")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/preprocessing_routes/routes_20261107_01")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.baseline, args.tokenizer_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
