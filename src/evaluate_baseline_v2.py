"""One prospective evaluation of a frozen baseline, gated by the V2 protocol.

No dataset is downloaded here. A custodian must supply a sealed release manifest
after all seven gates pass. Gate declarations do not independently prove the
human annotation/provenance audits; their receipts must be retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.experiments import file_fingerprint
from src.frozen_baselines import (BASELINE_ID, LABEL_TO_ID, NEW_COHORT, WINDOW_END, WINDOW_START,
    check_runtime, metric_table, new_output, predict_pair, read_json, text_hash, verify_bundle, write_json, write_table)


def iso_datetime(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps require an explicit timezone")
    return result


def validate_release(release, *, now=None):
    if release.get("status") != "SEALED_READY":
        raise ValueError("New test is NOT_READY; no input or labels will be opened")
    if release.get("baseline_id") != BASELINE_ID or release.get("cohort") != NEW_COHORT:
        raise ValueError("Baseline/cohort differs from the locked V2 protocol")
    now = now or datetime.now(timezone.utc)
    if now < WINDOW_END:
        raise ValueError("Protocol V2 test cannot open before 2026-12-06T00:00:00+07:00")
    opened = iso_datetime(release["opened_at"])
    if not WINDOW_END <= opened <= now:
        raise ValueError("opened_at must be after the protocol gate and no later than execution")
    for name in ("custodian", "evaluation_operator", "annotation_receipt", "contamination_audit_receipt"):
        if not isinstance(release.get(name), str) or not release[name].strip():
            raise ValueError(f"Missing role/audit receipt: {name}")
    if release["custodian"] == release["evaluation_operator"]:
        raise ValueError("Custodian and model evaluation operator must be separate roles")
    if release.get("gates") != {f"G{i}": True for i in range(1, 8)}:
        raise ValueError("All protocol gates G1-G7 must be explicitly true")


def ordered_ids_hash(rows):
    payload = [[row["source_dataset"], row["review_id"]] for row in rows]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_inputs(rows, registry):
    if len(rows) != 1000:
        raise ValueError("V2 requires exactly 1000 sealed reviews")
    seen = set()
    raw_seen = set()
    legacy_raw = set(registry["raw_text_hashes"])
    legacy_ids = set(registry["legacy_review_ids"])
    for row in rows:
        for name in ("source_dataset", "review_id", "product_key", "review_created_at", "collected_at", "source_reference"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Missing trusted new-test metadata: {name}")
        if "label" in row or not isinstance(row.get("review"), str):
            raise ValueError("Input must contain raw strings and no labels")
        identity = (row["source_dataset"], row["review_id"])
        if identity in seen or row["review_id"] in legacy_ids:
            raise ValueError("Duplicate/development review identity")
        seen.add(identity)
        timestamp = iso_datetime(row["review_created_at"])
        if not WINDOW_START <= timestamp < WINDOW_END or iso_datetime(row["collected_at"]) < timestamp:
            raise ValueError("Review is outside prospective window or has inconsistent collection time")
        raw_hash = text_hash(row["review"])
        if raw_hash in legacy_raw or raw_hash in raw_seen:
            raise ValueError("Raw-text overlap with development or within new test")
        raw_seen.add(raw_hash)
    groups = Counter(row["product_key"] for row in rows)
    if len(groups) < 200 or max(groups.values()) > 5:
        raise ValueError("V2 requires >=200 products and <=5 reviews per product")


def macro_from_confusion(matrix):
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.divide(2 * np.diag(matrix), denominator,
                          out=np.zeros(2, dtype=float), where=denominator != 0).mean())


def paired_product_bootstrap(truth, predictions, products, *, repeats=10000, seed=20261107):
    keys = sorted(set(products))
    positions = {key: i for i, key in enumerate(keys)}
    matrices = np.zeros((len(keys), 2, 2, 2), dtype=np.int64)  # product, model, true, predicted
    for i, product in enumerate(products):
        for model_index, name in enumerate(("ComplementNB", "MLP")):
            matrices[positions[product], model_index, LABEL_TO_ID[truth[i]], LABEL_TO_ID[predictions[name][i]]] += 1
    rng = np.random.default_rng(seed)
    scores = np.empty((repeats, 3))
    for i in range(repeats):
        sample = matrices[rng.integers(0, len(keys), size=len(keys))].sum(axis=0)
        nb, mlp = macro_from_confusion(sample[0]), macro_from_confusion(sample[1])
        scores[i] = nb, mlp, mlp - nb
    intervals = np.percentile(scores, [2.5, 97.5], axis=0, method="linear")
    return {"method": "paired cluster bootstrap by product_key; fixed two labels; zero_division=0",
            "seed": seed, "repeats": repeats, "products": len(keys),
            "ci95": {name: intervals[:, i].tolist() for i, name in enumerate(("ComplementNB", "MLP", "delta_mlp_minus_nb"))},
            "resamples": scores.tolist()}


def evaluate(bundle, release_path, output):
    bundle, release_path = Path(bundle).resolve(), Path(release_path).resolve()
    release = read_json(release_path)
    validate_release(release)  # Never touch new input/labels before this guard.
    manifest = verify_bundle(bundle)
    check_runtime(bundle)
    if release["baseline_manifest_sha256"] != file_fingerprint(bundle / "manifest.json")["sha256"]:
        raise ValueError("Release baseline fingerprint differs")
    if release["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("Release protocol fingerprint differs")
    output = new_output(output, bundle)
    input_path = (release_path.parent / release["input_file"]).resolve()
    if file_fingerprint(input_path) != release["input_fingerprint"]:
        raise ValueError("Sealed input fingerprint differs")
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    validate_inputs(rows, read_json(bundle / "development_registry.json"))
    if ordered_ids_hash(rows) != release["ordered_ids_sha256"]:
        raise ValueError("Ordered input identity fingerprint differs")
    predictions, mlp_rows = predict_pair(bundle, [row["review"] for row in rows])
    committed = [{"source_dataset": row["source_dataset"], "review_id": row["review_id"],
                  "product_key": row["product_key"], "ComplementNB": predictions["ComplementNB"][i],
                  "MLP": predictions["MLP"][i], "mlp_logits": mlp_rows[i]["logits"],
                  "empty_after_preprocessing": mlp_rows[i]["empty_after_preprocessing"],
                  "no_known_features": mlp_rows[i]["no_known_features"]} for i, row in enumerate(rows)]
    write_json(output / "predictions.json", committed)
    write_json(output / "prediction_commit.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "predictions": file_fingerprint(output / "predictions.json"), "input": release["input_fingerprint"],
        "baseline_manifest_sha256": release["baseline_manifest_sha256"], "labels_opened": False})
    # Both models' outputs are on disk and hashed before the label file is read.
    label_path = (release_path.parent / release["label_file"]).resolve()
    if file_fingerprint(label_path) != release["label_fingerprint"]:
        raise ValueError("Sealed label fingerprint differs")
    labels = [json.loads(line) for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if ordered_ids_hash(labels) != ordered_ids_hash(rows):
        raise ValueError("Label identities/order differ from committed predictions")
    truth = [row["label"] for row in labels]
    counts = Counter(truth)
    if set(counts) != set(LABEL_TO_ID) or min(counts.values()) < 100:
        raise ValueError("Sealed labels fail the two-class/support gate; do not resample or report partial results")
    table = metric_table(truth, predictions)
    write_table(output, table)
    bootstrap = paired_product_bootstrap(truth, predictions, [row["product_key"] for row in rows])
    write_json(output / "paired_bootstrap.json", bootstrap)
    write_json(output / "evaluation_receipt.json", {"baseline_id": BASELINE_ID, "cohort": NEW_COHORT,
        "status": "COMPLETED", "release_manifest": file_fingerprint(release_path),
        "rows": len(rows), "products": len({row["product_key"] for row in rows}),
        "label_counts": dict(counts), "empty_after_preprocessing": sum(row["empty_after_preprocessing"] for row in mlp_rows),
        "no_known_features": sum(row["no_known_features"] for row in mlp_rows), "processing_failures": 0,
        "primary_metric": "macro_f1", "delta_mlp_minus_nb": table[1]["macro_f1"] - table[0]["macro_f1"],
        "source_file": str(Path(__file__).resolve()), "refit": False,
        "audit_limit": "Clean-text/near-duplicate independence, sampling and annotation are certified by custodian G2-G4 receipts, not proved by this scorer."})
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        table = evaluate(args.bundle, args.release_manifest, args.output)
    print(json.dumps([{key: row[key] for key in ("model", "macro_f1")} for row in table], indent=2))


if __name__ == "__main__":
    main()
