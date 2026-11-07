"""Version, verify and replay fixed NB/MLP baselines; never refit during evaluation.

Run replay commands from the bundle's source/ directory to use the exact saved
code. V2 evaluation remains gated until the prospective test has been sealed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from src.data_pipeline import normalize_text
from src.experiments import PROJECT_ROOT, code_snapshot, environment_snapshot, file_fingerprint
from src.inference import load_model
from src.mlp import LABELS, LABEL_TO_ID
from src.mlp_pipeline import load_pipeline

BASELINE_ID = "baseline_v1_20261107"
PROTOCOL_NAME = "evaluation_protocol_v2_20261107.md"
NEW_COHORT = "shopee_prospective_20261108_20261205_v2"
WINDOW_START = datetime.fromisoformat("2026-11-08T00:00:00+07:00")
WINDOW_END = datetime.fromisoformat("2026-12-06T00:00:00+07:00")


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def text_hash(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("baseline_id") != BASELINE_ID:
        raise ValueError("Unsupported frozen baseline version")
    for name, fingerprint in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError(f"Manifest path escapes bundle: {name}")
        if file_fingerprint(path) != fingerprint:
            raise ValueError(f"Frozen bundle fingerprint mismatch: {name}")
    return manifest


def check_runtime(bundle):
    expected = read_json(Path(bundle) / "config.json")["runtime"]
    for name, target in expected.items():
        if version(name) != target:
            raise ValueError(f"Runtime mismatch: {name} requires {target}, found {version(name)}")


def new_output(output, bundle):
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    if output.is_relative_to(bundle):
        raise ValueError("Write evaluation results outside the immutable baseline bundle")
    output.mkdir(parents=True, exist_ok=False)
    return output


def predict_pair(bundle, texts):
    nb = load_model(Path(bundle) / "artifacts/complement_nb.joblib")
    mlp = load_pipeline(Path(bundle) / "artifacts/mlp")
    nb_rows = nb.infer(texts)
    mlp_rows = mlp.infer(texts, include_logits=True)
    return {"ComplementNB": [row["label"] for row in nb_rows],
            "MLP": [row["label"] for row in mlp_rows]}, mlp_rows


def metric_table(truth, predictions):
    result = []
    for model in ("ComplementNB", "MLP"):
        pred = predictions[model]
        report = classification_report(truth, pred, labels=list(LABELS), target_names=LABELS,
                                       zero_division=0, output_dict=True)
        result.append({"model": model, "rows": len(truth),
                       "accuracy": float(accuracy_score(truth, pred)),
                       "macro_f1": float(f1_score(truth, pred, labels=list(LABELS), average="macro", zero_division=0)),
                       "f1_negative": report["negative"]["f1-score"],
                       "f1_positive": report["positive"]["f1-score"],
                       "classification_report": report,
                       "confusion_matrix": confusion_matrix(truth, pred, labels=list(LABELS)).tolist(),
                       "confusion_matrix_axes": {"rows": "true", "columns": "predicted", "labels": list(LABELS)}})
    return result


def write_table(output, table):
    columns = ("model", "rows", "accuracy", "macro_f1", "f1_negative", "f1_positive")
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in table)
    write_json(output / "metrics.json", table)


def evaluate_validation(bundle, output, *, replay=False):
    bundle = Path(bundle).resolve()
    manifest = verify_bundle(bundle) if replay else None
    check_runtime(bundle)
    output = Path(output).resolve()
    if not replay and output == bundle / "results/legacy_validation":
        output.mkdir(parents=True, exist_ok=False)  # Initial build, before manifest is sealed.
    else:
        output = new_output(output, bundle)
    rows = read_csv(bundle / "data/validation_split.csv")
    texts = [row["review"] for row in rows]
    truth = [row["label"] for row in rows]
    predictions, mlp_rows = predict_pair(bundle, texts)
    table = metric_table(truth, predictions)
    write_table(output, table)
    output_rows = [{"source_id": row["source_id"], "label": row["label"],
                    "ComplementNB": predictions["ComplementNB"][i], "MLP": predictions["MLP"][i],
                    "mlp_logits": mlp_rows[i]["logits"],
                    "empty_after_preprocessing": mlp_rows[i]["empty_after_preprocessing"],
                    "no_known_features": mlp_rows[i]["no_known_features"]}
                   for i, row in enumerate(rows)]
    write_json(output / "predictions.json", output_rows)
    if replay:
        expected_rows = read_json(bundle / "results/legacy_validation/predictions.json")
        if [{k: v for k, v in row.items() if k != "mlp_logits"} for row in output_rows] != [
                {k: v for k, v in row.items() if k != "mlp_logits"} for row in expected_rows]:
            raise RuntimeError("Replayed validation identities/labels/flags differ from frozen results")
        actual = np.asarray([row["mlp_logits"] for row in output_rows])
        expected = np.asarray([row["mlp_logits"] for row in expected_rows])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        if table != read_json(bundle / "results/legacy_validation/metrics.json"):
            raise RuntimeError("Replayed metrics differ from frozen results")
        write_json(output / "replay_verification.json", {"baseline_id": BASELINE_ID, "rows": len(rows),
            "same_labels": True, "same_metrics": True, "rtol": 1e-6, "atol": 1e-7,
            "max_abs_logits_diff": float(np.max(np.abs(actual - expected))),
            "manifest_sha256": file_fingerprint(bundle / "manifest.json")["sha256"],
            "executed_source": str(Path(__file__).resolve()), "new_test_accessed": False})
    return table


def freeze(bundle):
    bundle = Path(bundle).resolve()
    if bundle.exists():
        raise FileExistsError("Baseline version exists; never overwrite a frozen release")
    root = PROJECT_ROOT
    package = root / "models/mlp_packaged/mlp_raw_20261106_02"
    split_dir = root / "models/evaluation_workflow/full_validation_20260929_01"
    # Verify the source package before copying any artifacts.
    for name, expected in read_json(package / "artifact_manifest.json").items():
        if file_fingerprint(package / name) != expected:
            raise ValueError(f"Source package changed: {name}")
    source_state = code_snapshot(root)
    bundle.mkdir(parents=True, exist_ok=False)
    copy_file(package / "baseline/sentiment_pipeline.joblib", bundle / "artifacts/complement_nb.joblib")
    for path in sorted((package / "pipeline").iterdir()):
        if path.is_file():
            copy_file(path, bundle / "artifacts/mlp" / path.name)
    copy_file(package / "training/best_training.pt", bundle / "checkpoints/mlp_best_training.pt")
    for name in ("train_split.csv", "validation_split.csv", "test_split.csv"):
        copy_file(split_dir / name, bundle / "data" / name)
    for name in ("shopee_reviews_dataset.jsonl", "sample_data.jsonl", "aug_unaccented_reviews.jsonl"):
        copy_file(root / "data" / name, bundle / "data/originals" / name)
    for directory in ("src", "assets", "tests"):
        for path in sorted((root / directory).rglob("*.py")):
            copy_file(path, bundle / "source" / path.relative_to(root))
    for name in ("requirements.txt", "app.py"):
        copy_file(root / name, bundle / "source" / name)
    for name in (PROTOCOL_NAME, "data_card_20261107.md", "data_card_evidence_20261107.json"):
        copy_file(root / "docs" / name, bundle / "documentation" / name)
    for name in ("comparison.json", "comparison.csv", "reload_verification.json", "metadata.json",
                 "requirements-resolved.txt", "fixed_raw_reviews.json"):
        copy_file(package / name, bundle / "provenance" / name)
    nb = load_model(bundle / "artifacts/complement_nb.joblib")
    mlp = load_pipeline(bundle / "artifacts/mlp")
    vectorizer = nb._pipeline.named_steps["tfidf"]
    if (nb.preprocessing_config != mlp.preprocessing_config or vectorizer.vocabulary_ != mlp.vectorizer.vocabulary_
            or not np.array_equal(vectorizer.idf_, mlp.vectorizer.idf_)):
        raise ValueError("Baseline features or preprocessing do not match")
    train, validation = (read_csv(bundle / "data" / name) for name in ("train_split.csv", "validation_split.csv"))
    if len(train) != 7660 or len(validation) != 958:
        raise ValueError("Unexpected development split sizes")
    if set(row["source_id"] for row in train) & set(row["source_id"] for row in validation):
        raise ValueError("Train/validation source overlap")
    for rows in (train, validation):
        if mlp._processor.transform([row["review"] for row in rows]) != [row["clean_text"] for row in rows]:
            raise ValueError("Frozen preprocessing differs from saved clean_text")
    checkpoint = torch.load(bundle / "checkpoints/mlp_best_training.pt", map_location="cpu", weights_only=True)
    if any(not torch.equal(value, mlp.model.state_dict()[key]) for key, value in checkpoint["model_state_dict"].items()):
        raise ValueError("Training checkpoint differs from selected inference weights")
    # A hash-only contamination registry: no future test text or labels are present.
    known_texts, known_ids = set(), set()
    for path in (bundle / "data/originals").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                known_texts.add(row["review"])
                if row.get("id") is not None:
                    known_ids.add(str(row["id"]))
    known_texts.update(read_json(package / "fixed_raw_reviews.json"))
    known_texts.update(["Sản phẩm tốt", "Không hài lòng", "Sản phẩm không tốt", "Máy khuấy tốt",
                        "sản_phẩm không tốt", "giao_hàng nhanh chất_lượng tốt"])
    cleaned = mlp._processor.transform(sorted(known_texts))
    write_json(bundle / "development_registry.json", {
        "raw_text_hashes": sorted({text_hash(text) for text in known_texts}),
        "clean_text_hashes": sorted({text_hash(text) for text in cleaned if text.strip()}),
        "legacy_review_ids": sorted(known_ids),
        "scope": "All three old JSONL files plus saved/manual fixtures; raw originals cover all old splits.",
        "limitation": "Additional unrecorded developer examples and near-duplicates require the independent G3 audit."})
    tfidf_config = {key: getattr(vectorizer, key) for key in ("ngram_range", "min_df", "max_features", "sublinear_tf",
                    "lowercase", "token_pattern", "norm", "smooth_idf", "use_idf")}
    config = {"labels": LABEL_TO_ID, "input_contract": "raw_review", "preprocessing": mlp.preprocessing_config,
              "tfidf": {**tfidf_config, "dtype": str(vectorizer.dtype), "vocabulary_size": len(vectorizer.vocabulary_)},
              "ComplementNB": nb._pipeline.named_steps["classifier"].get_params(),
              "MLP": {"architecture": checkpoint["architecture"], "epoch": checkpoint["epoch"],
                      "validation_macro_f1": checkpoint["validation_macro_f1"],
                      "training_config": checkpoint["training_config"]},
              "runtime": {name: version(name) for name in ("torch", "scikit-learn", "numpy", "underthesea")},
              "evaluation": {"threads": 4, "batch_size": 128, "label_rule": "saved predict/argmax", "refit": False}}
    write_json(bundle / "config.json", config)
    write_json(bundle / "environment.json", environment_snapshot())
    write_json(bundle / "source_state.json", source_state)
    table = evaluate_validation(bundle, bundle / "results/legacy_validation")
    expected_table = read_json(package / "comparison.json")
    for actual, expected in zip(table, expected_table):
        for key in ("accuracy", "macro_f1", "f1_negative", "f1_positive"):
            if abs(actual[key] - expected[key]) > 1e-12:
                raise RuntimeError(f"Frozen model changed historical {key}")
    write_json(bundle / "results/protocol_v2_status.json", {"status": "NOT_RUN", "cohort": NEW_COHORT,
        "reason": "No independent collected/annotated/sealed test exists; V2 G2-G7 pending.",
        "earliest_open_at": WINDOW_END.isoformat(), "models": ["ComplementNB", "MLP"], "metrics": None})
    write_json(bundle / "protocol_v2_release_template.json", {"status": "NOT_READY", "baseline_id": BASELINE_ID,
        "cohort": NEW_COHORT, "opened_at": None, "custodian": None, "evaluation_operator": None,
        "annotation_receipt": None, "contamination_audit_receipt": None,
        "gates": {f"G{i}": False for i in range(1, 8)},
        "baseline_manifest_sha256": "FILL_FROM_EXTERNAL_FROZEN_MANIFEST_HASH",
        "protocol_sha256": file_fingerprint(bundle / "documentation" / PROTOCOL_NAME)["sha256"],
        "input_file": "custodian_raw_reviews.jsonl", "input_fingerprint": {"sha256": None, "size_bytes": None},
        "label_file": "custodian_labels.jsonl", "label_fingerprint": {"sha256": None, "size_bytes": None},
        "ordered_ids_sha256": None})
    (bundle / "README.md").write_text(
        f"# {BASELINE_ID}\n\nImmutable fitted NB/MLP release. No refit or new-test evaluation occurred.\n\n"
        "Run from source/ in a matching environment:\n\n"
        "```powershell\npython -B -m src.frozen_baselines verify --bundle ..\n"
        "python -B -m src.frozen_baselines reproduce --bundle .. --output ../../baseline_v1_replay_01\n```\n\n"
        "data/test_split.csv is the ALREADY-CONSUMED historical test, not protocol V2 data.\n"
        "Legacy validation results are developmental, not an independent holdout.\n"
        "provenance/comparison.* preserve old measured timing/size; wall times are not deterministic.\n"
        "Source revision may be dirty; all actual source bytes are archived and hashed.\n"
        "Keep manifest.json and its separately recorded SHA-256 as the integrity anchor.\n"
        "Dataset license/provenance limitations are described in documentation/data_card_20261107.md.\n",
        encoding="utf-8")
    manifest = {"schema_version": 1, "baseline_id": BASELINE_ID, "frozen_on": "2026-11-07",
                "new_test_status": "NOT_RUN", "selection": "MLP dropout 0.3, seed 42, epoch 9 from OLD validation; NB alpha 0.5 fixed control",
                "data_roles": {"train_split.csv": "legacy training", "validation_split.csv": "consumed development/selection",
                               "test_split.csv": "consumed historical test; never independent V2 test"},
                "protocol_sha256": file_fingerprint(bundle / "documentation" / PROTOCOL_NAME)["sha256"],
                "files": {path.relative_to(bundle).as_posix(): file_fingerprint(path)
                          for path in sorted(bundle.rglob("*")) if path.is_file()}}
    write_json(bundle / "manifest.json", manifest)
    verify_bundle(bundle)
    return bundle, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify", "reproduce"):
        command = sub.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        if name == "reproduce":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.command == "freeze":
            bundle, table = freeze(args.bundle)
            print(json.dumps({"bundle": str(bundle), "manifest": file_fingerprint(bundle / "manifest.json"),
                              "validation": [{k: row[k] for k in ("model", "macro_f1", "f1_negative", "f1_positive")} for row in table]}, indent=2))
        elif args.command == "verify":
            manifest = verify_bundle(args.bundle)
            print(json.dumps({"baseline_id": manifest["baseline_id"], "verified_files": len(manifest["files"])}))
        else:
            evaluate_validation(args.bundle, args.output, replay=True)
            print(f"Reproduced frozen validation results: {args.output}")


if __name__ == "__main__":
    main()
