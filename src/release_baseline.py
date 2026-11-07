"""Fresh-clone baseline release workflow: doctor, train, evaluate and raw predict.

No command evaluates a prospective/new test. Evaluation defaults to the already
consumed validation set; use the separately gated V2 scorer for future test data.
"""
from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import shutil
import struct
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.experiments import code_snapshot, environment_snapshot, seed_everything
from src.frozen_baselines import metric_table, read_csv, write_table
from src.inference import SentimentModel, load_model
from src.mlp import LABEL_TO_ID, TfidfDataset, load_checkpoint
from src.mlp_pipeline import export_pipeline, load_pipeline
from src.prepare_data import verify_splits
from src.release_config import DEFAULT_CONFIG, ROOT, fingerprint, load_config, write_json
from src.run_mlp import fit_mlp


def doctor(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    expected = config["environment"]
    if (platform.python_version() != expected["python"] or platform.system() != expected["platform"]
            or struct.calcsize("P") * 8 != expected["bits"]):
        raise ValueError(f"Release requires {expected['platform']} {expected['bits']}-bit Python {expected['python']}")
    lock = ROOT / expected["lock"]
    packages = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ]+)", line)
        if match:
            name, target = match.groups()
            if version(name) != target:
                raise ValueError(f"Dependency differs from release lock: {name}, expected {target}, found {version(name)}")
            packages[name] = target
    if version("torch") != "2.15.0+cpu" or torch.cuda.is_available():
        raise ValueError("This reproducible profile requires PyTorch 2.15.0+cpu")
    packages["torch"] = version("torch")
    return {"status": "OK", "python": platform.python_version(), "platform": platform.system(),
            "isolated_venv": sys.prefix != sys.base_prefix, "lock": fingerprint(lock),
            "packages_checked": len(packages), "seed": config["seed"], "threads": config["threads"]}


def verified_run(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "release_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported release manifest")
    for name, expected in manifest["files"].items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir) or fingerprint(path) != expected:
            raise ValueError(f"Release artifact/source/split changed: {name}")
    return manifest


def output_directory(path, run_dir):
    path, run_dir = Path(path).resolve(), Path(run_dir).resolve()
    if path.is_relative_to(run_dir):
        raise ValueError("Evaluation outputs must be outside the immutable training run")
    path.mkdir(parents=True, exist_ok=False)
    return path


def infer_pair(run_dir, texts):
    run_dir = Path(run_dir)
    nb = load_model(run_dir / "artifacts/complement_nb.joblib")
    mlp = load_pipeline(run_dir / "artifacts/mlp")
    left = nb.infer(texts, include_probabilities=True)
    right = mlp.infer(texts, include_probabilities=True, include_logits=True)
    return {"ComplementNB": [row["label"] for row in left], "MLP": [row["label"] for row in right]}, left, right


def train(config_path, split_dir, output_dir):
    config = load_config(config_path)
    environment_check = doctor(config_path)
    split_dir, output_dir = Path(split_dir).resolve(), Path(output_dir).resolve()
    split_manifest = verify_splits(config_path, split_dir)
    if output_dir.exists():
        raise FileExistsError("Run already exists; use a fresh --output-dir")
    torch.set_num_threads(config["threads"])
    seed_everything(config["seed"], torch_enabled=True)
    train_rows, val_rows = (read_csv(split_dir / name) for name in ("train_split.csv", "validation_split.csv"))
    train_text = [row["clean_text"] for row in train_rows]
    val_text = [row["clean_text"] for row in val_rows]
    processor_config = split_manifest["model_preprocessing"]
    # Verify raw-to-clean compatibility before fitting any model.
    from src.preprocessor import VietnameseTextProcessor
    processor = VietnameseTextProcessor.from_config(processor_config)
    for rows, expected in ((train_rows, train_text), (val_rows, val_text)):
        if processor.transform([row["review"] for row in rows]) != expected:
            raise ValueError("Prepared clean_text differs from recorded preprocessing")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "training").mkdir()
    (output_dir / "artifacts").mkdir()
    (output_dir / "splits").mkdir()
    (output_dir / "validation").mkdir()
    shutil.copyfile(config_path, output_dir / "run_config.json")
    for name in ("train_split.csv", "validation_split.csv", "test_split.csv", "split_manifest.json"):
        shutil.copyfile(split_dir / name, output_dir / "splits" / name)
    write_json(output_dir / "environment_check.json", environment_check)
    options = config["tfidf"].copy()
    options["dtype"] = np.float32
    options["ngram_range"] = tuple(options["ngram_range"])
    vectorizer = TfidfVectorizer(**options)
    started = time.perf_counter()
    x_train = vectorizer.fit_transform(train_text)
    x_val = vectorizer.transform(val_text)
    feature_seconds = time.perf_counter() - started
    joblib.dump(vectorizer, output_dir / "training/tfidf.joblib")
    y_train_names = [row["label"] for row in train_rows]
    y_val_names = [row["label"] for row in val_rows]
    nb = ComplementNB(**config["nb"])
    started = time.perf_counter()
    nb.fit(x_train, y_train_names)
    nb_seconds = time.perf_counter() - started
    nb_raw = SentimentModel(Pipeline([("tfidf", vectorizer), ("classifier", nb)]), processor_config)
    joblib.dump(nb_raw, output_dir / "artifacts/complement_nb.joblib")
    train_data = TfidfDataset(x_train, [LABEL_TO_ID[label] for label in y_train_names])
    val_data = TfidfDataset(x_val, [LABEL_TO_ID[label] for label in y_val_names])
    selected = config["mlp"]
    mlp_config = {"name": "dropout", "learning_rate": selected["learning_rate"], "dropout": selected["dropout"]}
    summary, _, history = fit_mlp(mlp_config, train_data, val_data, output_dir / "training", config["seed"],
        selected["epochs"], selected["batch_size"],
        {"release_id": config["release_id"], "run_config": fingerprint(config_path),
         "tfidf_sha256": fingerprint(output_dir / "training/tfidf.joblib")["sha256"],
         "split_hashes": {name: split_manifest["files"][name] for name in ("train_split.csv", "validation_split.csv")}})
    _, checkpoint = load_checkpoint(output_dir / "training" / summary["checkpoint"])
    export_pipeline(output_dir / "artifacts/mlp", checkpoint, vectorizer, processor_config)
    predictions, left, right = infer_pair(output_dir, [row["review"] for row in val_rows])
    table = metric_table(y_val_names, predictions)
    write_table(output_dir / "validation", table)
    write_json(output_dir / "validation/predictions.json", [{"source_id": row["source_id"], "label": row["label"],
        "ComplementNB": left[i], "MLP": right[i]} for i, row in enumerate(val_rows)])
    for row in table:
        target = config["expected"]["validation_macro_f1"][row["model"]]
        if abs(row["macro_f1"] - target) > 1e-12:
            raise RuntimeError(f"Release reproduction drift: {row['model']} macro-F1 {row['macro_f1']} != {target}")
    if summary["best_epoch"] != config["expected"]["mlp_best_epoch"]:
        raise RuntimeError("Release reproduction selected a different epoch; inspect environment/source before accepting")
    metadata = {"release_id": config["release_id"], "config": fingerprint(config_path), "seed": config["seed"],
        "threads": config["threads"], "train_rows": len(train_rows), "validation_rows": len(val_rows),
        "features": x_train.shape[1], "tfidf_fit_transform_seconds": feature_seconds,
        "ComplementNB_fit_seconds": nb_seconds, "MLP": summary, "test_evaluated": False,
        "selection": "Fixed dropout config; highest OLD validation macro-F1, earliest epoch on ties; no refit on validation",
        "environment": environment_snapshot(), "code": code_snapshot(), "data": split_manifest["files"]}
    write_json(output_dir / "metadata.json", metadata)
    # Preserve actual code and lock/config files for this run, even before a Git commit.
    for name in ("src", "assets", "tests", "configs", "requirements"):
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                target = output_dir / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    for name in ("requirements.txt", ".python-version"):
        shutil.copyfile(ROOT / name, output_dir / "source" / name)
    write_json(output_dir / "release_manifest.json", {"schema_version": 1, "release_id": config["release_id"],
        "files": {path.relative_to(output_dir).as_posix(): fingerprint(path)
                  for path in sorted(output_dir.rglob("*")) if path.is_file()}})
    print(json.dumps({"run_dir": str(output_dir), "best_epoch": summary["best_epoch"],
        "validation": [{key: row[key] for key in ("model", "accuracy", "macro_f1")} for row in table]}, indent=2), flush=True)
    return output_dir


def evaluate(run_dir, output_dir, split="validation"):
    run_dir = Path(run_dir).resolve()
    verified_run(run_dir)
    doctor(run_dir / "run_config.json")
    if split not in ("train", "validation"):
        raise ValueError("This release evaluator does not open test; use the gated prospective protocol")
    output = output_directory(output_dir, run_dir)
    rows = read_csv(run_dir / "splits" / f"{split}_split.csv")
    predictions, left, right = infer_pair(run_dir, [row["review"] for row in rows])
    table = metric_table([row["label"] for row in rows], predictions)
    write_table(output, table)
    write_json(output / "predictions.json", [{"source_id": row["source_id"], "label": row["label"],
        "ComplementNB": left[i], "MLP": right[i]} for i, row in enumerate(rows)])
    if split == "validation":
        expected = json.loads((run_dir / "validation/predictions.json").read_text(encoding="utf-8"))
        for i, row in enumerate(rows):
            if left[i]["label"] != expected[i]["ComplementNB"]["label"] or right[i]["label"] != expected[i]["MLP"]["label"]:
                raise RuntimeError("Reloaded predictions differ from the saved release")
        np.testing.assert_allclose([row["logits"] for row in right], [row["MLP"]["logits"] for row in expected], rtol=1e-6, atol=1e-7)
    write_json(output / "evaluation_receipt.json", {"release": fingerprint(run_dir / "release_manifest.json"),
        "split": split, "rows": len(rows), "split_fingerprint": fingerprint(run_dir / "splits" / f"{split}_split.csv"),
        "loaded_predictions_match": split == "validation", "test_evaluated": False})
    return table


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    t = sub.add_parser("train")
    t.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    t.add_argument("--split-dir", type=Path, default=ROOT / "data/prepared/baseline-v1.0.0/splits")
    t.add_argument("--output-dir", type=Path, default=ROOT / "models/releases/baseline-v1.0.0")
    e = sub.add_parser("evaluate")
    e.add_argument("--run-dir", type=Path, required=True)
    e.add_argument("--output-dir", type=Path, required=True)
    e.add_argument("--split", choices=("train", "validation"), default="validation")
    p = sub.add_parser("predict")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--text", nargs="+", required=True)
    p.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.command in ("doctor", "train") else args.run_dir / "run_config.json"
    config = load_config(config_path)
    torch.set_num_threads(config["threads"])
    with threadpool_limits(limits=config["threads"]):
        if args.command == "doctor":
            print(json.dumps(doctor(args.config), indent=2))
        elif args.command == "train":
            train(args.config, args.split_dir, args.output_dir)
        elif args.command == "evaluate":
            print(json.dumps(evaluate(args.run_dir, args.output_dir, args.split), indent=2))
        else:
            verified_run(args.run_dir)
            doctor(config_path)
            _, left, right = infer_pair(args.run_dir, args.text)
            payload = [{"review": text, "ComplementNB": left[i], "MLP": right[i]} for i, text in enumerate(args.text)]
            if args.output_json:
                if args.output_json.resolve().is_relative_to(args.run_dir.resolve()):
                    raise ValueError("Write predictions outside the immutable run")
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output_json, payload)
            else:
                print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
