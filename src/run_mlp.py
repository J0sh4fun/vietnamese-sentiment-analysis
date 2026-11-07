"""Controlled MLP comparison using only existing train/validation clean_text.

No test file is opened. Every model shares one train-fitted TF-IDF and the
same seed/batches. Each MLP runs all 15 epochs; earliest maximum validation
macro-F1 wins. Run directories are immutable (a fresh name is required).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB

from src.experiments import PROJECT_ROOT, code_snapshot, environment_snapshot, file_fingerprint
from src.mlp import (LABELS, LABEL_TO_ID, SentimentMLP, TfidfDataset, evaluate,
                     load_checkpoint, make_loader, seed_everything, train_epoch, training_checkpoint)

CONFIGURATIONS = (
    {"name": "reference", "learning_rate": 0.001, "dropout": 0.0},
    {"name": "lower_lr", "learning_rate": 0.0003, "dropout": 0.0},
    {"name": "dropout", "learning_rate": 0.001, "dropout": 0.3},
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_splits(split_dir, expected_train=7660, expected_validation=958):
    frames = []
    for name, expected in (("train", expected_train), ("validation", expected_validation)):
        frame = pd.read_csv(Path(split_dir) / f"{name}_split.csv", keep_default_na=False,
                            dtype={"source_id": str, "clean_text": str, "label": str})
        required = {"source_id", "clean_text", "label"}
        if not required.issubset(frame):
            raise ValueError(f"{name}: missing required columns {required - set(frame)}")
        if len(frame) != expected:
            raise ValueError(f"{name}: expected {expected} rows, got {len(frame)}")
        if frame.source_id.str.strip().eq("").any() or frame.source_id.duplicated().any():
            raise ValueError(f"{name}: empty or duplicate source_id (original-only split required)")
        if set(frame.label) != set(LABELS):
            raise ValueError(f"{name}: expected both labels, exactly {LABELS}")
        if "is_augmented" in frame and frame.is_augmented.astype(str).str.lower().ne("false").any():
            raise ValueError(f"{name}: augmented rows are not permitted in this experiment")
        frames.append(frame)
    train, validation = frames
    if set(train.source_id) & set(validation.source_id):
        raise ValueError("source_id overlap between train and validation")
    for column in ("clean_text", "normalized_text", "normalized_clean_text"):
        if column in train and column in validation:
            left = set(train[column].str.strip().str.lower()) - {""}
            right = set(validation[column].str.strip().str.lower()) - {""}
            if left & right:
                raise ValueError(f"{column} overlap between train and validation")
    return train, validation


def fit_features(train_texts, validation_texts):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000,
                                sublinear_tf=True, lowercase=False,
                                token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    train_features = vectorizer.fit_transform(train_texts)
    validation_features = vectorizer.transform(validation_texts)
    return vectorizer, train_features, validation_features


def metric_summary(result):
    return {key: value for key, value in result.items() if key != "predictions"}


def overfit_32(train_dataset, seed):
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(np.flatnonzero(train_dataset.labels == label), 16, replace=False)
                              for label in (0, 1)])
    subset = TfidfDataset(train_dataset.features[indices], train_dataset.labels[indices])
    seed_everything(seed)
    model = SentimentMLP(train_dataset.features.shape[1])
    loader = make_loader(subset, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    initial = metric_summary(evaluate(model, loader))
    history = []
    for epoch in range(1, 201):
        train_epoch(model, loader, optimizer)
        result = metric_summary(evaluate(model, loader))
        history.append({"epoch": epoch, **result})
        if result["accuracy"] == 1.0 and result["loss"] < 0.05:
            break
    passed = result["accuracy"] == 1.0 and result["loss"] < initial["loss"] * 0.2
    return {"seed": seed, "learning_rate": 0.001, "dropout": 0, "size": 32,
            "indices": indices.tolist(), "class_counts": {"negative": 16, "positive": 16},
            "initial": initial, "final": result, "epochs": epoch, "passed": passed,
            "criterion": "accuracy == 1 and final loss < 20% of initial loss",
            "history": history}


def fit_mlp(config, train_dataset, validation_dataset, run_dir, seed, epochs, batch_size, training_context=None):
    seed_everything(seed)
    model = SentimentMLP(train_dataset.features.shape[1], dropout=config["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, seed=seed)
    train_eval_loader = make_loader(train_dataset, batch_size)
    val_loader = make_loader(validation_dataset, batch_size)
    best_f1, best_epoch, best_state = -1.0, None, None
    best_checkpoint = None
    training_config = {"optimizer": "Adam", "learning_rate": config["learning_rate"],
                       "weight_decay": 0, "seed": seed, "batch_size": batch_size,
                       "threads": torch.get_num_threads(), "epochs": epochs,
                       "loss": "CrossEntropyLoss", **(training_context or {})}
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        optimization_loss = train_epoch(model, train_loader, optimizer)
        train = evaluate(model, train_eval_loader)
        validation = evaluate(model, val_loader)
        row = {"epoch": epoch, "optimization_loss": optimization_loss,
               **{f"train_{k}": v for k, v in metric_summary(train).items()},
               **{f"validation_{k}": v for k, v in metric_summary(validation).items()}}
        history.append(row)
        if validation["macro_f1"] > best_f1:
            best_f1, best_epoch = validation["macro_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint = training_checkpoint(model, optimizer, train_loader, epoch=epoch,
                validation_macro_f1=best_f1, config=training_config)
        print(f"{config['name']} epoch={epoch:02d} train_loss={train['loss']:.4f} "
              f"val_loss={validation['loss']:.4f} val_f1={validation['macro_f1']:.6f}", flush=True)
    fit_seconds = time.perf_counter() - started
    torch.save(training_checkpoint(model, optimizer, train_loader, epoch=epochs,
        validation_macro_f1=validation["macro_f1"], config=training_config), run_dir / f"{config['name']}_last.pt")
    model.load_state_dict(best_state)
    restored = evaluate(model, val_loader)
    checkpoint_name = f"{config['name']}_best.pt"
    torch.save(best_checkpoint, run_dir / checkpoint_name)
    reloaded, _ = load_checkpoint(run_dir / checkpoint_name)
    verified = evaluate(reloaded, val_loader)
    if verified != restored:
        raise RuntimeError("Checkpoint reload changed validation predictions/metrics")
    pd.DataFrame(history).to_csv(run_dir / f"{config['name']}_history.csv", index=False)
    return {"model": config["name"], "seed": seed, **config, "best_epoch": best_epoch,
            "epochs_run": epochs, "checkpoint": checkpoint_name, "fit_seconds": fit_seconds,
            "checkpoint_verified": True, **metric_summary(restored)}, restored["predictions"], history


def plot_learning_curves(histories, summaries, output):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for row, config in enumerate(CONFIGURATIONS):
        name = config["name"]
        data = pd.DataFrame(histories[name])
        best_epoch = next(s["best_epoch"] for s in summaries if s["model"] == name)
        for col, (key, title) in enumerate((("train_loss", "Train loss (eval mode)"),
                                          ("validation_loss", "Validation loss"),
                                          ("validation_macro_f1", "Validation macro-F1"))):
            ax = axes[row, col]
            ax.plot(data.epoch, data[key], marker="o", markersize=3)
            ax.axvline(best_epoch, color="tab:red", linestyle="--", label=f"Best epoch: {best_epoch}")
            ax.set_title(f"{name}: {title}")
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle(f"TF-IDF MLP: seed {summaries[0]['seed']}; dashed line = selected validation macro-F1 epoch")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path,
                        default=PROJECT_ROOT / "models/evaluation_workflow/full_validation_20260929_01")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/mlp")
    parser.add_argument("--run-name", default="mlp_20261106_01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def run(args):
    from src.experiments import validate_run_name
    validate_run_name(args.run_name)
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("threads and batch-size must be positive")
    run_dir = args.output_dir / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}. Choose a fresh run name.")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    train, validation = load_splits(args.split_dir)
    vectorizer, x_train, x_val = fit_features(train.clean_text, validation.clean_text)
    y_train = train.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    y_val = validation.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    train_dataset, val_dataset = TfidfDataset(x_train, y_train), TfidfDataset(x_val, y_val)
    run_dir.mkdir(parents=True, exist_ok=False)
    batch_x, batch_y = next(iter(make_loader(train_dataset, args.batch_size)))
    probe = SentimentMLP(x_train.shape[1])
    with torch.no_grad():
        hidden_shape = list(probe.relu(probe.hidden(batch_x)).shape)
        output_shape = list(probe(batch_x).shape)
    audit = {"train_rows": len(train), "validation_rows": len(validation),
             "source_id_overlap": len(set(train.source_id) & set(validation.source_id)),
             "labels": LABEL_TO_ID, "vocabulary_size": x_train.shape[1],
             "train_label_counts": train.label.value_counts().to_dict(),
             "validation_label_counts": validation.label.value_counts().to_dict(),
             "empty_clean_text": {"train": int(train.clean_text.str.strip().eq("").sum()),
                                  "validation": int(validation.clean_text.str.strip().eq("").sum())},
             "batch_check": {"input_dtype": str(batch_x.dtype), "label_dtype": str(batch_y.dtype),
                             "input_shape": list(batch_x.shape), "hidden_shape": hidden_shape,
                             "output_shape": output_shape},
             "fit_scope": "TF-IDF.fit_transform(train.clean_text); transform(validation.clean_text)"}
    write_json(run_dir / "data_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    joblib.dump(vectorizer, run_dir / "tfidf.joblib")
    baseline = ComplementNB(alpha=0.5)
    started = time.perf_counter()
    baseline.fit(x_train, y_train)
    baseline_seconds = time.perf_counter() - started
    baseline_pred = baseline.predict(x_val)
    summaries = [{"model": "ComplementNB", "seed": args.seed, "learning_rate": None, "dropout": None,
                  "best_epoch": None, "alpha": 0.5, "fit_seconds": baseline_seconds,
                  "accuracy": float(accuracy_score(y_val, baseline_pred)),
                  "macro_f1": float(f1_score(y_val, baseline_pred, average="macro"))}]
    joblib.dump(baseline, run_dir / "complement_nb.joblib")
    overfit = overfit_32(train_dataset, args.seed)
    write_json(run_dir / "overfit_32.json", overfit)
    if not overfit["passed"]:
        raise RuntimeError("32-example overfit check failed; inspect overfit_32.json before full training")
    print(f"32-example memorization passed: {overfit['initial']['loss']:.4f} -> "
          f"{overfit['final']['loss']:.4f}, accuracy={overfit['final']['accuracy']:.3f}", flush=True)
    predictions, histories = {}, {}
    for config in CONFIGURATIONS:
        summary, pred, history = fit_mlp(config, train_dataset, val_dataset, run_dir,
                                         args.seed, 15, args.batch_size,
                                         {"tfidf_sha256": file_fingerprint(run_dir / "tfidf.joblib")["sha256"],
                                          "split_hashes": {f"{name}_split.csv": file_fingerprint(args.split_dir / f"{name}_split.csv")
                                                           for name in ("train", "validation")}})
        summaries.append(summary)
        predictions[config["name"]] = pred
        histories[config["name"]] = history
    best = max(summaries[1:], key=lambda row: row["macro_f1"])
    pred = predictions[best["model"]]
    report = classification_report(y_val, pred, labels=[0, 1], target_names=LABELS,
                                   zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_val, pred, labels=[0, 1]).tolist()
    write_json(run_dir / "classification_report.json", report)
    write_json(run_dir / "confusion_matrix.json", {"labels": list(LABELS), "rows": "true", "columns": "predicted", "matrix": matrix})
    result_frame = validation[[c for c in ("source_id", "review", "clean_text", "label") if c in validation]].copy()
    result_frame["prediction"] = [LABELS[index] for index in pred]
    result_frame["complement_nb_prediction"] = [LABELS[index] for index in baseline_pred]
    result_frame.to_csv(run_dir / "validation_predictions.csv", index=False)
    errors = result_frame[result_frame.label != result_frame.prediction]
    errors.to_csv(run_dir / "validation_errors.csv", index=False)
    sample = pd.concat([errors[errors.label == label].sample(n=min(6, sum(errors.label == label)), random_state=args.seed)
                        for label in LABELS])
    sample.to_csv(run_dir / "error_review_12.csv", index=False)
    pd.DataFrame(summaries).to_csv(run_dir / "comparison.csv", index=False)
    write_json(run_dir / "comparison.json", summaries)
    write_json(run_dir / "selection.json", {"selected_mlp": best["model"], "checkpoint": best["checkpoint"],
               "rule": "Highest validation macro-F1; earliest epoch on ties; declaration order for model ties.",
               "best_epoch": best["best_epoch"], "validation_macro_f1": best["macro_f1"]})
    plot_learning_curves(histories, summaries, run_dir / "learning_curves.png")
    metadata = {"seed": args.seed, "device": "cpu", "threads": args.threads, "batch_size": args.batch_size,
                "epochs": 15, "hidden_size": 64, "optimizer": "Adam", "weight_decay": 0,
                "loss": "CrossEntropyLoss; unweighted; logits input", "configurations": CONFIGURATIONS,
                "label_mapping": LABEL_TO_ID, "source_split_dir": str(args.split_dir.resolve()),
                "input_hashes": {f"{name}_split.csv": file_fingerprint(args.split_dir / f"{name}_split.csv")
                                 for name in ("train", "validation")},
                "test_accessed": False, "text_column": "clean_text", "features_dtype": "float32",
                "selection": "Validation macro-F1 only; no refit; same validation reused for 3 x 15 checkpoints.",
                "train_loss_definition": "Full training set, eval/no_grad after each epoch, dropout disabled, same as val loss.",
                "environment": environment_snapshot(), "code": code_snapshot()}
    write_json(run_dir / "metadata.json", metadata)
    (run_dir / "requirements-resolved.txt").write_text("".join(f"{k}=={v}\n" for k, v in metadata["environment"]["packages"].items()), encoding="utf-8")
    print(pd.DataFrame(summaries)[["model", "best_epoch", "accuracy", "macro_f1"]].to_string(index=False), flush=True)
    print(f"Saved artifacts: {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    run(parse_args())
