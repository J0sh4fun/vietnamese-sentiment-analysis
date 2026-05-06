from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Vietnamese sentiment model."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory created by src/train.py.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "test", "train"],
        default="test",
        help="Data split to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional model path override. Defaults to <run-dir>/sentiment_pipeline.joblib",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clean_text",
        help="Feature column to pass into model.predict.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Ground truth label column in split CSV.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    model_path = args.model_path or (run_dir / "sentiment_pipeline.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    split_filename = {
        "train": "train_split.csv",
        "validation": "validation_split.csv",
        "test": "test_split.csv",
    }[args.split]
    split_path = run_dir / split_filename
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")

    metrics_path = run_dir / f"{args.split}_metrics.json"
    predictions_path = run_dir / f"{args.split}_predictions.csv"
    return model_path, split_path, metrics_path, predictions_path


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> None:
    args = parse_args()
    model_path, split_path, metrics_path, predictions_path = resolve_paths(args)

    model = joblib.load(model_path)
    split_df = pd.read_csv(split_path)

    required_columns = {args.text_column, args.label_column}
    missing_columns = required_columns - set(split_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in split file: {missing}")

    features = split_df[args.text_column].astype(str)
    y_true = split_df[args.label_column].astype(str)
    y_pred = pd.Series(model.predict(features), index=split_df.index).astype(str)

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    metrics = evaluate_predictions(y_true=y_true, y_pred=y_pred, labels=labels)

    predictions_df = split_df.copy()
    predictions_df["predicted_label"] = y_pred
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    output_payload = {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "model_path": str(model_path),
        "split_path": str(split_path),
        "predictions_path": str(predictions_path),
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, ensure_ascii=False, indent=2)

    summary = {
        "message": "Evaluation finished successfully.",
        "split": args.split,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "f1_macro": metrics["f1_macro"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
