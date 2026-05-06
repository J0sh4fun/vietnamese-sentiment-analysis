from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessor import VietnameseTextProcessor


DEFAULT_TRAIN_DATA = PROJECT_ROOT / "data" / "shopee_reviews_dataset.jsonl"
DEFAULT_AUG_DATA = PROJECT_ROOT / "data" / "aug_unaccented_reviews.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
SUPPORTED_ALGORITHMS = ["logreg", "multinomial_nb", "complement_nb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and select the best Vietnamese sentiment model from JSONL datasets."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_TRAIN_DATA,
        help="Path to the primary JSONL dataset.",
    )
    parser.add_argument(
        "--aug-data",
        type=Path,
        nargs="*",
        default=[DEFAULT_AUG_DATA],
        help="Optional JSONL datasets to append to training data.",
    )
    parser.add_argument(
        "--disable-aug",
        action="store_true",
        help="Disable loading augmentation datasets.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="review",
        help="Text column name in input JSONL.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Label column name in input JSONL.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split ratio (from full dataset).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for number of rows used during training.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram length for TF-IDF.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram length for TF-IDF.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=150000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Ignore terms that appear in fewer than min_df documents.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=SUPPORTED_ALGORITHMS,
        default=None,
        help="Train only one algorithm (overrides --algorithms).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        choices=SUPPORTED_ALGORITHMS,
        default=SUPPORTED_ALGORITHMS,
        help="Algorithms to compare and select from.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["f1_macro", "accuracy"],
        default="f1_macro",
        help="Validation metric used for selecting the best model.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.5,
        help="Inverse regularization strength for Logistic Regression (C).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2500,
        help="Maximum iterations for Logistic Regression.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        choices=["balanced", "none"],
        default="balanced",
        help="Class weight strategy for Logistic Regression.",
    )
    parser.add_argument(
        "--nb-alpha",
        type=float,
        default=0.5,
        help="Smoothing alpha for Naive Bayes classifiers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write trained artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to timestamp.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    return pd.read_json(path, lines=True)


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    data_paths = [args.train_data]
    if not args.disable_aug:
        data_paths.extend(args.aug_data)

    frames = [read_jsonl(path) for path in data_paths]
    df = pd.concat(frames, ignore_index=True)

    required_columns = {args.text_column, args.label_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df.dropna(subset=[args.text_column, args.label_column]).copy()
    df[args.text_column] = df[args.text_column].astype(str)
    df[args.label_column] = df[args.label_column].astype(str)
    df = df.drop_duplicates(subset=[args.text_column, args.label_column], keep="first")

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be greater than 0.")
        sample_size = min(args.max_samples, len(df))
        df = df.sample(n=sample_size, random_state=args.random_state)

    if df.empty:
        raise ValueError("No rows remain after loading and filtering dataset.")

    return df.reset_index(drop=True)


def preprocess_dataset(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    processor = VietnameseTextProcessor()
    cleaned_text = processor.transform(df[text_column].tolist())

    processed_df = df.copy()
    processed_df["clean_text"] = cleaned_text
    processed_df = processed_df[processed_df["clean_text"].str.strip().ne("")]
    if processed_df.empty:
        raise ValueError("All rows became empty after preprocessing.")
    return processed_df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    label_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be in (0, 1).")
    if not 0 < val_size < 1:
        raise ValueError("--val-size must be in (0, 1).")
    if test_size + val_size >= 1:
        raise ValueError("--test-size + --val-size must be less than 1.")

    class_counts = df[label_column].value_counts()
    if (class_counts < 3).any():
        problematic_labels = class_counts[class_counts < 3].to_dict()
        raise ValueError(
            "Each label needs at least 3 samples for train/val/test stratified split. "
            f"Problematic labels: {problematic_labels}"
        )

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )

    val_ratio_in_train_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_in_train_val,
        random_state=random_state,
        stratify=train_val_df[label_column],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def resolve_candidate_algorithms(args: argparse.Namespace) -> list[str]:
    if args.algorithm is not None:
        return [args.algorithm]

    seen = set()
    resolved: list[str] = []
    for algo in args.algorithms:
        if algo not in seen:
            resolved.append(algo)
            seen.add(algo)
    if not resolved:
        raise ValueError("No candidate algorithms provided.")
    return resolved


def build_classifier(args: argparse.Namespace, num_classes: int, algorithm: str):
    if args.nb_alpha <= 0:
        raise ValueError("--nb-alpha must be greater than 0.")

    if algorithm == "logreg":
        class_weight = None if args.class_weight == "none" else "balanced"
        solver = "liblinear" if num_classes == 2 else "lbfgs"
        return LogisticRegression(
            C=args.regularization,
            max_iter=args.max_iter,
            class_weight=class_weight,
            solver=solver,
        )
    if algorithm == "multinomial_nb":
        return MultinomialNB(alpha=args.nb_alpha)
    if algorithm == "complement_nb":
        return ComplementNB(alpha=args.nb_alpha)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def build_model(args: argparse.Namespace, num_classes: int, algorithm: str) -> Pipeline:
    if args.ngram_min <= 0 or args.ngram_max < args.ngram_min:
        raise ValueError("n-gram range is invalid. Ensure 0 < ngram_min <= ngram_max.")

    vectorizer = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    classifier = build_classifier(args=args, num_classes=num_classes, algorithm=algorithm)
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_and_select_best_model(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_column: str,
) -> tuple[Pipeline, dict[str, object], list[dict[str, object]], list[str]]:
    candidate_algorithms = resolve_candidate_algorithms(args)
    num_classes = train_df[label_column].nunique()
    y_val_true = val_df[label_column].astype(str)

    leaderboard: list[dict[str, object]] = []
    best_entry: dict[str, object] | None = None
    best_model: Pipeline | None = None

    for algorithm in candidate_algorithms:
        model = build_model(args=args, num_classes=num_classes, algorithm=algorithm)
        model.fit(train_df["clean_text"], train_df[label_column])

        y_val_pred = pd.Series(model.predict(val_df["clean_text"]), index=val_df.index).astype(
            str
        )
        metrics = evaluate_predictions(y_true=y_val_true, y_pred=y_val_pred)
        entry = {
            "algorithm": algorithm,
            "validation_metrics": metrics,
            "selection_score": metrics[args.selection_metric],
        }
        leaderboard.append(entry)

        if best_entry is None:
            best_entry = entry
            best_model = model
            continue

        if entry["selection_score"] > best_entry["selection_score"]:
            best_entry = entry
            best_model = model
            continue

        if (
            entry["selection_score"] == best_entry["selection_score"]
            and entry["validation_metrics"]["accuracy"]
            > best_entry["validation_metrics"]["accuracy"]
        ):
            best_entry = entry
            best_model = model

    if best_entry is None or best_model is None:
        raise ValueError("No model trained during selection.")

    leaderboard = sorted(
        leaderboard,
        key=lambda item: (
            item["selection_score"],
            item["validation_metrics"]["accuracy"],
        ),
        reverse=True,
    )
    return best_model, best_entry, leaderboard, candidate_algorithms


def save_artifacts(
    args: argparse.Namespace,
    model: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_paths: list[Path],
    candidate_algorithms: list[str],
    best_entry: dict[str, object],
    leaderboard: list[dict[str, object]],
    test_metrics: dict[str, float],
) -> Path:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "sentiment_pipeline.joblib"
    train_split_path = run_dir / "train_split.csv"
    validation_split_path = run_dir / "validation_split.csv"
    test_split_path = run_dir / "test_split.csv"
    metadata_path = run_dir / "train_metadata.json"

    joblib.dump(model, model_path)
    train_df.to_csv(train_split_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(validation_split_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_split_path, index=False, encoding="utf-8-sig")

    output_payload = {
        "run_name": run_name,
        "dataset_paths": [str(path) for path in dataset_paths],
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "labels": sorted(train_df[args.label_column].unique().tolist()),
        "model_path": str(model_path),
        "selected_algorithm": best_entry["algorithm"],
        "split_paths": {
            "train": str(train_split_path),
            "validation": str(validation_split_path),
            "test": str(test_split_path),
        },
        "model_selection": {
            "selection_metric": args.selection_metric,
            "candidate_algorithms": candidate_algorithms,
            "leaderboard": leaderboard,
            "best_model": best_entry,
            "best_model_test_metrics": test_metrics,
        },
        "training_config": {
            "algorithm": args.algorithm,
            "algorithms": args.algorithms,
            "ngram_range": [args.ngram_min, args.ngram_max],
            "max_features": args.max_features,
            "min_df": args.min_df,
            "regularization": args.regularization,
            "max_iter": args.max_iter,
            "class_weight": args.class_weight,
            "nb_alpha": args.nb_alpha,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "max_samples": args.max_samples,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, ensure_ascii=False, indent=2)

    return run_dir


def main() -> None:
    args = parse_args()
    dataset_paths = [args.train_data]
    if not args.disable_aug:
        dataset_paths.extend(args.aug_data)

    raw_df = load_dataset(args)
    processed_df = preprocess_dataset(raw_df, args.text_column)

    train_df, val_df, test_df = split_dataset(
        processed_df,
        label_column=args.label_column,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    best_model, best_entry, leaderboard, candidate_algorithms = train_and_select_best_model(
        args=args,
        train_df=train_df,
        val_df=val_df,
        label_column=args.label_column,
    )

    y_test_pred = pd.Series(best_model.predict(test_df["clean_text"]), index=test_df.index).astype(
        str
    )
    test_metrics = evaluate_predictions(
        y_true=test_df[args.label_column].astype(str),
        y_pred=y_test_pred,
    )

    run_dir = save_artifacts(
        args=args,
        model=best_model,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        dataset_paths=dataset_paths,
        candidate_algorithms=candidate_algorithms,
        best_entry=best_entry,
        leaderboard=leaderboard,
        test_metrics=test_metrics,
    )

    summary = {
        "message": "Training finished successfully.",
        "run_dir": str(run_dir),
        "selected_algorithm": best_entry["algorithm"],
        "selection_metric": args.selection_metric,
        "best_validation_score": best_entry["selection_score"],
        "best_test_f1_macro": test_metrics["f1_macro"],
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
