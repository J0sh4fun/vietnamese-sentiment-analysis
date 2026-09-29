"""Fixed-cohort validation ablations followed by one selected-model test evaluation.

Run with ``python -m src.run_evaluation --help``. All outputs require a fresh run.
No automatic error diagnosis: the sampled errors contain blank fields for review.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.data_pipeline import assert_no_overlap, empty_text_report, normalize_text, prepare_dataset
from src.evaluate import evaluate_predictions
from src.experiments import (PROJECT_ROOT, code_snapshot, environment_snapshot,
    file_fingerprint, json_value, seed_everything, validate_training_args)
from src.inference import SentimentModel, load_model
from src.preprocessor import VietnameseTextProcessor
from src.train import build_model


ALGORITHMS = ("majority", "logreg", "multinomial_nb", "complement_nb")
CONFIGURATIONS = (
    {"name": "reference", "word_segmentation": True, "remove_stopwords": True, "augmentation": False,
     "change": "Reference: NFC, segmentation on, stopwords on, no augmentation; tone repositioning off."},
    {"name": "no_segmentation", "word_segmentation": False, "remove_stopwords": True, "augmentation": False,
     "change": "Only segmentation off: whitespace tokens; same whole-token stopword matching."},
    {"name": "no_stopwords", "word_segmentation": True, "remove_stopwords": False, "augmentation": False,
     "change": "Only stopword filtering off."},
    {"name": "unaccented_augmentation", "word_segmentation": True, "remove_stopwords": True, "augmentation": True,
     "change": "Only training augmentation on: validated unaccented variants with source_id."},
)
SELECTION_RULE = "Highest validation macro-F1, then accuracy, then configuration/algorithm declaration order. No refit on validation."
TIMING_METHOD = ("time.perf_counter wall seconds, sequential execution on this machine. Fit timing includes TF-IDF "
    "and classifier fit on prepared training text; preprocessing is reported separately per configuration. "
    "Inference includes raw-text preprocessing exactly once and label prediction, excludes disk I/O/model load/metrics. "
    "One full validation batch warm-up, then repeated full validation batches; median and individual durations recorded. "
    "Tokenizer/model are warm; ms/review is batch-amortized, not single-request latency. No probability calculation.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=PROJECT_ROOT / "data/shopee_reviews_dataset.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/evaluation_workflow")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--text-column", default="review")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--allowed-labels", nargs="+", default=["negative", "positive"])
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--error-count", type=int, default=40, help="30 to 50 real errors, or all if fewer exist.")
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args(argv)
    # Frozen hyperparameters shared by every configuration. Existing train CLI
    # remains available for other experiments; this workflow has a fixed protocol.
    for key, value in dict(aug_data=[], disable_aug=False, generate_aug=False,
        legacy_tone_repositioning=False, ngram_min=1, ngram_max=2, max_features=150000,
        regularization=1.5, max_iter=2500, class_weight="balanced", nb_alpha=0.5).items():
        setattr(args, key, value)
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive")
    if not 30 <= args.error_count <= 50:
        parser.error("--error-count must be between 30 and 50")
    validate_training_args(args)
    return args


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_value(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def representation(splits, processor, text_column):
    """Never re-filter or re-split an ablation's original cohort."""
    output = []
    for frame in splits:
        frame = frame.copy()
        frame["clean_text"] = processor.transform(frame[text_column].tolist())
        frame["normalized_clean_text"] = frame.clean_text.map(normalize_text)
        output.append(frame)
    assert_no_overlap(*output)  # Fail before fitting if a representation collides.
    return tuple(output)


def choose_validation_result(results):
    if not results:
        raise ValueError("No validation results available.")
    return max(results, key=lambda row: (row["metrics"]["f1_macro"], row["metrics"]["accuracy"]))


def benchmark(model, texts, repeats):
    texts = list(texts)
    model.predict(texts)  # Untimed warm-up, using validation only.
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(texts)
        durations.append(time.perf_counter() - start)
    median = statistics.median(durations)
    return {"rows": len(texts), "warmup_batches": 1, "repeats": repeats,
            "batch_seconds": durations, "median_batch_seconds": median,
            "median_ms_per_review": median * 1000 / len(texts),
            "reviews_per_second": len(texts) / median}


def processing_report(frame):
    # Schema-invalid inputs were rejected before splitting. Valid empty-feature
    # rows still receive a prediction and remain in every metric denominator.
    return {**empty_text_report(frame), "invalid_rows_in_split": 0,
            "processing_failures": 0, "unprocessable_rate": 0.0,
            "failure_policy": "abort on exception; never silently omit rows"}


def sample_errors(predictions, count, seed):
    errors = predictions.loc[predictions.true_label != predictions.predicted_label].copy()
    # Round robin over directions gives balanced review coverage without claiming
    # the sample reflects the natural prevalence of causes in the full error set.
    groups = [group.sort_values("source_id").sample(frac=1, random_state=seed).index.tolist()
              for _, group in errors.groupby(["true_label", "predicted_label"], sort=True)]
    indices = []
    while len(indices) < min(count, len(errors)):
        for group in groups:
            if group and len(indices) < min(count, len(errors)):
                indices.append(group.pop(0))
    selected = errors.loc[indices].copy()
    selected.insert(0, "error_id", range(1, len(selected) + 1))
    selected["observed_categories"] = ""
    selected["reviewer_notes"] = ""
    return errors, selected


def render_report(metadata, results, selected, final, sample):
    lines = ["# Validation-led sentiment evaluation", "", f"Run: `{metadata['run_name']}`.", "",
        "Selection rule: " + SELECTION_RULE, "",
        "The original cohort is validated/deduplicated using the reference preprocessing before splitting. "
        "All ablations reuse its exact source IDs and labels. Less aggressive ablations do not reinstate filtered rows. "
        "Splits are stratified by source; all representations pass source/raw/clean-text overlap checks before fitting. "
        "Accents are preserved when identifying originals. Legacy untraceable augmentation is excluded.", "",
        "The fixed test partition is used for deterministic overlap checks, then predictions only after selection.json "
        "and the selected model have been saved. No test scores select configuration. Earlier project smoke runs used "
        "examples from this dataset; this is not a claim of a historically pristine holdout.", "",
        "## Configurations", ""]
    lines += [f"- **{c['name']}**: {c['change']}" for c in CONFIGURATIONS]
    lines += ["", "Tone repositioning is excluded: the legacy heuristic still corrupts correctly spelled khuấy. "
        "NFC remains on. The segmentation-off ablation keeps the existing whole-token stopword rule; "
        "multiword underscore entries therefore match less often. This interaction is part of the measured change, "
        "not a new phrase-matching algorithm.", "",
        "TF-IDF: word 1–2 grams, min_df=" + str(metadata["effective_config"]["min_df"]) + ", max_features=150000, sublinear_tf=True. "
        "LR: C=1.5, balanced weights, max_iter=2500, seeded solver. Both NB models: alpha=0.5. "
        "DummyClassifier uses most_frequent, including TF-IDF fit for the same artifact interface; its classifier ignores features. "
        "No other hyperparameter search or ablation combinations were performed.", "", "## Cohort and processing", "",
        "| Configuration | Train rows (augmented) | Validation | Test | Empty train / validation / test |",
        "|---|---:|---:|---:|---|"]
    for c in metadata["configurations"]:
        counts = c["splits"]
        lines.append(f"| {c['name']} | {counts['train']['input_rows']} ({c['augmentation_rows']}) | "
            f"{counts['validation']['input_rows']} | {counts['test']['input_rows']} | "
            + " / ".join(f"{counts[s]['empty_rows']} ({counts[s]['empty_rate']:.2%})" for s in ("train", "validation", "test")) + " |")
    lines += ["", "Invalid raw values, duplicate/conflicting-label filtering, label distributions, augmentation exclusions "
        "and data fingerprints are in metadata.json. All split rows are evaluated, including empty outputs; "
        "per-split exclusions are zero. A processing exception aborts execution.", "", "## Validation results", "",
        "| Configuration | Algorithm | Macro-F1 | Accuracy | Fit seconds | Median raw inference ms/review |",
        "|---|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['configuration']} | {row['algorithm']} | {row['metrics']['f1_macro']:.6f} | "
            f"{row['metrics']['accuracy']:.6f} | {row['fit_seconds']:.4f} | {row['timing']['median_ms_per_review']:.4f} |")
    lines += ["", "Selected: **" + selected["configuration"] + " / " + selected["algorithm"] + "**. "
        "These are descriptive results from one split, without uncertainty estimates or repeated-seed validation. "
        "Small differences do not establish general algorithm superiority.", "", "### Validation per-class metrics", "",
        "| Configuration / algorithm | Class | Precision | Recall | F1 | Support |", "|---|---|---:|---:|---:|---:|"]
    for row in results:
        for label in row["metrics"]["labels"]:
            score = row["metrics"]["classification_report"][label]
            lines.append(f"| {row['configuration']} / {row['algorithm']} | {label} | {score['precision']:.6f} | "
                         f"{score['recall']:.6f} | {score['f1-score']:.6f} | {int(score['support'])} |")
    metrics = final["metrics"]
    lines += ["", "## Final test evaluation", "", f"Macro-F1: **{metrics['f1_macro']:.6f}**; accuracy: **{metrics['accuracy']:.6f}**.", "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in metrics["labels"]:
        score = metrics["classification_report"][label]
        lines.append(f"| {label} | {score['precision']:.6f} | {score['recall']:.6f} | {score['f1-score']:.6f} | {int(score['support'])} |")
    lines += ["", "Confusion matrix: rows = true labels; columns = predicted labels.", "",
        "| True / predicted | " + " | ".join(metrics["labels"]) + " |",
        "|---|" + "---:|" * len(metrics["labels"])]
    for label, values in zip(metrics["labels"], metrics["confusion_matrix"]):
        lines.append("| " + label + " | " + " | ".join(map(str, values)) + " |")
    p = final["preprocessing"]
    lines += ["", f"Test: {p['input_rows']} inputs, {p['processing_failures']} processing failures "
        f"({p['unprocessable_rate']:.2%}), {p['empty_rows']} empty outputs ({p['empty_rate']:.2%}), "
        f"{p['excluded_rows']} exclusions ({p['exclusion_rate']:.2%}). "
        "Empty means no cleaned lexical content, distinct from a processing failure; both rates are reported.", "",
        "## Timing", "", TIMING_METHOD, "",
        f"Final test inference: one raw batch, {final['inference_seconds']:.4f} seconds. "
        "No test timing repetitions. Timings depend on hardware, load, tokenizer caches and numerical libraries. "
        "Per-configuration preprocessing duration and individual validation timing repetitions are recorded in JSON.", "",
        "## Error review", "", f"{len(sample)} actual misclassifications were sampled deterministically across error directions. "
        "Full reviews and source IDs are in error_review.csv; annotate observed_categories and reviewer_notes. "
        "This stratified error sample is for qualitative review, not estimating population cause rates. "
        "No automated keyword heuristic is presented as a causal diagnosis.", "",
        "After reviewing these test errors, the test set has been inspected and is no longer an untouched holdout "
        "for the next development cycle. Any resulting changes require fresh evaluation data or a new explicitly "
        "declared evaluation protocol; no model changes in this run are driven by these errors.", "",
        "## Reproduction", "", "Effective settings, seeds, observed environment, code and data fingerprints are in metadata.json; "
        "the exact installed distributions are in requirements-resolved.txt. Match the recorded Python/platform. "
        "Seeded splits and estimators do not guarantee bitwise equality across dependency versions, operating systems "
        "or numerical libraries. The model embeds preprocessing; inference accepts raw reviews.", ""]
    return "\n".join(lines)


def run(args):
    validate_training_args(args)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.random_state)
    metadata = {"schema_version": 1, "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "effective_config": json_value(vars(args).copy()), "random_seed": args.random_state,
        "environment": environment_snapshot(), "code": code_snapshot(),
        "selection_rule": SELECTION_RULE, "timing_method": TIMING_METHOD,
        "configurations": [], "algorithms": list(ALGORITHMS)}
    write_json(run_dir / "protocol.json", metadata)
    start = time.perf_counter()
    reference = prepare_dataset(args, VietnameseTextProcessor())
    metadata["reference_preparation_seconds"] = time.perf_counter() - start
    metadata["data_sources"] = args.data_sources
    metadata["reference_data_audit"] = args.data_audit
    # Prepare every representation and check overlaps before any model selection.
    variants = []
    for config in CONFIGURATIONS:
        start = time.perf_counter()
        processor = VietnameseTextProcessor(word_segmentation=config["word_segmentation"],
                                             remove_stopwords=config["remove_stopwords"])
        if config["augmentation"]:
            augmented_args = copy.deepcopy(args)
            augmented_args.generate_aug = True
            splits = prepare_dataset(augmented_args, processor)
            for expected, actual in zip(reference, splits):
                original = actual.loc[~actual.is_augmented]
                if expected.source_id.tolist() != original.source_id.tolist():
                    raise ValueError("Augmentation preparation changed the frozen original split.")
            metadata["augmentation_data_audit"] = augmented_args.data_audit
        else:
            splits = representation(reference, processor, args.text_column)
        train, val, test = splits
        config_metadata = {**config, "preprocessing": processor.to_config(),
            "preparation_seconds": time.perf_counter() - start,
            "augmentation_rows": int(train.is_augmented.sum()),
            "overlap_checks": "source_id, normalized_text, normalized_clean_text passed",
            "splits": {name: {**processing_report(frame),
                "label_distribution": frame[args.label_column].value_counts().to_dict()}
                for name, frame in zip(("train", "validation", "test"), splits)}}
        metadata["configurations"].append(config_metadata)
        variants.append((config, processor, splits))
    results, best_model, best_entry, best_splits = [], None, None, None
    for config, processor, splits in variants:
        train, val, _ = splits  # Test labels/predictions never enter selection.
        for algorithm in ALGORITHMS:
            estimator = build_model(args, train[args.label_column].nunique(),
                                    "logreg" if algorithm == "majority" else algorithm)
            if algorithm == "majority":
                estimator.set_params(classifier=DummyClassifier(strategy="most_frequent"))
            start = time.perf_counter()
            estimator.fit(train.clean_text, train[args.label_column])
            fit_seconds = time.perf_counter() - start
            model = SentimentModel(estimator, processor.to_config(), args.text_column, args.label_column)
            # Evaluation, benchmark and exported artifact all use the public raw path.
            validation_predictions = model.predict(val[args.text_column])
            entry = {"configuration": config["name"], "algorithm": algorithm,
                "fit_seconds": fit_seconds,
                "estimator_parameters": {name: json_value(step.get_params(deep=False))
                                         for name, step in estimator.named_steps.items()},
                "metrics": evaluate_predictions(val[args.label_column], validation_predictions, model.classes_.tolist()),
                "timing": benchmark(model, val[args.text_column], args.timing_repeats)}
            results.append(entry)
            if choose_validation_result(results) is entry:
                best_model, best_entry, best_splits = model, entry, splits
            print(f"Validation {config['name']}/{algorithm}: macro-F1={entry['metrics']['f1_macro']:.6f}", flush=True)
    write_json(run_dir / "validation_results.json", results)
    write_json(run_dir / "selection.json", {"rule": SELECTION_RULE, "selected": best_entry,
        "test_predictions_not_yet_computed": True})
    model_path = run_dir / "sentiment_pipeline.joblib"
    joblib.dump(best_model, model_path)
    loaded = load_model(model_path)
    # Persistence verification is done on validation, before final test inference.
    raw_validation = best_splits[1][args.text_column]
    if best_model.infer(raw_validation) != loaded.infer(raw_validation):
        raise ValueError("Predictions changed after saving/loading the artifact.")
    metadata["persistence_validation"] = "All validation predictions and empty flags identical after save/load."
    metadata["selection_fingerprint_before_test"] = file_fingerprint(run_dir / "selection.json")
    for name, frame in zip(("train", "validation", "test"), best_splits):
        frame.to_csv(run_dir / f"{name}_split.csv", index=False, encoding="utf-8-sig", mode="x")
    train, _, test = best_splits
    train.loc[train.is_augmented].to_json(run_dir / "train_augmentation.jsonl", orient="records", lines=True, force_ascii=False)
    start = time.perf_counter()
    test_results = loaded.infer(test[args.text_column])
    test_seconds = time.perf_counter() - start
    predictions = test[["source_id", args.text_column, args.label_column]].rename(
        columns={args.text_column: "raw_text", args.label_column: "true_label"}).copy()
    predictions["predicted_label"] = [row["label"] for row in test_results]
    predictions["empty_after_preprocessing"] = [row["empty_after_preprocessing"] for row in test_results]
    predictions.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig", mode="x")
    final = {"selected_configuration": best_entry["configuration"], "selected_algorithm": best_entry["algorithm"],
        "metrics": evaluate_predictions(predictions.true_label, predictions.predicted_label, loaded.classes_.tolist()),
        "preprocessing": processing_report(test), "inference_seconds": test_seconds}
    write_json(run_dir / "test_metrics.json", final)
    errors, sample = sample_errors(predictions, args.error_count, args.random_state)
    errors.to_csv(run_dir / "test_errors.csv", index=False, encoding="utf-8-sig", mode="x")
    sample.to_csv(run_dir / "error_review.csv", index=False, encoding="utf-8-sig", mode="x")
    metadata["test_inspection"] = {"error_rows": len(errors), "sample_rows": len(sample),
        "sampling": "seeded shuffle within true/predicted direction, round robin",
        "human_review": "pending; see error_review.csv and any separately authored error_analysis.md",
        "historical_exposure": "Earlier project smoke experiments used examples from this same dataset.",
        "future_holdout_status": "Once these errors are reviewed, test is inspected; do not reuse as untouched holdout."}
    (run_dir / "requirements-resolved.txt").write_text("# Observed installed distributions, not a portable lockfile.\n" +
        "".join(f"{name}=={version}\n" for name, version in metadata["environment"]["packages"].items()), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(metadata, results, best_entry, final, sample), encoding="utf-8")
    metadata["artifact_fingerprints"] = {p.name: file_fingerprint(p) for p in sorted(run_dir.iterdir()) if p.is_file()}
    write_json(run_dir / "metadata.json", metadata)
    print(json.dumps({"run_dir": str(run_dir), "selected": [best_entry["configuration"], best_entry["algorithm"]],
        "test_macro_f1": final["metrics"]["f1_macro"], "test_accuracy": final["metrics"]["accuracy"],
        "errors": len(errors), "sampled_errors": len(sample)}, indent=2))
    return run_dir


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError) as error:
        raise SystemExit(f"Evaluation workflow error: {error}") from error
