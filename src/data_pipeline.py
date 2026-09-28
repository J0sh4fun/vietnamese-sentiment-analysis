"""Validation, provenance and leakage-safe splits (no learned preprocessing here)."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import warnings
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    """Identity normalization deliberately preserves Vietnamese accents."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def source_id_for(text: str) -> str:
    return "sha256:" + hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def remove_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def record(audit: list, stage: str, df: pd.DataFrame, label_column: str, **details) -> None:
    labels = df[label_column].map(lambda value: str(value)).value_counts().to_dict() if label_column in df else {}
    audit.append({"stage": stage, "samples": len(df), "label_distribution": labels, **details})


def read_jsonl(path: Path) -> pd.DataFrame:
    # Do not let pandas silently coerce IDs, labels or numeric-looking text.
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return pd.DataFrame(rows)


def filter_text_identity(df, key, label_column, audit, stage):
    # Missing lexical content is not evidence that two original reviews are
    # identical. Keep empty clean texts, with their distinct raw/source identity.
    comparable = df[key].ne("") if key == "normalized_clean_text" else pd.Series(True, index=df.index)
    conflicts = df.loc[comparable].groupby(key)[label_column].transform("nunique").gt(1)
    conflicts = conflicts.reindex(df.index, fill_value=False)
    duplicates = df.duplicated([key, label_column]) & comparable & ~conflicts
    result = df.loc[~(conflicts | duplicates)].copy()
    record(audit, stage, result, label_column,
           conflicting_rows=int(conflicts.sum()),
           duplicate_rows=int(duplicates.sum()))
    return result


def check_source_identity(df, key):
    """Do not discard an alias and accidentally break a supplied source group."""
    valid_ids = df["source_id"].map(lambda x: isinstance(x, str) and bool(x.strip()) and x == x.strip())
    if not valid_ids.all():
        raise ValueError("Original source_id values must be nonempty, trimmed strings.")
    comparable = df.loc[df[key].ne("")] if key == "normalized_clean_text" else df
    if comparable.groupby(key)["source_id"].nunique().gt(1).any():
        raise ValueError(f"Ambiguous original source_id: identical {key} has multiple source IDs; resolve aliases before training.")


def validate_frame(df, text_column, label_column, allowed_labels, audit, source, check_original_ids=False):
    """Identical policy for originals, imported and generated augmentation."""
    record(audit, f"{source}:before_validation", df, label_column)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(f"{source}: missing required columns: {sorted(missing)}")
    df = df.copy()
    valid_text = df[text_column].map(lambda x: isinstance(x, str) and bool(normalize_text(x))).astype(bool)
    valid_label = df[label_column].map(lambda x: isinstance(x, str) and x in allowed_labels).astype(bool)
    df = df.loc[valid_text & valid_label].copy()
    record(audit, f"{source}:valid_values", df, label_column,
           invalid_text_rows=int((~valid_text).sum()), invalid_label_rows=int((~valid_label).sum()))
    df["normalized_text"] = df[text_column].map(normalize_text)
    if check_original_ids and "source_id" in df:
        check_source_identity(df, "normalized_text")
    return filter_text_identity(df, "normalized_text", label_column, audit, f"{source}:after_validation")


def preprocess_frame(df, text_column, label_column, processor, audit, source, check_original_ids=False):
    df = df.copy()
    df["clean_text"] = pd.Series(processor.transform(df[text_column].tolist()), index=df.index, dtype="str")
    df["normalized_clean_text"] = df["clean_text"].map(normalize_text)
    if check_original_ids:
        check_source_identity(df, "normalized_clean_text")
    record(audit, f"{source}:preprocessed", df, label_column, **empty_text_report(df))
    return filter_text_identity(df, "normalized_clean_text", label_column, audit, f"{source}:after_preprocessing")


def empty_text_report(df):
    """Report all original rows, including zero-feature inputs, in every split."""
    empty = int(df["clean_text"].str.strip().eq("").sum())
    return {"input_rows": len(df), "empty_rows": empty,
            "empty_rate": empty / len(df) if len(df) else 0.0,
            "excluded_rows": 0, "exclusion_rate": 0.0, "empty_policy": "keep"}


def assert_no_overlap(train, validation, test):
    for (left_name, left), (right_name, right) in combinations(
        [("train", train), ("validation", validation), ("test", test)], 2
    ):
        for column in ("source_id", "normalized_text", "normalized_clean_text"):
            overlap = set(left[column]) & set(right[column])
            if column == "normalized_clean_text":
                overlap.discard("")  # Absence of features is not source identity.
            if overlap:
                raise ValueError(f"Leakage: {left_name}/{right_name} share {len(overlap)} {column} values")


def generate_unaccented(train, text_column, label_column):
    """Only call with the already assigned training originals."""
    result = train[[text_column, label_column, "source_id"]].copy()
    result["source_text"] = result[text_column]
    result[text_column] = result[text_column].map(remove_accents)
    result["augmentation_method"] = "unaccented_v1"
    return result


def prepare_dataset(args, processor):
    audit = []
    args.data_audit = audit
    text_column, label_column = args.text_column, args.label_column
    allowed_labels = getattr(args, "allowed_labels", ["negative", "positive"])
    if not 0 < args.test_size < 1 or not 0 < args.val_size < 1 or args.test_size + args.val_size >= 1:
        raise ValueError("Split sizes must be positive and --test-size + --val-size must be less than 1.")
    if text_column == label_column or {text_column, label_column} & {
        "source_id", "source_text", "augmentation_method", "is_augmented", "clean_text",
        "normalized_text", "normalized_clean_text"
    }:
        raise ValueError("Text and label columns must be distinct and cannot use reserved pipeline column names.")
    raw = read_jsonl(args.train_data)
    original = validate_frame(raw, text_column, label_column, allowed_labels, audit, "original", check_original_ids=True)
    if "source_id" not in original:
        original["source_id"] = original[text_column].map(source_id_for)
    if original.groupby("source_id")[label_column].nunique().gt(1).any():
        raise ValueError("Original source_id has conflicting labels.")
    original["is_augmented"] = False
    original = preprocess_frame(original, text_column, label_column, processor, audit, "original",
                                check_original_ids="source_id" in raw)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be greater than 0.")
        original = original.sample(n=min(args.max_samples, len(original)), random_state=args.random_state)
    record(audit, "original:after_sampling", original, label_column)
    if original.empty or original[label_column].nunique() < 2:
        raise ValueError("At least two labels must remain after original-data filtering.")

    # Split source groups, not rows, even if a caller supplies repeated source_id.
    groups = original[["source_id", label_column]].drop_duplicates("source_id")
    try:
        train_val, test_groups = train_test_split(groups, test_size=args.test_size,
            stratify=groups[label_column], random_state=args.random_state)
        train_groups, val_groups = train_test_split(train_val,
            # Count against all original groups; avoid float-ratio rounding
            # accidentally turning 30 validation examples into 31.
            test_size=math.ceil(len(groups) * args.val_size),
            stratify=train_val[label_column], random_state=args.random_state)
    except ValueError as error:
        raise ValueError(f"Cannot stratify original source groups; increase samples or adjust split sizes: {error}") from error
    train, val, test = [original.loc[original.source_id.isin(g.source_id)].copy()
                        for g in (train_groups, val_groups, test_groups)]
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        record(audit, f"split:{name}:originals", frame, label_column, **empty_text_report(frame))
    args.preprocessing_by_split = {name: empty_text_report(frame)
        for name, frame in (("train", train), ("validation", val), ("test", test))}
    assert_no_overlap(train, val, test)

    sources = []
    if not args.disable_aug:
        sources.extend((str(path), read_jsonl(path)) for path in args.aug_data)
        if getattr(args, "generate_aug", False):
            sources.append(("generated:unaccented_v1", generate_unaccented(train, text_column, label_column)))
    accepted = []
    train_labels = train.set_index("source_id")[label_column].to_dict()
    train_pairs = set(zip(train.source_id, train.normalized_text))
    for name, frame in sources:
        frame = validate_frame(frame, text_column, label_column, allowed_labels, audit, name)
        provenance_columns = {"source_id", "source_text", "augmentation_method"}
        if not provenance_columns.issubset(frame.columns):
            warnings.warn(f"Excluded {name}: untraceable augmentation. Regenerate with --generate-aug.", stacklevel=2)
            record(audit, f"{name}:provenance", frame.iloc[:0], label_column,
                   excluded_rows=len(frame), reason="missing provenance columns")
            continue
        def traceable(row):
            sid, source = row["source_id"], row["source_text"]
            return (isinstance(sid, str) and isinstance(source, str)
                    and (sid, normalize_text(source)) in train_pairs
                    and row[label_column] == train_labels[sid]
                    and row["augmentation_method"] == "unaccented_v1"
                    and normalize_text(row[text_column]) == normalize_text(remove_accents(source)))
        keep = frame.apply(traceable, axis=1) if len(frame) else pd.Series(False, index=frame.index, dtype=bool)
        eligible = frame.loc[keep].copy()
        record(audit, f"{name}:provenance", eligible, label_column,
               excluded_rows=len(frame) - len(eligible),
               exclusion_policy="require current training source, matching label and verified unaccented_v1 transform")
        eligible = preprocess_frame(eligible, text_column, label_column, processor, audit, name)
        empty = eligible["clean_text"].str.strip().eq("")
        input_count = len(eligible)
        eligible = eligible.loc[~empty].copy()
        record(audit, f"{name}:nonempty_augmentation", eligible, label_column,
               input_rows=input_count, excluded_rows=int(empty.sum()),
               exclusion_rate=int(empty.sum()) / input_count if input_count else 0.0)
        eligible["is_augmented"] = True
        accepted.append(eligible)
    if accepted:
        augmented = pd.concat(accepted, ignore_index=True)
        # Validate duplicates/conflicts across files as well as within each source.
        augmented = validate_frame(augmented, text_column, label_column, allowed_labels, audit, "augmentation:combined")
        augmented = filter_text_identity(augmented, "normalized_clean_text", label_column, audit, "augmentation:combined_clean")
        collision = pd.Series(False, index=augmented.index)
        for key in ("normalized_text", "normalized_clean_text"):
            collision |= augmented[key].isin(set(original[key]))
        augmented = augmented.loc[~collision].copy()
        record(audit, "augmentation:after_overlap_filter", augmented, label_column, excluded_rows=int(collision.sum()))
        train = pd.concat([train, augmented], ignore_index=True)
    assert_no_overlap(train, val, test)
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        record(audit, f"final:{name}", frame, label_column)
    args.overlap_checks = {"source_id": "passed", "normalized_text": "passed", "normalized_clean_text": "passed"}
    return tuple(frame.reset_index(drop=True) for frame in (train, val, test))
