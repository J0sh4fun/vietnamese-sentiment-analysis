"""Shared RAW sample/label contract. No linguistic preprocessing lives here."""
from __future__ import annotations

from collections.abc import Mapping, Set
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

LABEL_TO_ID = {"negative": 0, "positive": 1}


def raw_strings(values):
    if isinstance(values, str):
        return [values]
    if isinstance(values, (bytes, Mapping, Set)) or getattr(values, "ndim", 1) != 1:
        raise TypeError("Expected a raw string or ordered iterable of raw strings")
    try:
        values = list(values)
    except TypeError as error:
        raise TypeError("Expected raw strings") from error
    if any(not isinstance(text, str) for text in values):
        raise TypeError("Raw reviews must be strings; no missing, numeric or nested inputs")
    return values


@dataclass(frozen=True)
class RawReview:
    source_id: str
    review: str
    label: str | None = None

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a nonempty string")
        if not isinstance(self.review, str):
            raise TypeError("review must be a raw string")
        if self.label is not None and self.label not in LABEL_TO_ID:
            raise ValueError("label must be negative/positive or None for inference")


class RawReviewDataset(Dataset):
    def __init__(self, records):
        self.records = tuple(records)
        batch_identity(self.records)

    @classmethod
    def from_csv(cls, path: str | Path):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source_id", "review", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Expected source_id, RAW review and label; clean_text is never a fallback")
            return cls(RawReview(row["source_id"], row["review"], row["label"]) for row in reader)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


@dataclass
class BatchIdentity:
    source_ids: tuple[str, ...]
    raw_sha256: tuple[str, ...]
    labels: torch.Tensor | None


def batch_identity(records):
    records = tuple(records)
    if any(not isinstance(row, RawReview) for row in records):
        raise TypeError("Both model routes require RawReview records, not clean_text arrays")
    ids = tuple(row.source_id for row in records)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source_id; do not silently duplicate/reorder samples")
    labeled = [row.label is not None for row in records]
    if any(labeled) and not all(labeled):
        raise ValueError("Do not mix labeled and unlabeled reviews in a batch")
    labels = torch.tensor([LABEL_TO_ID[row.label] for row in records], dtype=torch.long) if records and all(labeled) else None
    return BatchIdentity(ids, tuple(hashlib.sha256(row.review.encode("utf-8")).hexdigest() for row in records), labels)


def assert_same_samples(left, right):
    if left.source_ids != right.source_ids or left.raw_sha256 != right.raw_sha256:
        raise ValueError("Model routes received different raw samples or order")
    if (left.labels is None) != (right.labels is None) or (
            left.labels is not None and not torch.equal(left.labels, right.labels)):
        raise ValueError("Model routes received different labels")
