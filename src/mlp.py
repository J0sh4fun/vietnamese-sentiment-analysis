"""PyTorch MLP on saved clean_text TF-IDF features (not raw-text inference).

The CSR matrix stays sparse in memory; only individual DataLoader batches
become dense. CrossEntropyLoss receives logits and integer class IDs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

LABELS = ("negative", "positive")
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class TfidfDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.tocsr().astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.features.shape[0] != len(self.labels) or self.labels.ndim != 1:
            raise ValueError("Features and one-dimensional labels must have matching lengths.")
        if not np.isfinite(self.features.data).all() or not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Expected finite features and labels in {0, 1}.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        dense = self.features[index].toarray().ravel()
        return torch.from_numpy(dense), torch.tensor(self.labels[index], dtype=torch.long)


def make_loader(dataset, batch_size=128, *, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


class SentimentMLP(nn.Module):
    def __init__(self, vocabulary_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.hidden = nn.Linear(vocabulary_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, features):
        return self.output(self.dropout(self.relu(self.hidden(features))))


def train_epoch(model, loader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    count = 0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        count += len(labels)
    if not count:
        raise ValueError("Cannot train on an empty DataLoader.")
    return loss_sum / count


def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, truth, predicted = 0.0, [], []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            loss_sum += criterion(logits, labels).item()
            truth.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).tolist())
    if not truth:
        raise ValueError("Cannot evaluate an empty DataLoader.")
    return {"loss": loss_sum / len(truth),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "predictions": predicted}


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_version") != 1 or checkpoint.get("input_contract") != "clean_text":
        raise ValueError("Expected a version-1 clean_text MLP checkpoint.")
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("Unsupported label mapping.")
    model = SentimentMLP(checkpoint["vocabulary_size"], 64, checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_clean_texts(run_dir, texts, batch_size=128):
    """Accept already-preprocessed strings; do not pass raw reviews here."""
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("clean_text inputs must be strings.")
    run_dir = Path(run_dir)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    model, checkpoint = load_checkpoint(run_dir / selection["checkpoint"])
    vectorizer = joblib.load(run_dir / "tfidf.joblib")
    if len(vectorizer.vocabulary_) != checkpoint["vocabulary_size"]:
        raise ValueError("Vectorizer/checkpoint dimension mismatch.")
    if not texts:
        return []
    dataset = TfidfDataset(vectorizer.transform(texts), np.zeros(len(texts), dtype=np.int64))
    probabilities = []
    with torch.no_grad():
        for features, _ in make_loader(dataset, batch_size):
            probabilities.extend(model(features).softmax(dim=1).tolist())
    return [{"label": LABELS[int(np.argmax(row))],
             "probabilities": dict(zip(LABELS, row)), "probability_kind": "uncalibrated"}
            for row in probabilities]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clean-text", nargs="+", required=True,
                        help="Already-preprocessed text from the same pipeline as the saved splits.")
    args = parser.parse_args()
    print(json.dumps(predict_clean_texts(args.run_dir, args.clean_text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
