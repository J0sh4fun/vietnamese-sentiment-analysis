"""PyTorch MLP on saved clean_text TF-IDF features (not raw-text inference).

The CSR matrix stays sparse in memory; only individual DataLoader batches
become dense. CrossEntropyLoss receives logits and integer class IDs.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.experiments import seed_everything as _shared_seed

LABELS = ("negative", "positive")
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}


def seed_everything(seed):
    _shared_seed(seed, torch_enabled=True)


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
    if checkpoint.get("artifact_version") not in (1, 2) or checkpoint.get("input_contract") != "clean_text":
        raise ValueError("Expected a version-1 or version-2 clean_text MLP checkpoint.")
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("Unsupported label mapping.")
    if checkpoint["artifact_version"] == 1:
        model = SentimentMLP(checkpoint["vocabulary_size"], 64, checkpoint["dropout"])
        state = checkpoint["state_dict"]
    else:
        arch = checkpoint["architecture"]
        if (arch["type"] != "SentimentMLP" or arch["activation"] != "relu"
                or arch["output_size"] != 2 or checkpoint["label_mapping"] != LABEL_TO_ID):
            raise ValueError("Unsupported architecture or label mapping.")
        model = SentimentMLP(arch["input_size"], arch["hidden_size"], arch["dropout"])
        state = checkpoint["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def training_checkpoint(model, optimizer, loader, *, epoch, validation_macro_f1, config):
    """Snapshot weights AND Adam/RNG state from the same epoch; safe for weights_only loading."""
    numpy_state = np.random.get_state()
    return {"artifact_version": 2, "input_contract": "clean_text", "labels": list(LABELS),
            "label_mapping": dict(LABEL_TO_ID),
            "architecture": {"type": "SentimentMLP", "input_size": model.hidden.in_features,
                             "hidden_size": model.hidden.out_features, "output_size": 2,
                             "activation": "relu", "dropout": model.dropout.p},
            "epoch": epoch, "validation_macro_f1": float(validation_macro_f1),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "training_config": copy.deepcopy(config),
            "rng_state": {"torch": torch.get_rng_state().clone(), "python": random.getstate(),
                          "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                                    "position": numpy_state[2], "has_gauss": numpy_state[3],
                                    "cached_gaussian": numpy_state[4]},
                          "loader_generator": loader.generator.get_state().clone()}}


def restore_training_checkpoint(path, dataset):
    model, checkpoint = load_checkpoint(path)
    if checkpoint["artifact_version"] != 2 or "optimizer_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no optimizer/RNG state; it cannot resume training.")
    config = checkpoint["training_config"]
    if config["optimizer"] != "Adam":
        raise ValueError("Only Adam checkpoints are supported.")
    if dataset.features.shape[1] != model.hidden.in_features:
        raise ValueError("Training features do not match the checkpoint architecture.")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loader = make_loader(dataset, config["batch_size"], shuffle=True, seed=config["seed"])
    state = checkpoint["rng_state"]
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["name"], np.asarray(ns["keys"], dtype=np.uint32), ns["position"],
                         ns["has_gauss"], ns["cached_gaussian"]))
    # Model construction consumes RNG; restore only after model/optimizer/loader exist.
    torch.set_rng_state(state["torch"])
    loader.generator.set_state(state["loader_generator"])
    return model, optimizer, loader, checkpoint


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
    if len(vectorizer.vocabulary_) != model.hidden.in_features:
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
