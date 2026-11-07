"""Continue an Adam/RNG checkpoint without refitting TF-IDF or changing saved splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from src.experiments import file_fingerprint
from src.mlp import (LABEL_TO_ID, TfidfDataset, evaluate, make_loader,
                     restore_training_checkpoint, training_checkpoint, train_epoch)
from src.run_mlp import load_splits


def resume(checkpoint_path, tfidf_path, split_dir, output_dir, epochs=1):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Resume output exists; choose a fresh directory")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("training_config", {})
    if not config.get("split_hashes") or not config.get("tfidf_sha256"):
        raise ValueError("Resume requires saved split/TF-IDF fingerprints")
    if file_fingerprint(tfidf_path)["sha256"] != config["tfidf_sha256"]:
        raise ValueError("TF-IDF fingerprint differs from checkpoint")
    for name, expected in config["split_hashes"].items():
        if name not in {"train_split.csv", "validation_split.csv"}:
            raise ValueError("Unexpected split name in checkpoint")
        if file_fingerprint(Path(split_dir) / name) != expected:
            raise ValueError(f"Saved split fingerprint differs: {name}")
    train, validation = load_splits(split_dir)
    vectorizer = joblib.load(tfidf_path)
    train_data = TfidfDataset(vectorizer.transform(train.clean_text), train.label.map(LABEL_TO_ID))
    val_data = TfidfDataset(vectorizer.transform(validation.clean_text), validation.label.map(LABEL_TO_ID))
    model, optimizer, loader, checkpoint = restore_training_checkpoint(checkpoint_path, train_data)
    output_dir.mkdir(parents=True, exist_ok=False)
    history = []
    for epoch in range(checkpoint["epoch"] + 1, checkpoint["epoch"] + epochs + 1):
        started = time.perf_counter()
        loss = train_epoch(model, loader, optimizer)
        train_seconds = time.perf_counter() - started
        metrics = evaluate(model, make_loader(val_data, config["batch_size"]))
        history.append({"epoch": epoch, "optimization_loss": loss, "train_seconds": train_seconds,
                        **{k: v for k, v in metrics.items() if k != "predictions"}})
    result = training_checkpoint(model, optimizer, loader, epoch=epoch,
        validation_macro_f1=metrics["macro_f1"], config=config)
    torch.save(result, output_dir / "resumed.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tfidf", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(resume(args.checkpoint, args.tfidf, args.split_dir, args.output_dir, args.epochs), indent=2))


if __name__ == "__main__":
    main()
