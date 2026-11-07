"""Versioned baseline config and hash helpers, usable before installing ML dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/baseline-v1.0.0.json"


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config["data"]["labels"] != {"negative": 0, "positive": 1}:
        raise ValueError("Unsupported release schema/label mapping")
    if type(config["seed"]) is not int or config["seed"] < 0 or config["threads"] < 1:
        raise ValueError("Invalid seed/threads")
    mlp = config["mlp"]
    if (mlp["hidden_size"] != 64 or mlp["optimizer"] != "Adam" or mlp["weight_decay"] != 0
            or mlp["selection_metric"] != "validation_macro_f1" or mlp["tie_break"] != "earliest_epoch"):
        raise ValueError("Unsupported MLP training contract; do not silently ignore config fields")
    if config["tfidf"]["dtype"] != "float32" or config["split"]["augmentation"]:
        raise ValueError("This baseline release requires float32 and no augmentation")
    return config
