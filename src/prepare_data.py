"""Restore pinned historical development data and byte-identical splits. Never creates a NEW test.

Download code uses only the standard library. Split preparation requires the
locked ML environment. Existing valid files are checked and reused, never overwritten.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import io
import json
from pathlib import Path
import urllib.request
import zipfile

from src.release_config import DEFAULT_CONFIG, ROOT, fingerprint, load_config, write_json


def ensure_originals(config, directory, *, archive_path=None, offline=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = config["data"]["files"]
    missing = []
    for name, info in expected.items():
        path = directory / name
        if path.exists():
            if fingerprint(path)["sha256"] != info["sha256"]:
                raise ValueError(f"Existing data hash mismatch: {name}; choose a clean output directory")
        else:
            missing.append(name)
    if not missing:
        return {name: fingerprint(directory / name) for name in expected}
    if archive_path is not None:
        content = Path(archive_path).read_bytes()
    elif offline:
        raise FileNotFoundError("Pinned data missing; supply --archive or allow the public download")
    else:
        request = urllib.request.Request(config["data"]["url"], headers={"User-Agent": "sentiment-baseline-release"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError("Unexpected/duplicate/path-traversal archive members")
        payloads = {name: archive.read(name) for name in expected}
    import hashlib
    for name, data in payloads.items():
        if hashlib.sha256(data).hexdigest() != expected[name]["sha256"]:
            raise ValueError(f"Downloaded dataset version/content differs: {name}")
        rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
        if len(rows) != expected[name]["rows"] or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Unexpected row count/schema: {name}")
    # Only whitelisted, hash-verified payloads are written; no extractall().
    for name in missing:
        with (directory / name).open("xb") as handle:
            handle.write(payloads[name])
    return {name: fingerprint(directory / name) for name in expected}


def verify_splits(config_path, directory):
    config, directory = load_config(config_path), Path(directory)
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["config"] != fingerprint(config_path) or manifest["seed"] != config["seed"]:
        raise ValueError("Prepared splits belong to a different config/seed")
    for name, digest in config["expected"]["split_sha256"].items():
        actual = fingerprint(directory / name)
        if actual != manifest["files"][name] or actual["sha256"] != digest:
            raise ValueError(f"Prepared split differs from release: {name}")
    return manifest


def prepare_splits(config_path, raw_dir, output_dir):
    config, output_dir = load_config(config_path), Path(output_dir)
    if output_dir.exists():
        return verify_splits(config_path, output_dir)
    from src.data_pipeline import assert_no_overlap, normalize_text, prepare_dataset
    from src.experiments import seed_everything
    from src.preprocessor import VietnameseTextProcessor
    seed_everything(config["seed"])
    args = Namespace(train_data=Path(raw_dir) / "shopee_reviews_dataset.jsonl", aug_data=[],
        allowed_labels=list(config["data"]["labels"]), text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"], random_state=config["seed"], max_samples=None,
        val_size=config["split"]["validation_size"], test_size=config["split"]["test_size"],
        disable_aug=True, generate_aug=False)
    reference = VietnameseTextProcessor(**config["preprocessing"]["cohort_reference"])
    frames = prepare_dataset(args, reference)
    processor = VietnameseTextProcessor(**config["preprocessing"]["model"])
    for frame in frames:
        frame["clean_text"] = processor.transform(frame[args.text_column].tolist())
        frame["normalized_clean_text"] = frame.clean_text.map(normalize_text)
    assert_no_overlap(*frames)
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name, frame in zip(("train", "validation", "test"), frames):
        if len(frame) != config["expected"][f"{name}_rows"]:
            raise ValueError(f"Unexpected {name} size")
        path = output_dir / f"{name}_split.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\r\n", mode="x")
        if fingerprint(path)["sha256"] != config["expected"]["split_sha256"][path.name]:
            raise ValueError(f"{name} split is not byte-identical; check locked runtime/source/data")
        counts[name] = {"rows": len(frame), "labels": frame.label.value_counts().to_dict()}
    manifest = {"schema_version": 1, "release_id": config["release_id"], "seed": config["seed"],
        "config": fingerprint(config_path), "source": fingerprint(args.train_data),
        "cohort_reference_preprocessing": reference.to_config(), "model_preprocessing": processor.to_config(),
        "counts": counts, "overlap_checks": args.overlap_checks, "filter_audit": args.data_audit,
        "files": {path.name: fingerprint(path) for path in output_dir.glob("*_split.csv")},
        "test_role": "already-consumed historical test; not protocol V2 test"}
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/prepared/baseline-v1.0.0")
    parser.add_argument("--archive", type=Path, help="Offline copy of public Kaggle v1 zip; file contents are still verified")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    originals = ensure_originals(config, args.output_dir / "raw", archive_path=args.archive, offline=args.offline)
    manifest = prepare_splits(args.config, args.output_dir / "raw", args.output_dir / "splits")
    print(json.dumps({"release_id": config["release_id"], "originals": originals,
                      "splits": manifest["files"], "seed": manifest["seed"]}, indent=2))


if __name__ == "__main__":
    main()
