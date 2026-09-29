"""File-based experiment provenance and validation; no external services."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
import random
import re
import subprocess
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def path_reference(path, base=PROJECT_ROOT):
    path, base = Path(path).resolve(), Path(base).resolve()
    try:
        return {"path": path.relative_to(base).as_posix(), "relative_to": "base_directory"}
    except ValueError:
        # External inputs must be relocated explicitly, using their fingerprint.
        return {"path": path.name, "relative_to": "external", "relocation_required": True}


def file_fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def environment_snapshot():
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "version_source": "installed distribution metadata; not a claim every package was exercised",
        "thread_environment": {key: os.environ.get(key) for key in
            ("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "numerical_libraries": [{key: value for key, value in pool.items() if key != "filepath"}
                                for pool in threadpool_info()],
    }


def code_snapshot(root=PROJECT_ROOT):
    root = Path(root)
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", check=True, timeout=10).stdout.strip()
    try:
        revision = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=normal")
        result = {"revision": revision, "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.SubprocessError):
        result = {"revision": None, "dirty": None, "status": None, "reason": "Git revision unavailable"}
    paths = list(root.glob("*.py")) + [root / "requirements.txt"]
    for directory in ("src", "assets", "tests"):
        paths.extend((root / directory).rglob("*.py"))
    result["file_fingerprints"] = {path.relative_to(root).as_posix(): file_fingerprint(path)
                                   for path in sorted(paths) if path.is_file()}
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def validate_split_parameters(args):
    for name in ("test_size", "val_size"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and strictly between 0 and 1.")
    if args.test_size + args.val_size >= 1:
        raise ValueError("--test-size + --val-size must be less than 1, leaving a training split.")
    if type(args.random_state) is not int or not 0 <= args.random_state <= 2**32 - 1:
        raise ValueError("--random-state must be an integer between 0 and 4294967295.")


def validate_run_name(name):
    if (not name or name in {".", ".."} or name != name.strip() or name.endswith(".")
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", name.split(".")[0], re.I)):
        raise ValueError("--run-name must be one portable directory name, without separators or reserved characters/names.")


def validate_training_args(args):
    validate_split_parameters(args)
    for name in ("ngram_min", "ngram_max", "max_features", "min_df", "max_iter", "max_samples"):
        value = getattr(args, name)
        if value is None and name == "max_samples":
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer.")
    if args.ngram_max < args.ngram_min:
        raise ValueError("--ngram-max must be at least --ngram-min.")
    for name in ("regularization", "nb_alpha"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and greater than zero.")
    if len(set(args.allowed_labels)) < 2 or any(not label.strip() for label in args.allowed_labels):
        raise ValueError("--allowed-labels must contain at least two distinct nonblank labels.")
    if not args.text_column.strip() or not args.label_column.strip():
        raise ValueError("Text and label column names cannot be blank.")
    if args.run_name is not None:
        validate_run_name(args.run_name)
        if (args.output_dir / args.run_name).exists():
            raise FileExistsError("Run directory already exists; choose a new --run-name. Overwriting is not supported.")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError("--output-dir must be a directory.")
    paths = [args.train_data] + ([] if args.disable_aug else args.aug_data)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input dataset is not a readable file: {path}")


def json_value(value):
    if isinstance(value, Path):
        return path_reference(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def experiment_config(args, algorithms):
    requested = json_value(vars(args).copy())  # Call before adding derived runtime state.
    effective = dict(requested, algorithms=algorithms, allowed_labels=list(dict.fromkeys(args.allowed_labels)))
    effective["generate_aug"] = args.generate_aug and not args.disable_aug
    effective["aug_data"] = [] if args.disable_aug else requested["aug_data"]
    return {"requested_config": requested, "effective_config": effective}
