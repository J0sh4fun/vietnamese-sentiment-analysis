"""Shared quality, latency, process-memory and training-cost measurement contract."""
from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ("negative", "positive")
MIB = 1024 ** 2
TRAIN_SCOPE = ("raw preprocessing + feature/token preparation + model/optimizer initialization + "
               "all training epochs + validation + checkpoint/artifact export and hashing; "
               "excludes data download, dependency install, hyperparameter search and final evaluation")


def quality_metrics(truth, predicted):
    truth, predicted = list(truth), list(predicted)
    if not truth or len(truth) != len(predicted):
        raise ValueError("Nonempty truth/predictions of equal length required")
    if set(truth + predicted) - set(LABELS):
        raise ValueError("Unknown labels; map class IDs explicitly to negative/positive")
    return {"rows": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)),
            "classification_report": classification_report(truth, predicted, labels=LABELS,
                target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
            "confusion_matrix_axes": {"rows": "true", "columns": "predicted", "labels": list(LABELS)},
            "zero_division": 0}


def latency_summary(seconds, batch_counts):
    seconds, counts = np.asarray(seconds, dtype=float), np.asarray(batch_counts, dtype=int)
    if (not len(seconds) or len(seconds) != len(counts) or np.any(counts < 1)
            or not np.isfinite(seconds).all() or np.any(seconds <= 0)):
        raise ValueError("Positive finite batch durations and sample counts required")
    total, samples = float(seconds.sum()), int(counts.sum())
    return {"timed_batches": len(seconds), "timed_reviews": samples, "total_seconds": total,
            "batch_mean_ms": float(seconds.mean() * 1000),
            "batch_p50_ms": float(np.percentile(seconds, 50) * 1000),
            "batch_p95_ms": float(np.percentile(seconds, 95) * 1000),
            "amortized_ms_per_review": total * 1000 / samples,
            "throughput_reviews_per_second": samples / total,
            "actual_batch_sizes": sorted(set(counts.tolist())),
            "percentile_unit": "batch; not individual-review latency"}


def process_memory():
    """Native OS counters, includes native NumPy/Torch allocations, not tracemalloc."""
    if platform.system() == "Windows":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

        kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": int(counter.WorkingSetSize), "peak_rss_bytes": int(counter.PeakWorkingSetSize),
                "source": "Windows GetProcessMemoryInfo working set"}
    if platform.system() == "Linux":
        values = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, value, _ = line.split()
                values[name] = int(value) * 1024
        return {"rss_bytes": values["VmRSS:"], "peak_rss_bytes": values["VmHWM:"],
                "source": "Linux /proc/self/status"}
    return {"rss_bytes": None, "peak_rss_bytes": None, "source": "unsupported OS; not measured"}


def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def gpu_memory(device, *, reset=False):
    if not str(device).startswith("cuda"):
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None, "scope": "CPU; not applicable"}
    import torch
    synchronize(device)
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "scope": "PyTorch allocator; includes resident model, not total device/driver VRAM"}


class TrainingMeasurement(AbstractContextManager):
    """Wrap the WHOLE per-model training run, in a fresh process, before preprocessing.

    Caller supplies resolved config/data hashes and hardware metadata in identity.
    The helper never fits a model, opens a test or infers cost from checkpoints.
    """
    def __init__(self, output, *, model, device, identity):
        self.output, self.model, self.device = Path(output), model, device
        self.identity = identity

    def __enter__(self):
        if self.output.exists():
            raise FileExistsError(self.output)
        self.before = process_memory()
        gpu_memory(self.device, reset=True)
        synchronize(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        synchronize(self.device)
        elapsed = time.perf_counter() - self.started
        payload = {"schema_version": 1, "kind": "standard_training_measurement", "model": self.model,
                   "status": "COMPLETE" if exc_type is None else "FAILED", "seconds": elapsed,
                   "scope": TRAIN_SCOPE, "device": str(self.device), "identity": self.identity,
                   "memory_before": self.before, "memory_after": process_memory(),
                   "cpu_peak_scope": "whole fresh training-process lifetime, including imports; not a delta",
                   "gpu_memory": gpu_memory(self.device), "pid": os.getpid()}
        with self.output.open("x", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        return False


def read_training_receipt(path, *, model, artifact, split):
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if (receipt.get("schema_version") != 1 or receipt.get("kind") != "standard_training_measurement"
            or receipt.get("model") != model or receipt.get("status") != "COMPLETE"
            or receipt.get("scope") != TRAIN_SCOPE):
        raise ValueError("Incompatible/incomplete standardized training receipt")
    seconds = receipt.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Training duration must be positive and finite")
    identity = receipt.get("identity", {})
    if identity.get("inference_artifact") != artifact or identity.get("validation_split") != split:
        raise ValueError("Training receipt is not bound to this artifact/validation split")
    for name in ("config", "train_split", "hardware"):
        if not identity.get(name):
            raise ValueError(f"Training receipt missing provenance: {name}")
    return receipt
