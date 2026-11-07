"""Common NB/MLP/local fine-tuned XLM-R validation and cost benchmark. No test access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.benchmarking import (LABELS, gpu_memory, latency_summary, process_memory,
                             quality_metrics, read_training_receipt, synchronize)
from src.experiments import code_snapshot, environment_snapshot, seed_everything
from src.release_config import ROOT, fingerprint
from src.review_inputs import RawReviewDataset


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def artifact_identity(path):
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Empty/missing inference artifact: {path}")
    entries = {(p.name if path.is_file() else p.relative_to(path).as_posix()): fingerprint(p) for p in files}
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "size_bytes": sum(p["size_bytes"] for p in entries.values()), "files": entries}


class TfidfAdapter:
    def __init__(self, kind, path, device):
        self.kind, self.device = kind, device
        if kind == "ComplementNB":
            if device != "cpu":
                raise ValueError("ComplementNB runs on CPU")
            from src.inference import load_model
            bundle = load_model(path)
            self.processor = bundle._processor
            self.vectorizer = bundle._pipeline.named_steps["tfidf"]
            self.model = bundle._pipeline.named_steps["classifier"]
        else:
            from src.mlp_pipeline import load_pipeline
            bundle = load_pipeline(path)
            self.processor, self.vectorizer = bundle._processor, bundle.vectorizer
            self.model = bundle.model.to(device).eval()
        self.contract = {"family": "tfidf", "preprocessing": bundle.preprocessing_config,
                         "precision": "saved estimator dtype" if kind == "ComplementNB" else "float32"}

    def prepare(self, texts):
        matrix = self.vectorizer.transform(self.processor.transform(texts)).tocsr()
        if self.kind == "ComplementNB":
            return matrix
        return torch.from_numpy(matrix.astype(np.float32).toarray()).to(self.device)

    def classify(self, features):
        if self.kind == "ComplementNB":
            return self.model.predict(features).tolist()
        with torch.inference_mode():
            ids = self.model(features).argmax(dim=1).cpu().tolist()
        return [LABELS[i] for i in ids]


class TransformerAdapter:
    def __init__(self, path, device):
        from transformers import AutoModelForSequenceClassification
        from src.transformer_inputs import XLMRReviewInputs
        path = Path(path)
        # A tokenizer-only/pretrained MLM directory is never a sentiment baseline.
        self.inputs = XLMRReviewInputs(path / "tokenizer")
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            path / "model", local_files_only=True, trust_remote_code=False, output_loading_info=True)
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError("Checkpoint has missing/mismatched weights; random classifier initialization rejected")
        if model.config.model_type != "xlm-roberta" or model.config.num_labels != 2:
            raise ValueError("Requires a two-class fine-tuned XLM-R checkpoint")
        mapping = {int(key): value for key, value in model.config.id2label.items()}
        if set(mapping) != {0, 1} or set(mapping.values()) != set(LABELS):
            raise ValueError("Explicit model.config.id2label negative/positive required; LABEL_0 is ambiguous")
        self.labels, self.device = mapping, device
        self.model = model.float().to(device).eval()
        self.contract = {"family": "xlm-roberta", "precision": "float32", "id2label": mapping,
                         "preprocessing": self.inputs.profile, "max_length": self.inputs.max_length}

    def prepare(self, texts):
        encoded = self.inputs.tokenize_raw(texts)[0]  # raw review; no TF-IDF clean_text
        return {key: value.to(self.device) for key, value in encoded.items()}

    def classify(self, encoded):
        with torch.inference_mode():
            ids = self.model(**encoded).logits.argmax(dim=1).cpu().tolist()
        return [self.labels[i] for i in ids]


def batches(values, size):
    return [values[start:start + size] for start in range(0, len(values), size)]


def measure_latency(adapter, texts, *, batch_size, warmup, repeats, mode):
    chunks = batches(texts, batch_size)
    def call(chunk):
        # classifier_only input is prepared outside the timer, including host-to-device copy.
        prepared = adapter.prepare(chunk) if mode == "classifier_only" else None
        synchronize(adapter.device)
        started = time.perf_counter()
        result = adapter.classify(adapter.prepare(chunk) if mode == "end_to_end" else prepared)
        synchronize(adapter.device)
        elapsed = time.perf_counter() - started
        if len(result) != len(chunk):
            raise ValueError("Adapter dropped/duplicated inputs")
        return elapsed
    for i in range(warmup):
        call(chunks[i % len(chunks)])
    gpu_memory(adapter.device, reset=True)
    seconds, counts = [], []
    for _ in range(repeats):
        for chunk in chunks:
            seconds.append(call(chunk))
            counts.append(len(chunk))
    return {"mode": mode, "includes_preprocessing": mode == "end_to_end",
            "includes_host_to_device": mode == "end_to_end", "includes_argmax_and_label_cpu_transfer": True,
            "warmup_batches": warmup, "repeats": repeats,
            **latency_summary(seconds, counts), "gpu_memory": gpu_memory(adapter.device),
            "samples": [{"batch_seconds": t, "reviews": n} for t, n in zip(seconds, counts)]}


def hardware(device):
    cpu = platform.processor()
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    elif platform.system() == "Linux":
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "platform": platform.platform(),
            "device": device, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "torch_cuda_build": torch.version.cuda, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def worker(spec):
    seed_everything(spec["seed"], torch_enabled=True)
    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    device = spec["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no silent CPU fallback")
    with threadpool_limits(limits=spec["threads"]):
        dataset = RawReviewDataset.from_csv(spec["split"])
        if not len(dataset):
            raise ValueError("Empty evaluation split")
        texts = [row.review for row in dataset]
        truth = [row.label for row in dataset]
        before = process_memory()
        synchronize(device)
        started = time.perf_counter()
        adapter = (TransformerAdapter(spec["artifact"], device) if spec["model"] == "Transformer"
                   else TfidfAdapter(spec["model"], spec["artifact"], device))
        synchronize(device)
        load_seconds = time.perf_counter() - started
        after_load = process_memory()
        predicted = []
        for chunk in batches(texts, spec["batch_size"]):
            predicted.extend(adapter.classify(adapter.prepare(chunk)))
        count = min(spec["benchmark_samples"], len(texts))
        indexes = np.sort(np.random.default_rng(spec["seed"]).choice(len(texts), count, replace=False))
        selected = [texts[i] for i in indexes]
        measurements = [measure_latency(adapter, selected, batch_size=spec["batch_size"],
            warmup=spec["warmup"], repeats=spec["repeats"], mode=mode)
            for mode in ("end_to_end", "classifier_only")]
        result = {"model": spec["model"], "status": "COMPLETE", "batch_size": spec["batch_size"],
                  "hardware": hardware(device), "load_seconds": load_seconds,
                  "input_contract": adapter.contract, "quality": quality_metrics(truth, predicted),
                  "ordered_predictions_sha256": hashlib.sha256(json.dumps(predicted).encode()).hexdigest(),
                  "predictions": [{"source_id": row.source_id, "truth": row.label, "predicted": pred}
                                  for row, pred in zip(dataset, predicted)],
                  "benchmark_indices": indexes.tolist(), "latency": measurements,
                  "memory": {"before_load": before, "after_load": after_load, "after_benchmark": process_memory(),
                      "cpu_peak_scope": "isolated worker lifetime: imports, loading, full quality pass, warmup, both latency modes; not inference-only/delta"},
                  "pid": os.getpid()}
        write_json(spec["result"], result)


TABLE_COLUMNS = ["model", "status", "split_sha256", "rows", "macro_f1", "accuracy",
    "precision_negative", "recall_negative", "f1_negative", "support_negative",
    "precision_positive", "recall_positive", "f1_positive", "support_positive",
    "device", "cpu", "gpu", "threads", "batch_size", "mode", "preprocessing_included",
    "warmup_batches", "repeats", "benchmark_reviews", "batch_p50_ms", "batch_p95_ms",
    "amortized_ms_per_review", "throughput_reviews_per_second", "load_seconds",
    "rss_after_load_mib", "process_lifetime_peak_rss_mib", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib",
    "inference_artifact_mib", "train_wall_seconds", "train_peak_rss_mib", "train_device",
    "training_status", "legacy_train_seconds", "legacy_train_scope", "result_json"]


def mib(value):
    return None if value is None else value / (1024 ** 2)


def comparison_rows(result, spec, training, legacy):
    q = result["quality"]
    report = q["classification_report"]
    common = {"model": result["model"], "status": result["status"], "rows": q["rows"],
              "split_sha256": spec["split_fingerprint"]["sha256"], "macro_f1": q["macro_f1"], "accuracy": q["accuracy"],
              "device": spec["device"], "cpu": result["hardware"]["cpu"], "gpu": result["hardware"]["gpu"],
              "threads": spec["threads"], "batch_size": spec["batch_size"], "load_seconds": result["load_seconds"],
              "inference_artifact_mib": mib(spec["artifact_identity"]["size_bytes"]),
              "rss_after_load_mib": mib(result["memory"]["after_load"]["rss_bytes"]),
              "process_lifetime_peak_rss_mib": mib(result["memory"]["after_benchmark"]["peak_rss_bytes"]),
              "train_wall_seconds": training["seconds"] if training else None,
              "train_peak_rss_mib": mib(training["memory_after"]["peak_rss_bytes"]) if training else None,
              "train_device": training["device"] if training else None,
              "training_status": "STANDARD_RECEIPT" if training else "NOT_MEASURED_UNDER_COMMON_SCOPE",
              "legacy_train_seconds": legacy.get("seconds"), "legacy_train_scope": legacy.get("scope"),
              "result_json": Path(spec["result"]).name}
    for label in LABELS:
        for metric in ("precision", "recall", "f1-score", "support"):
            common[f"{'f1' if metric == 'f1-score' else metric}_{label}"] = report[label][metric]
    return [{**common, "mode": m["mode"], "preprocessing_included": m["includes_preprocessing"],
             "warmup_batches": m["warmup_batches"], "repeats": m["repeats"],
             "benchmark_reviews": len(result["benchmark_indices"]),
             **{key: m[key] for key in ("batch_p50_ms", "batch_p95_ms", "amortized_ms_per_review", "throughput_reviews_per_second")},
             "gpu_peak_allocated_mib": mib(m["gpu_memory"]["peak_allocated_bytes"]),
             "gpu_peak_reserved_mib": mib(m["gpu_memory"]["peak_reserved_bytes"])} for m in result["latency"]]


def run(args):
    from src.release_baseline import verified_run
    run_dir = args.run_dir.resolve()
    verified_run(run_dir)  # Verify immutable model/source/split bytes before any worker.
    split = run_dir / "splits/validation_split.csv"
    split_fp = fingerprint(split)
    models = [("ComplementNB", run_dir / "artifacts/complement_nb.joblib", "cpu"),
              ("MLP", run_dir / "artifacts/mlp", args.neural_device)]
    if args.transformer_dir:
        models.append(("Transformer", args.transformer_dir.resolve(), args.neural_device))
    receipts = dict(item.split("=", 1) for item in args.training_receipt)
    if set(receipts) - {m[0] for m in models}:
        raise ValueError("Training receipt model is absent/unknown")
    identities = {name: artifact_identity(path) for name, path, _ in models}
    training = {name: read_training_receipt(path, model=name, artifact=identities[name], split=split_fp)
                for name, path in receipts.items()}
    output = args.output_dir.resolve()
    if output.is_relative_to(run_dir) or any(output.is_relative_to(path.resolve()) for _, path, _ in models if path.is_dir()):
        raise ValueError("Outputs must be outside immutable input artifacts")
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    legacy = {"ComplementNB": {"seconds": metadata["ComplementNB_fit_seconds"], "scope": "estimator.fit only; excludes TF-IDF/preprocessing"},
              "MLP": {"seconds": metadata["MLP"]["fit_seconds"], "scope": "fit_mlp epochs/validation/checkpoint; excludes TF-IDF/preprocessing"}}
    rows, specs, prediction_hashes = [], [], {}
    for name, artifact, device in models:
        for batch_size in args.batch_sizes:
            stem = f"{name.lower()}_b{batch_size}"
            spec = {"model": name, "artifact": str(artifact), "device": device, "split": str(split),
                    "split_fingerprint": split_fp, "artifact_identity": identities[name], "batch_size": batch_size,
                    "threads": args.threads, "seed": args.seed, "benchmark_samples": args.benchmark_samples,
                    "warmup": args.warmup, "repeats": args.repeats, "result": str(output / f"{stem}.json")}
            config_path = output / f"{stem}_config.json"
            write_json(config_path, spec)
            # Sequential, fresh process per model/batch; no concurrent timing contamination.
            with (output / f"{stem}.log").open("x", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-B", "-m", "src.benchmark_models", "--worker-config", str(config_path)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                    env={**os.environ, "PYTHONHASHSEED": str(args.seed), "TOKENIZERS_PARALLELISM": "false",
                         "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                         "OPENBLAS_NUM_THREADS": str(args.threads)})
            result = json.loads(Path(spec["result"]).read_text(encoding="utf-8"))
            digest = result["ordered_predictions_sha256"]
            if name in prediction_hashes and prediction_hashes[name] != digest:
                raise ValueError(f"Predictions changed with batch size for {name}; inspect numerical ties")
            prediction_hashes[name] = digest
            rows.extend(comparison_rows(result, spec, training.get(name), legacy.get(name, {})))
            specs.append(spec)
            print(f"{name} batch={batch_size}: macro-F1={result['quality']['macro_f1']:.6f}", flush=True)
    if not args.transformer_dir:
        for batch_size in args.batch_sizes:
            for mode in ("end_to_end", "classifier_only"):
                rows.append({"model": "Transformer", "status": "NOT_RUN_NO_FINETUNED_CHECKPOINT", "batch_size": batch_size,
                             "mode": mode, "split_sha256": split_fp["sha256"]})
    with (output / "comparison.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "benchmark_manifest.json", {"schema_version": 1, "status": "COMPLETE", "seed": args.seed,
        "release_manifest": fingerprint(run_dir / "release_manifest.json"), "split": split_fp,
        "split_role": "consumed development validation; no new-test claim", "test_accessed": False,
        "training_performed": False, "specs": specs, "training_receipts": training,
        "code": code_snapshot(), "environment": environment_snapshot(),
        "notes": ["Offline local inference artifacts; no model fit/download during benchmark.",
                  "Latency excludes artifact load, file I/O, scoring and report serialization.",
                  "CPU memory is process-lifetime peak, not per-mode inference peak.",
                  "Legacy training durations have different scopes; not directly comparable."]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--run-dir", type=Path, default=ROOT / "models/releases/baseline-v1.0.0")
    p.add_argument("--transformer-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--neural-device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--benchmark-samples", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--training-receipt", action="append", default=[], metavar="MODEL=JSON")
    args = p.parse_args()
    if args.worker_config:
        worker(json.loads(args.worker_config.read_text(encoding="utf-8")))
        return
    if (not args.output_dir or min(args.batch_sizes + [args.repeats, args.benchmark_samples, args.threads]) < 1
            or args.warmup < 1 or not 0 <= args.seed < 2 ** 32 or len(set(args.batch_sizes)) != len(args.batch_sizes)
            or any("=" not in item for item in args.training_receipt)):
        p.error("Need a fresh --output-dir, positive unique batch sizes/repeats/samples/threads/warmup and valid seed/receipts")
    run(args)


if __name__ == "__main__":
    main()
