"""Package the selected MLP, replay its optimizer state, verify raw inference and benchmark.

Only saved train/validation are read. Existing experiment directories are never
modified. Replaying the same seed must recover the old selected weights before
any new artifact is accepted. python -m src.package_mlp --help
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.experiments import PROJECT_ROOT, code_snapshot, environment_snapshot, file_fingerprint, validate_run_name
from src.inference import SentimentModel, load_model
from src.mlp import (LABELS, LABEL_TO_ID, SentimentMLP, TfidfDataset, evaluate, load_checkpoint,
                     make_loader, seed_everything, training_checkpoint, train_epoch)
from src.mlp_pipeline import RawTextMLP, export_pipeline
from src.preprocessor import VietnameseTextProcessor
from src.run_mlp import load_splits


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def predict_features(model, features, batch_size=128):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start:start + batch_size].toarray().astype(np.float32))
            predictions.extend(model(batch).argmax(dim=1).tolist())
    return np.asarray(predictions)


def benchmark(function, rows, repeats):
    function()  # One full warm-up; exclude disk/model loading and metrics.
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    median = statistics.median(times)
    return {"rows": rows, "warmups": 1, "repeats": repeats, "batch_seconds": times,
            "median_seconds": median, "median_ms_per_review": median * 1000 / rows}


def max_state_difference(left, right):
    return max(float((left[key] - right[key]).abs().max()) for key in left)


def fit_baseline(classifier, vectorizer, features, labels, preprocessing_config):
    """The shared raw-text wrapper expects named classes, not numeric MLP targets."""
    if set(labels) != set(LABELS):
        raise ValueError("Raw baseline requires negative/positive string labels")
    classifier = type(classifier)(**classifier.get_params())
    started = time.perf_counter()
    classifier.fit(features, labels)
    seconds = time.perf_counter() - started
    raw = SentimentModel(Pipeline([("tfidf", vectorizer), ("classifier", classifier)]), preprocessing_config)
    return raw, classifier, seconds


def render_report(result, verification, metadata):
    nb, mlp = result
    rows = []
    for item in result:
        rows.append(f"| {item['model']} | {item['macro_f1']:.6f} | {item['f1_negative']:.6f} | "
                    f"{item['f1_positive']:.6f} | {item['train_seconds']:.3f} | "
                    f"{item['raw_inference']['median_ms_per_review']:.4f} | "
                    f"{item['features_inference']['median_ms_per_review']:.4f} | "
                    f"{item['inference_artifact_bytes']/1048576:.3f} |")
    return "\n".join([
        "# Đóng gói MLP và so sánh triển khai — 06/11/2026", "",
        "Đã hoàn thành 8/8 mục. Mô hình được đóng gói độc lập, nhận review thô, giữ đúng TF-IDF và preprocessing của thí nghiệm nguồn. Không mở test.", "",
        "## Bằng chứng đóng gói và tiếp tục train", "",
        f"- Khôi phục optimizer bằng cách replay đúng cấu hình dropout/seed 42: sai lệch trọng số so với checkpoint epoch 9 cũ = **{verification['replay_weights_max_abs_diff']:.3g}**. Không giả lập trạng thái Adam từ trọng số.",
        "- `training/best_training.pt`: `model_state_dict`, `optimizer_state_dict`, `architecture`, `epoch`, `validation_macro_f1`, `label_mapping`, `training_config`, trạng thái RNG Python/NumPy/PyTorch và generator DataLoader cùng epoch 9.",
        "- `training/last_training.pt`: trạng thái epoch 15, dùng nếu muốn tiếp tục từ cuối run. Lưu best/last riêng; optimizer và trọng số luôn cùng epoch.",
        f"- Đã nạp checkpoint epoch 9 ở tiến trình Python mới và train tiếp đúng một epoch (đến epoch 10). Sai lệch trọng số so với epoch 10 chạy liền mạch = **{verification['resume_weights_max_abs_diff']:.3g}**; Adam step từ {verification['optimizer_step_before']} lên {verification['optimizer_step_after']}.",
        f"- Bộ cố định {verification['fixed_samples']} review: nhãn khớp hoàn toàn trước/sau tải lại ở tiến trình mới; sai lệch logits tối đa **{verification['reload_logits_max_abs_diff']:.3g}**; kiểm tra `rtol=1e-6, atol=1e-7`.",
        f"- Khôi phục preprocessing từ artifact nguồn; tái tạo khớp `clean_text` của toàn bộ {metadata['train_rows']} train và {metadata['validation_rows']} validation. Giữ snapshot stopword, segmentation, NFC, masking, negation và runtime từ cấu hình nguồn.",
        "- Bundle suy luận `pipeline/` chứa `model.pt`, `tfidf.joblib`, `preprocessing.json`, `manifest.json`; có fingerprint để phát hiện file trộn/sửa. Copy nguyên thư mục này sang vị trí khác không cần run nguồn hoặc dữ liệu để predict (vẫn cần mã và dependency tương ứng).",
        "- Mỗi lời gọi `infer` preprocess đúng một lần rồi TF-IDF transform → MLP. `predict` và `predict_proba` cũng dùng đường này. Không fit lại TF-IDF khi predict hoặc resume.", "",
        "## Đầu vào đặc biệt", "",
        "Chuỗi rỗng và chỉ dấu câu vẫn được giữ lại, có `empty_after_preprocessing=true` và `no_known_features=true`. Văn bản không rỗng nhưng không có đặc trưng trong vocabulary có `empty_after_preprocessing=false`, `no_known_features=true`. Mạng vẫn trả nhãn từ vector 0 (do bias đã học), kèm cảnh báo rõ ràng; đây không phải đánh giá có căn cứ về cảm xúc văn bản. Batch rỗng trả `[]`. Input None/số/nested/dict/set bị từ chối, không ép chuỗi hoặc âm thầm bỏ mẫu. Xác suất softmax chưa hiệu chỉnh. Xem ví dụ thực trong `special_inputs.json`.", "",
        "## So sánh trên cùng validation", "",
        "| Mô hình | Macro-F1 | F1 negative | F1 positive | Train (s) | Raw inference (ms/review) | Có sẵn TF-IDF (ms/review) | Artifact suy luận (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|", *rows, "",
        "Protocol: train 7.660 mẫu, validation 958 mẫu; seed 42; cùng TF-IDF float32, cùng `clean_text`, không augmentation. NB alpha=0,5; MLP V→64→2, ReLU, dropout=0,3, Adam LR=0,001, batch=128, weight_decay=0; lấy trọng số epoch 9 như lựa chọn đã chốt trong run trước.", "",
        "`Train (s)` chỉ đo fit classifier: NB một lần fit; MLP tổng thời gian cập nhật 15 epoch, loại trừ TF-IDF/preprocessing, đánh giá validation, lưu file và phép thử 32 mẫu. Đây là chi phí chạy cấu hình đã chọn, không bao gồm tìm kiếm 3 cấu hình của thí nghiệm cũ. Thời gian MLP đến epoch 9 cũng được lưu riêng trong comparison.json. Thời gian fit TF-IDF dùng chung được lưu ở metadata.json.", "",
        f"Suy luận đo bằng perf_counter trên CPU {metadata['threads']} threads, tokenizer/model ấm: một warm-up toàn validation, {metadata['timing_repeats']} lần đo. Raw inference gồm preprocessing + TF-IDF + dự đoán nhãn; cột có sẵn TF-IDF chỉ đo classifier (chia batch 128). Không tính tải đĩa, metrics, logits trả ra hoặc xác suất. ms/review là thời gian trung bình theo batch 958, không phải độ trễ một request; xem đầy đủ từng lần đo trong comparison.json. Các wrapper có chi phí kiểm tra/cảnh báo khác nhau nhỏ. Giới hạn BLAS threads bằng threadpoolctl cùng với torch.set_num_threads.", "",
        "Dung lượng tính file suy luận thực trên đĩa, không nén: NB gồm một joblib chứa cả TF-IDF, classifier và preprocessing; MLP gồm bốn file trong pipeline/. Không tính optimizer/resume checkpoint, dữ liệu, report hay mã nguồn. Chi phí checkpoint train được ghi riêng trong metadata.json. Cách lưu file có overhead khác nhau, không phải lượng RAM hay dung lượng cài dependency.", "",
        "## Kết luận thí nghiệm", "",
        f"Nếu ưu tiên chất lượng trên tập đánh giá hiện có, MLP dropout là ứng viên tốt hơn: macro-F1 {mlp['macro_f1']:.4f}, so với {nb['macro_f1']:.4f} của ComplementNB; hai F1 từng lớp đều được trình bày để tránh chỉ nhìn accuracy. Tuy nhiên, ưu thế này chỉ được quan sát trên validation đã dùng chọn cấu hình và epoch. Nó chưa chứng minh MLP tốt hơn trên dữ liệu Shopee mới hoặc miền khác.", "",
        "Nếu ưu tiên huấn luyện nhanh, artifact nhỏ và cách vận hành đơn giản, ComplementNB là lựa chọn đối chứng rất cạnh tranh. MLP cần nhiều bước cập nhật và lưu nhiều trọng số hơn. Vì cả hai dùng chung tách từ/TF-IDF, chi phí preprocessing có thể chiếm phần lớn raw inference; cần đọc đồng thời thời gian raw và classifier thay vì suy ra độ trễ toàn pipeline từ riêng forward của MLP. Không kết luận khả năng chịu tải production từ benchmark batch ấm này.", "",
        "Checkpoint đủ optimizer/RNG đã cho phép tiếp tục train tái lập, nhưng train thêm không mặc nhiên làm mô hình tốt hơn. Run tiếp một epoch chỉ kiểm tra resume, không thay thế model epoch 9 được bàn giao. Mô hình vẫn có các giới hạn bag-of-ngrams với phủ định, câu nhiều khía cạnh và slang; văn bản ngoài vocabulary tạo vector 0. Xác suất chưa calibration, mới một seed/split, chưa kiểm chứng domain/time/product generalization. Test lịch sử từng được đọc lỗi; cần protocol đánh giá mới được chốt trước khi khẳng định khả năng tổng quát hóa.", "",
        "## Checklist", "",
        "| Hoàn thành | Việc cần làm | Bằng chứng |", "|---|---|---|",
        "| ☑ | Lưu model tốt nhất | training/best_training.pt và pipeline/model.pt, epoch 9 |",
        "| ☑ | Lưu trạng thái tiếp tục train | optimizer/RNG/config; resume_check/resumed.pt epoch 10, khớp chạy liền |",
        "| ☑ | Lưu toàn bộ suy luận | pipeline/ với TF-IDF, preprocessing snapshot và manifest |",
        "| ☑ | Kiểm tra tải lại | Tiến trình mới, nhãn/logits khớp; reload_verification.json |",
        "| ☑ | Dự đoán từ văn bản thô | src.mlp_pipeline.load_pipeline(...).infer(raw_reviews) và CLI |",
        "| ☑ | Đầu vào đặc biệt | special_inputs.json; cờ empty/no_known_features và cảnh báo |",
        "| ☑ | So sánh mô hình | comparison.json/csv: macro-F1, F1 từng lớp, train/inference/size |",
        "| ☑ | Kết luận | Phần trên nêu lựa chọn theo chất lượng/chi phí và giới hạn |", "",
        "API loading/resume tham khảo: [PyTorch saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).",
    ]) + "\n"


def run(args):
    validate_run_name(args.run_name)
    if args.threads < 1 or args.timing_repeats < 1:
        raise ValueError("threads/timing-repeats must be positive")
    output = args.output_dir / args.run_name
    if output.exists():
        raise FileExistsError("Output exists; use a fresh run name")
    torch.set_num_threads(args.threads)
    metadata_source = json.loads((args.source_run / "metadata.json").read_text(encoding="utf-8"))
    selection = json.loads((args.source_run / "selection.json").read_text(encoding="utf-8"))
    old_model, old_checkpoint = load_checkpoint(args.source_run / selection["checkpoint"])
    source_results = json.loads((args.source_run / "comparison.json").read_text(encoding="utf-8"))
    config_source = next(row for row in source_results if row["model"] == selection["selected_mlp"])
    train, validation = load_splits(args.split_dir)
    split_hashes = {f"{name}_split.csv": file_fingerprint(args.split_dir / f"{name}_split.csv")
                    for name in ("train", "validation")}
    if split_hashes != metadata_source["input_hashes"]:
        raise ValueError("Source experiment split fingerprints do not match")
    reference = load_model(args.split_dir / "sentiment_pipeline.joblib")
    preprocessing_config = reference.preprocessing_config
    processor = VietnameseTextProcessor.from_config(preprocessing_config)
    for name, frame in (("train", train), ("validation", validation)):
        if processor.transform(frame.review.tolist()) != frame.clean_text.tolist():
            raise ValueError(f"Preprocessing does not reproduce saved {name} clean_text")
    print("Preprocessing matches all 7660 train and 958 validation texts", flush=True)
    vectorizer = joblib.load(args.source_run / "tfidf.joblib")
    fitted_check = copy.deepcopy(vectorizer)
    started = time.perf_counter()
    fitted_check.fit(train.clean_text)
    tfidf_fit_seconds = time.perf_counter() - started
    if fitted_check.vocabulary_ != vectorizer.vocabulary_ or not np.array_equal(fitted_check.idf_, vectorizer.idf_):
        raise ValueError("Saved TF-IDF differs from train-only fit")
    x_train, x_val = vectorizer.transform(train.clean_text), vectorizer.transform(validation.clean_text)
    y_train = train.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    y_val = validation.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    train_data, val_data = TfidfDataset(x_train, y_train), TfidfDataset(x_val, y_val)
    output.mkdir(parents=True, exist_ok=False)
    training_dir = output / "training"
    training_dir.mkdir()
    shutil.copyfile(args.source_run / "tfidf.joblib", training_dir / "tfidf.joblib")
    config = {"optimizer": "Adam", "learning_rate": config_source["learning_rate"], "weight_decay": 0,
              "batch_size": metadata_source["batch_size"], "seed": metadata_source["seed"],
              "threads": args.threads, "epochs": metadata_source["epochs"], "loss": "CrossEntropyLoss",
              "split_hashes": split_hashes, "tfidf_sha256": file_fingerprint(training_dir / "tfidf.joblib")["sha256"]}
    if config["epochs"] != 15 or selection["best_epoch"] >= config["epochs"]:
        raise ValueError("This recovery protocol requires a best epoch before the end of a 15-epoch source run")
    seed_everything(config["seed"])
    model = SentimentMLP(x_train.shape[1], dropout=config_source["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    loader = make_loader(train_data, config["batch_size"], shuffle=True, seed=config["seed"])
    epoch_times, best_checkpoint, expected_next_state, expected_next_optimizer = [], None, None, None
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        loss = train_epoch(model, loader, optimizer)
        epoch_times.append(time.perf_counter() - started)
        print(f"Recover epoch={epoch:02d} optimization_loss={loss:.6f}", flush=True)
        if epoch == selection["best_epoch"]:
            difference = max_state_difference(model.state_dict(), old_model.state_dict())
            if difference > 1e-7:
                raise RuntimeError(f"Optimizer recovery failed: model differs from source by {difference}")
            metrics = evaluate(model, make_loader(val_data, config["batch_size"]))
            if abs(metrics["macro_f1"] - selection["validation_macro_f1"]) > 1e-12:
                raise RuntimeError("Recovered validation score differs from source")
            best_checkpoint = training_checkpoint(model, optimizer, loader, epoch=epoch,
                validation_macro_f1=metrics["macro_f1"], config=config)
            torch.save(best_checkpoint, training_dir / "best_training.pt")
        if epoch == selection["best_epoch"] + 1:
            expected_next_state = copy.deepcopy(model.state_dict())
            expected_next_optimizer = copy.deepcopy(optimizer.state_dict())
    last_metrics = evaluate(model, make_loader(val_data, config["batch_size"]))
    torch.save(training_checkpoint(model, optimizer, loader, epoch=config["epochs"],
        validation_macro_f1=last_metrics["macro_f1"], config=config), training_dir / "last_training.pt")
    export_pipeline(output / "pipeline", best_checkpoint, vectorizer, preprocessing_config)
    print("Saved raw-text bundle and best/last Adam checkpoints", flush=True)
    # A separate Python process must actually load optimizer/RNG and train the next epoch.
    command = [sys.executable, "-m", "src.resume_mlp", "--checkpoint", str(training_dir / "best_training.pt"),
               "--tfidf", str(training_dir / "tfidf.joblib"), "--split-dir", str(args.split_dir),
               "--output-dir", str(output / "resume_check"), "--epochs", "1"]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    resumed = torch.load(output / "resume_check/resumed.pt", weights_only=True, map_location="cpu")
    resume_difference = max_state_difference(resumed["model_state_dict"], expected_next_state)
    if resume_difference > 1e-7:
        raise RuntimeError(f"Resumed model differs from uninterrupted training: {resume_difference}")
    for key, state in expected_next_optimizer["state"].items():
        for name, value in state.items():
            if not torch.equal(value, resumed["optimizer_state_dict"]["state"][key][name]):
                raise RuntimeError("Resumed optimizer differs from uninterrupted optimizer")
    raw_mlp = RawTextMLP(old_model, vectorizer, preprocessing_config)
    fixed_texts = ["", "!!!", "xqzvvvzzrandomtoken20261106", "Sản phẩm không tốt", "Giao hàng nhanh, chất lượng tốt!"] + validation.review.iloc[:32].tolist()
    expected = raw_mlp.infer(fixed_texts, include_probabilities=True, include_logits=True)
    fixture_path = output / "fixed_raw_reviews.json"
    write_json(fixture_path, fixed_texts)
    write_json(output / "expected_fixed_predictions.json", expected)
    subprocess.run([sys.executable, "-m", "src.mlp_pipeline", "--model-dir", str(output / "pipeline"),
                    "--input-json", str(fixture_path), "--output-json", str(output / "reloaded_fixed_predictions.json"),
                    "--logits", "--threads", str(args.threads)], cwd=PROJECT_ROOT, check=True)
    actual = json.loads((output / "reloaded_fixed_predictions.json").read_text(encoding="utf-8"))
    expected_logits = np.asarray([row["logits"] for row in expected])
    actual_logits = np.asarray([row["logits"] for row in actual])
    if [row["label"] for row in expected] != [row["label"] for row in actual]:
        raise RuntimeError("Reload changed fixed-sample labels")
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-6, atol=1e-7)
    assert all(row["no_known_features"] for row in actual[:3])
    assert actual[0]["empty_after_preprocessing"] and actual[1]["empty_after_preprocessing"]
    assert not actual[2]["empty_after_preprocessing"]
    write_json(output / "special_inputs.json", [{"raw_review": text, **row} for text, row in zip(fixed_texts[:5], actual[:5])])
    # Fit and time NB on exactly the same features; MLP uses the unchanged selected epoch 9.
    baseline = joblib.load(args.source_run / "complement_nb.joblib")
    raw_nb, baseline, baseline_train_seconds = fit_baseline(
        baseline, vectorizer, x_train, train.label.tolist(), preprocessing_config)
    baseline_dir = output / "baseline"
    baseline_dir.mkdir()
    joblib.dump(raw_nb, baseline_dir / "sentiment_pipeline.joblib")
    def predict_nb_features():
        labels = np.concatenate([baseline.predict(x_val[start:start + 128]) for start in range(0, len(validation), 128)])
        return np.asarray([LABEL_TO_ID[label] for label in labels])
    results = []
    for name, estimator, predict_function, train_seconds in (
        ("ComplementNB", raw_nb, predict_nb_features, baseline_train_seconds),
        ("MLP dropout 0.3, epoch 9", raw_mlp, lambda: predict_features(old_model, x_val), sum(epoch_times)),
    ):
        predictions = predict_function()
        raw_predictions = estimator.predict(validation.review.tolist())
        if not np.array_equal(predictions, np.asarray([LABEL_TO_ID[label] for label in raw_predictions])):
            raise RuntimeError("Raw and saved clean_text predictions differ")
        report = classification_report(y_val, predictions, labels=[0, 1], target_names=LABELS,
                                       zero_division=0, output_dict=True)
        directory = baseline_dir if name == "ComplementNB" else output / "pipeline"
        item = {"model": name, "macro_f1": float(f1_score(y_val, predictions, average="macro")),
                "accuracy": float(accuracy_score(y_val, predictions)),
                "f1_negative": report["negative"]["f1-score"], "f1_positive": report["positive"]["f1-score"],
                "train_seconds": train_seconds, "classification_report": report,
                "inference_artifact_bytes": sum(path.stat().st_size for path in directory.iterdir() if path.is_file()),
                "features_inference": benchmark(predict_function, len(validation), args.timing_repeats),
                "raw_inference": benchmark(lambda: estimator.infer(validation.review.tolist()), len(validation), args.timing_repeats)}
        if name != "ComplementNB":
            item["train_seconds_to_selected_epoch"] = sum(epoch_times[:selection["best_epoch"]])
        results.append(item)
        print(f"{name}: macro-F1={item['macro_f1']:.6f}, raw_ms/review={item['raw_inference']['median_ms_per_review']:.4f}", flush=True)
    import pandas as pd
    pd.DataFrame([{key: value for key, value in row.items() if key not in
                   ("classification_report", "features_inference", "raw_inference")} |
                  {"raw_ms_per_review": row["raw_inference"]["median_ms_per_review"],
                   "features_ms_per_review": row["features_inference"]["median_ms_per_review"]}
                  for row in results]).to_csv(output / "comparison.csv", index=False)
    write_json(output / "comparison.json", results)
    before_step = int(next(iter(best_checkpoint["optimizer_state_dict"]["state"].values()))["step"].item())
    after_step = int(next(iter(resumed["optimizer_state_dict"]["state"].values()))["step"].item())
    verification = {"fixed_samples": len(fixed_texts), "labels_equal": True, "rtol": 1e-6, "atol": 1e-7,
                    "reload_logits_max_abs_diff": float(np.max(np.abs(actual_logits - expected_logits))),
                    "replay_weights_max_abs_diff": difference, "resume_weights_max_abs_diff": resume_difference,
                    "optimizer_state_matches_uninterrupted": True,
                    "optimizer_step_before": before_step, "optimizer_step_after": after_step,
                    "resume_epochs": 1, "resume_from_epoch": selection["best_epoch"],
                    "resumed_epoch": resumed["epoch"], "fresh_process_inference": True, "fresh_process_resume": True,
                    "preprocessing_matches_saved_splits": True, "tfidf_matches_train_only_fit": True}
    write_json(output / "reload_verification.json", verification)
    metadata = {"source_run": str(args.source_run.resolve()), "source_checkpoint": file_fingerprint(args.source_run / selection["checkpoint"]),
                "source_splits": split_hashes, "preprocessing_source": file_fingerprint(args.split_dir / "sentiment_pipeline.joblib"),
                "train_rows": len(train), "validation_rows": len(validation), "test_accessed": False,
                "tfidf_fit_seconds_shared": tfidf_fit_seconds, "mlp_epoch_train_seconds": epoch_times,
                "threads": args.threads, "timing_repeats": args.timing_repeats,
                "training_checkpoint_bytes": {name: (training_dir / name).stat().st_size for name in
                                              ("best_training.pt", "last_training.pt")},
                "environment": environment_snapshot(), "code": code_snapshot()}
    write_json(output / "metadata.json", metadata)
    (output / "report_vi.md").write_text(render_report(results, verification, metadata), encoding="utf-8")
    (output / "requirements-resolved.txt").write_text("".join(f"{key}=={value}\n" for key, value in
        metadata["environment"]["packages"].items()), encoding="utf-8")
    write_json(output / "artifact_manifest.json", {path.relative_to(output).as_posix(): file_fingerprint(path)
               for path in sorted(output.rglob("*")) if path.is_file()})
    print(f"Completed package: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=PROJECT_ROOT / "models/mlp/mlp_20261106_01")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "models/evaluation_workflow/full_validation_20260929_01")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/mlp_packaged")
    parser.add_argument("--run-name", default="mlp_raw_20261106_01")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=5)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
