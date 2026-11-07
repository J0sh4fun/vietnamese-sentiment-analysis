# Bộ đo chung cho NB → MLP → Transformer

Script: [`src/benchmark_models.py`](../src/benchmark_models.py). Hàm dùng lại và bộ đo train: [`src/benchmarking.py`](../src/benchmarking.py). [Mẫu bảng CSV](templates/model_comparison.csv) và [bảng chạy thực tế](benchmark_20261107_comparison.csv).

## Chất lượng: cùng mẫu, cùng nhãn

- Metric chính: **Macro-F1**, trung bình F1 của đúng hai lớp theo thứ tự `negative`, `positive`, kể cả khi một lớp không xuất hiện. `zero_division=0`.
- Báo cáo thêm accuracy; precision, recall, F1, support cho từng lớp. Confusion matrix có **hàng = nhãn thật**, **cột = dự đoán**, thứ tự `[negative, positive]`. Các file kết quả JSON chứa đầy đủ classification report và ma trận số nguyên.
- Toàn bộ **958 mẫu validation cũ** được chấm chất lượng; không lấy 256 mẫu benchmark latency để thay thế tập chấm chất lượng. Không fit vectorizer, đổi threshold, chọn lại model hoặc mở test trong script này.
- Chỉ đọc `source_id`, raw `review`, `label`; không fallback sang `clean_text`. NB/MLP dùng preprocessing và TF-IDF của artifact. Transformer dùng nhánh raw XLM-R riêng. Tất cả giữ nguyên thứ tự/số mẫu và nhãn; khi đổi batch size phải cho cùng dự đoán, nếu khác script dừng để kiểm tra sai số/tie.
- Hash release manifest, split, tất cả file inference artifact, config từng worker, source và môi trường được ghi lại. Dữ liệu/weights đóng băng không bị sửa.

Đây là validation đã dùng phát triển. Kết quả **không phải** chất lượng trên tập test độc lập. Test V2 tiếp tục `NOT_RUN`; script không có tùy chọn mở test. Scorer V2 có gate riêng, không dùng script này để bỏ qua các gate đó.

## Độ trễ và throughput

Mặc định chạy **CPU**, 4 thread tính toán, Torch inter-op=1; batch **1 và 32**; **5 batch warm-up cho mỗi chế độ**, không tính warm-up vào thời gian. Chọn cùng 256 review validation không lặp bằng seed 42, lưu các chỉ số dòng rồi chạy **3 lượt** theo cùng thứ tự. Batch cuối nhỏ hơn vẫn được giữ và ghi `actual_batch_sizes`. Số batch đo, số review thực đo và từng thời gian thô đều được lưu.

| Chế độ | Phần nằm trong timer |
|---|---|
| `end_to_end` | Raw review trong RAM → preprocessing/tokenization → TF-IDF hoặc dense tensor → chuyển CPU/GPU → forward/predict → argmax và nhãn về CPU |
| `classifier_only` | Input đã chuẩn bị và đã ở đúng device → forward/predict → argmax và nhãn về CPU |

Trong `classifier_only`, chuẩn bị input riêng cho từng batch **trước** timer; không giữ toàn bộ dense corpus trên RAM/GPU. Preprocessing chạy đúng một lần mỗi lời gọi, không áp `clean_text` vào Transformer. Không tính đọc CSV, khởi tạo worker, tải artifact, metric hay serialize JSON vào latency. `load_seconds` được ghi riêng, gồm import adapter và load artifact trong worker; không tuyên bố đó là cold start vì OS disk cache chưa bị xóa.

Dùng wall clock `perf_counter`; CUDA được synchronize trước và sau đoạn đo để tránh chỉ đo thời gian enqueue. Các model/batch chạy **tuần tự**, mỗi cặp model/batch ở một tiến trình Python mới. Không chạy benchmark đồng thời với train, test hoặc tác vụ nặng khác. Không trộn kết quả CPU và GPU thành một xếp hạng duy nhất; NB luôn CPU, device thực tế nằm trong từng hàng.

Các số đo:

- `batch_p50_ms`, `batch_p95_ms`: percentile của **thời gian cả batch**, không phải percentile latency của từng review. Batch 1 mới gần với latency một request đơn; đây vẫn là benchmark offline, không đo queue/network/server.
- `amortized_ms_per_review = 1000 × tổng giây / tổng review thực đo`. Đây là chi phí trung bình phân bổ theo throughput, không phải p50/p95 của từng review trong batch 32.
- `throughput_reviews_per_second = tổng review / tổng giây`. Tổng review bao gồm các lượt lặp; không lấy trung bình các throughput batch có kích thước khác nhau.

Ít batch thì p95 chỉ mang tính mô tả. Lặp toàn bộ benchmark vào thư mục mới để xem độ biến động trên cùng máy; báo config và số mẫu cùng với kết quả.

## RAM, VRAM và dung lượng artifact

**RAM:** dùng native OS counter (Windows `GetProcessMemoryInfo`, Linux `/proc/self/status`), có tính allocation native của NumPy/Torch, không dùng `tracemalloc` làm tổng RAM. Ghi RSS trước/sau load và peak RSS. Peak là **toàn bộ vòng đời worker riêng**: import thư viện, load, quality pass, warm-up và hai chế độ latency; **không gọi là peak inference thuần, không trừ RSS nền và không cộng RAM với VRAM**. Peak của hai chế độ trong CSV giống nhau vì thuộc cùng worker. Shared library pages/OS cache khiến số này không phải dung lượng weights hoặc tổng RAM hệ thống. OS chưa hỗ trợ thì ghi null.

**VRAM:** với CUDA, reset peak allocator sau warm-up cho từng chế độ; ghi `max_memory_allocated` và `max_memory_reserved`. Bao gồm model còn resident và chuẩn bị input trong các lượt đo, kể cả khi prep ở ngoài latency timer của `classifier_only`. Đây là bộ đếm allocator PyTorch, **không phải toàn bộ VRAM của driver/device**. CPU ghi null thay vì 0 giả. Build CPU của baseline hiện tại sẽ từ chối CUDA; không tự chuyển về CPU nếu người dùng yêu cầu CUDA.

**Artifact:** tổng byte các file inference cần mang đi: NB joblib; MLP weights+TF-IDF+preprocessing+manifest; Transformer thư mục model+tokenizer+metadata. Không tính optimizer checkpoint, source, split hay ZIP của cả run. Dùng **MiB = 1.048.576 byte**. Thư mục Transformer nên chỉ chứa file inference để không vô tình tính log/checkpoint train thừa.

## Thời gian và bộ nhớ train

Không suy thời gian train từ weights/checkpoint. Một phép đo train chuẩn phải chạy **một model trong tiến trình mới**, dùng `TrainingMeasurement` bọc từ trước preprocessing đến sau export/hash artifact. Scope thống nhất:

**Raw preprocessing + fit/transform feature hoặc token preparation + khởi tạo model/optimizer + toàn bộ epoch + validation + ghi checkpoint/export/hash artifact.** Không tính cài dependency, tải dữ liệu/pretrained weights, tìm siêu tham số hoặc đánh giá cuối. NB không có epoch nhưng vẫn tính toàn bộ bước fit và validation của run. MLP/Transformer tính mọi epoch đã chạy, không chỉ đến epoch được chọn. Chi phí tìm cấu hình/seed khác cần báo riêng nếu có.

Ví dụ tích hợp vào entry point train mới (các hàm train là phần triển khai của từng họ model):

```python
from src.benchmarking import TrainingMeasurement
from src.benchmark_models import artifact_identity, hardware
from src.release_config import fingerprint

identity = {
    "config": fingerprint(config_path),
    "train_split": fingerprint(train_csv),
    "validation_split": fingerprint(validation_csv),
    "hardware": hardware(device),
    "seed": 42, "batch_size": 128, "threads": 4,
}
with TrainingMeasurement("mlp_training_cost.json", model="MLP", device=device, identity=identity):
    # preprocess raw reviews; fit train-only features; create model/optimizer;
    # train + validate all epochs; save selected inference artifact and checkpoints
    identity["inference_artifact"] = artifact_identity(inference_directory)
```

Helper synchronize CUDA, đo wall time, ghi peak RAM theo vòng đời tiến trình và peak allocated/reserved GPU trong scope train. Exception tạo receipt `FAILED`, không được đưa vào bảng train thành công. Receipt phải chứa hash config/train/validation, hardware và identity của artifact cuối. Evaluator kiểm tra receipt có đúng model/artifact/validation và scope; không coi lời khai trong receipt là bằng chứng nguồn dữ liệu độc lập.

Các thời gian train có sẵn của release v1 không có scope trên: NB chỉ `estimator.fit`, MLP là vòng epoch+validation/checkpoint, đều bỏ preprocessing/TF-IDF. Script giữ riêng ở `legacy_train_seconds`/`legacy_train_scope`; cột chuẩn `train_wall_seconds` và peak RAM train **để trống**, trạng thái `NOT_MEASURED_UNDER_COMMON_SCOPE`. Chưa chạy lại train bằng bộ đo mới; không dùng hai số lịch sử để kết luận model nào rẻ hơn toàn pipeline.

## Chạy trên baseline hiện tại

Sau các bước cài đặt/prepare/train trong [README](../README.md), dùng source hiện tại có script benchmark:

```powershell
.venv/Scripts/python.exe -B -m src.benchmark_models --run-dir models/releases/baseline-v1.0.0 --output-dir models/benchmarks/quality_cost_01 --batch-sizes 1 32 --warmup 5 --repeats 3 --benchmark-samples 256 --threads 4 --seed 42
```

Output gồm `comparison.csv`, `benchmark_manifest.json`, config/log và kết quả từng worker. JSON mỗi worker chứa report, confusion matrix, source_id/nhãn/dự đoán, latency thô, hardware, preprocessing và memory scope. Có thể kèm receipt train mới bằng `--training-receipt MLP=D:/runs/mlp_training_cost.json` (lặp tùy chọn cho từng model). Mỗi lần chạy dùng thư mục output mới, ở ngoài run/artifact bất biến.

## Thêm Transformer sentiment khi có

Project hiện **chưa có** classifier Transformer đã fine-tune. Mẫu bảng và lần chạy này ghi `NOT_RUN_NO_FINETUNED_CHECKPOINT`, không ghi F1/latency bằng 0 và không lấy kết quả backbone MLM làm baseline sentiment.

Đầu vào adapter đang hỗ trợ **XLM-R đã fine-tune**, đúng nhánh đã chốt:

```text
my_sentiment_xlmr/
  model/       # save_pretrained classifier: config + complete local weights
  tokenizer/   # pinned XLM-R tokenizer files + processor_config.json
```

Tokenizer là bản đã kiểm chứng từ `src.transformer_inputs --prepare`; giữ nguyên profile/hash/runtime, max length 512, padding theo batch, nhận raw review. Model config cần `model_type=xlm-roberta`, 2 nhãn và `id2label` rõ ràng `negative`/`positive`; adapter tôn trọng thứ tự ID đã lưu, không giả định ID 0 luôn là negative. Checkpoint thiếu classifier weights/khởi tạo head mới bị từ chối. Chỉ đọc file local, `trust_remote_code=False`, không tải hay train model lúc benchmark. Cần giữ bằng chứng fine-tuning, split/config/seed và tokenizer tương thích cùng checkpoint; một file config tự khai không chứng minh model đã được train đúng.

```powershell
.venv/Scripts/python.exe -B -m src.benchmark_models --output-dir models/benchmarks/with_xlmr_01 --transformer-dir D:/runs/my_sentiment_xlmr --neural-device cpu --batch-sizes 1 32 --warmup 5 --repeats 3
```

Mặc định neural model chạy float32, không autocast/quantize. `--neural-device cuda:0` dành cho môi trường CUDA được kiểm chứng riêng, không thay đổi lock CPU của release v1. Adapter đã được test bằng checkpoint XLM-R tổng hợp nhỏ để kiểm tra serialization, raw input và đảo thứ tự nhãn; **chưa đo classifier sentiment Transformer thực hoặc GPU**. Họ model khác cần adapter/preprocessing riêng trước khi đưa vào cùng bảng.

## Lần chạy 07/11/2026

Máy **Intel Core i7-9850H @ 2,60 GHz**, Windows 11, CPU, 4 thread, inter-op=1. Quality trên toàn bộ 958 validation cũ; latency trên cùng 256 review, seed 42, warm-up 5, repeat 3, batch 1/32. [Bảng đầy đủ](benchmark_20261107_comparison.csv) chứa từng lớp, memory, artifact và hai scope latency; [biên bản JSON](benchmark_20261107_summary.json) chứa confusion matrix và hash kết quả gốc. Lần benchmark chạy tuần tự, hoàn tất trước khi chạy test adapter.

| Model | Macro-F1 | Batch | E2E p50/p95 batch (ms) | E2E ms/review phân bổ | Classifier ms/review phân bổ |
|---|---:|---:|---:|---:|---:|
| ComplementNB | 0,923094 | 1 | 2,059 / 4,903 | 2,432 | 0,288 |
| MLP | 0,939045 | 1 | 2,229 / 5,316 | 2,580 | 0,273 |
| ComplementNB | 0,923094 | 32 | 49,384 / 64,153 | 1,614 | 0,0115 |
| MLP | 0,939045 | 32 | 58,554 / 72,064 | 1,892 | 0,0385 |
| Transformer | NOT_RUN | — | — | — | — |

MLP có Macro-F1 cao hơn; NB rẻ hơn về thời gian phân bổ ở batch 32 trong lần đo này. Khi tính preprocessing, chênh lệch nhỏ hơn đáng kể so với classifier-only: xử lý văn bản chiếm phần lớn thời gian. Batch 32 tăng throughput nhưng một batch chờ lâu hơn; không đọc ms/review phân bổ thành latency của một request đơn. Đây là một lượt benchmark trên cùng máy, không có khoảng tin cậy hay kết luận chi phí GPU/Transformer/train toàn pipeline.

Kiểm chứng tại thời điểm thêm benchmark: **100 test đạt**, gồm 7 test mới về thứ tự lớp, metric khi thiếu lớp, timer có/không preprocessing, batch cuối nhỏ, bộ nhớ native, receipt train ràng buộc artifact/split và adapter Transformer local với thứ tự class ID đảo. Lệnh lúc đó: `.venv-release/Scripts/python.exe -B -m unittest discover -s tests -v`. Không có benchmark latency chạy đồng thời với bộ test này. Suite hiện tại và policy CI được cập nhật trong [README](../README.md).

Tham khảo phương pháp: [CUDA synchronization và allocator](https://docs.pytorch.org/docs/main/notes/cuda.html), [peak memory allocated/reset](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html), [Hugging Face AutoModel với checkpoint local](https://huggingface.co/docs/transformers/main/model_doc/auto). Phạm vi đo được cố định bởi code của project, không dựa vào việc các API này tự đo toàn bộ chi phí hệ thống.
