# Đóng gói MLP và so sánh triển khai — 06/11/2026

Đã hoàn thành 8/8 mục. Mô hình được đóng gói độc lập, nhận review thô, giữ đúng TF-IDF và preprocessing của thí nghiệm nguồn. Không mở test.

## Bằng chứng đóng gói và tiếp tục train

- Khôi phục optimizer bằng cách replay đúng cấu hình dropout/seed 42: sai lệch trọng số so với checkpoint epoch 9 cũ = **0**. Không giả lập trạng thái Adam từ trọng số.
- `training/best_training.pt`: `model_state_dict`, `optimizer_state_dict`, `architecture`, `epoch`, `validation_macro_f1`, `label_mapping`, `training_config`, trạng thái RNG Python/NumPy/PyTorch và generator DataLoader cùng epoch 9.
- `training/last_training.pt`: trạng thái epoch 15, dùng nếu muốn tiếp tục từ cuối run. Lưu best/last riêng; optimizer và trọng số luôn cùng epoch.
- Đã nạp checkpoint epoch 9 ở tiến trình Python mới và train tiếp đúng một epoch (đến epoch 10). Sai lệch trọng số so với epoch 10 chạy liền mạch = **0**; Adam step từ 540 lên 600.
- Bộ cố định 37 review: nhãn khớp hoàn toàn trước/sau tải lại ở tiến trình mới; sai lệch logits tối đa **0**; kiểm tra `rtol=1e-6, atol=1e-7`.
- Khôi phục preprocessing từ artifact nguồn; tái tạo khớp `clean_text` của toàn bộ 7660 train và 958 validation. Giữ snapshot stopword, segmentation, NFC, masking, negation và runtime từ cấu hình nguồn.
- Bundle suy luận `pipeline/` chứa `model.pt`, `tfidf.joblib`, `preprocessing.json`, `manifest.json`; có fingerprint để phát hiện file trộn/sửa. Copy nguyên thư mục này sang vị trí khác không cần run nguồn hoặc dữ liệu để predict (vẫn cần mã và dependency tương ứng).
- Mỗi lời gọi `infer` preprocess đúng một lần rồi TF-IDF transform → MLP. `predict` và `predict_proba` cũng dùng đường này. Không fit lại TF-IDF khi predict hoặc resume.

## Đầu vào đặc biệt

Chuỗi rỗng và chỉ dấu câu vẫn được giữ lại, có `empty_after_preprocessing=true` và `no_known_features=true`. Văn bản không rỗng nhưng không có đặc trưng trong vocabulary có `empty_after_preprocessing=false`, `no_known_features=true`. Mạng vẫn trả nhãn từ vector 0 (do bias đã học), kèm cảnh báo rõ ràng; đây không phải đánh giá có căn cứ về cảm xúc văn bản. Batch rỗng trả `[]`. Input None/số/nested/dict/set bị từ chối, không ép chuỗi hoặc âm thầm bỏ mẫu. Xác suất softmax chưa hiệu chỉnh. Xem ví dụ thực trong `special_inputs.json`.

## So sánh trên cùng validation

| Mô hình | Macro-F1 | F1 negative | F1 positive | Train (s) | Raw inference (ms/review) | Có sẵn TF-IDF (ms/review) | Artifact suy luận (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ComplementNB | 0.923094 | 0.940273 | 0.905914 | 0.012 | 2.2556 | 0.0054 | 1.853 |
| MLP dropout 0.3, epoch 9 | 0.939045 | 0.953743 | 0.924347 | 45.669 | 1.9036 | 0.1097 | 7.013 |

Protocol: train 7.660 mẫu, validation 958 mẫu; seed 42; cùng TF-IDF float32, cùng `clean_text`, không augmentation. NB alpha=0,5; MLP V→64→2, ReLU, dropout=0,3, Adam LR=0,001, batch=128, weight_decay=0; lấy trọng số epoch 9 như lựa chọn đã chốt trong run trước.

`Train (s)` chỉ đo fit classifier: NB một lần fit; MLP tổng thời gian cập nhật 15 epoch, loại trừ TF-IDF/preprocessing, đánh giá validation, lưu file và phép thử 32 mẫu. Đây là chi phí chạy cấu hình đã chọn, không bao gồm tìm kiếm 3 cấu hình của thí nghiệm cũ. Thời gian MLP đến epoch 9 cũng được lưu riêng trong comparison.json. Thời gian fit TF-IDF dùng chung được lưu ở metadata.json.

Suy luận đo bằng perf_counter trên CPU 4 threads, tokenizer/model ấm: một warm-up toàn validation, 5 lần đo. Raw inference gồm preprocessing + TF-IDF + dự đoán nhãn; cột có sẵn TF-IDF chỉ đo classifier (chia batch 128). Không tính tải đĩa, metrics, logits trả ra hoặc xác suất. ms/review là thời gian trung bình theo batch 958, không phải độ trễ một request; xem đầy đủ từng lần đo trong comparison.json. Các wrapper có chi phí kiểm tra/cảnh báo khác nhau nhỏ. Giới hạn BLAS threads bằng threadpoolctl cùng với torch.set_num_threads.

Dung lượng tính file suy luận thực trên đĩa, không nén: NB gồm một joblib chứa cả TF-IDF, classifier và preprocessing; MLP gồm bốn file trong pipeline/. Không tính optimizer/resume checkpoint, dữ liệu, report hay mã nguồn. Chi phí checkpoint train được ghi riêng trong metadata.json. Cách lưu file có overhead khác nhau, không phải lượng RAM hay dung lượng cài dependency.

## Kết luận thí nghiệm

Nếu ưu tiên chất lượng trên tập đánh giá hiện có, MLP dropout là ứng viên tốt hơn: macro-F1 0.9390, so với 0.9231 của ComplementNB; hai F1 từng lớp đều được trình bày để tránh chỉ nhìn accuracy. Tuy nhiên, ưu thế này chỉ được quan sát trên validation đã dùng chọn cấu hình và epoch. Nó chưa chứng minh MLP tốt hơn trên dữ liệu Shopee mới hoặc miền khác.

Nếu ưu tiên huấn luyện nhanh, artifact nhỏ và cách vận hành đơn giản, ComplementNB là lựa chọn đối chứng rất cạnh tranh. MLP cần nhiều bước cập nhật và lưu nhiều trọng số hơn. Vì cả hai dùng chung tách từ/TF-IDF, chi phí preprocessing có thể chiếm phần lớn raw inference; cần đọc đồng thời thời gian raw và classifier thay vì suy ra độ trễ toàn pipeline từ riêng forward của MLP. Không kết luận khả năng chịu tải production từ benchmark batch ấm này.

Checkpoint đủ optimizer/RNG đã cho phép tiếp tục train tái lập, nhưng train thêm không mặc nhiên làm mô hình tốt hơn. Run tiếp một epoch chỉ kiểm tra resume, không thay thế model epoch 9 được bàn giao. Mô hình vẫn có các giới hạn bag-of-ngrams với phủ định, câu nhiều khía cạnh và slang; văn bản ngoài vocabulary tạo vector 0. Xác suất chưa calibration, mới một seed/split, chưa kiểm chứng domain/time/product generalization. Test lịch sử từng được đọc lỗi; cần protocol đánh giá mới được chốt trước khi khẳng định khả năng tổng quát hóa.

## Checklist

| Hoàn thành | Việc cần làm | Bằng chứng |
|---|---|---|
| ☑ | Lưu model tốt nhất | training/best_training.pt và pipeline/model.pt, epoch 9 |
| ☑ | Lưu trạng thái tiếp tục train | optimizer/RNG/config; resume_check/resumed.pt epoch 10, khớp chạy liền |
| ☑ | Lưu toàn bộ suy luận | pipeline/ với TF-IDF, preprocessing snapshot và manifest |
| ☑ | Kiểm tra tải lại | Tiến trình mới, nhãn/logits khớp; reload_verification.json |
| ☑ | Dự đoán từ văn bản thô | src.mlp_pipeline.load_pipeline(...).infer(raw_reviews) và CLI |
| ☑ | Đầu vào đặc biệt | special_inputs.json; cờ empty/no_known_features và cảnh báo |
| ☑ | So sánh mô hình | comparison.json/csv: macro-F1, F1 từng lớp, train/inference/size |
| ☑ | Kết luận | Phần trên nêu lựa chọn theo chất lượng/chi phí và giới hạn |

API loading/resume tham khảo: [PyTorch saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

## Sử dụng artifact đã kiểm chứng

Bundle hoàn tất: [`models/mlp_packaged/mlp_raw_20261106_02/`](../models/mlp_packaged/mlp_raw_20261106_02/). Run chẩn đoán thất bại `_01` đã được dọn khỏi workspace. Pipeline hoàn tất `_02` và các checkpoint đã được kiểm chứng độc lập.

```python
from src.mlp_pipeline import load_pipeline

model = load_pipeline("models/mlp_packaged/mlp_raw_20261106_02/pipeline")
results = model.infer(["Sản phẩm không tốt", "Giao hàng nhanh, chất lượng tốt!", "!!!", ""],
                      include_probabilities=True, include_logits=True)
```

```powershell
# Dự đoán review thô, không tự làm sạch trước khi gọi.
.venv/Scripts/python.exe -m src.mlp_pipeline --model-dir models/mlp_packaged/mlp_raw_20261106_02/pipeline --text "Sản phẩm không tốt" "Giao hàng nhanh, chất lượng tốt!" --logits

# Input JSON UTF-8 dùng được cho batch có chuỗi rỗng; output phải là file mới.
.venv/Scripts/python.exe -m src.mlp_pipeline --model-dir models/mlp_packaged/mlp_raw_20261106_02/pipeline --input-json models/mlp_packaged/mlp_raw_20261106_02/fixed_raw_reviews.json --output-json work/mlp_raw_predictions_01.json --logits

# Resume thêm một epoch từ best (9 → 10); không ghi đè model suy luận.
.venv/Scripts/python.exe -m src.resume_mlp --checkpoint models/mlp_packaged/mlp_raw_20261106_02/training/best_training.pt --tfidf models/mlp_packaged/mlp_raw_20261106_02/training/tfidf.joblib --split-dir models/evaluation_workflow/full_validation_20260929_01 --output-dir models/mlp_resume/resume_01 --epochs 1

# Tạo lại toàn bộ package/kiểm chứng/benchmark; cần có run nguồn và split.
.venv/Scripts/python.exe -u -m src.package_mlp --run-name mlp_raw_reproduction_01
```

Các lệnh đều chạy từ thư mục gốc project. Inference chỉ cần `pipeline/` + mã/dependency; resume cần đúng split và TF-IDF được nhận dạng bằng SHA-256. Chọn output/run name mới mỗi lần. Để tiếp tục từ cuối thay vì best, dùng `training/last_training.pt` (epoch 15). `src.run_mlp` hiện lưu checkpoint version 2 cho các run mới; checkpoint version 1 của thí nghiệm cũ vẫn đọc được để infer, nhưng không đủ trạng thái để resume trực tiếp.

Đã chạy **72 tests, OK** sau thay đổi. Các test mới kiểm tra preprocessing một lần cho từng API, input invalid/empty/OOV, gradient/dropout trong inference, bundle di chuyển không còn thư mục nguồn, fingerprint/runtime mismatch, đúng nhãn baseline và Adam/RNG/shuffle khôi phục chính xác. Kiểm chứng thực trên dữ liệu project nằm trong [`reload_verification.json`](../models/mlp_packaged/mlp_raw_20261106_02/reload_verification.json). Sai lệch logits, trọng số replay và trọng số resume đều **0** trong môi trường đã chạy; không bảo đảm bitwise khi đổi hệ điều hành hoặc dependency.

[Bảng so sánh CSV được lưu trong Git](mlp_packaging_20261106_comparison.csv). JSON chi tiết gồm classification report, từng lần đo và chi phí train đến epoch tốt nhất nằm trong bundle. Runtime/package versions và code/input fingerprints có ở `metadata.json`; môi trường quan sát có trong `requirements-resolved.txt`. Bộ dependency không được tính vào dung lượng artifact.

Mã chính: [`mlp_pipeline.py`](../src/mlp_pipeline.py), [`package_mlp.py`](../src/package_mlp.py), [`resume_mlp.py`](../src/resume_mlp.py). Giao diện Streamlit cũ vẫn dùng loader scikit-learn; MLP raw-text dùng CLI/API ở đây.
