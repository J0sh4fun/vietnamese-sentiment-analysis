# Baseline release v1.0.0 — kiểm chứng tái lập

Ngày kiểm chứng: 2026-11-07. Git ref của mã nguồn release: **`refs/tags/baseline-v1.0.0`**. Đây là release cục bộ; chưa push tag hoặc xuất bản artifact lên dịch vụ bên ngoài.

## Phạm vi đã kiểm chứng

Đã tạo virtualenv mới `.venv-release`, không kế thừa site-packages, trên Windows x64 / CPython 3.13.5 / CPU. Cài bằng `pip install --require-hashes -r requirements.txt`, không dùng môi trường phát triển cũ. Lock gồm 70 distribution với phiên bản chính xác và hash wheel, bao gồm PyTorch 2.15.0+cpu. `pip check` và `src.release_baseline doctor` đều đạt.

Đã chạy các bước prepare → train → evaluate → predict trong [README](../../README.md). Sau đó cài lock Transformer tùy chọn, kiểm tra lại dependency và chạy toàn bộ 93 test: **OK, không skip**. Trước khi cài phần tùy chọn, bộ 93 test đạt với 6 test tokenizer được skip. Đây là kiểm chứng trên virtualenv mới của cùng máy, chưa phải kiểm chứng trên máy/OS khác.

## Dữ liệu, seed và cấu hình

[Cấu hình duy nhất](../../configs/baseline-v1.0.0.json) cố định seed 42, 4 CPU thread, preprocessing, TF-IDF, split, NB và MLP. Python/NumPy/PyTorch dùng cùng helper seed; DataLoader có generator riêng có seed; checkpoint giữ optimizer và trạng thái RNG để resume. Mỗi run lưu cấu hình thực tế, môi trường, source snapshot và manifest hash.

Tải lại dữ liệu công khai Kaggle **version 1** và kiểm tra cả hai JSONL theo SHA-256/số dòng. Bộ gốc có 9.599 mẫu; cohort tham chiếu còn 9.576. Không dùng augmentation. Bước chuẩn bị đã tái tạo CSV khớp từng byte với split đã đóng băng:

| Tập | Số mẫu | SHA-256 |
|---|---:|---|
| Train | 7.660 | `5a4e1a492f32d532090e462bb36db3373c22b8c078b324fb8aa8cdd0b400fcc6` |
| Validation cũ | 958 | `ff1f197113875c8a57483648d21d1e66b00a8195f78de3b27b92762ce06526c9` |
| Test cũ, đã tiêu thụ trước release | 958 | `ce1378a3aac65972ad97fded13746fdff51ec0e80554aa1d3f2b931dbb96afdc` |

Không trùng `source_id` giữa các tập. Preprocessing tham chiếu dùng để xác định cohort được ghi riêng với preprocessing đầu vào mô hình. Không âm thầm thay split khi đổi môi trường; hash không khớp thì dừng. Xem [data card](../data_card_20261107.md) về nguồn, nhãn và những điểm chưa xác minh.

## Kết quả chạy lại

Cùng TF-IDF fit trên train, 24.982 đặc trưng float32. ComplementNB alpha=0,5; MLP hidden=64, dropout=0,3, Adam LR=0,001, batch=128, chạy 15 epoch và chọn epoch 9 bằng macro-F1 validation cũ. Không tìm lại cấu hình trong release này.

| Mô hình | Accuracy | Macro-F1 | F1 negative | F1 positive |
|---|---:|---:|---:|---:|
| ComplementNB | 0,926931 | 0,923094 | 0,940273 | 0,905914 |
| MLP, epoch 9 | 0,942589 | 0,939045 | 0,953743 | 0,924347 |

Các metric và epoch khớp kết quả trước release. Thời gian fit riêng lần này: NB 0,0115 giây, MLP 80,42 giây; chưa bao gồm preprocessing/TF-IDF và không phải benchmark phần cứng. Bảng máy đọc được nằm trong [CSV](baseline-v1.0.0-comparison.csv).

Lệnh evaluate ở tiến trình Python mới kiểm tra hash, chạy từ raw `review` trên 958 mẫu validation, xác nhận dự đoán sau tải lại giống tham chiếu; logits được so sánh với `rtol=1e-6`, `atol=1e-7`. CLI predict đã kiểm tra hai review thô: “Sản phẩm không tốt” → negative; “Giao hàng nhanh, chất lượng tốt!” → positive ở cả hai mô hình. Preprocessing chỉ chạy một lần trong mỗi pipeline.

Nhánh Transformer tùy chọn đã kiểm tra cùng 7.660/958 mẫu và nhãn với release mới: TF-IDF khớp `clean_text` đã lưu, XLM-R nhận văn bản thô; 1 review train bị cắt ở giới hạn 512 token, validation không có mẫu bị cắt. Chưa huấn luyện classifier Transformer.

## Artifact và cách sử dụng

Run cục bộ: `models/releases/baseline-v1.0.0/`, gồm model NB, MLP inference bundle, checkpoint best/last có optimizer/RNG, TF-IDF, split, history, metric và source/environment/config snapshot. Run được kiểm tra bất biến bằng `release_manifest.json`, SHA-256:

`e42cefcf0061b24be08bc58352a89fc3f4e680fd861a4ddfcca5f996c068cd3e`

Bản source ZIP, model ZIP và descriptor chứa commit/hash thực tế được đặt cạnh run dưới `models/releases/` sau khi commit/tag. Các artifact lớn được Git-ignore; fresh clone tái tạo chúng bằng README, không phụ thuộc bundle cũ. Bundle đóng băng trước đó `models/baselines/baseline_v1_20261107` được giữ nguyên.

[Biên bản JSON](baseline-v1.0.0-verification.json) lưu hash lock/config/dữ liệu/split/run, kết quả kiểm thử và biên nhận evaluate. Không yêu cầu hash file checkpoint/ZIP giống nhau giữa mọi lần train: kiểm tra dữ liệu, cấu hình, metric, nhãn và logits mới là tiêu chí tái lập ở profile đã nêu.

## Giới hạn đánh giá

Validation đã dùng chọn cấu hình/epoch, test cũ đã từng được đọc và phân tích. Các số trên chỉ xác nhận tái lập baseline, không ước lượng chất lượng trên test độc lập. Release evaluator chỉ nhận train/validation. Test mới theo [protocol V2](../evaluation_protocol_v2_20261107.md) vẫn **NOT_RUN**; chưa có dữ liệu mới độc lập để chấm lại baseline theo V2. Chưa xác nhận tính tương đương trên Linux/macOS, CUDA hoặc phiên bản Python khác.
