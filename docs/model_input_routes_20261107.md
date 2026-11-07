# Hai đường xử lý review thô theo họ mô hình — 07/11/2026

Đã triển khai hai adapter riêng, cùng nhận `RawReview(source_id, review, label)` và cùng ánh xạ `negative=0`, `positive=1`. **Không sửa preprocessing/TF-IDF cũ hoặc 59 file của baseline v1. Không lấy `clean_text` làm input mặc định cho Transformer.** [Kết quả kiểm tra đầy đủ](model_input_routes_20261107.json).

## Model và yêu cầu đã đọc trước khi triển khai

Chọn **`FacebookAI/xlm-roberta-base`**, revision cố định **`e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`**. Metadata/model card có tiếng Việt và công bố license MIT. Đây là backbone pretrained MLM, **chưa phải classifier sentiment đã fine-tune**; việc chọn ở đây nhằm chốt contract tokenizer, không tuyên bố model tốt hơn NB/MLP.

- [Model card XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base): ví dụ truyền raw text trực tiếp vào tokenizer; model dùng subword/SentencePiece, không yêu cầu segmenter tiếng Việt bên ngoài.
- [Tokenizer config của revision](https://huggingface.co/FacebookAI/xlm-roberta-base/blob/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/tokenizer_config.json): `model_max_length=512`.
- [Model config của revision](https://huggingface.co/FacebookAI/xlm-roberta-base/blob/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/config.json): `max_position_embeddings=514`, pad ID 1. Với cách đánh position ID của XLM-R, giới hạn là **512 token**, không phải 514.
- [Padding/truncation](https://huggingface.co/docs/transformers/main/pad_truncation): padding theo batch dài nhất, truncation có max length rõ ràng. Đã kiểm tra hành vi trên bản cài thực `transformers=5.22.2`, `tokenizers=0.23.3`.
- Đã tải và kiểm tra tokenizer JSON thật: normalizer **Precompiled**, pre-tokenizer WhitespaceSplit/Metaspace, và 2 token đặc biệt cho một review. Nhánh mới **không thêm chuẩn hóa ngoài**; dùng đúng normalizer tích hợp trong file tokenizer đã pin/hash. Giữ case, dấu câu, emoji, URL/phone ở đầu vào; không áp dụng heuristic sửa dấu hoặc bộ lọc của TF-IDF.

Contract này chỉ dành cho XLM-R. Không chuyển nguyên cấu hình này sang model khác: ví dụ [PhoBERT](https://huggingface.co/vinai/phobert-base-v2) yêu cầu input đã tách từ và khuyến nghị RDRSegmenter/VnCoreNLP. Nếu chọn họ đó cần adapter/contract mới theo tài liệu của chính model.

## Hai nhánh rõ ràng

```text
RawReview(source_id, review thô, label)
       ├─ TF-IDF: snapshot VietnameseTextProcessor → TF-IDF đã fit → NB/MLP
       └─ XLM-R: raw text nguyên trạng → tokenizer model đã pin → input_ids + attention_mask

source_id, hash review thô và labels được đối chiếu giữa hai nhánh.
Không có mũi tên từ clean_text của TF-IDF sang tokenizer XLM-R.
```

| Thuộc tính | TF-IDF hiện tại | XLM-R riêng |
|---|---|---|
| Input mặc định | `review` thô | Chính `review` thô đó |
| Chữ hoa/thường | Lowercase theo snapshot cũ | Không tự lowercase |
| Chuẩn hóa | NFC và quy tắc cũ | Native normalizer của tokenizer đã pin |
| Tách từ tiếng Việt | underthesea, compound underscore theo snapshot | Không gọi underthesea, không tự ghép underscore |
| Dấu câu/emoji | Làm sạch theo code cũ | Không xóa trước tokenizer |
| URL/email/phone | Mask theo code cũ | Không tự mask |
| Stopwords | Tắt trong artifact đang chọn; giữ nguyên | Không lọc |
| Đặc trưng | CSR float32 `[B,24982]`, vectorizer không fit lại | IDs/mask int64 `[B,L]`, L ≤ 512 |
| Độ dài | Dùng toàn review theo pipeline cũ | Cắt phải, giữ tối đa 510 content tokens + 2 special tokens |
| Padding | Không áp dụng cho CSR | Bên phải, dài bằng sequence dài nhất trong batch |
| Nhãn | int64 `[B]`, negative=0/positive=1 | Cùng giá trị và thứ tự đó; key `labels` trong model inputs |

Hai model nhận cùng **mẫu/nhãn**, nhưng cách biểu diễn và lượng nội dung giữ lại khác nhau. TF-IDF không bị ép cắt theo context của Transformer. Nếu sau này so sánh điểm, phải công bố tỷ lệ truncation; không ngầm coi hai nhánh có cùng cửa sổ thông tin.

## Contract dữ liệu và API

[`review_inputs.py`](../src/review_inputs.py) chỉ quản lý mẫu/nhãn, không xử lý ngôn ngữ. `RawReviewDataset.from_csv()` yêu cầu rõ `source_id`, **`review`**, `label`; nếu chỉ có `clean_text` thì báo lỗi. Có cả hai cột thì luôn lấy `review`. Không drop/augment/chia lại mẫu, không sửa nhãn. Batch trộn mẫu có nhãn và không nhãn, source ID trùng hoặc nhãn ngoài hai lớp bị từ chối.

[`tfidf_inputs.py`](../src/tfidf_inputs.py) khôi phục đúng processor và vectorizer từ baseline, gọi `transform()`; không thay code cũ hay fit lại vocabulary/IDF. [`transformer_inputs.py`](../src/transformer_inputs.py) có tokenizer/profile riêng, không import hoặc gọi processor TF-IDF. Mỗi batch gọi tokenizer một lần trên raw strings; phần overflow chỉ dùng audit độ dài, **không trả thành các review bổ sung**.

```python
from src.review_inputs import RawReviewDataset, assert_same_samples
from src.tfidf_inputs import TfidfReviewInputs
from src.transformer_inputs import XLMRReviewInputs

baseline = "models/baselines/baseline_v1_20261107"
dataset = RawReviewDataset.from_csv(f"{baseline}/data/train_split.csv")
records = dataset[:16]  # Cùng một batch cho cả hai nhánh.

tfidf = TfidfReviewInputs.from_baseline(baseline)
transformer = XLMRReviewInputs()
left = tfidf.encode(records)
right = transformer.encode(records)
assert_same_samples(left.identity, right.identity)

X = left.features                   # CSR float32
y = left.identity.labels            # torch.long
inputs = right.model_inputs         # input_ids, attention_mask, labels
# Khi đã có classifier Transformer được train: output = model(**inputs)
```

Với inference, tạo `RawReview("new-001", "Sản phẩm tốt!")` không label; Transformer batch sẽ không có key `labels`. Thông tin source ID/hash/length nằm ngoài `model_inputs`, không đưa vào encoder. Caller cố tình truyền văn bản đã làm sạch vào trường `review` không thể được nhận biết chắc chắn bằng heuristic; trách nhiệm giữ raw input được bảo vệ bằng loader cố định và kiểm tra identity/hash, không đoán từ dấu underscore.

## Độ dài và đầu vào đặc biệt

`max_length=512` **bao gồm `<s>`/`</s>`**. Cắt phải, stride 0, một sequence/review; không head-tail sampling hoặc sliding-window aggregation trong bản này. `max_length >512` hoặc ≤2 bị từ chối. Mask bằng 1 ở token thật/special, 0 ở padding; pad ID=1. Không trả `token_type_ids` không cần thiết cho route này.

Độ dài trước cắt được tính từ fast-tokenizer encoding và overflow, loại phần special token lặp ở từng chunk. Không dùng dấu hiệu “length bằng 512” để kết luận chắc chắn đã cắt: test boundary **512 → không cắt**, **513 → có cắt** đã pass. Không nhân đôi labels khi có overflow.

| Input | TF-IDF | Transformer |
|---|---|---|
| Chuỗi rỗng | Zero features, cờ empty/OOV | Chỉ 2 special tokens; vẫn giữ mẫu và label |
| `!!!` | Thành chuỗi sạch rỗng | Vẫn có token dấu câu; không coi là chuỗi rỗng |
| Từ mới không có trong TF-IDF | Có thể zero features | Tách thành subword; có thể có `<unk>`, không đồng nhất với OOV TF-IDF |
| Review quá dài | Theo quy tắc cũ | Cắt phải, có cờ truncated và độ dài trước/sau |
| Batch rỗng | CSR `[0,V]` | IDs/mask `[0,0]`; caller không forward batch rỗng |
| None/số/list lồng | Báo lỗi | Báo lỗi, không ép chuỗi hoặc bỏ mẫu |

Ví dụ đã kiểm tra: `SẢN PHẨM này KHÔNG tốt!!! 😞` vào TF-IDF thành `sản_phẩm này không tốt`, còn raw string gốc được truyền nguyên trạng vào tokenizer XLM-R. Đây là khác biệt có chủ đích theo họ mô hình.

## Kết quả kiểm tra trên split phát triển đã lưu

| Tập | Số mẫu giữ nguyên | Số bị cắt | Median / P95 / max token trước cắt | Số token `<unk>` |
|---|---:|---:|---|---:|
| Train | 7.660 | 1 (0,0131%) | 35 / 98 / 538 | 110 |
| Validation cũ | 958 | 0 | 34 / 92,3 / 298 | 12 |

Mọi độ dài bao gồm special tokens; `<unk>` là **số token**, không phải số review. Cả hai nhánh khớp toàn bộ source ID, thứ tự, hash raw review và labels. TF-IDF tái tạo đúng `clean_text` đã lưu trên mọi mẫu; không có dòng nào bị loại vì empty/OOV/truncation. Test độc lập mới không được mở, không xử lý ví dụ từ test lịch sử; kiểm tra manifest có thể đọc hash file test cũ để xác minh tính toàn vẹn baseline.

Đã chạy **87 tests, OK** sau khi cài optional stack. Replay NB/MLP bằng source snapshot trong tiến trình mới vẫn khớp toàn bộ nhãn/logits/metrics validation cũ; cả **59 fingerprint baseline v1 không đổi**.

Unit/integration tests kiểm tra raw input không qua TF-IDF trước Transformer, một tokenizer call/batch, profile/hash, label alignment, padding/boundary/overflow, empty/OOV, và forward ở context 512 qua một **kiến trúc XLM-R nhỏ khởi tạo ngẫu nhiên**. Forward này chỉ kiểm tra shape/position IDs, không phải đánh giá pretrained model hay sentiment. Các test Transformer cần optional dependencies + tokenizer local; không tự tải mạng trong test.

## Cài và tái lập

```powershell
# Từ root project, sau requirements.txt.
.venv/Scripts/python.exe -m pip install -r requirements-transformer.txt
.venv/Scripts/python.exe -B -m src.transformer_inputs --prepare

# Dùng --output-dir mới mỗi lần.
.venv/Scripts/python.exe -B -m src.check_model_inputs --output-dir models/preprocessing_routes/routes_recheck_01
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

Tokenizer local: `models/transformer_tokenizers/xlm_roberta_base_e73636d4/`. Chỉ có ba file `config.json`, `tokenizer_config.json`, `tokenizer.json` và `processor_config.json`, với commit/hash cụ thể và runtime. Bộ fast tokenizer JSON đã đủ vocab/normalizer/postprocessor; không cần Java hay thư viện sentencepiece Python cho đường đang dùng. Chuẩn bị có thể tải ba file công khai đã pin; encoding luôn `local_files_only=True`, `trust_remote_code=False`, không cần token đăng nhập. Không tải pretrained weights (~model đầy đủ), không train classifier, không tạo metric sentiment mới trong tác vụ này.

Baseline v1 và protocol V2 **không được sửa**. Nhánh Transformer mới là phần chuẩn bị ứng viên nghiên cứu ngoài baseline v1. Nếu muốn thêm classifier Transformer vào đợt test xác nhận, cần train/chọn bằng dữ liệu phát triển, đóng băng phiên bản mới và cập nhật danh sách ứng viên/protocol trước khi mở test; không tự đưa nó vào scorer V2 hiện chỉ khóa NB/MLP.
