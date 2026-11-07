# Protocol đánh giá mới — V2.0, chốt ngày 07/11/2026

**Trạng thái: đã chốt thiết kế; chưa có dữ liệu test mới, chưa mở test.**

Protocol này đánh giá hai artifact đã cố định trên review Shopee tiếng Việt được thu thập mới, ngoài toàn bộ dữ liệu phát triển. Không đổi seed rồi chia lại dữ liệu cũ để gọi là test mới. Không chạy huấn luyện, tải dữ liệu bên ngoài hoặc đánh giá mới khi viết tài liệu này.

Múi giờ quy ước: **Asia/Bangkok (UTC+07:00)**. Tên cohort: `shopee_prospective_20261108_20261205_v2`. Mọi thay đổi protocol phải có phiên bản và lý do, được lập trước khi tiếp cận nội dung hoặc kết quả của cohort chịu ảnh hưởng.

## 1. Câu hỏi và phạm vi kết luận

Câu hỏi chính: **MLP dropout đã chọn có macro-F1 cao hơn ComplementNB cố định trên các review mới sau mốc khóa mô hình hay không?** Đơn vị dự đoán là một review; hai nhãn là `negative=0`, `positive=1`. Input mô hình chỉ là review thô.

Đây là đánh giá prospective theo thời gian, trên các review đủ điều kiện phân loại nhị phân. Không đánh giá neutral/multiclass, sentiment theo khía cạnh, calibration hay tải production. Việc lấy tối đa 5 review/sản phẩm làm thay đổi cơ cấu so với toàn bộ lưu lượng Shopee; kết quả chỉ đại diện cho khung lấy mẫu và tiêu chí dưới đây, không tự động đại diện toàn nền tảng.

## 2. Kiểm kê dữ liệu hiện có và vai trò từng tập

Đã kiểm tra toàn bộ dòng JSONL trong `data/`: cả ba file chỉ có `id`, `review`, `rating`, `label`. Không có `product_id`, `review_created_at` hoặc `author_id`. `id` chưa được xác minh là mã sản phẩm; **không dùng nó làm nhóm sản phẩm, không suy thời gian từ ID hay ngày sửa file**.

| Tập / nguồn | Quy mô đã biết | Vai trò trong protocol V2 |
|---|---:|---|
| Train cũ đã lưu | 7.660 | Dữ liệu đã fit TF-IDF và classifier; giữ nguyên cho truy vết/replay, không fit lại trong lần đánh giá xác nhận này |
| Validation cũ đã lưu | 958 | Đã chọn cấu hình/epoch MLP; thuộc dữ liệu phát triển, không còn là holdout xác nhận |
| Test cũ đã lưu | 958 | Đã đọc và phân tích lỗi; chỉ còn giá trị lịch sử/phát triển, không dùng cho kết luận xác nhận V2 |
| Toàn bộ originals cũ | 9.599, gồm 23 dòng đã bị lọc trước split | Toàn bộ được coi là đã tiếp xúc trong phát triển; dùng làm danh sách chống trùng, không tái chế dòng đã lọc thành test |
| Sample cũ | 20 | Ví dụ schema/kiểm thử; cũng nằm trong danh sách chống trùng |
| Augmentation cũ | 1.348 | Dữ liệu dẫn xuất đã biết; cấm đưa vào test mới; đưa vào danh sách chống trùng |
| Augmentation, split, fixture, ví dụ thủ công và error review đã lưu khác | Có thể lặp từ các nguồn trên | Đều là dữ liệu phát triển đã tiếp xúc; đưa vào danh sách chống trùng với nguồn gốc tương ứng |
| Candidate pool mới | **Chưa có** | Export mới, chỉ người quản lý test và người gán nhãn được tiếp cận; chưa được coi là test chính thức |
| Test mới `T_new` | **Chưa có; chốt 1.000 review đủ điều kiện** | Tập xác nhận duy nhất của V2; không tham gia fit, chọn epoch, threshold, calibration hoặc phân tích lỗi trước lần đánh giá chính thức |

Các số 7.660/958/958 là split có sẵn trong `models/evaluation_workflow/full_validation_20260929_01/`. Fingerprint originals cũ:

```text
data/shopee_reviews_dataset.jsonl
9e9c1c19b68c9d758ad5b006ae65777f97fce0f6b8ed8bbd75fe368321c11270

data/sample_data.jsonl
358b05410397b3ced7a661d82936cc7c3c53a8e58d127abd55247f3a74dc3ece

data/aug_unaccented_reviews.jsonl
2125c656d29e05fb1b5c0f40ee042d2b94a1a5bd0fdda3916c8e880606db215d
```

**Không tạo thêm train/validation từ candidate pool mới trong V2.** Hai mô hình đã được chọn trước; toàn bộ cohort mới được dành cho test hoặc loại bởi tiêu chí dữ liệu cố định. Nếu cần chỉnh mô hình trước khi đánh giá, phải lập protocol mới và dành một cohort khác cho phát triển; không lấy dần mẫu từ `T_new` để tune.

## 3. Nguồn mới và cách chia theo thời gian/sản phẩm

### 3.1. Nguồn và cửa sổ thu thập đã chốt

- Nguồn mục tiêu duy nhất: **một đợt export mới các review Shopee tiếng Việt**, có provenance và metadata nguồn xác minh được. Không tải lại bản khác của dataset Kaggle cũ, không dùng câu sinh tổng hợp hoặc biến thể bỏ dấu làm test.
- Dùng **thời gian review được đăng tại nguồn**, không phải thời gian download: từ `2026-11-08T00:00:00+07:00` (bao gồm) đến `2026-12-06T00:00:00+07:00` (không bao gồm), tức 28 ngày.
- Mốc bắt đầu nằm sau ngày khóa protocol/artifact. Tất cả file phát triển đã có trước ngày 08/11 phải được fingerprint trước khi nhận export mới.
- Trước khi thu thập, người quản lý test ghi tên export/nguồn, phạm vi sản phẩm, bộ lọc, phương thức lấy mẫu, người quản lý và người gán nhãn vào manifest thu thập. Đây là điền thông tin vận hành của nguồn Shopee đã chốt, không được chọn nguồn sau khi xem kết quả mô hình.
- Chưa có export hoặc kết nối nguồn review mới trong workspace. Mốc thời gian là **kế hoạch thu thập**, không phải khẳng định đã có review thuộc khoảng này.

Metadata tối thiểu cho mỗi candidate:

| Trường | Ý nghĩa và kiểm tra |
|---|---|
| `source_dataset` | Định danh export/đợt thu thập; khác dữ liệu phát triển |
| `review_id` | ID review tại nguồn, dạng chuỗi, xác minh đúng loại ID |
| `product_key` | Khóa sản phẩm có namespace đủ duy nhất, ví dụ platform/shop/item; không dùng tên sản phẩm làm khóa |
| `review_created_at` | Thời gian đăng từ nguồn, ISO-8601 có timezone, nằm trong cửa sổ đã chốt |
| `collected_at` | Thời điểm export/thu thập, tách riêng khỏi thời gian đăng |
| `review` | Review thô nguyên văn, không thêm/bớt dấu, không tiền xử lý trước khi lưu |
| `source_reference` | Tham chiếu nguồn/export phục vụ kiểm chứng provenance |
| `source_id` | Định danh nguồn có namespace hoặc hash văn bản theo quy tắc chống trùng; không tái sử dụng nhầm ID augmentation |

`rating`, danh mục và `author_key` có thể giữ làm metadata kiểm toán, không làm feature. Nếu có `author_key`, dùng mã không chứa thông tin định danh trực tiếp; không cố suy tác giả khi trường này vắng mặt. Nhãn cuối và biên bản annotation được lưu tách khỏi input cho mô hình.

**Nếu không lấy được `product_key` hoặc timestamp tin cậy, V2 chưa đủ điều kiện thực hiện.** Không tự chuyển sang random split. Muốn dùng nguồn khác hoặc đánh giá text-only phải lập phiên bản mới trước khi phát triển nhìn dữ liệu đó.

### 3.2. Chia và lấy mẫu cố định

1. Khóa toàn bộ dữ liệu cũ thành `D_dev_legacy`; không chia lại. Candidate hợp lệ theo nguồn và thời gian thuộc vùng test prospective, không có phần nào được chuyển sang train/validation.
2. Chuẩn hóa khóa nguồn và loại các bản ghi trùng nguồn/văn bản theo mục 4 trước khi gán nhãn cuối. Audit chỉ nhìn dữ liệu và provenance, tuyệt đối không dùng dự đoán hoặc loss của mô hình.
3. Với mỗi candidate, tính khóa thứ tự `SHA256("v2-20261107|" + source_dataset + "|" + review_id)`. Sắp tăng theo khóa; nếu hòa, dùng khóa nguồn dạng chuỗi theo thứ tự từ điển.
4. Giữ tối đa **5 review mỗi `product_key`**, theo thứ tự trên, trước annotation. Không xem rating/nhãn để chọn 5 mẫu, không ưu tiên review dễ hoặc khó. Nếu một mẫu bị loại sau annotation, không đưa review thứ 6 của sản phẩm đó vào để bù.
5. Gán nhãn theo mục 5. Trong các review có nhãn nhị phân hợp lệ, lấy **1.000 review đầu** theo khóa thứ tự đã chốt. Không stratify theo nhãn, không cân bằng 50/50 và không chọn lại vì kết quả dự đoán.
6. Điều kiện đủ: đúng 1.000 review, ít nhất **200 sản phẩm** (được bảo đảm bởi cap 5 nếu đủ 1.000 review), ít nhất **100 review mỗi nhãn**. Ngưỡng support do người quản lý test kiểm tra; không gửi văn bản/nhãn chi tiết cho nhóm phát triển.
7. Nếu không đủ quy mô/support, giữ trạng thái `NOT_READY`, **không mở test, không chạy mô hình**. Không đổi seed, lấy lại mẫu hay tự kéo dài cửa sổ để đạt kết quả mong muốn. Muốn thu thêm phải ban hành amendment trước đợt thu thập mới; cohort này vẫn không được dùng tune.

Các candidate không được chọn không trở thành dữ liệu phát triển trong chu kỳ V2; tiếp tục được giữ kín. Kích thước 1.000 là ngân sách đánh giá cố định, **không phải bảo đảm statistical power** cho một chênh lệch F1 nhỏ.

### 3.3. Giới hạn của chia theo sản phẩm

Trong cohort mới, khóa sản phẩm kiểm soát cap lấy mẫu và là đơn vị resampling khi tính khoảng tin cậy. Không có sản phẩm mới nào được chia sang train/validation mới vì V2 không có hai tập đó.

**Không thể khẳng định `product_key(T_new)` không giao với sản phẩm của `D_dev_legacy`**, do dữ liệu cũ không có product metadata. Không suy khóa sản phẩm từ review ID, từ vựng hoặc tên trong câu. Chỉ được báo cáo đánh giá review mới theo thời gian; không gọi đây là benchmark “unseen products”. Muốn đánh giá sản phẩm hoàn toàn mới cần nguồn liên kết metadata sản phẩm cũ đã xác minh và protocol riêng, được chốt trước khi xem test đó.

Tương tự, khi thiếu author metadata ở dữ liệu cũ, không tuyên bố độc lập theo tác giả. Group theo sản phẩm và loại near-duplicate không chứng minh độc lập ngữ nghĩa tuyệt đối.

## 4. Chống trùng và chuẩn bị dữ liệu độc lập

Người quản lý test tạo registry từ toàn bộ dữ liệu phát triển đã liệt kê ở mục 2, gồm raw text, source identity, các biến thể augmentation đã lưu và fixture/report có văn bản. Ghi danh sách file và SHA-256 trước khi tiếp nhận cohort mới. Tất cả split chỉ là dẫn xuất, không phải nguồn độc lập.

Quy tắc áp dụng trước dự đoán, với audit đếm riêng từng loại:

- Loại trùng `source_dataset/review_id` và source identity đã xác minh. Các bản export lặp cùng review không trở thành mẫu mới.
- So raw text bằng **NFC → casefold → chuẩn hóa khoảng trắng**, giống quy tắc source identity hiện có. Giữ dấu tiếng Việt; không dùng bỏ dấu để đồng nhất nguồn gốc.
- So thêm văn bản được xử lý bằng preprocessing snapshot đã khóa của hai mô hình. Candidate có clean text không rỗng trùng dữ liệu phát triển bị loại. Không coi các chuỗi sạch rỗng là cùng một nguồn.
- Trong candidate pool, các bản trùng không rỗng được gộp, giữ bản có khóa thứ tự nhỏ nhất; nếu có mâu thuẫn nguồn hoặc nhãn cuối thì đưa cả nhóm vào audit, không tự chọn nhãn thuận tiện.
- Screen near-duplicate bằng tập character 5-grams của raw text đã chuẩn hóa: nếu cả hai chuỗi dài ít nhất 30 ký tự và Jaccard ≥ **0,90**, gắn cờ để kiểm tra mù mô hình. Dùng truy hồi gần đúng để tăng tốc chỉ được phép nếu không bỏ qua bước xác minh exact similarity; ghi rõ phương pháp và giới hạn kiểm tra.
- Reviewer kiểm tra các cặp gắn cờ, không biết model prediction; loại bản sao/sửa nhẹ hoặc biến thể cùng review. Cặp không xác minh được nguồn gốc được loại theo hướng bảo thủ, với lý do ghi sẵn. Với chuỗi dưới 30 ký tự, vẫn áp dụng exact/source/clean-text checks; không tuyên bố đã loại hết near-duplicate ngắn.
- Không loại review chỉ vì MLP/NB không có feature, predicted confidence thấp, hai mô hình bất đồng hoặc câu khó. Empty-after-preprocessing và OOV hợp lệ vẫn nằm trong mẫu số đánh giá.

Không fit vocabulary/IDF, stopword list, normalizer hoặc threshold bằng candidate/test. Chỉ dùng preprocessing đóng băng để kiểm tra trùng và suy luận. Không dùng clean text mới để thay đổi cách lọc cohort dựa vào kết quả mô hình.

## 5. Nhãn và eligibility

- Hai người gán nhãn độc lập đọc review thô, **không xem prediction, probability, feature attribution, nhãn cũ hoặc rating**. Chỉ dùng ngữ cảnh sản phẩm tối thiểu đã quy định đồng nhất nếu cần hiểu câu.
- `positive`: có đánh giá hài lòng, khuyến nghị hoặc thiện cảm tổng thể rõ. `negative`: có bất mãn, không khuyến nghị hoặc vấn đề sử dụng tổng thể rõ.
- Với nhiều khía cạnh, chỉ gán nhị phân khi ý định tổng thể đủ rõ. Nếu công dụng/chất lượng sản phẩm được đánh giá rõ, ưu tiên đánh giá đó hơn lời khen giao hàng mang tính phụ; không áp dụng máy móc khi văn bản không cho phép kết luận.
- Neutral, không đủ ngữ cảnh, mixed không phân xử được hoặc ngoài phạm vi tiếng Việt được đánh dấu exclusion có lý do; không ép thành positive/negative. Báo cáo tỷ lệ loại này để thể hiện phạm vi nhị phân có điều kiện.
- Bất đồng được người thứ ba phân xử, vẫn mù mô hình. Lưu hai nhãn ban đầu, nhãn adjudicated và lý do; báo cáo agreement thô và Cohen's kappa trước phân xử.
- Không suy nhãn từ rating; không tự sửa nhãn sau khi xem lỗi mô hình. Bộ hướng dẫn và người phụ trách phải được ghi vào annotation manifest trước annotation chính thức. Training annotator dùng ví dụ cũ, không dùng `T_new` làm ví dụ phát triển mô hình.

## 6. Quy tắc chọn và khóa mô hình

**Không có vòng chọn mô hình mới trên `T_new`.** Ứng viên đã chọn trước test là MLP dropout 0,3, hidden 64, LR 0,001, seed 42, **epoch 9**, theo macro-F1 validation cũ. ComplementNB alpha 0,5 là đối chứng cố định, dùng đúng TF-IDF và preprocessing tương ứng. Không chọn lại epoch, seed, ngưỡng, checkpoint resume epoch 10 hoặc ensemble sau khi thấy test.

Hai artifact đánh giá xác nhận:

| Thành phần | Đường dẫn local đã kiểm tra ngày 07/11/2026 |
|---|---|
| MLP raw-review bundle | `models/mlp_packaged/mlp_raw_20261106_02/pipeline/` |
| NB raw-review pipeline | `models/mlp_packaged/mlp_raw_20261106_02/baseline/sentiment_pipeline.joblib` |

SHA-256 phải khớp trước mở test:

```text
pipeline/model.pt
ae902eaabe4f2a79c34ba7c3078a5918d04f45348d6f749216853a16d9848fe8
pipeline/tfidf.joblib
2a280ca97fd17eb21f77461dc2c95df3da20c1fee12787e6593d4efe61878ea1
pipeline/preprocessing.json
0b91df5919314e18350ee0f0aca92e8db6555477d66569c02ae53cd722ca6119
pipeline/manifest.json
5a6fd600b2a154bf43b7cb1d78d0a36a7bba8b3739e4daa30f518063ca19e4cf
baseline/sentiment_pipeline.joblib
b7b6b35e9952ac618e39d66bdc752c9e92406e21d6b97943f9c918a63f0d85a7

src/mlp_pipeline.py
edb5919dde7ac13dc4c667ba82c49cf5ec984f6542dbc180fb1a46a97217989a
src/preprocessor.py
ee5ec64b9963e44275b9e1f4fd44531c155b8eda9aaa1d3749a5a9fcc20600b3
src/inference.py
dc6ab0dec9febbb82c89d842045bbd4878525dbbc3278eaa5036b95a9935041e
```

Runtime tham chiếu của bundle: Python 3.13.5, torch 2.15.0+cpu, scikit-learn 1.9.0, NumPy 2.5.1, underthesea 9.5.0. Trước mở test, lưu toàn bộ environment snapshot, source snapshot, code revision/dirty state và SHA-256 của mọi file scorer/dependency snapshot thực dùng. Nếu source/runtime cần đổi, kiểm tra tương thích và fixed-sample logits bằng dữ liệu phát triển trước; thay đổi nội dung suy luận bắt buộc có amendment trước khi chạm test.

Suy luận nhận review thô; preprocessing chạy đúng một lần mỗi model/call. MLP dùng `eval()` và `no_grad()`, CPU 4 threads, batch 128. Dùng nhãn `predict`/argmax đã có, không chọn threshold trên test. Thứ tự nhãn là `[negative, positive]`; trường hợp logits bằng nhau lấy negative theo thứ tự này. Không calibration hoặc cập nhật trọng số/IDF. Không refit trên train+validation trước đánh giá V2.

## 7. Metric chính, phụ và so sánh định trước

**Metric chính: macro-F1** trên toàn bộ 1.000 review của `T_new`:

`macro-F1 = (F1_negative + F1_positive) / 2`.

Luôn dùng tập nhãn cố định `negative, positive`, quy ước `zero_division=0`. Point estimate tính theo review, không lấy trung bình F1 của từng sản phẩm. Không loại empty/OOV hoặc dự đoán sai khỏi mẫu số.

Metric phụ bắt buộc: accuracy; precision/recall/F1/support từng lớp; confusion matrix (hàng thật, cột dự đoán, thứ tự negative/positive); tổng lỗi; empty-after-preprocessing/OOV và processing-failure counts; số sản phẩm, phân bố nhãn, số review/sản phẩm; số lượng bị loại ở từng bước và annotation agreement. Không thay metric chính bằng metric phụ khi kết quả bất lợi.

So sánh duy nhất định trước: **Δ = macro-F1(MLP) − macro-F1(NB)** trên cùng các review. Báo cáo điểm và khoảng tin cậy 95% bằng **paired cluster bootstrap theo `product_key`**:

1. Có G sản phẩm; mỗi lần lấy G khóa sản phẩm có hoàn lại, giữ mọi review của mỗi lần xuất hiện sản phẩm.
2. Dùng cùng resample cho cả hai mô hình; tính lại hai macro-F1 và Δ từ dự đoán đã khóa.
3. **10.000** lần, NumPy `default_rng(20261107)`; percentile 2,5/97,5, phương pháp percentile tuyến tính của NumPy. Lưu seed, phiên bản và kết quả resampling. Không bootstrap riêng hai mô hình rồi trừ hai CI.
4. Giữ đủ 10.000 lần, kể cả lần resample thiếu một lớp; vẫn dùng hai nhãn cố định và zero_division=0, không rút lại mẫu bootstrap cho đẹp số.

Nếu CI của Δ nằm hoàn toàn trên 0, báo cáo bằng chứng ủng hộ MLP trên cohort này, với các giả định sampling/grouping đã nêu. Nếu CI chứa 0, không kết luận MLP vượt trội, cũng không kết luận hai mô hình tương đương. Không thêm cặp so sánh hoặc lựa chọn subgroup sau khi thấy kết quả. CI không khắc phục được bias nguồn, annotation hoặc tác giả/sản phẩm không độc lập.

Thời gian và dung lượng artifact hiện có là thông tin vận hành phụ. Không lặp inference trên test để tune batch size hoặc chọn runtime. Nếu cần benchmark, dùng validation cũ hoặc fixture cố định riêng, ghi rõ không phải điểm xác nhận V2.

## 8. Thời điểm mở test và kiểm soát truy cập

Phân biệt **đọc test để annotation/kiểm toán** với **mở test cho người phát triển mô hình**. Annotation là bắt buộc nhưng không được biến thành tín hiệu tune.

- **Nhóm phát triển** chỉ thấy dữ liệu cũ, protocol và audit tổng hợp đủ/không đủ; không đọc raw text, nhãn từng mẫu, class-conditioned errors hoặc prediction của `T_new` trước lần chạy chính thức.
- **Người quản lý test** giữ export, nhãn, registry chống trùng và split manifest, không tham gia chỉnh mô hình từ nội dung test. Hai annotator/adjudicator được đọc review theo quy trình mù mô hình.
- Trước thu thập phải gán tên người thực hiện các vai trò trong access manifest. Nếu chưa có cơ chế tách quyền đọc, test chưa đủ điều kiện niêm phong. Việc đặt file trong `data/` hoặc Git ignore **không phải** kiểm soát truy cập; không để test plaintext trong workspace phát triển trước ngày mở.

**Chỉ mở test cho lần đánh giá chính thức sau `2026-12-06T00:00:00+07:00`**, và chỉ khi tất cả gate sau đạt; đây là thời điểm sớm nhất, không phải lịch mở tự động:

| Gate trước mở test | Bằng chứng bắt buộc |
|---|---|
| G1 — Protocol khóa | SHA-256 tài liệu này, thời điểm khóa và version trong access/evaluation manifest |
| G2 — Nguồn độc lập | Export provenance, timestamps, product keys, raw export fingerprint, phạm vi sampling đã ghi trước thu thập |
| G3 — Không rò rỉ | Registry dữ liệu phát triển đã khóa, audit exact/source/clean/near-duplicate, log loại trừ; không có dữ liệu mới đi vào fit/tune |
| G4 — Test và nhãn khóa | 1.000 mẫu; ≥200 sản phẩm; ≥100 mẫu mỗi nhãn; annotation/adjudication hoàn tất; SHA-256 ordered review IDs, raw-text file và label file |
| G5 — Hai mô hình khóa | Hash ở mục 6, môi trường/source/scorer snapshot, fixed-sample reload checks; không dùng resumed epoch 10 |
| G6 — Scorer sẵn sàng | Kiểm thử scorer/metric/label mapping trên fixture phát triển hoặc dữ liệu giả; không debug bằng `T_new` |
| G7 — Truy cập sẵn sàng | Người quản lý và người chạy được chỉ định; log chưa có mở sớm; thời điểm chạy chính thức được ghi trước khi chạy |

Tất cả tên file, fingerprint, người phụ trách và thời điểm gate phải được điền bằng dữ liệu thực, không dùng placeholder làm bằng chứng đã hoàn tất. **Hiện G2–G7 chưa hoàn tất; không được ghi protocol này là một lần đánh giá đã chạy.**

## 9. Một lần chạy xác nhận và xử lý sự cố

1. Người quản lý khóa ordered input và nhãn. Người chạy nhận đúng input thô cho scorer đã khóa, chưa nhận nhãn; mỗi model chạy một lần trên cùng thứ tự, giữ đủ mọi sample ID.
2. Ghi file dự đoán, hash và log số hàng cho cả hai model trước khi mở label file cho scorer metrics. Không hiển thị/lướt prediction để chọn lại artifact.
3. Sau khi prediction của cả hai đã bất biến, mở nhãn, tính metrics/paired bootstrap từ các file đã lưu và xuất báo cáo cho cả hai mô hình.
4. Nếu có lỗi đọc file/hạ tầng **trước khi mở nhãn**, ghi incident; chỉ được chạy lại đúng input, code và artifact. Không bỏ hàng lỗi hoặc đổi model. Nếu không thể hoàn thành cho đủ 1.000 mẫu, run là chưa hợp lệ; công khai số lỗi và trạng thái, không báo F1 trên phần thành công như kết quả chính.
5. Nếu sửa logic suy luận, dữ liệu hoặc nhãn sau khi đã xem kết quả, `T_new` đã bị tiêu thụ cho phát triển. Có thể báo một phân tích sửa lỗi được gắn nhãn riêng, nhưng không gọi lần sửa đó là xác nhận độc lập; muốn xác nhận cần cohort/protocol mới.
6. Chỉ phân tích lỗi định tính sau khi báo cáo chính và mọi artifact đã khóa. Từ lúc đó `T_new` trở thành test đã mở; không dùng nó để chọn cấu hình tiếp theo rồi tiếp tục gọi là holdout mới.

Kết quả test đánh giá ứng viên đã chọn trước, không phải vòng thi chọn epoch/model mới. Nếu test không ủng hộ MLP, báo trung thực kết quả đó; không chọn một checkpoint khác để thay kết luận. Protocol này không tự phê duyệt triển khai production.

## 10. Artifact bắt buộc khi thực hiện

Đây là danh mục chuẩn bị, **chưa phải các file test đã được tạo**:

- Manifest thu thập: nguồn, phạm vi, cửa sổ, người phụ trách, raw export SHA-256 và schema.
- Registry contamination và audit chống trùng/loại trừ; snapshot annotation guide, hai nhãn ban đầu/adjudication và agreement.
- Manifest test đã niêm phong: ordered review IDs, product keys, số mẫu/nhóm/support, fingerprint input và label, người giữ test, access log.
- Manifest đánh giá: protocol SHA-256, thời điểm mở, hashes của hai artifact/scorer/source/environment, khai báo đã/không tiếp xúc test trước đó.
- Hai file predictions cùng ID, metrics và confusion matrices, paired bootstrap/CI, incident log nếu có, báo cáo chính có giới hạn nguồn và độc lập metadata.

**Việc cần làm tiếp theo để có test thật:** thu được export Shopee trong cửa sổ đã chốt, với `product_key` và `review_created_at` đáng tin cậy; chỉ định người giữ/gán nhãn test tách khỏi phát triển; thực hiện kiểm tra và niêm phong theo các gate. Hiện không có dữ liệu mới để hoàn thành các bước này trong workspace. Các artifact MLP/NB đã có sẵn để khóa; không cần train thêm để thực hiện protocol V2.
