# CI kiểm thử — Windows / Python 3.13.5 / CPU

Workflow [tests.yml](../.github/workflows/tests.yml) thay thế hoàn toàn workflow cũ chỉ in trạng thái Streamlit. Chạy trên push, pull request và manual dispatch, không deploy ứng dụng. Runner chọn `windows-2025`, cài chính xác CPython 3.13.5 x64 và tạo `.venv` không kế thừa site-packages. Dependency dùng hai lock Windows/CPU đã kiểm chứng; không có job Linux dùng nhầm wheel lock Windows.

## Hai job thực

| Job | Dependency | Chuẩn bị | Kiểm tra |
|---|---|---|---|
| `baseline` | `requirements.txt`, 70 distribution | Không cần dữ liệu/model cũ hoặc tokenizer | `pip check`, doctor, discover toàn suite, chỉ cho phép 7 skip tùy chọn đã định danh |
| `integration` | `requirements-transformer.txt`, 81 distribution | Tải/kiểm hash dữ liệu public v1; tái tạo split; train NB+MLP; tải tokenizer có revision/hash cố định | `pip check`, doctor, evaluate sau reload ở tiến trình mới, predict raw review, discover toàn suite và cấm mọi skip |

Script [`tools/run_tests.py`](../tools/run_tests.py) dùng `unittest.TextTestRunner` thật. Assert lỗi, import lỗi, không discover test nào, tất cả test bị skip hoặc skip ngoài policy đều trả exit code khác 0. Profile `full` không cho skip. Ba test regression của runner cố tình dựng suite rỗng, assertion sai và suite bị skip để kiểm tra các trường hợp báo lỗi, không dựa vào thông báo in ra.

Suite hiện có **103 test**. Job baseline sạch chạy 96 và skip 7: sáu test `test_model_inputs.ModelFamilyRouteTests` cần tokenizer/baseline, một test Transformer adapter cần optional dependency. Job integration có đủ artifact nên chạy 103, không skip. Runner không hardcode tổng số test để tránh chặn việc thêm test hợp lệ; nó kiểm tra suite có thực thi và policy skip. JSON receipt lưu số đã discover/thực thi, ID/reason skip, failure/error, thời gian, Python/platform và trạng thái virtualenv.

Mỗi lệnh native được đặt trong một step PowerShell riêng để exit code được truyền thành trạng thái job. Không dùng `continue-on-error`, không chuyển lỗi tải dependency/dữ liệu thành thành công. Các action chính thức checkout/setup-python/upload-artifact được khóa bằng commit SHA tương ứng release đã xác minh; pip cache chỉ cache download, không cache kết quả test hay model/data.

## Kiểm tra luồng model

Job integration tạo mới `models/releases/baseline-v1.0.0` từ chính source PR/commit đang kiểm tra. Không phụ thuộc model bị Git-ignore trên máy tác giả. Bước train kiểm tra số mẫu, hash split, fit TF-IDF trên train và metric/epoch cố định của cấu hình baseline. Evaluate kiểm tra manifest, nhãn và logits tải lại trên validation. Predict nhận review thô, dùng NB joblib và cả thư mục MLP inference bundle. Snapshot source của run mới bao gồm `tools/` để giữ runner được test sử dụng.

Tokenizer được tải trước khi chạy profile `full`, không tải pretrained weights. Những test XLM-R dùng kiến trúc rất nhỏ tạo tại chỗ để kiểm tra forward, serialization và input contract; không có kết quả sentiment Transformer mới. `test_split.csv` lịch sử chỉ được tái tạo/kiểm hash trong preparation; không chạy scorer test. Test độc lập V2 không được mở.

Artifact Actions giữ 14 ngày: JSON receipt tests; split manifest; environment/model metadata; release manifest; bảng validation; receipt reload và kết quả predict. `always()` cố upload các receipt đã có khi job lỗi; nếu chưa đến bước tạo file thì log job vẫn là bằng chứng lỗi. Không upload toàn bộ raw dataset/weights và không cần secret cho dữ liệu public hoặc tokenizer.

## Chạy lại cục bộ

Tại root source hiện tại, dùng môi trường mới đúng [README](../README.md):

```powershell
.venv/Scripts/python.exe -m pip install --require-hashes -r requirements.txt
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -B -m src.release_baseline doctor
.venv/Scripts/python.exe -B tools/run_tests.py --profile baseline --report work/ci/baseline-tests.json
```

Để chạy như integration, tiếp tục cài optional lock, prepare dữ liệu, train, evaluate/predict và chuẩn bị tokenizer theo README, sau đó:

```powershell
.venv/Scripts/python.exe -B tools/run_tests.py --profile full --report work/ci/full-tests.json
```

Output model/evaluate và receipt dùng đường dẫn chưa tồn tại; không xóa/ghi đè baseline đóng băng để chạy lại. Các số 61/65/72/87/93 trong tài liệu thí nghiệm cũ là lịch sử, không phải số test hiện tại. README gốc chỉ dẫn hai loader hiện tại: `src.inference.load_model` cho NB và `src.mlp_pipeline.load_pipeline` cho thư mục MLP.

## Trạng thái kiểm chứng

Đã chạy lại cục bộ từ bản source sạch `work/ci_validation_20261107`, ban đầu không có `data/` hoặc `models/`, bằng virtualenv mới không kế thừa site-packages. Baseline lock được cài trước; sau khi kiểm tra baseline, optional lock được cài thêm trong cùng virtualenv cho lượt integration. Job hosted được cấu hình tạo virtualenv riêng cho mỗi profile.

| Kiểm tra cục bộ | Kết quả |
|---|---|
| Baseline `pip check` + doctor; xác nhận Transformers chưa cài | PASS |
| Baseline tests | 103 discover, 96 thực thi, 7 skip trong allowlist; không failure/error |
| Optional lock + `pip check` + doctor | PASS |
| Tải public-v1 và tái tạo split đúng hash | PASS |
| Train mới NB+MLP, epoch/metric cố định | PASS; NB Macro-F1 0,923094, MLP 0,939045, epoch 9 |
| New-process validation reload và raw predict | PASS; nhãn/logits khớp tham chiếu |
| Tokenizer pin + full tests | PASS; 103 thực thi, 0 skip |

[Biên bản JSON](ci_verification_20261107.json) giữ receipts, lệnh/exit code, hash và giới hạn kiểm chứng. Source runtime/workflow trong bản source kiểm tra được đối chiếu với các file hiện tại. Không sửa run baseline đóng băng gốc.

Workflow đã qua **actionlint 1.7.12** (binary Windows chính thức, SHA-256 đối chiếu checksum release). Các SHA action được đối chiếu với tag trong repo chính thức bằng `git ls-remote`:

- checkout v6.0.2: `de0fac2e4500dabe0009e67214ff5f5447ce83dd`
- setup-python v6.2.0: `a309ff8b426b58ec0e2a45f0f869d46889d02405`
- upload-artifact v7.0.2: `cf430e030ddbb5b0abf93d22962f4752f3646cd9`

Chưa có hosted GitHub Actions run cho những thay đổi chưa push này. Kiểm chứng cục bộ dùng Windows 11/Python 3.13.5 CPU; `windows-2025` của GitHub là Windows Server 2025 nên kết quả hosted chỉ được xác nhận sau khi workflow thật chạy. Dù cùng API Windows/lock/kiến trúc, không coi test trên máy tác giả là bằng chứng đã chạy trên hosted runner.

Nguồn tham khảo: [runner Windows Server 2025](https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md), [setup-python advanced usage](https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md), [actionlint release](https://github.com/rhysd/actionlint/releases/tag/v1.7.12).
