# Vietnamese E-Commerce Sentiment Analysis

Reproducible Vietnamese review classification with **ComplementNB** and a **PyTorch TF-IDF MLP**. Both accept raw reviews and predict `negative` or `positive`. The release uses one versioned config, verified data/split hashes and an exact dependency lock.

## Release profile

**`baseline-v1.0.0`: Windows x64, CPython 3.13.5, CPU.** This is the tested profile. The wheel lock is intentionally platform-specific; Linux/macOS, other Python versions and CUDA builds are not verified by this release. Install Python 3.13.5 before following these PowerShell commands from the repository root.

The current checkout includes CI and benchmark additions after the frozen `baseline-v1.0.0` snapshot. Follow the commands below from this checkout for the current tests/CI. The frozen source ZIP reproduces the older release and its recorded 93-test suite; it does not contain the later CI/benchmark additions. The tag and ZIPs remain local until published.

No pre-existing files under `models/`, Kaggle account, API key, Transformer weights or old experiment bundle are required. Internet access is needed to install wheels and fetch the public dataset on first preparation. Verified local data/archive can be used offline afterward.

## 1. Create a clean environment

```powershell
# Open PowerShell at the root of the current source checkout.
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install --require-hashes -r requirements.txt
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -B -m src.release_baseline doctor
```

`doctor` checks Python/platform, all 70 locked runtime distributions and the CPU Torch build. `.python-version` records 3.13.5. `requirements.txt` includes the full transitive wheel-hash lock, including the official PyTorch CPU wheel. It does not install optional Transformer packages. Installing into an existing environment can retain unrelated packages; use a new virtualenv for the release checks.

**PyTorch 2.15.0+cpu and matplotlib are required baseline dependencies**, already installed by this lock. Do not add an unpinned `pip install torch` step. Only the Transformers/tokenizer stack is optional; it uses the separate full lock below.

## 2. Download, verify and prepare data

```powershell
.venv/Scripts/python.exe -B -m src.prepare_data --output-dir data/prepared/baseline-v1.0.0
```

The preparation command downloads **Kaggle dataset version 1**, checks the SHA-256 and row count of both JSONL members before writing them, then recreates the historical source-group split. Existing correct files are reused; mismatches fail without overwriting. It creates:

```text
data/prepared/baseline-v1.0.0/
  raw/shopee_reviews_dataset.jsonl
  raw/aug_unaccented_reviews.jsonl
  splits/train_split.csv          # 7,660
  splits/validation_split.csv     # 958
  splits/test_split.csv           # 958; already-consumed historical test
  splits/split_manifest.json
```

Originals: **9,599**; reference filtering retains **9,576**. Source file hash is `9e9c1c19b68c9d758ad5b006ae65777f97fce0f6b8ed8bbd75fe368321c11270`. All three prepared CSVs must match the full hashes in [the release config](configs/baseline-v1.0.0.json), including row order, BOM and CRLF. The reference cohort uses stopword filtering; the model representation then disables stopword filtering on those same retained rows. No augmentation is used in this release.

For an offline public-v1 archive:

```powershell
.venv/Scripts/python.exe -B -m src.prepare_data --archive D:/datasets/shopee-v1.zip --output-dir data/prepared/baseline-v1.0.0
# Once raw files exist and match, preparation needs no network:
.venv/Scripts/python.exe -B -m src.prepare_data --offline --output-dir data/prepared/baseline-v1.0.0
```

These are **old development data**, not a new independent test. Labels come from the dataset's supplied semantic labels, not a rating threshold; rating is not a feature. The publisher declares Apache 2.0, but human-vs-automated annotation, neutral/mixed rules and third-party content permissions remain partly unverified. See the [data card](docs/data_card_20261107.md).

## 3. Train the two fixed baselines

```powershell
.venv/Scripts/python.exe -B -m src.release_baseline train --output-dir models/releases/baseline-v1.0.0
```

All settings come from [configs/baseline-v1.0.0.json](configs/baseline-v1.0.0.json). Both models use the same train-fitted float32 TF-IDF and exact saved preprocessing. ComplementNB uses alpha=0.5. MLP uses 64 hidden units, ReLU, dropout=0.3, Adam LR=0.001, batch=128, at most 15 epochs. Selection is highest old-validation macro-F1; exact ties choose the earliest epoch. The expected selected epoch is 9. There is no refit on validation or test evaluation.

A shared seed helper sets Python/NumPy/PyTorch to **42**, with seeded DataLoader and deterministic Torch algorithms; CPU threads=4. Hash seed, platform and installed packages are recorded in metadata. This does not promise bitwise equivalence on other systems.

The run contains raw-text NB/MLP artifacts, best/last MLP optimizer+RNG checkpoints, training TF-IDF, copied splits, effective config, environment/source snapshots, learning history, validation metrics/predictions and a SHA-256 release manifest. Choose a fresh output directory when repeating; existing runs are never overwritten.

## 4. Evaluate after loading the saved artifacts

```powershell
.venv/Scripts/python.exe -B -m src.release_baseline evaluate --run-dir models/releases/baseline-v1.0.0 --split validation --output-dir models/release_checks/validation_01
```

This new process verifies hashes, reads **raw `review`**, applies preprocessing once per model, evaluates both baselines on the same rows and checks labels/logits against the training-run receipt. It writes CSV/JSON metrics, per-class reports, confusion matrices and predictions. Outputs must be outside the immutable run and use a new directory.

Expected results on the already-consumed validation split:

| Model | Accuracy | Macro-F1 | F1 negative | F1 positive |
|---|---:|---:|---:|---:|
| ComplementNB | 0.926931 | 0.923094 | 0.940273 | 0.905914 |
| MLP, epoch 9 | 0.942589 | 0.939045 | 0.953743 | 0.924347 |

These are reproducibility checks, not independent generalization estimates. Runtime/artifact container hashes can differ across runs; the release records actual hashes and checks discrete predictions and logits within numerical tolerance.

## 5. Predict raw reviews

```powershell
.venv/Scripts/python.exe -B -m src.release_baseline predict --run-dir models/releases/baseline-v1.0.0 --text "Sản phẩm không tốt" "Giao hàng nhanh, chất lượng tốt!"
```

The command returns both models' labels and uncalibrated probabilities; MLP also returns logits and explicit empty/OOV warnings. Do not preprocess input yourself. Empty and zero-feature strings are retained; the resulting classifier defaults (MLP biases or NB zero-feature scores/tie behavior) are not evidence of sentiment.

Python APIs use different artifact types:

```python
from src.inference import load_model
from src.mlp_pipeline import load_pipeline

nb = load_model("models/releases/baseline-v1.0.0/artifacts/complement_nb.joblib")
mlp = load_pipeline("models/releases/baseline-v1.0.0/artifacts/mlp")
reviews = ["Sản phẩm không tốt", "!!!", ""]  # raw text, no prior clean_text
nb_results = nb.infer(reviews, include_probabilities=True)
mlp_results = mlp.infer(reviews, include_probabilities=True, include_logits=True)
```

`load_model` expects the versioned raw-text NB joblib; `load_pipeline` expects the **whole MLP directory**, including `model.pt`, `tfidf.joblib`, `preprocessing.json` and `manifest.json`. A `.pt` file alone cannot predict raw reviews. Use artifacts produced by this project and preserve their manifests. Training checkpoints with optimizer/RNG state are separate from inference bundles; resume with `src.resume_mlp`, not the inference loader.

The Streamlit demo uses the NB loader; MLP is available through the CLI/API:

```powershell
$env:SENTIMENT_MODEL_PATH = "models/releases/baseline-v1.0.0/artifacts/complement_nb.joblib"
.venv/Scripts/python.exe -m streamlit run app.py
```

## Tests and optional Transformer input route

```powershell
.venv/Scripts/python.exe -B tools/run_tests.py --profile baseline --report work/ci/local-baseline-tests.json
```

The current source discovers **103 tests**. With a clean baseline-only install, **96 execute and 7 skip**: six tokenizer/model-route tests and one local Transformer-adapter test. No old model/data bundle is required. The runner fails on empty/all-skipped discovery, assertions/import errors or a skip outside this exact allowlist; counts/reasons are written to JSON. Use a new report path for each run.

To execute all 103 tests, complete baseline training first and install/prepare the optional assets:

```powershell
.venv/Scripts/python.exe -m pip install --require-hashes -r requirements-transformer.txt
.venv/Scripts/python.exe -B -m src.transformer_inputs --prepare
.venv/Scripts/python.exe -B -m src.check_model_inputs --baseline models/releases/baseline-v1.0.0 --output-dir models/preprocessing_routes/release_01
.venv/Scripts/python.exe -B tools/run_tests.py --profile full --report work/ci/local-full-tests.json
```

The optional lock includes the same baseline versions plus the pinned tokenizer stack. XLM-R receives raw reviews directly, never TF-IDF `clean_text`; preprocessing/length requirements are documented in [the two-route report](docs/model_input_routes_20261107.md). No Transformer classifier is trained by this release workflow.

The `full` profile requires **zero skipped tests**. It includes actual PyTorch forward/backward/checkpoint checks and tiny local XLM-R architecture/serialization checks. These tiny test weights do not establish sentiment quality. Dated experiment reports retain the test counts observed at their publication time; they are not the current suite count.

## Continuous integration

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs on pushes, pull requests and manual dispatch, using **`windows-2025`, x64 CPython 3.13.5, CPU** and a fresh `.venv` in each job. Action revisions and dependency wheel hashes are pinned. The workflow replaces the previous echo-only Streamlit deployment status; it does not deploy the app.

| Job | What actually runs | Expected current suite |
|---|---|---|
| `baseline` | Baseline hash-lock install, `pip check`, environment doctor, real unittest discovery | 103 discovered; 96 execute, 7 explicitly allowed skips |
| `integration` | Optional hash-lock install, data download/hash/split verification, fixed NB+MLP training, new-process validation reload/logit check, raw prediction, pinned tokenizer preparation, full tests | 103 execute; every skip fails the job |

Each external command is a separate PowerShell step so a nonzero exit fails the job. Tests produce JSON receipts, uploaded with integration config/hash/metric receipts even after a failure when available. Installation/download failures remain job failures; pip download caching does not substitute for running tests. No data/model cache, secrets, production deployment or new-test access is needed. The integration job depends on the public Kaggle/tokenizer endpoints being reachable.

Run the same commands locally as shown above; see [CI verification and limitations](docs/ci_testing.md). A workflow file/local run does not establish a successful hosted Actions run; that status is available after these changes are pushed and the jobs finish.

## Shared quality and cost benchmark

The current source adds a common NB/MLP/local fine-tuned XLM-R evaluator, beyond the frozen v1.0.0 source ZIP. After preparing/training the baseline:

```powershell
.venv/Scripts/python.exe -B -m src.benchmark_models --run-dir models/releases/baseline-v1.0.0 --output-dir models/benchmarks/quality_cost_01 --batch-sizes 1 32 --warmup 5 --repeats 3 --benchmark-samples 256 --threads 4 --seed 42
```

It scores all old-validation rows and separately benchmarks raw-to-label versus classifier-only latency on a seeded subset. Reports include per-class metrics, confusion matrices, batch p50/p95, throughput, CPU/GPU/thread metadata, native peak RSS, GPU allocator peaks and artifact size. The Transformer row stays `NOT_RUN` until a compatible sentiment checkpoint is supplied. Standardized training time needs a bound `TrainingMeasurement` receipt; legacy train timings remain explicitly separate. See [measurement rules and results](docs/benchmark_quality_cost.md) and the [comparison table template](docs/templates/model_comparison.csv). The immutable v1.0.0 release and prospective test gate are unchanged.

## Versioning, artifacts and future evaluation

- [Release verification](docs/releases/baseline-v1.0.0.md): exact fresh-environment checks and artifact fingerprints.
- [Frozen baseline snapshot](docs/baseline_v1_20261107.md): previous immutable source/model/data bundle and replay instructions.
- [Evaluation protocol V2](docs/evaluation_protocol_v2_20261107.md): prospective review collection and gated independent evaluation. Its test is **NOT_RUN**; the release evaluator intentionally accepts only train/validation, not a test flag.
- [MLP experiment](docs/mlp_20261106.md), [raw-text packaging](docs/mlp_packaging_20261106.md), [previous README](docs/history/README_pre_release_20261107.md): historical results and legacy commands, separated from this fresh-install workflow.

Large raw data, trained models, virtualenvs and temporary downloads are Git-ignored. A fresh clone recreates them with the preparation/training steps above. Preserve the complete trained run if moving it; source/data/lock hashes are recorded so copying just weights does not silently substitute a different pipeline. No deployment or new-test reliability claim is made from these checks.

Lock maintenance: edit the exact roots/constraints under `requirements/`, download a new wheel set into a fresh directory, and generate a hash lock with `tools/lock_wheels.py`. Revalidate in a new virtualenv before accepting a changed lock; do not replace the verified platform-specific lock with unconstrained upgrades.

## Author

josh4fun · [GitHub](https://github.com/J0sh4fun)
