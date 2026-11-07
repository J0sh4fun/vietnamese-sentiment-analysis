# Vietnamese E-Commerce Sentiment Analysis

Reproducible Vietnamese review classification with **ComplementNB** and a **PyTorch TF-IDF MLP**. Both accept raw reviews and predict `negative` or `positive`. The release uses one versioned config, verified data/split hashes and an exact dependency lock.

## Release profile

**`baseline-v1.0.0`: Windows x64, CPython 3.13.5, CPU.** This is the tested profile. The wheel lock is intentionally platform-specific; Linux/macOS, other Python versions and CUDA builds are not verified by this release. Install Python 3.13.5 before following these PowerShell commands from the repository root.

The current release does not require pre-existing files under `models/`, a Kaggle account, an API key, Transformer weights, or the old experiment bundle. Internet access is needed to install wheels and fetch the public dataset on first preparation. Verified local data/archive can be used offline afterward. The release tag and ZIPs are currently local; the tag has not been pushed to GitHub. Use the delivered source ZIP on another machine, or clone the local tagged repository.

## 1. Create a clean environment

```powershell
Expand-Archive -LiteralPath ./baseline-v1.0.0-source.zip -DestinationPath ./vietnamese-sentiment-analysis
cd vietnamese-sentiment-analysis
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install --require-hashes -r requirements.txt
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -B -m src.release_baseline doctor
```

`doctor` checks Python/platform, all 70 locked runtime distributions and the CPU Torch build. `.python-version` records 3.13.5. `requirements.txt` includes the full transitive wheel-hash lock, including the official PyTorch CPU wheel. It does not install optional Transformer packages. Installing into an existing environment can retain unrelated packages; use a new virtualenv for the release checks.

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

The command returns both models' labels and uncalibrated probabilities; MLP also returns logits and explicit empty/OOV warnings. Do not preprocess input yourself. Empty and zero-feature strings are retained; their predictions use learned biases and are not evidence of sentiment.

Python API:

```python
from src.mlp_pipeline import load_pipeline
model = load_pipeline("models/releases/baseline-v1.0.0/artifacts/mlp")
results = model.infer(["Sản phẩm không tốt", "!!!", ""], include_probabilities=True)
```

For the existing Streamlit demo, set `SENTIMENT_MODEL_PATH` to `models/releases/baseline-v1.0.0/artifacts/complement_nb.joblib` and run `.venv/Scripts/python.exe -m streamlit run app.py`. The current demo uses the NB loader; use the CLI/API for MLP.

## Tests and optional Transformer input route

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

Baseline tests need no old model/data bundle. The six optional Transformer route tests skip unless tokenizer assets and the optional stack are present. To enable them after baseline training:

```powershell
.venv/Scripts/python.exe -m pip install --require-hashes -r requirements-transformer.txt
.venv/Scripts/python.exe -B -m src.transformer_inputs --prepare
.venv/Scripts/python.exe -B -m src.check_model_inputs --baseline models/releases/baseline-v1.0.0 --output-dir models/preprocessing_routes/release_01
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

The optional lock includes the same baseline versions plus the pinned tokenizer stack. XLM-R receives raw reviews directly, never TF-IDF `clean_text`; preprocessing/length requirements are documented in [the two-route report](docs/model_input_routes_20261107.md). No Transformer classifier is trained by this release workflow.

## Versioning, artifacts and future evaluation

- [Release verification](docs/releases/baseline-v1.0.0.md): exact fresh-environment checks and artifact fingerprints.
- [Frozen baseline snapshot](docs/baseline_v1_20261107.md): previous immutable source/model/data bundle and replay instructions.
- [Evaluation protocol V2](docs/evaluation_protocol_v2_20261107.md): prospective review collection and gated independent evaluation. Its test is **NOT_RUN**; the release evaluator intentionally accepts only train/validation, not a test flag.
- [MLP experiment](docs/mlp_20261106.md), [raw-text packaging](docs/mlp_packaging_20261106.md), [previous README](docs/history/README_pre_release_20261107.md): historical results and legacy commands, separated from this fresh-install workflow.

Large raw data, trained models, virtualenvs and temporary downloads are Git-ignored. A fresh clone recreates them with the preparation/training steps above. Preserve the complete trained run if moving it; source/data/lock hashes are recorded so copying just weights does not silently substitute a different pipeline. No deployment or new-test reliability claim is made from these checks.

Lock maintenance: edit the exact roots/constraints under `requirements/`, download a new wheel set into a fresh directory, and generate a hash lock with `tools/lock_wheels.py`. Revalidate in a new virtualenv before accepting a changed lock; do not replace the verified platform-specific lock with unconstrained upgrades.

## Author

josh4fun · [GitHub](https://github.com/J0sh4fun)
