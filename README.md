# Vietnamese E-Commerce Sentiment Analysis

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Vietnamese-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](đường_link_đến_notebook_kaggle_của_bạn)

An end-to-end Machine Learning pipeline designed to classify Vietnamese e-commerce reviews (e.g., Shopee) into Positive or Negative sentiments. 

This project heavily emphasizes a **data-centric approach**, featuring a robust custom text preprocessor for the Vietnamese language, and an automated Model Selection pipeline to handle imbalanced datasets.

## Key Features

*   **Tailored Vietnamese Preprocessing:** HTML cleanup, structured URL/email/phone masking, Unicode NFC normalization, and word segmentation using `underthesea`. Custom tone repositioning is disabled by default.
*   **Smart Stopword Filtering:** Filters noise while preserving crucial negation words (e.g., *không*, *chẳng*, *chưa*) to prevent sentiment flip errors.
*   **Automated Model Selection:** Automatically trains and evaluates multiple algorithms (`Logistic Regression`, `MultinomialNB`, `ComplementNB`), selecting the best performer based on the `F1-macro` score to combat class imbalance.
*   **MLOps-Ready Structure:** CLI-driven execution using `argparse`, isolated source code, and automated artifact logging (saving `.joblib` models, metrics in JSON, and train/val/test splits).
*   **Error Analysis Support:** Generates detailed `predictions.csv` files alongside Confusion Matrices to facilitate deep dive error analysis.

## Project Structure

```text
vietnamese-sentiment-analysis/
│
├── data/                   # (Ignored in Git) Contains .jsonl datasets
│   └── sample_data.jsonl   # Sample format for reference
├── models/                 # (Ignored in Git) Auto-generated artifacts & weights
│
├── src/                    # Core pipeline source code
│   ├── __init__.py         
│   ├── preprocessor.py     # Vietnamese text cleaning & tokenization
│   ├── train.py            # Training & Model Selection pipeline
│   └── evaluate.py         # Evaluation & Error Analysis scripts
│
├── config.py               # Stopwords & global configurations
├── requirements.txt        # Project dependencies
├── .gitignore              
└── README.md
```

## Getting Started
### 🚀 Quick Start & Demo (Run on Kaggle)
Skip the local setup and explore the code directly in your browser! We provide a complete Kaggle Notebook demonstrating the Exploratory Data Analysis (EDA), Text Preprocessing, and Model Training steps:
👉 **[Open Kaggle Notebook Demo](https://www.kaggle.com/code/josh4fun/vietnamese-text-processing)**

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/J0sh4fun/vietnamese-sentiment-analysis.git
cd vietnamese-sentiment-analysis
pip install -r requirements.txt
```
### 2. Dataset Preparation
Place your raw dataset in the data/ directory. The pipeline expects a .jsonl format with at least two columns:

- review: The raw text of the comment.

- label: The sentiment class (e.g., positive, negative).

Note: Due to file size and privacy, the full training dataset is not included in this repository. Please refer to data/sample_data.jsonl for the expected schema.

Credit: https://www.kaggle.com/datasets/dduongdev/shopee-vietnamese-product-reviews-sentiment

### 3. Exploratory Data Analysis (EDA)
To understand the dataset's distribution, class balance, and vocabulary characteristics, an in-depth Exploratory Data Analysis was conducted. This includes generating sentiment-specific WordClouds and analyzing text lengths.
Detailed visual analysis and data mixing strategies can be found in our interactive notebook:
🔗 **[View EDA Notebook on Kaggle](https://www.kaggle.com/code/josh4fun/vietnamese-text-processing)**

## Train the model

Run the training pipeline. The script validates original data, creates fixed stratified train/validation/test splits, and selects a model using validation performance. The test split is evaluated only once after selection.

```bash
python src/train.py --train-data data/shopee_reviews_dataset.jsonl
```

Optional Arguments:

--algorithms: Choose specific models (e.g., --algorithms logreg complement_nb).

--nb-alpha: Set smoothing parameter for Naive Bayes.

--regularization: Set inverse regularization strength (C) for Logistic Regression.

Training will:

1. Validate and clean the original JSONL data, then assign stable source IDs.
2. Split original source groups into training, validation and test sets.
3. Optionally generate or import verified augmentation for training sources only.
4. Check source-ID, normalized-text and preprocessed-text overlap across splits.
5. Fit TF-IDF/classifiers on training only and select using validation metrics.
6. Evaluate the selected model on test once; save artifacts in a new `models/<run_name>/` directory.

Output: The best model (sentiment_pipeline.joblib) and metrics will be saved in a timestamped folder inside models/ (e.g., models/20260507_120000).

Training artifacts include:

- `sentiment_pipeline.joblib`
- `train_split.csv`
- `validation_split.csv`
- `test_split.csv`
- `train_metadata.json` (filtering/split/augmentation counts, label distributions and overlap checks)
- `train_augmentation.jsonl` (accepted variants with provenance; empty if none)
- `environment.json` (observed Python, platform, installed dependency versions and numerical-library/thread information)
- `requirements-resolved.txt` (installed-version snapshot for that run's environment)

## Experiment provenance and safe reruns

Metadata schema version 2 records requested options and the **effective configuration** after overrides (`--algorithm` overrides `--algorithms`; `--disable-aug` disables generated and imported augmentation). It also stores the selected estimator's actual parameters, random seed, exact preprocessing settings, per-split label distributions, imported/generated/accepted/excluded augmentation counts, and the detailed filtering audit. Split distributions describe the final rows, including training augmentation; original-only distributions remain in the audit.

Each loaded dataset gets a SHA-256 digest and byte count from the exact bytes consumed by the JSONL reader. These hashes include row order, line endings and any BOM. Disabled augmentation files are not read or fingerprinted. Exported model, splits, augmentation and environment files have their own hashes in metadata. Git HEAD and dirty status are captured when available; otherwise revision/dirty are explicitly null. Source and test file fingerprints distinguish uncommitted working-tree code from the recorded commit. Preserve the matching source changes as well as the commit to reproduce a dirty-tree run.

The seed is applied to Python `random`, NumPy, pandas sampling, stratified splitting and Logistic Regression's supported `random_state`. Naive Bayes and the fixed CRF tokenizer have no stochastic fitting step in this pipeline. Repeated seeds do not guarantee bit-for-bit results across Python/package versions, operating systems, BLAS/OpenMP builds, CPU architectures or thread settings. The environment snapshot records those numerical libraries and relevant environment variables when observable. `PYTHONHASHSEED` must be set **before** Python starts if you want to control hash randomization; setting it inside the program would not do so. The code does not claim to force universal deterministic execution.

Run directories are never reused or overwritten, and there is no overwrite flag. Default names include microseconds. An existing `--run-name` fails before data preparation; directory creation also refuses reuse if another process creates it during training. A failed save may leave a partial new directory: retain it for diagnosis and choose a new name. Evaluation likewise refuses existing metrics or prediction files; use `--output-dir` to write a repeat evaluation elsewhere without changing the run.

```powershell
# New run with the same seed/configuration; use another run-name for each rerun.
.venv/Scripts/python.exe src/train.py --random-state 42 --generate-aug --run-name repro_seed42_a

# Write evaluation outside the run directory.
.venv/Scripts/python.exe src/evaluate.py --run-dir models/repro_seed42_a --split test --output-dir models/repro_seed42_a_evaluation

# Focused provenance, seed, relocation and overwrite-protection tests.
.venv/Scripts/python.exe -m unittest discover -s tests -p test_experiments.py -v
```

The split ratios must be finite, strictly between zero and one, and sum to less than one. Seeds must be unsigned 32-bit integers. Sample/feature/iteration counts and minimum document frequency must be positive integers; regularization and NB smoothing must be finite and positive. The CLI also checks input files and portable run names. Datasets too small for stratification still fail with an explanation of the required adjustment.

Artifact filenames in training metadata are relative to the run directory. Copy the **whole run directory** and resolve these filenames against its new location. Input references marked `base_directory` are relative to the project root; external inputs carry a basename plus `relocation_required` and must be remapped using their fingerprints. Evaluation references are relative to the evaluation JSON's directory where possible; external references are explicitly marked. The saved inference object has no dependency on its original filesystem location. Metadata schema version 2 is separate from the inference artifact's version 2.

## Verified dependency environment

`requirements.txt` contains broad dependency ranges, not a tested lockfile. [docs/verified-environment.json](docs/verified-environment.json) records the actual inspected Windows environment used for these regression tests: CPython 3.13.5, NumPy 2.5.1, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0, underthesea 9.5.0, underthesea_core 3.3.2, joblib 1.5.3, Streamlit 1.59.2 and threadpoolctl 3.6.0. These values were read from the running interpreter/installed distribution metadata, not inferred from requirements. Other installed distributions in the snapshot were not all independently exercised. Although the original requirements declare `torch`, it was **not installed** in this test environment; no PyTorch behavior was verified, and the current TF-IDF/CRF path did not require it.

For a historical run, use its own `environment.json` and `requirements-resolved.txt`, matching the recorded Python version and platform in an isolated environment. Installing that snapshot is a starting point, not a cross-platform lock or a guarantee of future wheel availability. No clean-environment reinstall or non-Windows run was performed. This environment emits a joblib/NumPy deprecation warning during loading; the artifact round-trip tests pass. Keep package versions and data/code fingerprints fixed when comparing reruns.

## Leakage-safe data and augmentation

The old pipeline split off test, appended augmentation, then split training/validation. A variant could therefore enter validation while its original was in training, or enter training while its original was in test. Augmentation files also bypassed validation. Existing models and scores from that procedure should be rerun before comparison.

Every input source, including generated variants, uses the same validation policy: required text/label columns; nonempty string text; exact allowed string labels (default `negative positive`); accent-preserving NFC, case and whitespace normalization; removal of duplicate text/label rows; and removal of **all** rows with conflicting labels for identical normalized text. Schema errors fail the run. Missing/invalid values are filtered and counted. Stateless preprocessing also filters duplicates and conflicting labels that become identical after cleaning, except empty cleaned originals are retained (see below). These checks happen before splitting originals. No TF-IDF vocabulary or IDF is fitted during this stage.

`source_id` is retained when supplied as a nonempty trimmed string; otherwise it is a SHA-256 hash of normalized original text. The old `id` column is not assumed to identify a source. Repeated source IDs stay in one split; conflicting labels within a source ID fail the run. Supplied source IDs that disagree for identical raw or cleaned text are rejected before deduplication, so dropping an alias cannot break a source group. Split ratios apply to source groups (normally one row each), with validation/test counts rounded up from the full original group count; row ratios can differ when a source has multiple original rows. `--max-samples` caps filtered originals before splitting. Small datasets that cannot support stratification fail with an explanatory error.

Accent removal is **only** an augmentation operation. For example, `má` and `ma` remain separate originals and source IDs. Original examples that merely share accent-stripped text are never merged. Candidate augmentation that collides with any retained original's normalized raw or preprocessed text is excluded; validation and test remain original-only. Checks fail closed if any source ID or normalized text still crosses splits.

Legacy augmentation is no longer loaded automatically. `--aug-data` is preserved, but files without `source_id`, `source_text` and `augmentation_method` are excluded with a warning. Do not infer provenance by matching accent-stripped reviews. To regenerate safely, use:

```bash
python src/train.py --train-data data/shopee_reviews_dataset.jsonl --disable-aug
python src/train.py --train-data data/shopee_reviews_dataset.jsonl --generate-aug
```

`--generate-aug` creates unaccented variants **after** splitting, only from training originals. Accepted variants are saved to the new run's `train_augmentation.jsonl`; the legacy file is untouched. Each variant carries its original's `source_id`, exact `source_text`, label, and `augmentation_method: unaccented_v1`. Imported variants must match a current training source, its label and the declared transformation. Unknown sources, held-out sources, mismatched labels/text and unsupported methods are excluded and counted. Adding a new augmentation method requires an explicit provenance/transform verifier.

To reuse a generated file, keep the original dataset and split settings fixed (seed, ratios, cap and labels). Imports are rechecked against the current split regardless:

```bash
python src/train.py --train-data data/shopee_reviews_dataset.jsonl --aug-data models/<prior_run>/train_augmentation.jsonl
```

An empty augmentation export contains no examples and should be omitted from `--aug-data`. `--aug-data` with no paths is valid. `--disable-aug` disables both imported and generated variants. Use `--allowed-labels` to declare other exact string labels; numeric labels must first be converted explicitly in the input. All existing model-selection options remain available. The metadata records counts and label distributions at each filtering stage, original splits and final splits, as well as exclusion reasons and overlap-check outcomes.

Runs default to unique microsecond timestamps. An existing `--run-name` directory is rejected rather than overwritten. Choose a fresh name for every experiment. Use validation for further tuning; repeated test-driven tuning invalidates the final test estimate.

Run the focused regression tests (synthetic fixtures, temporary artifacts):

```bash
python -m unittest discover -s tests -v
```

## Preprocessing behavior and reproducibility

The original custom tone heuristic changed correctly spelled `khuấy` to `khúây`: its three-vowel rule selected `u` from `uâ y`. A regression test reproduced this before the implementation changed. There is no evidence establishing this heuristic's general correctness. `normalize_unicode()` now performs only NFC composition, and `normalize_word_tone()` preserves NFC spelling by default. Neither tries to correct alternative placements such as `hòa`/`hoà` or `thủy`/`thuỷ`. The segmenter's separate implicit tone/token normalization is also explicitly disabled with `use_token_normalize=False`.

The order is NFC, lowercase and HTML cleanup, identify structured entities, clean punctuation in the remaining text, segment words, then filter whole-token stopwords. NFC comes first because punctuation filtering can otherwise discard decomposed combining marks. Segmentation sees the final spelling, and stopword filtering sees the resulting compound tokens. URL/email/contiguous Vietnamese phone spans bypass segmentation as `<url>`, `<email>`, `<phone>`; uppercase URLs are recognized and literal `TOKURL` substrings are not rewritten. Phone masking covers contiguous `0...`/`+84...` forms, not every spaced or punctuated phone notation.

The checked stopword list has 1,942 entries: 1,571 contain underscores and none contain spaces. Stopword entries and segmented tokens share one NFC/lowercase/underscore comparison form; a custom space-separated entry such as `bây giờ` therefore matches `bây_giờ`. Filtering matches whole tokens, not arbitrary phrases across token boundaries. Tokens containing the syllables `không`, `chẳng`, `chưa`, `chớ`, or `đừng` are protected, including compounds such as `không_phải` and `chưa_từng`. The source stopword list is unchanged.

`transform()` returns exactly one string per input, including `""` for reviews reduced to no lexical content. Original empty results remain in training, validation and test; TF-IDF represents them as zero-feature rows and metrics include them. `preprocessing_by_split` in metadata and training output reports the number of assigned originals, empty count/rate, and empty-exclusion count/rate (zero under the `keep` policy). Earlier schema/conflict/duplicate filtering has its own global audit counts and does not contribute to these empty-exclusion rates. Empty cleaned strings alone do not identify a shared source; raw-text and source-ID overlap checks still apply. Empty generated variants add no information and are excluded with separate augmentation counts/rates. An all-empty training corpus fails with an explicit error. CSV evaluation preserves empty strings instead of turning them into the token `nan`, and reports the evaluated denominator and empty-input rate.

Every new run saves a JSON-safe `preprocessing` configuration, including implementation version, normalization/masking/filtering policies, exact effective stopwords and protected negations, and tokenizer/Unicode versions. Restore it with:

```python
import json
from pathlib import Path
from src.preprocessor import VietnameseTextProcessor

metadata = json.loads(Path("models/<run_name>/train_metadata.json").read_text(encoding="utf-8"))
processor = VietnameseTextProcessor.from_config(metadata["preprocessing"])
```

Restoration rejects unsupported settings or runtime mismatches. This standalone processor API is for inspection and data preparation; do not use it before calling the inference artifact. The artifact restores its own saved configuration. Existing artifacts are not rewritten. For an explicit diagnostic comparison only, `--legacy-tone-repositioning` enables the still-defective custom heuristic before segmentation and emits a warning. This flag does not recreate every behavior of the old pipeline. These correctness changes do not establish an F1 improvement.

## Raw-text inference and artifact migration

New `sentiment_pipeline.joblib` files contain a **version-2 `SentimentModel` wrapper**, bundling the fitted TF-IDF/classifier and its exact preprocessing configuration. A wrapper preserves the audited training-data preparation while giving evaluation, CLI prediction and Streamlit one raw-review path. Preprocessing runs exactly once per inference request. Training still fits TF-IDF only on the prepared training features; the selected model's final test evaluation uses the raw-review wrapper.

Run from the repository root with the recorded dependency versions:

```bash
python src/train.py --disable-aug
python src/predict.py --model-path models/<new_run>/sentiment_pipeline.joblib --text "Máy khuấy tốt" "Không hài lòng" --probabilities
```

The training command prints its new run directory. To generate training-only augmentation, replace `--disable-aug` with `--generate-aug`. Training refuses to overwrite an existing run.

```python
from src.inference import load_model

model = load_model("models/<new_run>/sentiment_pipeline.joblib")
labels = model.predict(["Máy khuấy tốt", "Không hài lòng"])
results = model.infer(["Máy khuấy tốt", "!!!"], include_probabilities=True)
```

Pass **raw text**, never `clean_text` or a call to `processor.transform()`. `predict()` returns labels. `predict_proba()` returns a matrix whose columns follow `model.classes_`. `infer()` returns labels, `empty_after_preprocessing` flags and optional label-keyed probabilities, using `classes_` for the mapping. Use `infer()` when requesting both labels and probabilities so preprocessing runs once. These are uncalibrated model probabilities, not a guarantee that a prediction is correct. The predicted label comes from the classifier's `predict()` rather than a second UI-specific threshold rule.

One raw string or an ordered iterable of raw strings is accepted. Missing values (`None`, NaN), numbers, bytes, nested inputs, mappings and unordered sets raise `TypeError`; inputs are never silently stringified. An empty batch returns an empty result. Empty/whitespace or fully filtered reviews remain zero-feature inputs, are flagged by `infer()`, and are included in evaluation. Streamlit displays the same prediction with an empty-input warning.

Legacy unversioned TF-IDF-only artifacts are **rejected**, even if a preprocessing metadata file exists. Retrain into a fresh run and point clients to that version-2 artifact. There is no automatic conversion that guesses which preprocessing was used. Unsupported artifact/configuration/runtime versions fail explicitly. Version and raw input contract are recorded both in the artifact and `train_metadata.json`; loading does not depend on a separate metadata file. Direct `joblib.load()` of a new version-2 artifact also restores the recorded processor, but `load_model()` is the supported entry point because it additionally rejects legacy bare pipelines.

## Evaluate the model

Evaluate the trained model on the test split to generate the Classification Report, Confusion Matrix, and Prediction outputs. Evaluation reads the original raw-review column recorded in the artifact (`review` by default) and calls the same `infer()` method as Streamlit. The saved `clean_text` column is retained only for audit purposes. `--text-column` can name another raw column; explicitly selecting a known preprocessed column is rejected. The label-column default also comes from the artifact.

```bash
python src/evaluate.py --run-dir models/<run_name> --split test
```

Evaluation outputs:

- `<split>_predictions.csv`
- `<split>_metrics.json`

The metrics JSON includes `metrics.labels`, in classifier class order, alongside `metrics.confusion_matrix`. Matrix rows are true labels and columns are predicted labels, both in that recorded order. Absent model classes remain in the matrix, and unknown or empty ground-truth labels are rejected. Macro evaluation scores and the classification report use the same explicit label list.

### Useful options

```bash
python src/train.py --disable-aug --max-samples 50000 --run-name baseline
python src/evaluate.py --run-dir models/baseline --split validation
```

Use Naive Bayes models:

```bash
python src/train.py --algorithm multinomial_nb --nb-alpha 0.5 --run-name nb_multinomial
python src/train.py --algorithm complement_nb --nb-alpha 0.5 --run-name nb_complement
```

Compare multiple models and auto-select the best:

```bash
python src/train.py --algorithms logreg multinomial_nb complement_nb --selection-metric f1_macro --run-name model_selection
```

## Interactive Web App (Streamlit)

You can test the trained model through an interactive web application. It displays sentiment predictions and uncalibrated model probabilities, labelled as estimates rather than prediction-correctness guarantees.

### 1. Start the App
Select a newly trained version-2 artifact using `SENTIMENT_MODEL_PATH`; editing `app.py` is unnecessary.

Run the following command from the root directory:

```powershell
$env:SENTIMENT_MODEL_PATH = "models/<new_run>/sentiment_pipeline.joblib"
python -m streamlit run app.py
```

## Author 
josh4fun

GitHub: https://github.com/J0sh4fun

LinkedIn: [www.linkedin.com/in/tiến-đạt-nguyễn-0b7241373](https://www.linkedin.com/in/ti%E1%BA%BFn-%C4%91%E1%BA%A1t-nguy%E1%BB%85n-0b7241373/)

Email: tiendat9320@gmail.com

If you found this project helpful, feel free to give it a ⭐!

