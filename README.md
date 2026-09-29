# Vietnamese E-Commerce Sentiment Analysis

Classify Vietnamese e-commerce review text as **`negative`** or **`positive`** using TF-IDF and scikit-learn classifiers. This is an experimental project with a local Streamlit demo and a reproducible, file-based evaluation workflow. Operational deployment, monitoring and production reliability have not been evaluated.

The reported task is binary: there is no neutral, mixed-sentiment, aspect-level or abstention class. A review can praise delivery while criticising the product, yet the model must return one label. CLI support for other allowed labels does not establish performance beyond the two evaluated labels.

The latest verified full-data experiment, `full_validation_20260929_01` (29 September 2026), selected **ComplementNB with stopword filtering disabled** using validation macro-F1. Its final test macro-F1 was **0.928263**, with **0.932150 accuracy** on 958 reviews. These are single-split results, not cross-validation or evidence of general algorithm superiority. The test errors have since been inspected.

## Data, labels and provenance

The project's attributed upstream source is [Shopee Vietnamese Product Reviews Sentiment on Kaggle](https://www.kaggle.com/datasets/dduongdev/shopee-vietnamese-product-reviews-sentiment). The current local files are:

| File | Rows | Role |
|---|---:|---|
| `data/shopee_reviews_dataset.jsonl` | 9599 | Originals used in the verified experiment |
| `data/sample_data.jsonl` | 20 | Local schema example; not the full-data benchmark |
| `data/aug_unaccented_reviews.jsonl` | 1348 | Legacy augmentation, excluded from the verified experiment |

These JSONL files and the `models/` directory are git-ignored; a fresh clone may not contain them. Obtain the data separately. The repository does not include a verified download-to-local conversion or annotation script. Local originals contain `id`, `review`, `rating` and `label`, but the model uses review text and the supplied label, **not rating as a feature**. The exact label-generation methodology, rating-to-label thresholds, any upstream filtering and independent human annotation quality remain unverified. Do not infer that methodology merely from the presence of ratings. The upstream page resolves, but its detailed methodology was not available in the documentation check.

The verified input's SHA-256 is:

```text
9e9c1c19b68c9d758ad5b006ae65777f97fce0f6b8ed8bbd75fe368321c11270
```

This identifies the exact local file, including row order and line endings; downloading another upstream version need not reproduce it.

### Counts and label distributions

The verified run found no invalid text/label values and no duplicates or label conflicts under raw NFC/case/whitespace normalization. Reference preprocessing then removed 18 same-label duplicate rows and all 5 rows in conflicting-label groups. The retained cohort is shared by every ablation.

| Stage or split | Negative | Positive | Total |
|---|---:|---:|---:|
| Raw originals | 5965 | 3634 | 9599 |
| Filtered originals | 5944 | 3632 | 9576 |
| Training originals | 4754 | 2906 | 7660 |
| Validation | 595 | 363 | 958 |
| Test | 595 | 363 | 958 |
| Training with augmentation, augmentation ablation only | 9503 | 5807 | 15310 |

Source groups were stratified with seed 42, validation fraction 0.1 and test fraction 0.1. In this dataset, these yield an approximately 80/10/10 split. The augmentation ablation generated 7,660 training variants, accepted 7,650 and excluded 10 text collisions with retained originals. The selected final configuration uses **no augmentation**.

### Leakage prevention

The previous procedure appended augmentation before the train/validation split, allowing related originals and variants to cross boundaries. Historical scores from that procedure are not presented as results of the updated implementation.

The current [data pipeline](src/data_pipeline.py):

1. Applies the same required-column, string-value, allowed-label, duplicate and conflicting-label checks to originals and augmentation. Schema errors fail; invalid rows are filtered and counted. Cleaned-text identity is also checked; empty cleaned originals are retained.
2. Keeps a supplied, valid `source_id`, or derives it from SHA-256 of the original's NFC/case/whitespace-normalized text. The old `id` field is not assumed to identify an augmentation source. Conflicting source labels or ambiguous supplied identities fail.
3. Splits original source groups into train, validation and test **before generating augmentation**. Generated/imported variants must trace to current training originals through `source_id`, `source_text`, matching labels and a verified `unaccented_v1` transformation.
4. Checks cross-split `source_id`, normalized raw text and normalized cleaned text. Accent removal does not merge original identities. Augmentation colliding with retained original texts is excluded.
5. Fits TF-IDF vocabulary and IDF only on training features. Validation selects configuration; test predictions are computed after selection.

The evaluation workflow freezes the reference-filtered cohort and split for all ablations. It does not restore rows when a less aggressive preprocessor would retain them. Every alternate representation is checked for overlap before fitting; a collision aborts the run. These checks do not establish independence between semantically similar reviews, shared authors/products or collection periods.

## Preprocessing, representation and inference

[VietnameseTextProcessor](src/preprocessor.py) applies:

1. Unicode **NFC composition**, lowercase and HTML-tag cleanup.
2. Identification of URL, email and contiguous Vietnamese phone spans, represented as `<url>`, `<email>` and `<phone>`; these bypass segmentation.
3. Punctuation cleanup, preserving underscores, and whitespace normalization in the remaining text.
4. Optional `underthesea.word_tokenize` segmentation with `use_token_normalize=False`. With segmentation disabled, whitespace tokens are used.
5. Optional whole-token stopword filtering using a common NFC/lowercase/underscore key.

NFC precedes punctuation cleanup so decomposed combining marks are not discarded. Segmentation sees the final spelling, and stopword matching sees its compound tokens. The stopword source is [assets/vietnamese_stopwords.py](assets/vietnamese_stopwords.py), not `config.py`. Its 1,942 entries include 1,571 with underscores and none with spaces. Custom space-separated entries are canonicalized to underscores. Tokens containing `không`, `chẳng`, `chưa`, `chớ` or `đừng` are protected from filtering; this does not claim full negation-scope understanding or normalization of every shorthand form.

**Custom tone repositioning is disabled.** The legacy heuristic corrupts `khuấy` into `khúây` and has not established general correctness. NFC is separate from that heuristic. Alternative tone placements are not automatically rewritten, and this is not a general Vietnamese spell checker. The diagnostic `train.py --legacy-tone-repositioning` flag remains available with a warning but was excluded from the verified study. Phone masking does not cover every spaced or punctuated notation.

The reference configuration and ordinary `train.py` defaults enable segmentation and stopword filtering. The selected study artifact enables segmentation but **disables stopword filtering**. Use the recorded artifact or the full study workflow to reproduce that result; the ordinary training CLI has no stopword/segmentation toggle.

### TF-IDF and classifiers

The verified study uses word unigrams and bigrams, `min_df=2`, `max_features=150000`, `sublinear_tf=True`, L2 normalization and smoothed IDF. The vectorizer has `lowercase=False` because preprocessing lowercases text, and uses `(?u)\b\w+\b` as its token pattern, retaining underscore compounds.

| Classifier | Study settings |
|---|---|
| Majority baseline | `DummyClassifier(strategy="most_frequent")`; ignores features |
| Logistic Regression | `C=1.5`, `class_weight="balanced"`, `max_iter=2500`, binary `liblinear` solver, seed 42 |
| MultinomialNB | `alpha=0.5` |
| ComplementNB | `alpha=0.5` |

Selection uses **one validation holdout**, maximising macro-F1, then accuracy, then configuration/algorithm declaration order on exact ties. There is no cross-validation, repeated-seed comparison or exhaustive hyperparameter search. The selected estimator is not refitted on validation. The ordinary training CLI can select by accuracy instead; that was not the protocol used for the results below.

### Raw-text inference contract

The saved `sentiment_pipeline.joblib` is a version-2 [SentimentModel](src/inference.py) wrapper containing the fitted TF-IDF/classifier pipeline and its exact preprocessing configuration. `predict()` and `infer()` accept raw strings and apply preprocessing exactly once per call. Evaluation and [Streamlit](app.py) use this same interface. Do not preprocess text before passing it to the artifact.

`infer()` flags empty preprocessing outputs. Empty batches return no predictions; empty/fully filtered strings remain zero-feature inputs and receive predictions. Invalid non-string inputs, missing values, mappings, unordered sets and nested inputs are rejected. All retained split rows, including empty outputs, remain in evaluation denominators. Empty augmentation is excluded separately; processing exceptions abort rather than silently omit examples.

Optional probabilities are mapped by `model.classes_`; they are **uncalibrated model estimates**, not guaranteed prediction correctness. Legacy bare TF-IDF/classifier artifacts are rejected rather than silently receiving new preprocessing. Retrain into a fresh run. Preprocessing schema 3 records segmentation explicitly; schema-2 configurations retain their known always-on segmentation behavior. Unsupported configurations/runtime mismatches fail explicitly.

## Verified measurements

All measurements here come from `models/evaluation_workflow/full_validation_20260929_01/`, not older models or notebook results. The saved input hash, all 18 recorded code-file hashes and 14 generated-artifact hashes matched during this README review. The run recorded Git revision `2fe062a64e04ace19650501633322980f19643d8` **with working-tree changes**; the revision alone is insufficient to reproduce it.

See the [tracked experiment summary](docs/evaluation_20260929.md), [full report](models/evaluation_workflow/full_validation_20260929_01/report.md), [metadata](models/evaluation_workflow/full_validation_20260929_01/metadata.json), [validation results](models/evaluation_workflow/full_validation_20260929_01/validation_results.json) and [final test metrics](models/evaluation_workflow/full_validation_20260929_01/test_metrics.json). Links into `models/` require the local artifact bundle; it is not tracked in Git.

### Validation baseline and controlled ablations

Reference: NFC, segmentation on, stopword filtering on, no augmentation, tone repositioning off. Each ablation changes one setting relative to that reference. All comparisons use the same 958 validation originals.

| Configuration | Classifier | Macro-F1 | Accuracy |
|---|---|---:|---:|
| Reference | Majority | 0.383129 | 0.621086 |
| Reference | Logistic Regression | 0.897570 | 0.902923 |
| Reference | MultinomialNB | 0.895072 | 0.901879 |
| Reference | ComplementNB | 0.889818 | 0.894572 |
| Segmentation off | Majority | 0.383129 | 0.621086 |
| Segmentation off | Logistic Regression | 0.913248 | 0.917537 |
| Segmentation off | MultinomialNB | 0.894713 | 0.900835 |
| Segmentation off | ComplementNB | 0.898727 | 0.902923 |
| Stopword filtering off | Majority | 0.383129 | 0.621086 |
| Stopword filtering off | Logistic Regression | 0.918537 | 0.922756 |
| Stopword filtering off | MultinomialNB | 0.916423 | 0.921712 |
| Stopword filtering off | **ComplementNB — selected** | **0.923094** | **0.926931** |
| Training accent-removal augmentation | Majority | 0.383129 | 0.621086 |
| Training accent-removal augmentation | Logistic Regression | 0.900924 | 0.906054 |
| Training accent-removal augmentation | MultinomialNB | 0.891537 | 0.898747 |
| Training accent-removal augmentation | ComplementNB | 0.894528 | 0.899791 |

The majority baseline predicts `negative`: validation negative-class F1 is 0.766259 and positive precision/recall/F1 are zero. Baselines and non-selected candidates were **not evaluated on test** in this workflow. Per-class validation metrics/support for all 16 candidates are in the full report and JSON.

On this split, disabling stopword filtering increased macro-F1 for all three learned classifiers. Disabling segmentation produced mixed changes across classifiers and also changed which multiword stopwords matched. Accent-removal augmentation had mixed results and was not combined with the selected stopword-off setting. These observations do not establish universal benefits, statistical significance or general algorithm superiority. Tone repositioning was not an evaluated ablation.

### Final test, selected configuration only

The selection decision and model were saved before final test inference. **Macro-F1: 0.928263; accuracy: 0.932150.**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.952218 | 0.937815 | 0.944962 | 595 |
| positive | 0.900538 | 0.922865 | 0.911565 | 363 |

Confusion matrix: rows are true labels, columns are predictions, in saved order `negative`, `positive`.

| True / predicted | negative | positive |
|---|---:|---:|
| negative | 558 | 37 |
| positive | 28 | 335 |

There were 65 errors among 958 inputs. Processing failures, empty preprocessing outputs and per-split exclusions were each **0/958 (0%)** on test. Empty/exclusion rates were also zero for train and validation under every ablation. The earlier 23-row quality-filter exclusion is separate from these per-split processing rates.

### Timing

The selected TF-IDF/classifier fit took **0.6937 seconds**, on already-prepared training text. Preparing the stopword-off representation across all splits took **14.9842 seconds**, excluding shared initial cohort preparation. These are not an end-to-end cold training time.

Raw validation inference had a median **1.5663 ms/review**, amortised over full batches of 958. Measurement used `time.perf_counter`, one untimed warm-up batch and three measured batches. It includes preprocessing and label prediction, excludes disk/model loading, metrics and probability calculation, and uses a warm tokenizer/model. Final test inference ran once, taking **1.5250 seconds**. The full report/JSON records every candidate's fit time and individual timing repetitions. Hardware/load effects apply; this is not a single-request latency or production capacity benchmark.

## Error analysis and remaining evidence gaps

A completed assistant-authored qualitative review examined **40 actual test errors**, 20 in each direction, sampled from the 37 negative→positive and 28 positive→negative errors. [The analysis](models/evaluation_workflow/full_validation_20260929_01/error_analysis.md) records each review's interpretation and literal excerpts; [the source CSV](models/evaluation_workflow/full_validation_20260929_01/error_review.csv) supplies full text and source IDs. [The completion receipt](models/evaluation_workflow/full_validation_20260929_01/review_completion.json) verifies 89 quoted excerpts. Generated CSV annotation columns remain blank because the completed review was saved separately.

| Observed category | Annotated examples out of 40 |
|---|---:|
| Multiple aspects, contrast or qualified sentiment | 21 |
| Negation, shorthand negation or scope | 16 |
| Limited experience, implicit sentiment or missing context | 9 |
| Fairly clear sentiment missed; mechanism undetermined | 8 |
| Partially missing accents | 7 |
| Label/text alignment merits adjudication | 3 |

Groups overlap and the sample balances directions, so these are not population cause rates. Examples include criticism of `hơi mùi dầu` alongside `Giao hàng nhanh lắm`, praise expressed by `ko bị móp méo`, and the sparse positive-labelled `Quà khuyến mãi tặng kèm.`. They illustrate competing aspects, negation scope and missing context. They do not prove causal feature explanations or annotation errors. No independent human adjudication or inter-reviewer agreement study was performed; no labels or models were changed based on this review.

**The test set has now been inspected and is no longer an untouched holdout for the next development cycle.** Earlier smoke experiments also used examples from the same dataset. Selection in this run used validation only, but historical holdout purity is not claimed. Improvements prompted by these errors require fresh evaluation data or an explicitly declared new evaluation protocol.

Still unevaluated: neutral/multiclass behavior, aspect-level outputs, other domains, author/product/time-grouped generalization, repeated-seed uncertainty, dedicated accent/shorthand robustness sets, probability calibration and operational deployment. A Streamlit demo and passing tests do not establish these properties. No general spelling-correction or tone-repositioning benefit is claimed.

## Installation and data preparation

Commands below are for **PowerShell**, from the repository root. In a fresh checkout, create a virtual environment using an installed Python 3.13 interpreter:

```powershell
git clone https://github.com/J0sh4fun/vietnamese-sentiment-analysis.git
cd vietnamese-sentiment-analysis
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`requirements.txt` uses broad lower bounds, not a tested lockfile. The recorded run used Windows 11 and CPython **3.13.5**, with NumPy 2.5.1, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0, underthesea 9.5.0, underthesea_core 3.3.2, joblib 1.5.3 and Streamlit 1.59.2. Versions were read from the actual environment, not inferred from dependency ranges. The requirements also declare `torch`, but it was **not installed** in the verified environment and was not needed by the exercised TF-IDF/CRF path. No PyTorch behavior, clean-environment reinstall or non-Windows execution is verified.

See [the inspected environment snapshot](docs/verified-environment.json). To reproduce this particular study, use its `metadata.json` environment and `requirements-resolved.txt`, rather than assuming an unconstrained install will match:

```powershell
# Requires the saved run bundle; install into an isolated environment.
.venv/Scripts/python.exe -m pip install -r models/evaluation_workflow/full_validation_20260929_01/requirements-resolved.txt
```

That snapshot lists observed installed distributions, including packages not independently exercised; it is not a portable lockfile or a guarantee of future wheel availability.

Place UTF-8 JSONL originals at `data/shopee_reviews_dataset.jsonl`, or supply `--train-data`. Each line must be an object with nonempty string `review` and an exact string `label` (`negative` or `positive`). This is an **illustrative schema example**, not a reported dataset observation:

```json
{"review": "Sản phẩm tốt", "label": "positive"}
{"review": "Không hài lòng", "label": "negative"}
```

Extra columns are allowed; `source_id` is optional and must follow the identity rules above. `--text-column` and `--label-column` support alternative field names. The local 20-row sample is too small to stand in for the reported experiment. No preprocessing or accent-stripping pass should be applied to the raw inference inputs.

## Training, evaluation and demo commands

### Reproduce the full study

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m src.run_evaluation --train-data data/shopee_reviews_dataset.jsonl --random-state 42 --test-size 0.1 --val-size 0.1 --min-df 2 --timing-repeats 3 --error-count 40 --run-name readme_study_01
```

This trains all 16 validation candidates, selects by validation, checks save/load consistency on validation and evaluates the selected model once on test. Outputs go to `models/evaluation_workflow/readme_study_01/`. It generates a report and an error-review template; it does **not** automatically author qualitative error diagnoses. `--error-count` accepts 30–50 and samples fewer if fewer errors exist.

Choose a fresh name on every run, or omit `--run-name` for a timestamp. Existing runs are rejected; there is no overwrite flag. For a separate smoke run, add `--max-samples 300 --min-df 1` and use a different name. A rerun on this dataset is an already-inspected evaluation, not a fresh holdout.

### Ordinary training and augmentation

```powershell
# Reference preprocessing; compare the three learned classifiers.
.venv/Scripts/python.exe -m src.train --train-data data/shopee_reviews_dataset.jsonl --disable-aug --algorithms logreg multinomial_nb complement_nb --selection-metric f1_macro --random-state 42 --run-name train_reference_01

# Regenerate training-only unaccented variants, saved in a different new run.
.venv/Scripts/python.exe -m src.train --train-data data/shopee_reviews_dataset.jsonl --generate-aug --run-name train_augmented_01
```

Ordinary training writes `models/<run-name>/` and evaluates its selected model on test at the end. It does not run the four-configuration ablation study or include DummyClassifier. These commands therefore do not reproduce the selected stopword-off configuration by themselves. Use validation for development; repeatedly choosing settings after looking at these test outputs is test-driven tuning.

`--algorithm` selects one classifier and overrides `--algorithms`. Other training options include `--regularization`, `--class-weight`, `--nb-alpha`, `--max-iter`, n-gram/feature limits, split sizes and `--max-samples`. The full study CLI intentionally freezes classifier hyperparameters; those training-only flags are not accepted by `src.run_evaluation`.

Legacy augmentation is not loaded automatically. Files supplied through `--aug-data` without reliable provenance are excluded with a warning. Do not infer source IDs by matching accent-stripped text. Generated variants carry `source_id`, `source_text`, `label` and `augmentation_method: unaccented_v1`. A nonempty accepted export can be supplied to a later run with the same originals/split settings:

```powershell
.venv/Scripts/python.exe -m src.train --train-data data/shopee_reviews_dataset.jsonl --aug-data models/train_augmented_01/train_augmentation.jsonl --run-name train_imported_01
```

Imports are still rechecked against current training sources. `--disable-aug` overrides both imported and generated augmentation. Omit empty augmentation exports.

### Evaluate a saved artifact

```powershell
.venv/Scripts/python.exe -m src.evaluate --run-dir models/evaluation_workflow/readme_study_01 --split validation --output-dir models/evaluation_reports/readme_study_validation_01
```

The evaluator reads the saved raw review column, not `clean_text`, and writes `<split>_metrics.json` and `<split>_predictions.csv`. It supports `train`, `validation` and `test`; use validation for further comparisons. The full study already produced final test results. If repeating a frozen test audit, use `--split test` with a fresh `--output-dir`; do not use the output to tune and still call the test untouched. Existing output files are rejected. JSON records label order alongside confusion-matrix axes.

### Raw prediction and Streamlit

After the study command above completes:

```powershell
.venv/Scripts/python.exe -m src.predict --model-path models/evaluation_workflow/readme_study_01/sentiment_pipeline.joblib --text "Sản phẩm không tốt" "Máy khuấy tốt" --probabilities

$env:SENTIMENT_MODEL_PATH = "models/evaluation_workflow/readme_study_01/sentiment_pipeline.joblib"
.venv/Scripts/python.exe -m streamlit run app.py
```

There is no hard-coded production model. Streamlit requires `SENTIMENT_MODEL_PATH`, uses `infer()` and displays empty-input warnings and uncalibrated label-mapped probabilities. Headless app tests were run; an interactive browser session and deployed service were not evaluated.

The Python API also accepts raw text:

```python
from src.inference import load_model

model = load_model("models/evaluation_workflow/readme_study_01/sentiment_pipeline.joblib")
results = model.infer(["Sản phẩm không tốt", "!!!"], include_probabilities=True)
```

All four command interfaces can be inspected with `-m src.train --help`, `-m src.run_evaluation --help`, `-m src.evaluate --help` and `-m src.predict --help` using the same Python executable.

## Artifacts and reproducibility

| Producer | Main outputs |
|---|---|
| `src.train` | `sentiment_pipeline.joblib`, three split CSVs, `train_augmentation.jsonl`, `train_metadata.json`, `environment.json`, `requirements-resolved.txt` |
| `src.run_evaluation` | Selected model and split CSVs, selected-run `train_augmentation.jsonl`, `protocol.json`, `selection.json`, `validation_results.json`, `test_metrics.json`, predictions/errors CSVs, `error_review.csv`, `report.md`, `metadata.json`, `requirements-resolved.txt` |
| `src.evaluate` | `<split>_metrics.json`, `<split>_predictions.csv` |

The workflow metadata stores environment information inside `metadata.json`; it does not create the separate `environment.json` used by ordinary training. Its configuration records include each ablation's settings, preparation time, augmentation count and per-split distributions. The top-level base `generate_aug=false` does not mean the augmentation ablation was skipped. The selected configuration here has no augmentation, so its exported augmentation file has no examples.

Metadata records effective settings, seed, observed Python/dependency/platform versions, code revision/dirty state and file hashes, exact input fingerprints, preprocessing/stopword snapshots, filtering/augmentation audits and split label distributions. Preserve the whole run bundle **and matching source**, especially for a dirty-tree run. Artifact paths are relative; external data references require relocation by fingerprint. Copying only the `.joblib` file preserves its embedded processor but loses the audit trail.

Python/NumPy randomness, sampling, source-group splits and Logistic Regression are seeded where supported. This does not guarantee bitwise equality across operating systems, dependencies, BLAS implementations or thread settings. `PYTHONHASHSEED`, if needed, must be set before Python starts. Ratios must lie strictly between zero and one and sum to less than one; numerical limits and run names are validated, and insufficient data for stratification raises an explanatory error. Failed runs may leave a partial new directory; retain it and use a new name.

The last full regression execution passed **61 tests** in 9.547 seconds before the reported experiment. Coverage includes provenance/split leakage, preprocessing correctness and negation protection, invalid/empty inputs, raw inference, save/load consistency, Streamlit's shared path, metadata and overwrite rejection. The full study also confirmed identical predictions and empty flags for all 958 validation rows before/after serialization. Known joblib/NumPy deprecation and headless Streamlit context warnings did not fail those checks. For this documentation update, CLI help, code, hashes, metrics and reports were rechecked; training and final test inference were not rerun.

## Repository structure

```text
vietnamese-sentiment-analysis/
├── app.py                         # Streamlit raw-text demo
├── config.py                      # Empty legacy placeholder; not the stopword source
├── requirements.txt               # Broad dependency requirements
├── assets/
│   └── vietnamese_stopwords.py
├── src/
│   ├── data_pipeline.py           # Validation, source identities, splitting, augmentation
│   ├── preprocessor.py            # NFC, masking, segmentation and stopword settings
│   ├── train.py                   # Ordinary training and validation-based model selection
│   ├── run_evaluation.py          # Fixed-cohort comparison and ablation workflow
│   ├── experiments.py             # Seeds, configuration checks and provenance
│   ├── inference.py               # Versioned raw-text model wrapper
│   ├── predict.py                 # Raw-text prediction CLI
│   └── evaluate.py                # Saved-model metrics and predictions
├── tests/                         # Six unittest modules
├── docs/
│   ├── verified-environment.json
│   └── evaluation_20260929.md
├── data/                          # Local JSONL files; git-ignored
├── models/                        # Runs, artifacts and reports; git-ignored
├── .gitignore
└── README.md
```

The older [Kaggle notebook](https://www.kaggle.com/code/josh4fun/vietnamese-text-processing) is a historical project reference. Its contents were not accessible during this check; it is not verified as implementing the current split, preprocessing, inference or evaluation protocol. The CLI and recorded run above are the basis for the results in this README.

## Author

josh4fun

GitHub: https://github.com/J0sh4fun

LinkedIn: [www.linkedin.com/in/tiến-đạt-nguyễn-0b7241373](https://www.linkedin.com/in/ti%E1%BA%BFn-%C4%91%E1%BA%A1t-nguy%E1%BB%85n-0b7241373/)

Email: tiendat9320@gmail.com

If you found this project helpful, feel free to give it a ⭐!
