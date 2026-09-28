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

Restoration rejects unsupported settings or runtime mismatches. The Streamlit app now restores this configuration alongside the model; models without it require retraining and updating `model_path`. Existing artifacts are not rewritten. For an explicit diagnostic comparison only, `--legacy-tone-repositioning` enables the still-defective custom heuristic before segmentation and emits a warning. This flag does not recreate every behavior of the old pipeline. These correctness changes do not establish an F1 improvement.

## Evaluate the model

Evaluate the trained model on the test split to generate the Classification Report, Confusion Matrix, and Prediction outputs

```bash
python src/evaluate.py --run-dir models/<run_name> --split test
```

Evaluation outputs:

- `<split>_predictions.csv`
- `<split>_metrics.json`

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

You can test the trained model directly through an interactive web application. The interface allows you to input custom Vietnamese reviews and provides real-time sentiment predictions along with confidence scores (probability percentages).

### 1. Start the App
Before running the app, ensure you have successfully trained a model and that the `model_path` in `app.py` points to your latest `.joblib` artifact. 

Run the following command from the root directory:

```bash
streamlit run app.py

## Author 
josh4fun

GitHub: https://github.com/J0sh4fun

LinkedIn: [www.linkedin.com/in/tiến-đạt-nguyễn-0b7241373](https://www.linkedin.com/in/ti%E1%BA%BFn-%C4%91%E1%BA%A1t-nguy%E1%BB%85n-0b7241373/)

Email: tiendat9320@gmail.com

If you found this project helpful, feel free to give it a ⭐!

