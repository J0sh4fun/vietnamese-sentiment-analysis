# Executed evaluation: 29 September 2026

The full-data run `full_validation_20260929_01` completed all 16 planned validation comparisons. Validation selected **ComplementNB with stopword filtering disabled**, segmentation enabled, no augmentation and no tone repositioning. The model was saved before one final test evaluation; it was not refitted on validation.

Artifacts are in [the new run directory](../models/evaluation_workflow/full_validation_20260929_01/). Previous runs were not modified.

- [Full report: validation per-class metrics, timings and final confusion matrix](../models/evaluation_workflow/full_validation_20260929_01/report.md)
- [Completed qualitative analysis of 40 real test errors](../models/evaluation_workflow/full_validation_20260929_01/error_analysis.md)
- [Exact sampled reviews and source IDs](../models/evaluation_workflow/full_validation_20260929_01/error_review.csv)
- [Settings, environment, audits, distributions and fingerprints](../models/evaluation_workflow/full_validation_20260929_01/metadata.json)
- [Validation selection saved before test evaluation](../models/evaluation_workflow/full_validation_20260929_01/selection.json)

The `models/` directory is git-ignored. Preserve or copy the whole run directory to retain the model, full reviews and detailed reports; this document alone is not the artifact bundle.

## Data and protocol

The source contained 9,599 rows. No invalid review/label values or raw normalized-text duplicates were found. Reference preprocessing identified 18 same-label duplicate rows and 5 rows in conflicting-label groups; these 23 rows were excluded before splitting (0.240% of input). All ablations use the same remaining 9,576 originals. This reference-filtered cohort does not measure how each ablation would behave on the excluded rows.

| Stage/split | Negative | Positive | Total |
|---|---:|---:|---:|
| Raw originals | 5965 | 3634 | 9599 |
| Filtered originals | 5944 | 3632 | 9576 |
| Training originals | 4754 | 2906 | 7660 |
| Validation | 595 | 363 | 958 |
| Test | 595 | 363 | 958 |
| Training with augmentation, augmentation ablation only | 9503 | 5807 | 15310 |

Seed 42; validation and test fractions each 0.1. Source IDs and accent-preserving raw/clean normalized text are checked across splits for all configurations. The augmentation ablation generated 7,660 candidates exclusively from training originals, accepted 7,650, and excluded 10 collisions with original texts. No original identities were merged through accent removal. The old untraceable augmentation file was not used.

Reference: NFC, segmentation on, stopwords on, augmentation off, tone repositioning off. Each ablation changes one setting. Disabling segmentation retains whole-token stopword matching, which changes how often underscore multiword entries match. The legacy tone heuristic remains excluded because its correctness is not established.

All classifiers use the same TF-IDF configuration: word 1–2 grams, min_df=2, at most 150,000 features, sublinear TF. LR uses C=1.5, balanced weights and max_iter=2500; NB alpha=0.5. The majority baseline uses DummyClassifier(strategy="most_frequent"). TF-IDF fits on training only. Selection maximises validation macro-F1, then accuracy, then declaration order. There was no joint ablation search or other hyperparameter tuning.

## Validation macro-F1

| Configuration | Majority | Logistic Regression | MultinomialNB | ComplementNB |
|---|---:|---:|---:|---:|
| Reference | 0.383129 | 0.897570 | 0.895072 | 0.889818 |
| Segmentation off | 0.383129 | 0.913248 | 0.894713 | 0.898727 |
| Stopwords off | 0.383129 | 0.918537 | 0.916423 | **0.923094** |
| Training accent-removal augmentation | 0.383129 | 0.900924 | 0.891537 | 0.894528 |

The selected validation accuracy was 0.926931. All 16 accuracies and per-class precision/recall/F1/support are in the full report and `validation_results.json`. These single-split differences are descriptive; they establish neither general algorithm superiority nor confidence intervals. The augmentation effect is measured relative to the reference, not in combination with the selected stopword-off setting.

## Final test

**Macro-F1: 0.928263. Accuracy: 0.932150.** All 958 test rows were included. There were 0 processing failures, 0 empty preprocessing outputs and 0 split-row exclusions (each 0%). The same rates were zero for training and validation under every ablation. Pre-split quality filtering is separate from these per-split rates.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.952218 | 0.937815 | 0.944962 | 595 |
| positive | 0.900538 | 0.922865 | 0.911565 | 363 |

Confusion matrix, rows true and columns predicted:

| True / predicted | negative | positive |
|---|---:|---:|
| negative | 558 | 37 |
| positive | 28 | 335 |

The selected TF-IDF/classifier fit took 0.6937 seconds on prepared training text. Preparing the stopword-off representation for all three splits took 14.9842 seconds, excluding the shared initial reference-cohort preparation. These timings cover different work and should not be interpreted as an end-to-end cold training benchmark.

Selected-model validation inference took a median 1.5663 ms/review, amortised over batches of 958. Method: `time.perf_counter`, one untimed whole-validation warm-up followed by three full raw-text batches; includes preprocessing and prediction, excludes model/disk loading, metrics and probability calculation. The final test used one raw batch, taking 1.5250 seconds. Timings reflect this Windows environment and machine load, not a single-request latency guarantee. All other fit durations and individual inference repetitions are saved.

## Completed error analysis and limitations

All 40 sampled full reviews were read: 20 negative→positive and 20 positive→negative, out of 65 total errors. The separate analysis documents every example, literal excerpts, groups and tentative interpretations. Observations include mixed aspects (21 annotations), negation/scope (16), partially missing accents (7), limited experience/context (9), possible label/text ambiguity (3) and fairly clear sentiment missed with mechanism undetermined (8). Groups overlap; their counts are not population error-cause estimates. No label corrections or model improvements were made from these observations.

Examples include negative `hơi mùi dầu` alongside `Giao hàng nhanh lắm`, positive `ko bị móp méo` denying damage, and the context-poor positive-labelled `Quà khuyến mãi tặng kèm.`. Label ambiguity is not proof of annotation error; no independent adjudication was performed.

**The test set has now been inspected and is no longer an untouched holdout for the next development cycle.** Earlier smoke experiments also used examples from this dataset. Current selection used validation only, but historical holdout purity is not claimed. Any improvements prompted by these examples require fresh evaluation data or a clearly declared new protocol.

## Verification and rerun

Actual verification: `python -m unittest discover -s tests -v` completed **61 tests, all passing, in 9.547 seconds** before this full run. Tests cover fixed ablation cohorts, source/variant boundaries, alternate-representation collision rejection, validation-only stable selection, majority behaviour, save/load consistency, metadata, empty denominators and overwrite rejection. The initial focused test run exposed a test fixture whose word was removed by stopword filtering; the fixture was corrected to disable filtering for that overlap test, then the complete suite passed.

The full run also verified that all 958 validation predictions and empty flags remained identical after saving and loading the selected artifact. Existing tests exercise the shared inference and headless Streamlit path. No fresh-environment installation, interactive browser session, repeated-seed experiment or probability calibration was performed. The environment reports CPython 3.13.5, scikit-learn 1.9.0 and underthesea 9.5.0; exact observed distributions are saved, not claimed to be universally compatible. Joblib/NumPy deprecation and headless Streamlit context warnings were non-failing.

Final artifact checks matched every recorded artifact and code fingerprint. All 89 literal excerpts in the 40 individual error-review rows matched their source reviews, with both directions verified; [the completion receipt](../models/evaluation_workflow/full_validation_20260929_01/review_completion.json) records those checks. The raw inference command below was executed successfully and returned `negative`, with `empty_after_preprocessing=false`.

From the project root, in the recorded environment:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m src.run_evaluation --run-name full_validation_repeat_01
.venv/Scripts/python.exe -m src.predict --model-path models/evaluation_workflow/full_validation_20260929_01/sentiment_pipeline.joblib --text "Sản phẩm không tốt"
```

Choose a fresh name on each execution. A rerun on this data reproduces an already-inspected evaluation; it does not create a fresh holdout. Existing files are rejected rather than overwritten. Generated error-review columns start blank; the completed analysis for this run is a separate authored file so generated CSVs and their recorded fingerprints remain intact.
