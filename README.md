# Vietnamese E-Commerce Sentiment Analysis

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Vietnamese-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](đường_link_đến_notebook_kaggle_của_bạn)

An end-to-end Machine Learning pipeline designed to classify Vietnamese e-commerce reviews (e.g., Shopee) into Positive or Negative sentiments. 

This project heavily emphasizes a **data-centric approach**, featuring a robust custom text preprocessor for the Vietnamese language, and an automated Model Selection pipeline to handle imbalanced datasets.

## Key Features

*   **Tailored Vietnamese Preprocessing:** Built from scratch to handle real-world "dirty" data. Includes HTML/URL masking, Unicode (NFC) normalization, Vietnamese tone standardization, and tokenization using `underthesea`.
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

Run the training pipeline. The script will automatically preprocess the data, run Cross-Validation across multiple algorithms, and save the best model.

```bash
python src/train.py --train-data data/shopee_reviews_dataset.jsonl
```

Optional Arguments:

--algorithms: Choose specific models (e.g., --algorithms logreg complement_nb).

--nb-alpha: Set smoothing parameter for Naive Bayes.

--regularization: Set inverse regularization strength (C) for Logistic Regression.

Training will:

1. Load `data/shopee_reviews_dataset.jsonl` and `data/aug_unaccented_reviews.jsonl`
2. Clean and normalize Vietnamese text with `src/preprocessor.py`
3. Split data into train/validation/test sets (stratified)
4. Train and compare candidate models on validation split
5. Automatically select the best model
6. Save artifacts in `models/<run_name>/`

Output: The best model (sentiment_pipeline.joblib) and metrics will be saved in a timestamped folder inside models/ (e.g., models/20260507_120000).

Training artifacts include:

- `sentiment_pipeline.joblib`
- `train_split.csv`
- `validation_split.csv`
- `test_split.csv`
- `train_metadata.json`

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

