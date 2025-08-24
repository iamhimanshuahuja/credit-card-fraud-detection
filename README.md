
# Credit Card Fraud Detection

An end-to-end machine learning pipeline for detecting credit card fraud, built to demonstrate advanced data science and engineering skills. This project leverages the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset and implements robust solutions for real-world imbalanced classification problems.

---

## 🚀 Skills Demonstrated

- **Data Engineering:** Modular data preprocessing, feature engineering, and pipeline design
- **Imbalanced Learning:** SMOTE, undersampling, and baseline strategies
- **Modeling:** Naïve Bayes, Logistic Regression, SVM, Random Forest
- **Evaluation:** Stratified splits, cross-validation, PR-AUC, ROC-AUC, precision, recall, F1
- **Automation:** CLI tools for training/evaluation, reproducible experiments
- **Best Practices:** Code modularity, documentation, pre-commit hooks, and version control
- **Visualization:** Metrics plots, leaderboard generation

---


## Features
- Handles class imbalance (SMOTE, undersampling, baseline)
- Modular pipelines with scaling, resampling, and model selection
- Stratified train/val/test split
- Grid/randomized search with cross-validation
- Metrics: PR-AUC (main), ROC-AUC, precision, recall, F1
- Saves best model, metrics, plots, and leaderboard
- CLI for training and evaluation

## Project Structure

```
credit-card-fraud-detection/
├── data/           # Raw data (not tracked by git)
├── models/         # Saved models and experiment runs
├── notebooks/      # EDA and experimentation notebooks
│   └── EDA.ipynb
├── src/            # Source code
│   ├── config.py
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---


## Setup
1. **Python ≥3.10** required.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download `creditcard.csv` from Kaggle and place it in `data/` (not tracked by git).


## Usage

### Train a model
```bash
python -m src.train --resampling smote --model logreg --cv 5
```

### Evaluate a saved model
```bash
python -m src.evaluate --run_dir models/logreg/ --threshold auto
```

### Example: Try different models and resampling
```bash
# Train Random Forest with undersampling
python -m src.train --resampling undersample --model rf --cv 3

# Train SVM with SMOTE
python -m src.train --resampling smote --model svm --cv 5
```

### Example: Evaluate with custom threshold
```bash
python -m src.evaluate --run_dir models/rf/ --threshold 0.2
```


## Code Quality & Linting

This repo uses [pre-commit](https://pre-commit.com/) with Black, isort, and flake8. To enable:

```bash
pip install pre-commit
pre-commit install
```

Run all hooks manually:
```bash
pre-commit run --all-files
```


---

## Notes & Highlights
- Data is highly imbalanced. PR-AUC is more informative than ROC-AUC.
- Threshold tuning can improve recall while maintaining high precision.
- Modular code and CLI make it easy to extend or adapt to new datasets.
- See code comments for implementation details and rationale.

---
