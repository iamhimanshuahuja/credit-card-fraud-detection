# Credit Card Fraud Detection

This project trains and compares Naïve Bayes, Logistic Regression, SVM, and Random Forest on the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.

## Features
- Handles class imbalance (SMOTE, undersampling, baseline)
- Modular pipelines with scaling, resampling, and model
- Stratified train/val/test split
- Grid/randomized search with cross-validation
- Metrics: PR-AUC (main), ROC-AUC, precision, recall, F1
- Saves best model, metrics, plots, and leaderboard
- CLI for training and evaluation

## Setup
1. **Python ≥3.10** required.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download `creditcard.csv` from Kaggle and place it in `data/` (not tracked by git).

## Usage
Train a model (example):
```bash
python -m src.train --resampling smote --model logreg --cv 5
```

Evaluate a saved model (example):
```bash
python -m src.evaluate --run_dir models/logreg/ --threshold auto
```

## Code style & linting

This repo uses [pre-commit](https://pre-commit.com/) with Black, isort, and flake8. To enable:

```bash
pip install pre-commit
pre-commit install
```

Run all hooks manually:
```bash
pre-commit run --all-files
```

## Notes
- Data is highly imbalanced. PR-AUC is more informative than ROC-AUC.
- Threshold tuning can improve recall while maintaining high precision.
- See code comments for details.
