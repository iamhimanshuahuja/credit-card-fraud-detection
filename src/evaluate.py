# evaluate.py: CLI for evaluating saved models
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score
from .config import MODELS_DIR, DATA_PATH, SEED
from .data_prep import load_data, stratified_split
from .utils import plot_roc, plot_pr, save_json, setup_logger


def find_best_threshold(y_true, y_score, min_precision=0.8):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    mask = precision >= min_precision
    if np.any(mask):
        idx = np.argmax(recall * mask)  # highest recall with precision >= min_precision
    else:
        idx = np.argmax(f1s)
    return thresholds[idx], f1s[idx], precision[idx], recall[idx]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--threshold', default='auto')
    parser.add_argument('--precision-min', type=float, default=None, help='Find highest-recall threshold with precision ≥ value (overrides --threshold if set)')
    args = parser.parse_args()

    setup_logger()
    X, y = load_data()
    _, _, X_test, _, _, y_test = stratified_split(*load_data())

    model = joblib.load(os.path.join(args.run_dir, 'model.joblib'))
    y_score = model.decision_function(X_test) if hasattr(model.named_steps['clf'], 'decision_function') else model.predict_proba(X_test)[:,1]
    y_pred = (y_score > 0.5).astype(int)

    # Threshold tuning
    if args.precision_min is not None:
        threshold, f1, prec, rec = find_best_threshold(y_test, y_score, min_precision=args.precision_min)
        y_pred = (y_score > threshold).astype(int)
        print(f"Threshold for precision≥{args.precision_min:.2f}: {threshold:.3f} (Recall={rec:.3f}, F1={f1:.3f}, Precision={prec:.3f})")
    elif args.threshold == 'auto':
        threshold, f1, prec, rec = find_best_threshold(y_test, y_score)
        y_pred = (y_score > threshold).astype(int)
        print(f"Auto threshold: {threshold:.3f} (F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f})")

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # Save
    save_json({'classification_report': report, 'confusion_matrix': cm.tolist(), 'roc_auc': roc_auc, 'pr_auc': pr_auc}, os.path.join(args.run_dir, 'test_metrics.json'))
    plot_roc(y_test, y_score, os.path.join(args.run_dir, 'test_roc.png'))
    plot_pr(y_test, y_score, os.path.join(args.run_dir, 'test_pr.png'))
    print(f"Test ROC-AUC: {roc_auc:.3f}, PR-AUC: {pr_auc:.3f}")

if __name__ == '__main__':
    main()
