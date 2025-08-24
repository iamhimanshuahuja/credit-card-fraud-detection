# train.py: CLI for training models
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score
from .config import MODELS_DIR, SEED, CV_FOLDS
from .data_prep import load_data, stratified_split, get_pipeline
from .models import get_model_and_params
from .utils import set_seed, save_json, save_csv, setup_logger, plot_roc, plot_pr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resampling', choices=['none', 'smote', 'undersample'], default='none')
    parser.add_argument('--model', choices=['nb', 'logreg', 'svm', 'rbfsvm', 'rf'], required=True)
    parser.add_argument('--cv', type=int, default=CV_FOLDS)
    args = parser.parse_args()

    setup_logger()
    set_seed(SEED)

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    model, param_grid = get_model_and_params(args.model)
    pipe = get_pipeline(model, resampling=args.resampling)

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, param_grid, scoring='average_precision', cv=skf, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    # Fit on train+val with best params
    X_trval = pd.concat([X_train, X_val])
    y_trval = pd.concat([y_train, y_val])
    best_pipe = get_pipeline(model, resampling=args.resampling)
    best_pipe.set_params(**search.best_params_)
    best_pipe.fit(X_trval, y_trval)

    # Save artifacts
    run_dir = os.path.join(MODELS_DIR, args.model)
    os.makedirs(run_dir, exist_ok=True)
    joblib.dump(best_pipe, os.path.join(run_dir, 'model.joblib'))
    save_json({'best_params': search.best_params_, 'cv_score': search.best_score_}, os.path.join(run_dir, 'metrics.json'))


    # Predict on val for curves and metrics
    y_val_pred = best_pipe.decision_function(X_val) if hasattr(best_pipe.named_steps['clf'], 'decision_function') else best_pipe.predict_proba(X_val)[:,1]
    from sklearn.metrics import roc_auc_score
    pr_auc = average_precision_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred)
    plot_roc(y_val, y_val_pred, os.path.join(run_dir, 'roc.png'))
    plot_pr(y_val, y_val_pred, os.path.join(run_dir, 'pr.png'))

    # Leaderboard row
    import hashlib, datetime
    params_str = str(search.best_params_)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    leaderboard_path = os.path.join(MODELS_DIR, 'leaderboard.csv')
    row = [timestamp, args.model, args.resampling, f"{pr_auc:.5f}", f"{roc_auc:.5f}", params_hash, run_dir]
    header = ["timestamp", "model", "resampling", "pr_auc", "roc_auc", "params_hash", "run_dir"]
    # Append or create leaderboard
    if not os.path.exists(leaderboard_path):
        save_csv([row], leaderboard_path, header=header)
    else:
        with open(leaderboard_path, 'a', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(row)

    print(f"Model saved to {run_dir}")

if __name__ == '__main__':
    main()
