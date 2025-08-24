# utils.py: plotting, saving, logging, seeds
import os
import json
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def set_seed(seed):
    import random
    import torch
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_csv(rows, path, header=None):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def plot_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(path)
    plt.close()


def plot_pr(y_true, y_score, path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(path)
    plt.close()
