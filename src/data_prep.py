# data_prep.py: load, split, build pipelines
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from .config import DATA_PATH, SEED, TRAIN_SIZE, VAL_SIZE, TEST_SIZE


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


def stratified_split(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1-TRAIN_SIZE, stratify=y, random_state=SEED)
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio, stratify=y_temp, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_pipeline(estimator, resampling='none'):
    steps = [('scaler', StandardScaler())]
    if resampling == 'smote':
        steps.append(('sampler', SMOTE(random_state=SEED)))
    elif resampling == 'undersample':
        steps.append(('sampler', RandomUnderSampler(random_state=SEED)))
    steps.append(('clf', estimator))
    if resampling == 'none':
        return Pipeline(steps)
    else:
        return ImbPipeline(steps)
