# config.py: project-wide constants
import os

SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
TEST_SIZE = 0.15
VAL_SIZE = 0.15
TRAIN_SIZE = 0.7
CV_FOLDS = 5
RANDOM_STATE = SEED
