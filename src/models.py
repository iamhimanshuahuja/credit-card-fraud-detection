# models.py: model factories and param grids
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier


def get_model_and_params(name):
    if name == 'nb':
        model = GaussianNB()
        param_grid = {}
    elif name == 'logreg':
        model = LogisticRegression(solver='liblinear', class_weight=None, random_state=0, max_iter=200)
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__solver': ['liblinear', 'saga'],
            'clf__class_weight': [None, 'balanced']
        }
    elif name == 'svm':
        model = LinearSVC(dual=False, class_weight=None, random_state=0, max_iter=2000)
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__class_weight': [None, 'balanced']
        }
    elif name == 'rbfsvm':
        model = SVC(kernel='rbf', probability=True, class_weight=None, random_state=0)
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__gamma': ['scale', 'auto'],
            'clf__class_weight': [None, 'balanced']
        }
    elif name == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        param_grid = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [None, 5, 10],
            'clf__class_weight': [None, 'balanced']
        }
    else:
        raise ValueError(f'Unknown model: {name}')
    return model, param_grid
