"""Models: a keyword rule (baseline), logistic regression, gradient boosting."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from salary.data import ROOT
from salary.features import FEATURE_COLS

THRESHOLD = 0.5
MODEL_PATH = ROOT / "models" / "salary_model.joblib"


def rule_baseline(X: pd.DataFrame) -> np.ndarray:
    """What a first version in production might do: 'lön'/'salary' in the text,
    unless it is a Swish or a transfer between own accounts."""
    return ((X["kw_salary"] == 1) & (X["kw_swish"] == 0) & (X["kw_own_transfer"] == 0)).astype(float).to_numpy()


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def make_gbm(seed: int = 0):
    # Small trees + strong regularisation: only ~2k credits and 100 persons.
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
        l2_regularization=1.0, random_state=seed,
    )


MODELS = {"logreg": make_logreg, "gbm": make_gbm}


def fit(model, X: pd.DataFrame, cols=FEATURE_COLS):
    return model.fit(X[cols], X["is_salary"].astype(int))


def predict_proba(model, X: pd.DataFrame, cols=FEATURE_COLS) -> np.ndarray:
    return model.predict_proba(X[cols])[:, 1]


def save_model(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLS, "threshold": THRESHOLD}, path)


def load_model(path=MODEL_PATH) -> dict:
    return joblib.load(path)
