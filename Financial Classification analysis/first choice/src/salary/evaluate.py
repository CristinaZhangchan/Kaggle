"""Validation helpers: person-grouped cross-validation and income-level metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from salary.model import THRESHOLD, fit, predict_proba


def oof_predict(make_model, X: pd.DataFrame, cols, seed: int, n_splits: int = 5,
                X_eval: pd.DataFrame | None = None) -> np.ndarray:
    """Out-of-fold probabilities. Folds are split by person: a person's
    transactions are never in both train and test. That mirrors production
    (a new applicant is a new person) and stops the model from memorising a
    specific employer name it saw in training.

    X_eval (same rows as X, e.g. with damaged text) is what the held-out fold is
    scored on; the model is always trained on X.
    """
    X_eval = X if X_eval is None else X_eval
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for train_idx, test_idx in cv.split(X, X["is_salary"], groups=X["person_id"]):
        model = fit(make_model(), X.iloc[train_idx], cols)
        oof[test_idx] = predict_proba(model, X_eval.iloc[test_idx], cols)
    return oof


def damage_text(txns: pd.DataFrame, mode: str, seed: int = 0) -> pd.DataFrame:
    """Stress test: simulate worse bank data than we trained on."""
    out = txns.copy()
    rng = np.random.default_rng(seed)
    if mode == "no text":
        out["text"] = ""
    elif mode == "50% of texts blank":
        out.loc[rng.random(len(out)) < 0.5, "text"] = ""
    elif mode == "Swedish letters lost":
        out["text"] = out["text"].str.replace(r"[åäöÅÄÖ]", " ", regex=True)
    elif mode == "texts cut to 8 chars":
        out["text"] = out["text"].str[:8]
    else:
        raise ValueError(mode)
    return out


def txn_metrics(y_true, proba, threshold: float = THRESHOLD) -> dict:
    pred = proba >= threshold
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "pr_auc": average_precision_score(y_true, proba),
        "fp": int((pred & ~y_true).sum()),
        "fn": int((~pred & y_true).sum()),
    }


def person_income(X: pd.DataFrame, pred: np.ndarray, all_txns: pd.DataFrame) -> pd.DataFrame:
    """Average monthly salary per person, true vs predicted.

    Months = calendar months in which the person has any transaction, so a
    person who joined in August is not divided by 12.
    """
    months = all_txns.groupby("person_id")["date"].apply(lambda d: d.dt.to_period("M").nunique())
    amt = X["amount"].to_numpy()
    df = pd.DataFrame({
        "person_id": X["person_id"].to_numpy(),
        "true_amt": np.where(X["is_salary"].to_numpy(), amt, 0.0),
        "pred_amt": np.where(pred, amt, 0.0),
    })
    per = df.groupby("person_id")[["true_amt", "pred_amt"]].sum()
    per = per.reindex(months.index, fill_value=0.0)
    per["months"] = months
    per["true_monthly"] = per["true_amt"] / per["months"]
    per["pred_monthly"] = per["pred_amt"] / per["months"]
    per["has_salary"] = per["true_amt"] > 0
    per["pred_has_salary"] = per["pred_amt"] > 0
    per["abs_pct_err"] = np.where(per["has_salary"],
                                  (per["pred_monthly"] - per["true_monthly"]).abs() / per["true_monthly"],
                                  np.nan)
    return per


def income_metrics(per: pd.DataFrame) -> dict:
    sal = per[per["has_salary"]]
    return {
        "person_has_salary_acc": float((per["has_salary"] == per["pred_has_salary"]).mean()),
        "no_salary_persons_with_false_income": int((~per["has_salary"] & per["pred_has_salary"]).sum()),
        "income_median_abs_pct_err": float(sal["abs_pct_err"].median()),
        "income_share_within_5pct": float((sal["abs_pct_err"] <= 0.05).mean()),
        "income_share_within_10pct": float((sal["abs_pct_err"] <= 0.10).mean()),
    }


def bootstrap_ci(X: pd.DataFrame, proba: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CI by resampling *persons* (the unit that varies between applicants)."""
    rng = np.random.default_rng(seed)
    persons = X["person_id"].unique()
    idx_by_person = X.groupby("person_id").indices
    y = X["is_salary"].to_numpy()
    stats = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        sample = rng.choice(persons, size=len(persons), replace=True)
        idx = np.concatenate([idx_by_person[p] for p in sample])
        m = txn_metrics(y[idx], proba[idx])
        for k in stats:
            stats[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in stats.items()}


def text_slice(X: pd.DataFrame) -> pd.Series:
    """Which kind of evidence the text gives: used to break recall/FP down."""
    s = pd.Series("other text", index=X.index)
    s[X["kw_employer"] == 1] = "employer name"
    s[X["kw_salary"] == 1] = "salary keyword"
    s[(X["kw_salary"] == 1) & ((X["kw_swish"] == 1) | (X["kw_own_transfer"] == 1))] = "'lön' via Swish/transfer"
    benefit = X[["kw_pension", "kw_student_aid", "kw_unemployment", "kw_social_insurance"]].sum(axis=1) > 0
    s[benefit] = "pension/CSN/benefit"
    s[X["text_empty"] == 1] = "empty text"
    s[X["text_corrupted"] == 1] = "corrupted text"
    return s
