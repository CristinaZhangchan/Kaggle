"""Task 3: validate the salary classifier.

Run from the repo root:  python src/validate.py
Writes reports/validation_*.csv|json and reports/figures/*.png
"""
import json

import matplotlib

matplotlib.use("Agg")  # scripts only write image files; no GUI backend needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

from salary.data import ROOT, load_transactions
from salary.evaluate import (bootstrap_ci, damage_text, income_metrics, oof_predict,
                             person_income, text_slice, txn_metrics)
from salary.features import FEATURE_COLS, PATTERN_COLS, TEXT_COLS, build_features
from salary.model import THRESHOLD, fit, make_gbm, make_logreg, rule_baseline
from salary.style import COLORS, save, setup

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
N_REPEATS = 5

VARIANTS = {
    # name: (model factory, feature columns)
    "LogReg - text + pattern": (make_logreg, FEATURE_COLS),
    "LogReg - text only": (make_logreg, TEXT_COLS),
    "LogReg - pattern only": (make_logreg, PATTERN_COLS),
    "GBM - text + pattern": (make_gbm, FEATURE_COLS),
}
MAIN = "LogReg - text + pattern"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    setup()
    txns = load_transactions()
    X = build_features(txns).reset_index(drop=True)
    y = X["is_salary"].to_numpy()
    print(f"{len(X)} credits, {y.sum()} salaries, {X.person_id.nunique()} persons")

    # 1. Baseline + model variants, 5 x repeated 5-fold CV grouped by person.
    rows = []
    rule = rule_baseline(X)
    rows.append({"model": "Rule: 'lön/salary' in text", "repeat": 0, **txn_metrics(y, rule)})
    oof = {}
    for name, (factory, cols) in VARIANTS.items():
        oof[name] = []
        for r in range(N_REPEATS):
            p = oof_predict(factory, X, cols, seed=r)
            oof[name].append(p)
            rows.append({"model": name, "repeat": r, **txn_metrics(y, p)})
    res = pd.DataFrame(rows)
    summary = res.groupby("model", sort=False).agg(["mean", "min", "max"]).drop(columns="repeat")
    summary.to_csv(REPORTS / "validation_models.csv")
    print(summary.xs("mean", axis=1, level=1).round(3).to_string())

    # 2. Main model, repeat 0: confidence interval, income level, slices, errors.
    p = oof[MAIN][0]
    pred = p >= THRESHOLD
    ci = bootstrap_ci(X, p)
    per = person_income(X, pred, txns)
    per_rule = person_income(X, rule >= THRESHOLD, txns)
    inc, inc_rule = income_metrics(per), income_metrics(per_rule)
    per.to_csv(REPORTS / "validation_person_income.csv")

    X["slice"] = text_slice(X)
    X["proba"] = p
    X["pred"] = pred
    slices = X.groupby("slice").apply(lambda g: pd.Series({
        "n": len(g),
        "salaries": int(g.is_salary.sum()),
        "recall": g.loc[g.is_salary, "pred"].mean() if g.is_salary.any() else np.nan,
        "false_pos": int((g.pred & ~g.is_salary).sum()),
        "rule_recall": rule[g.index][g.is_salary].mean() if g.is_salary.any() else np.nan,
        "rule_false_pos": int(((rule[g.index] == 1) & ~g.is_salary).sum()),
    }), include_groups=False)
    slices.to_csv(REPORTS / "validation_slices.csv")
    print(slices.round(3).to_string())

    # Coverage slice: persons with < 11 months of data.
    short = per.index[per["months"] < 11]
    in_short = X.person_id.isin(short).to_numpy()
    coverage = {
        "short_coverage_persons": int(len(short)),
        "short": txn_metrics(y[in_short], p[in_short]),
        "full": txn_metrics(y[~in_short], p[~in_short]),
    }

    errors = X.loc[pred != y, ["person_id", "date", "amount", "text", "is_salary", "proba", "slice",
                               "recur_loose_months", "key_n_months"]]
    errors.sort_values("proba").to_csv(REPORTS / "validation_errors.csv", index=False)
    print(errors.to_string())

    # 3. Threshold trade-off on the main model.
    thr = pd.DataFrame([{"threshold": t, **txn_metrics(y, p, t)} for t in [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]])
    thr.to_csv(REPORTS / "validation_thresholds.csv", index=False)

    # 4. Stress test: train on clean data, score held-out persons with damaged text.
    stress = [{"scenario": "clean", **txn_metrics(y, p)}]
    for mode in ["Swedish letters lost", "texts cut to 8 chars", "50% of texts blank", "no text"]:
        X_bad = build_features(damage_text(txns, mode)).reset_index(drop=True)
        p_bad = oof_predict(VARIANTS[MAIN][0], X, FEATURE_COLS, seed=0, X_eval=X_bad)
        p_rule = rule_baseline(X_bad)
        stress.append({"scenario": mode, **txn_metrics(y, p_bad),
                       "rule_recall": txn_metrics(y, p_rule)["recall"]})
        if mode == "no text":
            # Fallback: a pattern-only model trained on text-less data too.
            p_fb = oof_predict(VARIANTS[MAIN][0], X_bad, PATTERN_COLS, seed=0)
            stress.append({"scenario": "no text, model retrained without text", **txn_metrics(y, p_fb)})
    stress = pd.DataFrame(stress)
    stress.to_csv(REPORTS / "validation_stress.csv", index=False)
    print(stress.round(3).to_string())

    # 5. Permutation importance on held-out folds (grouped importance per feature).
    imp = held_out_importance(X, seed=0)
    imp.to_csv(REPORTS / "validation_importance.csv")

    main_mean = res[res.model == MAIN][["precision", "recall", "f1", "pr_auc"]].mean()
    out = {
        "main_model": MAIN,
        "threshold": THRESHOLD,
        "cv": f"{N_REPEATS}x repeated 5-fold StratifiedGroupKFold by person_id",
        "txn_metrics_mean": main_mean.round(4).to_dict(),
        "txn_metrics_repeat0": txn_metrics(y, p),
        "bootstrap_95ci_by_person": ci,
        "income_model": inc,
        "income_rule_baseline": inc_rule,
        "coverage": coverage,
    }
    (REPORTS / "validation_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))

    plot_pr(X, y, oof, rule)
    plot_income(per)
    plot_importance(imp)


def held_out_importance(X, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X, X["is_salary"], groups=X["person_id"]):
        m = fit(VARIANTS[MAIN][0](), X.iloc[tr])
        r = permutation_importance(m, X.iloc[te][FEATURE_COLS], X.iloc[te]["is_salary"].astype(int),
                                   scoring="average_precision", n_repeats=10, random_state=seed)
        scores.append(r.importances_mean)
    return pd.Series(np.mean(scores, axis=0), index=FEATURE_COLS, name="pr_auc_drop").sort_values(ascending=False)


def plot_pr(X, y, oof, rule):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, color in [(MAIN, COLORS["blue"]), ("LogReg - text only", COLORS["orange"]),
                        ("LogReg - pattern only", COLORS["aqua"])]:
        prec, rec, _ = precision_recall_curve(y, oof[name][0])
        ax.plot(rec, prec, color=color, lw=2, label=name.replace("LogReg - ", ""))
    m = txn_metrics(y, rule)
    ax.plot(m["recall"], m["precision"], "o", ms=9, color=COLORS["ink"], label="keyword rule")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0.5, 1.02),
           title="Precision-recall, held-out persons")
    ax.legend(loc="lower left", frameon=False)
    save(fig, FIG / "pr_curves.png")
    plt.close(fig)


def plot_income(per):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    k = per[["true_monthly", "pred_monthly"]] / 1000
    lim = k.max().max() * 1.05
    ax.plot([0, lim], [0, lim], color=COLORS["grid"], lw=1, zorder=0)
    ax.scatter(k.true_monthly, k.pred_monthly, s=40, color=COLORS["blue"], edgecolor="white", lw=1)
    ax.set(xlabel="True salary, kSEK / month", ylabel="Predicted salary, kSEK / month",
           xlim=(-1, lim), ylim=(-1, lim), title="Monthly salary per person, held-out")
    save(fig, FIG / "income_scatter.png")
    plt.close(fig)


def plot_importance(imp):
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [COLORS["orange"] if c in TEXT_COLS else COLORS["blue"] for c in top.index]
    ax.barh(top.index, top.values, color=colors, height=0.6)
    ax.set(xlabel="Drop in PR-AUC when shuffled", title="What the model relies on")
    ax.text(0.98, 0.05, "orange = text\nblue = pattern", transform=ax.transAxes, ha="right",
            color=COLORS["muted"], fontsize=9)
    save(fig, FIG / "importance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
