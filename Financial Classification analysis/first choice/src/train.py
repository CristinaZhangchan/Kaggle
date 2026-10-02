"""Task 2: train the final salary classifier on all 100 persons and save it.

Run from the repo root:  python src/train.py
Writes models/salary_model.joblib, outputs/predictions.csv,
outputs/person_income.csv, reports/model_coefficients.csv, reports/figures/coefficients.png
"""
import matplotlib

matplotlib.use("Agg")  # scripts only write image files; no GUI backend needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from salary.data import ROOT, load_transactions
from salary.evaluate import oof_predict, person_income
from salary.features import FEATURE_COLS, TEXT_COLS, build_features
from salary.model import MODEL_PATH, THRESHOLD, fit, make_logreg, predict_proba, save_model
from salary.style import COLORS, save, setup

OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main():
    for d in (OUT, REPORTS / "figures"):
        d.mkdir(parents=True, exist_ok=True)
    setup()
    txns = load_transactions()
    X = build_features(txns)

    model = fit(make_logreg(), X)
    save_model(model)

    # proba_oof: honest score from a model that never saw this person (use for evaluation).
    # proba: final model trained on everyone (what production would output).
    X = X.reset_index(drop=True)
    X["proba_oof"] = oof_predict(make_logreg, X, FEATURE_COLS, seed=0)
    X["proba"] = predict_proba(model, X)
    preds = txns.merge(X[["txn_id", "proba", "proba_oof"]], on="txn_id", how="left")
    preds[["proba", "proba_oof"]] = preds[["proba", "proba_oof"]].fillna(0.0)  # debits are never salary
    preds["pred_is_salary"] = preds["proba"] >= THRESHOLD
    preds.drop(columns=["n_accounts"]).to_csv(OUT / "predictions.csv", index=False)

    per = person_income(X, X["proba_oof"].to_numpy() >= THRESHOLD, txns)
    per.round(2).to_csv(OUT / "person_income.csv")

    # Standardised coefficients: effect of +1 std of a feature on the log-odds.
    coef = pd.Series(model[-1].coef_[0], index=FEATURE_COLS, name="coef").sort_values()
    coef.round(3).to_csv(REPORTS / "model_coefficients.csv")
    plot_coefficients(coef)

    n_pred = int(preds["pred_is_salary"].sum())
    print(f"Saved {MODEL_PATH.relative_to(ROOT)}; {n_pred} transactions predicted as salary "
          f"(labelled: {int(preds['is_salary'].sum())}).")


def plot_coefficients(coef):
    top = pd.concat([coef.head(7), coef.tail(8)])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    colors = [COLORS["orange"] if c in TEXT_COLS else COLORS["blue"] for c in top.index]
    ax.barh(top.index, top.values, color=colors, height=0.6)
    ax.axvline(0, color=COLORS["muted"], lw=0.8)
    ax.set(xlabel="Coefficient (log-odds per 1 std)", title="Evidence for / against salary")
    ax.text(0.97, 0.03, "orange = text feature\nblue = pattern feature", transform=ax.transAxes,
            ha="right", va="bottom", color=COLORS["muted"], fontsize=9)
    save(fig, REPORTS / "figures" / "coefficients.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
