"""Task 1: characteristics of the data.

Run from the repo root:  python src/eda.py
Prints the numbers used in SOLUTION.md and the slides, writes reports/eda_findings.json
and reports/figures/{payday,salary_text}.png
"""
import json

import matplotlib

matplotlib.use("Agg")  # scripts only write image files; no GUI backend needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from salary.data import ROOT, load_transactions
from salary.evaluate import text_slice
from salary.features import build_features, payday_25
from salary.style import COLORS, save, setup

REPORTS = ROOT / "reports"


def income_type(X: pd.DataFrame) -> pd.Series:
    t = pd.Series("Swish / private", index=X.index)
    t[X.kw_own_transfer == 1] = "Own transfer"
    t[X.kw_student_aid == 1] = "CSN (student aid)"
    t[X.kw_unemployment == 1] = "A-kassa"
    t[X.kw_pension == 1] = "Pension"
    t[(X.kw_tax_refund == 1) | (X.kw_insurance == 1)] = "Tax refund / insurance"
    t[X.kw_social_insurance == 1] = "Försäkringskassan"
    t[X.is_salary] = "Salary"
    return t


def person_segment(X: pd.DataFrame) -> pd.Series:
    """Main income source per person (by total amount, excluding own transfers)."""
    ext = X[X.income_type != "Own transfer"]
    main = ext.groupby(["person_id", "income_type"])["amount"].sum().reset_index()
    return main.sort_values("amount").groupby("person_id").last()["income_type"]


def main():
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    setup()
    txns = load_transactions()
    X = build_features(txns)
    X["income_type"] = income_type(X)
    X["on_payday25"] = X["date"] == payday_25(X["date"])
    sal = X[X.is_salary]

    months = txns.groupby("person_id")["date"].agg(["min", "max"])
    sal_month = sal.assign(m=sal.date.dt.to_period("M"))
    per_sal = sal_month.groupby("person_id").agg(n=("amount", "size"), months=("m", "nunique"),
                                                 median=("amount", "median"),
                                                 formats=("text_key", "nunique"))
    X["slice"] = text_slice(X)

    findings = {
        "transactions": len(txns),
        "debit_share": round(float((~txns.is_credit).mean()), 3),
        "credits": len(X),
        "salary_transactions": int(X.is_salary.sum()),
        "persons": int(txns.person_id.nunique()),
        "persons_with_salary": int(sal.person_id.nunique()),
        "persons_two_accounts": int((txns.groupby("person_id").n_accounts.first() == 2).sum()),
        "persons_start_after_jan": int((months["min"] > "2025-01-31").sum()),
        "persons_end_before_dec": int((months["max"] < "2025-12-01").sum()),
        "salary_only_debit_or_credit": "salary is always a credit to the main (Privat) account",
        "salary_share_on_payday25": round(float(sal.on_payday25.mean()), 3),
        "persons_paid_twice_a_month": int((per_sal.n / per_sal.months > 1.5).sum()),
        "salary_median_monthly_payment": round(float(per_sal["median"].median())),
        "salary_payment_range": [round(float(sal.amount.min())), round(float(sal.amount.max()))],
        "text_formats_per_salary_person_median": float(per_sal["formats"].median()),
        "text_formats_per_salary_person_max": int(per_sal["formats"].max()),
        "salary_text_evidence": X[X.is_salary]["slice"].value_counts().to_dict(),
        "lon_keyword_not_salary": int(((X.kw_salary == 1) & ~X.is_salary).sum()),
        "lon_keyword_not_salary_persons": int(X[(X.kw_salary == 1) & ~X.is_salary].person_id.nunique()),
        "empty_text_credits": int((X.text_empty == 1).sum()),
        "empty_text_salary_share": round(float(X[X.text_empty == 1].is_salary.mean()), 3),
        "bonus_or_holiday_pay_months": sal[sal.kw_bonus_holiday == 1].date.dt.month.value_counts().to_dict(),
        "payday25_share_by_income_type": X.groupby("income_type").on_payday25.mean().round(2).to_dict(),
        "csn_amounts": X[X.income_type == "CSN (student aid)"].amount.value_counts().head(3).to_dict(),
        "person_main_income": person_segment(X).value_counts().to_dict(),
        "no_salary_person_main_income": person_segment(X)
        .drop(sal.person_id.unique()).value_counts().to_dict(),
    }
    (REPORTS / "eda_findings.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(findings, indent=2, ensure_ascii=False, default=str))

    plot_payday(X)
    plot_salary_text(X)


def plot_payday(X):
    """When in the month does each income type arrive? Salary and CSN share the 25th."""
    order = ["Salary", "CSN (student aid)", "A-kassa", "Pension", "Försäkringskassan", "Swish / private"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    rng = np.random.default_rng(0)
    for i, typ in enumerate(order):
        d = X[X.income_type == typ]
        color = COLORS["blue"] if typ == "Salary" else COLORS["muted"]
        ax.scatter(d.date.dt.day + rng.uniform(-0.3, 0.3, len(d)), i + rng.uniform(-0.25, 0.25, len(d)),
                   s=10, alpha=0.45, color=color, lw=0)
        ax.text(31.8, i, f"{d.on_payday25.mean():.0%}", va="center", color=COLORS["ink"], fontsize=9)
    ax.text(31.8, -0.9, "on 25th", va="center", color=COLORS["muted"], fontsize=8)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set(xlabel="Day of month", xlim=(0.5, 31.5), title="When does money arrive?")
    ax.grid(axis="y", visible=False)
    save(fig, REPORTS / "figures" / "payday.png")
    plt.close(fig)


def plot_salary_text(X):
    """What the text of a real salary looks like."""
    labels = {"salary keyword": "'Lön' / 'Salary'", "employer name": "Employer name only",
              "empty text": "Empty", "other text": "Other (bonus, truncated…)",
              "corrupted text": "Broken letters ('L N')"}
    counts = X[X.is_salary]["slice"].map(labels).value_counts()[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(counts.index, counts.values, color=COLORS["blue"], height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + 8, i, f"{v}  ({v / counts.sum():.0%})", va="center", fontsize=9, color=COLORS["ink"])
    ax.set(xlim=(0, counts.max() * 1.3), xlabel="Salary transactions",
           title="Text on the 876 salary payments")
    ax.grid(axis="y", visible=False)
    save(fig, REPORTS / "figures" / "salary_text.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
