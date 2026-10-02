"""Score a transactions file (same nested JSON format) with the saved model.

    python src/predict.py data/transactions.json --out outputs/scored.csv

Prints each person's estimated average monthly salary. The `is_salary` label
is not needed in the input.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from salary.data import load_transactions
from salary.features import build_features
from salary.model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bundle = load_model()
    txns = load_transactions(args.path)
    X = build_features(txns)
    X["proba"] = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]

    scored = txns.merge(X[["txn_id", "proba"]], on="txn_id", how="left").fillna({"proba": 0.0})
    scored["pred_is_salary"] = scored["proba"] >= bundle["threshold"]

    months = scored.groupby("person_id")["date"].apply(lambda d: d.dt.to_period("M").nunique())
    salary = scored[scored.pred_is_salary].groupby("person_id")["amount"].sum()
    income = pd.DataFrame({"months": months, "monthly_salary": (salary / months).fillna(0.0).round(0)})
    print(income.to_string())

    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"Wrote {len(scored)} rows, {int(np.sum(scored.pred_is_salary))} salary, to {args.out}")


if __name__ == "__main__":
    main()
