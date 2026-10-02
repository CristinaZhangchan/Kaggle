"""Load the nested transactions JSON into one flat table (one row per transaction)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "transactions.json"


def load_transactions(path: Path = DATA_PATH) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        people = json.load(f)

    rows = []
    for person in people:
        products = {a["uid"]: a["product"] for a in person["accounts"]}
        for t in person["transactions"]:
            rows.append({
                "person_id": person["person_id"],
                "bank": person["bank"]["name"],
                "n_accounts": len(person["accounts"]),
                "account_id": t["bank_account_id"],
                "product": products.get(t["bank_account_id"]),
                "date": t["transaction_date"],
                "amount": t["transaction_amount"]["amount"],
                "currency": t["transaction_amount"]["currency"],
                "is_credit": t["credit_debit_indicator"] == "CRDT",
                "text": " ".join(t.get("remittance_information") or []),
                # New data to score has no label, so default to False.
                "is_salary": bool(t.get("is_salary", False)),
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # Stable id so predictions can be joined back to the raw data.
    df = df.sort_values(["person_id", "date", "account_id"], kind="stable").reset_index(drop=True)
    df["txn_id"] = df.index
    return df
