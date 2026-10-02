"""Feature engineering for salary classification.

Only credits (incoming money) can be salary, so every feature is computed on a
person's credits. Two families of features:

* Text features: keyword groups on a normalised remittance text. The text is
  noisy (Swedish letters dropped, truncated, empty), so the patterns are
  written to survive that, e.g. "LÖN", "Lon", "L N" all count as "lön".
* Pattern features: how a transaction fits the person's own history. Salary is
  the large credit that comes back every month on about the same day with
  about the same amount. These features use no labels, only the person's
  transactions, so they are available for a new applicant at inference time.
"""

import numpy as np
import pandas as pd

# --- Text -------------------------------------------------------------------

MONTHS = r"jan|feb|mar|apr|maj|may|jun|jul|aug|sep|okt|oct|nov|dec"

# Each group is one regex on the normalised text. "[ö o]?" and "[ä a]?" accept a
# Swedish letter, its ASCII fallback, or a blank (corrupted encoding).
KEYWORDS = {
    "kw_salary": r"\bl[öo ]?n\b|l[öo ]?neins|\bsalary\b|\blon\b|payroll",
    "kw_bonus_holiday": r"\bbonus\b|semesterers",
    "kw_payout": r"\butbetalning\b",
    "kw_employer": r"\b(?:ab|hb)\b|\bgroup\b|bemanning",
    "kw_pension": r"pension",
    "kw_student_aid": r"\bcsn\b|centrala stud",
    "kw_unemployment": r"a-kassa|alfakassan|l[öo ]?negaranti",
    "kw_social_insurance": r"f[öo ]?rs[äa ]?kringskas",
    "kw_tax_refund": r"skatteverket|skatte[åa ]?terb",
    "kw_insurance": r"f[öo ]?rs[äa ]?kring|folksam|trygg-hansa|l[äa ]?nsf",
    "kw_swish": r"swish",
    "kw_private": r"tack|middag|fika|present|grattis|utl[äa ]?gg|resa|hyra",
    "kw_own_transfer": r"[öo ]?verf[öo ]?ring|fr[åa ]?n (?:spar|privat)|^se\d",
}
# Truncation cuts words short ("Försäkringskas", "Semesterersätt"), so every
# pattern above matches on a prefix rather than the full word.


def normalise_text(text: pd.Series) -> pd.Series:
    """Lowercase, collapse whitespace. Keeps digits (needed for account numbers)."""
    return text.fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def text_key(norm: pd.Series) -> pd.Series:
    """Group key for "same payer": drop dates, refs, ids, month names, punctuation.

    "LÖN 2025-03-25" and "Lön 2025-04-25" get the same key "lön".
    """
    key = norm.str.replace(r"(ref|id:?|p)\s*[\d-]+", " ", regex=True)
    key = key.str.replace(r"[\d/:+.-]+", " ", regex=True)
    key = key.str.replace(rf"\b({MONTHS})\b", " ", regex=True)
    return key.str.replace(r"\s+", " ", regex=True).str.strip()


def text_features(norm: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=norm.index)
    for name, pattern in KEYWORDS.items():
        out[name] = norm.str.contains(pattern, regex=True).astype(int)
    out["text_empty"] = (norm == "").astype(int)
    out["text_len"] = norm.str.len()
    # Encoding damage shows up as a single letter between blanks ("l n", "v st").
    out["text_corrupted"] = norm.str.contains(r"\b[a-z]\s[a-z]{1,2}\b").astype(int)
    return out


# --- Dates ------------------------------------------------------------------

# Swedish bank holidays that can move the 25th payday (2025). Extend per year.
SE_HOLIDAYS = np.array(
    ["2025-01-01", "2025-01-06", "2025-04-18", "2025-04-21", "2025-05-01",
     "2025-05-29", "2025-06-06", "2025-06-20", "2025-12-24", "2025-12-25",
     "2025-12-26", "2025-12-31"],
    dtype="datetime64[D]",
)


def payday_25(dates: pd.Series) -> pd.Series:
    """Swedish convention: salary on the 25th, or the last bank day before it."""
    d25 = dates.dt.to_period("M").dt.to_timestamp() + pd.Timedelta(days=24)
    rolled = np.busday_offset(d25.values.astype("datetime64[D]"), 0,
                              roll="backward", holidays=SE_HOLIDAYS)
    return pd.to_datetime(rolled)


# --- Pattern features -------------------------------------------------------

def _similar_count(g: pd.DataFrame, day_tol: int, amt_tol: float) -> np.ndarray:
    """For each credit: in how many *other* months does the person get a credit on
    about the same day of month with about the same amount?"""
    dom = g["dom"].to_numpy()
    amt = g["amount"].to_numpy()
    month = g["month"].to_numpy()
    acct = g["account_id"].to_numpy()

    day_diff = np.abs(dom[:, None] - dom[None, :])
    day_diff = np.minimum(day_diff, 31 - day_diff)  # the 30th and the 2nd are close
    ratio = amt[:, None] / amt[None, :]
    similar = ((day_diff <= day_tol)
               & (ratio >= 1 - amt_tol) & (ratio <= 1 / (1 - amt_tol))
               & (month[:, None] != month[None, :])
               & (acct[:, None] == acct[None, :]))
    # Count distinct months, not transactions.
    counts = np.zeros(len(g), dtype=int)
    for i in range(len(g)):
        counts[i] = len(set(month[similar[i]]))
    return counts


def pattern_features(cr: pd.DataFrame) -> pd.DataFrame:
    """cr: credits of all persons, with columns from load_transactions + text_key."""
    cr = cr.copy()
    cr["dom"] = cr["date"].dt.day
    cr["month"] = cr["date"].dt.to_period("M")
    out = pd.DataFrame(index=cr.index)

    out["log_amount"] = np.log10(cr["amount"])
    out["cents_zero"] = (cr["amount"] % 1 == 0).astype(int)
    out["is_savings_account"] = (cr["product"] == "Sparkonto").astype(int)
    out["weekday"] = cr["date"].dt.dayofweek
    out["day_of_month"] = cr["dom"]
    out["days_from_payday25"] = (cr["date"] - payday_25(cr["date"])).dt.days.to_numpy()
    out["is_payday25"] = (out["days_from_payday25"] == 0).astype(int)

    by_person = cr.groupby("person_id")
    # Coverage: an applicant with 5 months of data cannot have 11 repeats.
    n_months = by_person["month"].transform("nunique")
    person_max = by_person["amount"].transform("max")
    out["amount_vs_person_max"] = cr["amount"] / person_max
    out["amount_rank_in_month"] = (cr.groupby(["person_id", "month"])["amount"]
                                   .rank(ascending=False, method="min"))
    month_total = cr.groupby(["person_id", "month"])["amount"].transform("sum")
    out["share_of_month_inflow"] = cr["amount"] / month_total

    # Same payer (text key) seen in other months, and how stable it is.
    grp = cr.groupby(["person_id", "text_key"])
    out["key_n_months"] = grp["month"].transform("nunique")
    out["key_month_frac"] = out["key_n_months"] / n_months
    out["key_per_month"] = grp["amount"].transform("size") / out["key_n_months"]
    out["key_amount_cv"] = (grp["amount"].transform("std") / grp["amount"].transform("mean")).fillna(1.0)
    out["key_dom_std"] = grp["dom"].transform("std").fillna(15.0)
    out["amount_vs_key_median"] = cr["amount"] / grp["amount"].transform("median")

    # Same day + similar amount in other months, independent of text. This is what
    # finds a salary with an empty or damaged text, and what rejects a one-off
    # "lön" Swish from a friend.
    loose = pd.Series(0, index=cr.index)
    strict = pd.Series(0, index=cr.index)
    for _, g in by_person:
        loose.loc[g.index] = _similar_count(g, day_tol=3, amt_tol=0.35)
        strict.loc[g.index] = _similar_count(g, day_tol=2, amt_tol=0.10)
    out["recur_loose_months"] = loose
    out["recur_strict_months"] = strict
    out["recur_loose_frac"] = loose / n_months
    out["person_n_months"] = n_months
    # Split pay: exactly the same amount twice in a month (e.g. 10th and 25th).
    out["same_amount_same_month"] = (cr.groupby(["person_id", "month", "amount"])["amount"]
                                     .transform("size") > 1).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the credit rows of df with all model features and the label."""
    cr = df[df["is_credit"]].copy()
    cr["text_norm"] = normalise_text(cr["text"])
    cr["text_key"] = text_key(cr["text_norm"])
    feats = pd.concat([text_features(cr["text_norm"]), pattern_features(cr)], axis=1)
    meta = cr[["txn_id", "person_id", "date", "amount", "text", "text_key", "is_salary"]]
    return pd.concat([meta, feats], axis=1)


TEXT_COLS = list(KEYWORDS) + ["text_empty", "text_len", "text_corrupted"]
PATTERN_COLS = [
    "log_amount", "cents_zero", "is_savings_account", "weekday", "day_of_month",
    "days_from_payday25", "is_payday25", "amount_vs_person_max", "amount_rank_in_month",
    "share_of_month_inflow", "key_n_months", "key_month_frac", "key_per_month",
    "key_amount_cv", "key_dom_std", "amount_vs_key_median", "recur_loose_months",
    "recur_strict_months", "recur_loose_frac", "person_n_months", "same_amount_same_month",
]
FEATURE_COLS = TEXT_COLS + PATTERN_COLS
