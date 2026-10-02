"""Salary features and scoring. No labels, identities, or future rows in features."""
from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = ['log_amount', 'day', 'pay_window', 'salary_word', 'extra_pay',
        'excluded_income', 'private_transfer', 'company_suffix', 'missing_text']
HISTORY = ['prior_credit_count', 'history_days', 'similar_monthly_count',
           'monthly_amount_distance', 'same_text_monthly', 'same_text_prior_count']


def normalize(text):
    text = unicodedata.normalize('NFKD', str(text).lower())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z ]', ' ', text)).strip()


def flatten(path, require_labels=True):
    people = json.loads(Path(path).read_text())
    rows, seen = [], set()
    for person in people:
        pid = person['person_id']
        if pid in seen:
            raise ValueError(f'Duplicate person_id: {pid}')
        seen.add(pid)
        accounts = {a['uid']: a for a in person['accounts']}
        for i, t in enumerate(person['transactions']):
            aid = t['bank_account_id']
            if aid not in accounts:
                raise ValueError(f'Unknown account: {aid}')
            amount = float(t['transaction_amount']['amount'])
            currency = t['transaction_amount']['currency']
            direction = t['credit_debit_indicator']
            if currency != 'SEK' or direction not in ('CRDT', 'DBIT') or not np.isfinite(amount) or amount < 0:
                raise ValueError('Expected finite nonnegative SEK amount and CRDT/DBIT direction')
            label = t.get('is_salary')
            if require_labels and type(label) is not bool:
                raise ValueError('Expected Boolean is_salary')
            remittance = t.get('remittance_information')
            if remittance is None:
                text = ''
            elif isinstance(remittance, list):
                text = ' '.join(str(v) for v in remittance if v is not None)
            else:
                text = str(remittance)
            rows.append(dict(transaction_id=f'{pid}:{i:04d}', person_id=pid,
                             bank=person['bank']['name'], account_id=aid,
                             account_name=accounts[aid]['name'], amount=amount,
                             date=t['transaction_date'], credit=direction == 'CRDT',
                             text=text, is_salary=label))
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    return df


def features(df):
    """Strictly earlier dates, per person across accounts; same-day order is unknown.

    History is observable input, not learned state. Computing it before a person
    split is safe because no person shares rows with another split. For temporal
    evaluation, later rows cannot alter earlier features.
    """
    f = pd.DataFrame(index=df.index)
    n = df.text.map(normalize)
    f['log_amount'] = np.log1p(df.amount)
    f['day'] = df.date.dt.day
    f['pay_window'] = f.day.between(23, 28).astype(int)
    patterns = {
        'salary_word': r'\b(?:lon|l n|salary|payroll|loneutbetalning)\b',
        'extra_pay': r'semester|bonus|retroaktiv|overtid',
        'excluded_income': r'pension|\bcsn\b|forsakringskassan|a kassa|akassa|bidrag|skatte|aterbaring|ersattning fran',
        'private_transfer': r'swish|overfor|eget konto|sparkonto|privatkonto|present|utlagg|tack|middag',
        'company_suffix': r'\b(?:ab|hb|kommun|region)\b',
    }
    for name, pattern in patterns.items():
        f[name] = n.str.contains(pattern, regex=True).astype(int)
    f['missing_text'] = n.eq('').astype(int)
    for col in HISTORY:
        f[col] = 0.0
    f['monthly_amount_distance'] = 1.0
    for _, person in df.groupby('person_id', sort=False):
        incoming = person[person.credit].sort_values('date')
        for idx, row in incoming.iterrows():
            past = incoming[(incoming.date < row.date) & (incoming.date >= row.date - pd.Timedelta(days=180))]
            first_date = person.date.min()
            monthly = past[(row.date - past.date).dt.days.between(20, 40)]
            distance = (monthly.amount - row.amount).abs() / max(row.amount, 1)
            f.loc[idx, HISTORY] = [len(past), min((row.date-first_date).days, 180),
                                  int((distance <= .20).sum()),
                                  min(float(distance.min()), 1) if len(distance) else 1,
                                  int(bool(n[idx]) and (n.loc[monthly.index] == n[idx]).any()),
                                  int((n.loc[past.index] == n[idx]).sum()) if n[idx] else 0]
    return f.astype(float)


def keyword(df, f):
    return (df.credit & ((f.salary_word > 0) | (f.extra_pay > 0))).astype(int).to_numpy()


def make_model(name):
    if name == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
    return HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
        learning_rate=.06, l2_regularization=5, min_samples_leaf=20,
        early_stopping=False, random_state=42)


def columns(name):
    return BASE if name == 'boost_no_history' else BASE + HISTORY


def scores(model, df, f, cols):
    result = np.zeros(len(df))
    eligible = df.credit.to_numpy()
    if eligible.any():
        result[eligible] = model.predict_proba(f.loc[df.index[eligible], cols])[:, 1]
    return result
