"""Run EDA, person-grouped model selection, frozen holdout evaluation and export."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, fbeta_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance
from .salary import flatten, features, keyword, make_model, columns, scores, BASE, HISTORY

SEED = 42


def metrics(y, pred, score=None):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    result = dict(n=len(y), positives=int(y.sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                  precision=float(precision), recall=float(recall), f1=float(f1),
                  f05=float(fbeta_score(y, pred, beta=.5, zero_division=0)))
    if score is not None and y.sum():
        result['average_precision'] = float(average_precision_score(y, score))
    return result


def choose_threshold(y, score):
    candidates = []
    for t in np.arange(.05, .951, .01):
        m = metrics(y, score >= t)
        candidates.append(dict(threshold=round(float(t), 2), **m))
    # Precision breaks F0.5 ties, then the higher threshold (conservative).
    best = max(candidates, key=lambda x: (x['f05'], x['precision'], x['threshold']))
    return best, candidates


def bootstrap(df, iterations=2000):
    """Percentile intervals resampling people, preserving within-person dependence."""
    rng = np.random.default_rng(SEED)
    groups = [g for _, g in df.groupby('person_id')]
    counts = np.array([[sum(g.is_salary & g.predicted_salary), sum(~g.is_salary & g.predicted_salary),
                        sum(g.is_salary & ~g.predicted_salary)] for g in groups])
    values = []
    for _ in range(iterations):
        tp, fp, fn = counts[rng.integers(0, len(groups), len(groups))].sum(axis=0)
        # Undefined precision/recall remains undefined, not perfect.
        p = tp/(tp+fp) if tp+fp else np.nan
        r = tp/(tp+fn) if tp+fn else np.nan
        values.append([p, r])
    a = np.array(values)
    return {k: np.nanpercentile(a[:, j], [2.5, 97.5]).tolist() for j, k in enumerate(['precision', 'recall'])}


def analysis(df, f, out):
    people = df.groupby('person_id').agg(has_salary=('is_salary', 'max'), months=('date', lambda s: s.dt.to_period('M').nunique()),
        transactions=('transaction_id', 'size'), bank=('bank', 'first'), name=('account_name', 'first'))
    credits = df[df.credit]
    facts = dict(people=len(people), transactions=len(df), accounts=int(df.account_id.nunique()),
        salary_transactions=int(df.is_salary.sum()), credits=int(df.credit.sum()),
        salary_rate=float(df.is_salary.mean()), salary_share_of_credits=float(credits.is_salary.mean()),
        no_salary_people=int((~people.has_salary).sum()), distinct_holder_names=int(people.name.nunique()),
        min_observed_months=int(people.months.min()), max_observed_months=int(people.months.max()),
        partial_year_people=int((people.months < 12).sum()), missing_text=int(df.text.eq('').sum()),
        salary_missing_text=int((df.is_salary & df.text.eq('')).sum()),
        salary_without_salary_word=int((df.is_salary & (f.salary_word == 0) & (f.extra_pay == 0)).sum()),
        misleading_salary_words=int((~df.is_salary & df.credit & ((f.salary_word > 0) | (f.extra_pay > 0))).sum()),
        salary_outside_pay_window=int((df.is_salary & (f.pay_window == 0)).sum()),
        salary_debits=int((df.is_salary & ~df.credit).sum()),
        duplicate_rows=int(df.duplicated(['person_id', 'account_id', 'date', 'amount', 'credit', 'text']).sum()),
        salary_amount_quantiles=df.loc[df.is_salary, 'amount'].quantile([0,.25,.5,.75,1]).to_dict())
    people.to_csv(out/'people_summary.csv')
    df.assign(month=df.date.dt.strftime('%Y-%m')).groupby('month').agg(
        active_people=('person_id','nunique'), transactions=('transaction_id','size'), salaries=('is_salary','sum')).to_csv(out/'monthly_coverage.csv')
    df.groupby('bank').agg(people=('person_id','nunique'), transactions=('transaction_id','size'), salaries=('is_salary','sum')).to_csv(out/'bank_summary.csv')
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/transactions.json')
    parser.add_argument('--output', default='reports')
    parser.add_argument('--model', default='models/salary.joblib')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = flatten(args.data)
    f = features(df)
    y = df.is_salary.astype(int)
    person_y = df.groupby('person_id').is_salary.max().astype(int)
    dev_ids, test_ids = train_test_split(person_y.index.to_numpy(), test_size=.2, stratify=person_y, random_state=SEED)
    dev = df.person_id.isin(dev_ids); test = ~dev
    assert not set(dev_ids) & set(test_ids)
    assert not (df.is_salary & ~df.credit).any(), 'Credit gate contradicts labels'
    split = {'seed': SEED, 'development': sorted(dev_ids.tolist()), 'test': sorted(test_ids.tolist()), 'folds': []}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for tr, va in cv.split(dev_ids, person_y.loc[dev_ids]):
        train_ids, val_ids = dev_ids[tr], dev_ids[va]
        folds.append((df.person_id.isin(train_ids), df.person_id.isin(val_ids)))
        split['folds'].append({'train': sorted(train_ids.tolist()), 'validation': sorted(val_ids.tolist())})
    (out/'splits.json').write_text(json.dumps(split, indent=2))
    names = ['logistic', 'boost_no_history', 'boost_history']
    candidates, curves, oof_scores = {}, [], {}
    for name in names:
        oof = np.zeros(len(df)); cols = columns(name)
        for tr, va in folds:
            model = make_model(name)
            eligible = tr & df.credit
            model.fit(f.loc[eligible, cols], y[eligible])
            oof[va] = scores(model, df[va], f, cols)
        best, curve = choose_threshold(y[dev], oof[dev])
        candidates[name] = best
        curves.extend([dict(model=name, **m) for m in curve])
        oof_scores[name] = oof
    selected = max(names, key=lambda name: (candidates[name]['f05'], candidates[name]['precision']))
    threshold = candidates[selected]['threshold']
    cols = columns(selected)
    model = make_model(selected)
    model.fit(f.loc[dev & df.credit, cols], y[dev & df.credit])
    # All choices frozen above; test is scored once, no tuning below.
    score = scores(model, df[test], f, cols)
    pred = score >= threshold
    result_df = df[test].copy()
    result_df['salary_score'] = score
    result_df['predicted_salary'] = pred
    result_df['error'] = np.where(result_df.is_salary & ~pred, 'FN', np.where(~result_df.is_salary & pred, 'FP', 'correct'))
    result_df.to_csv(out/'test_predictions.csv', index=False)
    result_df[result_df.error != 'correct'].to_csv(out/'test_errors.csv', index=False)
    baseline = keyword(df, f)
    slices = {'all': np.ones(test.sum(), dtype=bool), 'credits': df[test].credit.to_numpy(),
        'missing_text': df[test].text.eq('').to_numpy(),
        'first_40_days': ((f.loc[test, 'history_days'] < 40) & df[test].credit).to_numpy(),
        'later_history': (f.loc[test, 'history_days'] >= 40).to_numpy(),
        'salary_word_present': (f.loc[test, 'salary_word'] > 0).to_numpy()}
    for bank in df.bank.unique():
        slices['bank_'+bank] = df[test].bank.eq(bank).to_numpy()
    slice_metrics = {k: metrics(y[test].to_numpy()[mask], pred[mask], score[mask]) for k,mask in slices.items() if mask.any()}
    monthly = result_df.assign(month=result_df.date.dt.strftime('%Y-%m'),
        actual_income=np.where(result_df.is_salary, result_df.amount, 0),
        predicted_income=np.where(pred, result_df.amount, 0)).groupby(['person_id','month']).agg(
            actual_income=('actual_income','sum'), predicted_income=('predicted_income','sum'))
    monthly['error_sek'] = monthly.predicted_income-monthly.actual_income
    monthly.to_csv(out/'test_monthly_income.csv')
    person = result_df.groupby('person_id').agg(actual=('is_salary','max'), predicted=('predicted_salary','max'))
    person.to_csv(out/'test_person_predictions.csv')
    income = {'observed_person_months': len(monthly), 'monthly_mae_sek': float(monthly.error_sek.abs().mean()),
        'monthly_bias_sek': float(monthly.error_sek.mean()),
        'false_positive_amount_sek': float(result_df.loc[result_df.error == 'FP', 'amount'].sum()),
        'missed_salary_amount_sek': float(result_df.loc[result_df.error == 'FN', 'amount'].sum()),
        'no_salary_test_people': int((~person.actual).sum()),
        'no_salary_people_flagged': int((~person.actual & person.predicted).sum()),
        'person_metrics': metrics(person.actual,person.predicted)}
    fold_results = []
    for i, (_, va) in enumerate(folds):
        fold_results.append({'fold': i+1, **metrics(y[va], oof_scores[selected][va] >= threshold, oof_scores[selected][va])})
    # Development-only permutation importance (one validation fold), not test tuning.
    tr, va = folds[0]; eligible = tr & df.credit; validation = va & df.credit
    probe = make_model(selected).fit(f.loc[eligible, cols], y[eligible])
    importance = permutation_importance(probe, f.loc[validation, cols], y[validation],
        scoring='average_precision', n_repeats=5, random_state=SEED)
    pd.DataFrame({'feature': cols, 'mean_ap_drop': importance.importances_mean,
        'std_ap_drop': importance.importances_std}).sort_values('mean_ap_drop', ascending=False).to_csv(out/'development_importance.csv', index=False)
    # Fixed-model chronological diagnostic: held-out people, Q4 only. No new fit/tuning.
    late = result_df.date >= '2025-10-01'
    summary = {'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'characteristics': analysis(df, f, out), 'selection_objective': 'Maximum pooled development OOF F0.5; precision then higher threshold breaks threshold ties',
        'development_candidates': candidates, 'selected_model': selected, 'threshold': threshold,
        'development_keyword': metrics(y[dev], baseline[dev]),
        'development_missing_text': metrics(y[dev & df.text.eq('')], oof_scores[selected][dev & df.text.eq('')] >= threshold), 'development_selected_folds': fold_results,
        'test': metrics(y[test], pred, score), 'test_keyword': metrics(y[test], baseline[test]),
        'test_person_bootstrap_95_percent': bootstrap(result_df), 'test_slices': slice_metrics,
        'test_q4_diagnostic': metrics(result_df.loc[late,'is_salary'], pred[late], score[late]),
        'income': income}
    (out/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    pd.DataFrame(curves).to_csv(out/'development_thresholds.csv', index=False)
    oof_df = df.loc[dev, ['transaction_id','person_id','is_salary']].copy()
    for name in names: oof_df[name+'_score'] = oof_scores[name][dev]
    oof_df.to_csv(out/'development_oof.csv', index=False)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'columns': cols, 'threshold': threshold, 'selected_model': selected,
                 'training_person_ids': sorted(dev_ids.tolist()), 'data_sha256': summary['data_sha256']}, args.model)
    print(json.dumps({k: summary[k] for k in ['characteristics','development_candidates','selected_model','threshold','test','test_keyword','test_person_bootstrap_95_percent','income']}, indent=2))


if __name__ == '__main__':
    main()
