"""Score unlabeled nested JSON using the frozen development-trained artifact."""
import argparse
from pathlib import Path
import joblib
from .salary import flatten, features, scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--model', default='models/salary.joblib')
    p.add_argument('--output', default='predictions.csv')
    args = p.parse_args()
    artifact = joblib.load(args.model)  # Only load trusted model files.
    df = flatten(args.input, require_labels=False)
    f = features(df)
    df['salary_score'] = scores(artifact['model'], df, f, artifact['columns'])
    df['predicted_salary'] = df.salary_score >= artifact['threshold']
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df[['transaction_id', 'person_id', 'account_id', 'date', 'amount', 'salary_score', 'predicted_salary']].to_csv(args.output, index=False)
    print(f'Wrote {len(df)} predictions to {args.output}')


if __name__ == '__main__':
    main()
