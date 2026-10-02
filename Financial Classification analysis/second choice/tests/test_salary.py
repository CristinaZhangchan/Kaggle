"""Behavioral checks for leakage, identity boundaries and the inference contract."""
import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.salary import flatten, features, scores, HISTORY


def example():
    return pd.DataFrame([
        dict(person_id='a', date=pd.Timestamp(d), amount=a, text=t, credit=True, is_salary=y)
        for d,a,t,y in [('2025-01-25',30000,'LÖN',True), ('2025-02-25',30200,'LÖN',True),
                         ('2025-02-25',30100,'LÖN',False), ('2025-03-25',29900,'LÖN',True)]])


class FeatureTests(unittest.TestCase):
    def test_labels_do_not_change_features(self):
        df = example()
        pd.testing.assert_frame_equal(features(df), features(df.assign(is_salary=~df.is_salary)))
        pd.testing.assert_frame_equal(features(df), features(df.drop(columns='is_salary')))

    def test_future_rows_do_not_change_past(self):
        df = example()
        pd.testing.assert_frame_equal(features(df.iloc[:3]), features(df).iloc[:3])

    def test_same_day_rows_do_not_see_each_other(self):
        f = features(example())
        self.assertEqual(f.loc[1, 'prior_credit_count'], 1)
        self.assertEqual(f.loc[2, 'prior_credit_count'], 1)

    def test_persons_are_isolated_and_order_independent(self):
        df = example()
        extra = df.assign(person_id='b', amount=30000)
        combined = pd.concat([df, extra], ignore_index=True)
        pd.testing.assert_frame_equal(features(df), features(combined).iloc[:4])
        pd.testing.assert_frame_equal(features(combined), features(combined.sample(frac=1,random_state=1)).sort_index())

    def test_recurrence_does_not_require_salary_label(self):
        f = features(example().assign(is_salary=False))
        self.assertEqual(f.loc[1,'similar_monthly_count'],1)
        self.assertEqual(f.loc[1,'same_text_monthly'],1)
        self.assertEqual(f.loc[0,'history_days'],0)

    def test_null_and_multiline_description_unlabeled(self):
        person = {'person_id':'x','bank':{'name':'Bank'},'accounts':[{'uid':'a','name':'Same Name'}],
            'transactions':[{'bank_account_id':'a','transaction_date':'2025-01-25',
                'transaction_amount':{'amount':123,'currency':'SEK'},'credit_debit_indicator':'CRDT',
                'remittance_information':None}]}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.json'; path.write_text(json.dumps([person]))
            df=flatten(path,require_labels=False)
            self.assertEqual(df.text.iloc[0],'')
            self.assertEqual(features(df).missing_text.iloc[0],1)
            with self.assertRaises(ValueError): flatten(path)
            person['transactions'][0]['remittance_information']=['LÖN','ACME AB']
            path.write_text(json.dumps([person]))
            self.assertEqual(flatten(path,False).text.iloc[0],'LÖN ACME AB')
            person['transactions'][0]['bank_account_id']='unknown'
            path.write_text(json.dumps([person]))
            with self.assertRaises(ValueError): flatten(path,False)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = joblib.load('models/salary.joblib')

    def test_group_splits_disjoint(self):
        split=json.loads(Path('reports/splits.json').read_text())
        self.assertFalse(set(split['development']) & set(split['test']))
        self.assertEqual(len(set(split['development']+split['test'])),100)
        validation=[]
        for fold in split['folds']:
            self.assertFalse(set(fold['train']) & set(fold['validation']))
            self.assertEqual(set(fold['train']+fold['validation']),set(split['development']))
            validation.extend(fold['validation'])
        self.assertEqual(sorted(validation),sorted(split['development']))
        self.assertEqual(self.artifact['training_person_ids'],split['development'])

    def test_saved_artifact_reproduces_test_predictions_without_labels(self):
        expected=pd.read_csv('reports/test_predictions.csv')
        df=flatten('data/transactions.json',False)
        df=df[df.person_id.isin(expected.person_id.unique())].drop(columns='is_salary')
        actual=scores(self.artifact['model'],df,features(df),self.artifact['columns'])
        np.testing.assert_allclose(actual,expected.salary_score,rtol=1e-10)
        self.assertTrue((actual[~df.credit.to_numpy()]==0).all())


if __name__ == '__main__':
    unittest.main()
