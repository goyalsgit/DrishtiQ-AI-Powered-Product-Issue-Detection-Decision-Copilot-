import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src import analytics as a

class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.df, _ = a.clean(pd.DataFrame([
            ['a', 'P', '2026-01-01', 1, 'broken lid'],
            ['b', 'P', '2026-01-02', 5, 'good'],
            ['c', 'P', '2026-02-01', 2, 'leaked jar'],
        ], columns=['review_id','product','date','rating','text']))

    def test_metrics_and_empty(self):
        n, avg, neg, share = a.metrics(self.df)
        self.assertEqual((n, neg), (3, 2))
        self.assertAlmostEqual(avg, 8/3)
        self.assertAlmostEqual(share, 200/3)
        self.assertEqual(a.metrics(self.df.iloc[:0]), (0, None, 0, None))

    def test_invalid_and_duplicates(self):
        raw = self.df.copy()
        raw.loc[0, 'rating'] = 6
        cleaned, removed = a.clean(pd.concat([raw, raw.iloc[[1]]]))
        self.assertEqual((len(cleaned), removed), (2, 2))
        raw.loc[0, 'review_id'] = 'b'
        raw.loc[0, 'rating'] = 1
        with self.assertRaises(ValueError):
            a.clean(raw)

    def test_evidence(self):
        self.assertEqual(len(a.topics(self.df)['Packaging']), 2)
        self.assertEqual(a.retrieve(self.df, 'broken')['review_id'].tolist(), ['a'])
        self.assertTrue(a.retrieve(self.df, 'nonexistent').empty)

    def test_persistence_and_duplicate_decisions(self):
        old = a.DB
        try:
            with tempfile.TemporaryDirectory() as tmp:
                a.DB = Path(tmp) / 'test.sqlite'
                a.save_decision({'evidence':['a']}, 'Investigate', 'Approve', 'Tester')
                a.save_decision({'evidence':['a']}, 'Investigate', 'Reject', 'Tester')
                self.assertEqual(len(a.history()), 1)
                self.assertEqual(a.history().iloc[0]['decision'], 'Approve')
        finally:
            a.DB = old

if __name__ == '__main__':
    unittest.main()
