"""Dashboard SQL analytics: filters, live updates and safe text search."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.database import ReviewStore

ROWS = [
    ('r1', 'CREAM', '2024-01-05', 1, 'Bottle leaked everywhere'),
    ('r2', 'CREAM', '2024-01-20', 5, 'Love it'),
    ('r3', 'CREAM', '2024-02-02', 2, 'Smell is too strong'),
    ('r4', 'SOAP', '2024-02-10', 4, 'Nice lather, 100% happy'),
    ('r5', 'SOAP', '2024-03-01', 1, "Arrived broken_box, it's bad"),
]


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = ReviewStore(Path(tmp.name) / 'r.sqlite')
        frame = pd.DataFrame(ROWS, columns=['review_id', 'product', 'date', 'rating', 'text'])
        self.ds, _ = self.store.ingest(frame, 'test')

    def test_summary_all_products(self):
        s = self.store.summary(self.ds)
        self.assertEqual((s['reviews'], s['products'], s['low_rating_reviews']), (5, 2, 3))
        self.assertAlmostEqual(s['low_rating_share_percent'], 60.0)

    def test_filters_combine(self):
        s = self.store.summary(self.ds, products=['CREAM'], ratings=[1, 2])
        self.assertEqual(s['reviews'], 2)
        s = self.store.summary(self.ds, start='2024-02-01', end='2024-02-28T23:59:59')
        self.assertEqual(s['reviews'], 2)

    def test_product_table_ranks_worst_first(self):
        table = self.store.product_table(self.ds)
        self.assertEqual(table['product'].tolist(), ['CREAM', 'SOAP'])
        self.assertAlmostEqual(table.iloc[0]['low_rating_share_percent'], 66.7)

    def test_monthly_trend_and_distribution(self):
        self.assertEqual(self.store.monthly_trend(self.ds)['reviews'].tolist(), [2, 2, 1])
        dist = self.store.rating_distribution(self.ds).set_index('rating')['reviews']
        self.assertEqual(dist[1], 2)

    def test_text_search_is_literal_and_safe(self):
        self.assertEqual(self.store.summary(self.ds, text='LEAK')['reviews'], 1)
        # % and _ are matched literally, not as SQL wildcards.
        self.assertEqual(self.store.summary(self.ds, text='100%')['reviews'], 1)
        self.assertEqual(self.store.summary(self.ds, text='broken_box')['reviews'], 1)
        self.assertEqual(self.store.summary(self.ds, text='_')['reviews'], 1)
        # Injection attempt is just text that matches nothing; the table survives.
        self.assertEqual(self.store.summary(self.ds, text="x' OR 1=1; DROP TABLE reviews; --")['reviews'], 0)
        self.assertEqual(self.store.summary(self.ds)['reviews'], 5)

    def test_page_is_newest_first_and_bounded(self):
        page = self.store.filtered_page(self.ds, limit=2)
        self.assertEqual(page['review_id'].tolist(), ['r5', 'r4'])
        with self.assertRaises(ValueError):
            self.store.filtered_page(self.ds, limit=500)

    def test_request_reviews_find_wish_language(self):
        live = self.store.create_live_dataset('requests')
        self.store.append(live, pd.DataFrame([
            ('q1', 'CREAM', '2024-01-01', 5, 'Great, but I wish it came in a larger size'),
            ('q2', 'CREAM', '2024-02-01', 4, 'Would be nice to have a pump'),
            ('q3', 'CREAM', '2024-02-05', 5, 'Perfect as it is'),
            ('q4', 'SOAP', '2024-02-05', 3, 'I wish it lathered more'),
        ], columns=['review_id', 'product', 'date', 'rating', 'text']))
        self.assertEqual(self.store.request_reviews(live, 'CREAM')['review_id'].tolist(), ['q2', 'q1'])
        self.assertEqual(self.store.request_reviews(live, 'CREAM', '2024-01')['review_id'].tolist(), ['q1'])

    def test_metrics_all_months(self):
        n, _, low, _ = self.store.metrics(self.ds, 'CREAM', None)
        self.assertEqual((n, low), (3, 2))

    def test_new_live_reviews_show_up_immediately(self):
        live = self.store.create_live_dataset('shop')
        frame = pd.DataFrame(ROWS[:2], columns=['review_id', 'product', 'date', 'rating', 'text'])
        self.store.append(live, frame)
        self.assertEqual(self.store.summary(live)['reviews'], 2)
        self.store.append(live, pd.DataFrame([('r9', 'CREAM', '2024-04-01', 1, 'Leaked again')],
                                             columns=frame.columns))
        s = self.store.summary(live)
        self.assertEqual((s['reviews'], s['last_date'][:10]), (3, '2024-04-01'))


if __name__ == '__main__':
    unittest.main()


class CompareProductsToolTests(unittest.TestCase):
    def test_ranks_by_low_rating_count_with_quotable_examples(self):
        from src.tools import make_tools
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ReviewStore(Path(tmp.name) / 'r.sqlite')
        rows = ([('a%d' % i, 'BIG', '2024-01-0%d' % (1 + i % 9), 1, 'Broke fast') for i in range(5)]
                + [('b%d' % i, 'BIG', '2024-01-05', 5, 'Great') for i in range(95)]
                + [('c1', 'TINY', '2024-01-05', 1, 'Bad'), ('c2', 'TINY', '2024-01-06', 5, 'Ok')])
        ds, _ = store.ingest(pd.DataFrame(rows, columns=['review_id', 'product', 'date',
                                                         'rating', 'text']), 't')
        tool = next(t for t in make_tools(store, ds, None, None) if t.name == 'compare_products')
        out = tool.invoke({})
        # BIG has 5 complaints (5%), TINY 1 (50%): ranked by count, BIG first.
        self.assertEqual(out['products'][0]['product'], 'BIG')
        self.assertEqual(len([r for r in out['reviews'] if r['product'] == 'BIG']), 3)
        self.assertTrue(all(r['rating'] <= 2 and len(r['month']) == 7 for r in out['reviews']))
