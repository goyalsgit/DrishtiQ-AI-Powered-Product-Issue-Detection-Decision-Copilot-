"""A product that did not exist before is picked up everywhere, with no code change."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.database import ReviewStore
from src.embeddings import ReviewSearch
from src.themes import problems_by_product, summarize_themes
from src.tools import make_tools
from tests.test_live_sync import FakeModel

COLUMNS = ['review_id', 'product', 'date', 'rating', 'text']


class NewProductTests(unittest.TestCase):
    def test_new_product_and_reviews_flow_through_sql_search_themes_and_tools(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ReviewStore(Path(tmp.name) / 'r.sqlite')
        search = ReviewSearch(FakeModel(), Path(tmp.name) / 'chroma')
        live = store.create_live_dataset('shop')
        old = [(f'o{i}', 'SOAP', '2024-01-05', 5, 'love soft skin') for i in range(10)]
        added, _ = store.append(live, pd.DataFrame(old, columns=COLUMNS))
        search.add_reviews(store, live, added)

        # A brand-new product arrives later, with complaints.
        new = [(f'n{i}', 'NEW PUMP', '2024-02-0%d' % (1 + i % 9), 1,
                ['pump leak bottle', 'bottle leak leak', 'rash skin rash'][i % 3]) for i in range(12)]
        added, _ = store.append(live, pd.DataFrame(new, columns=COLUMNS))
        search.add_reviews(store, live, added)

        self.assertEqual(store.metrics(live, 'NEW PUMP', None)[0], 12)                  # SQL
        hits = search.search(store, live, 'NEW PUMP', None, 'leak bottle', limit=3)      # search
        self.assertTrue(all(h['product'] == 'NEW PUMP' for h in hits))
        themes = summarize_themes(store, search, live, 'NEW PUMP', None, 'problems')     # themes
        self.assertEqual(themes['reviews_analyzed'], 12)
        top = problems_by_product(store, search, live)                                    # per product
        self.assertEqual(top[0]['product'], 'NEW PUMP')
        ranking = next(t for t in make_tools(store, live, None, None, search)
                       if t.name == 'compare_products').invoke({})                        # AI tool
        self.assertEqual(ranking['products'][0]['product'], 'NEW PUMP')


if __name__ == '__main__':
    unittest.main()
