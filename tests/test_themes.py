"""Theme grouping covers every review in scope and counts exactly."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.database import ReviewStore
from src.embeddings import ReviewSearch
from src.themes import summarize_themes
from tests.test_live_sync import FakeModel

TEXTS = ['leak bottle leak', 'bottle leak pump', 'leak leak bottle', 'pump leak bottle',
         'soft skin love', 'love soft skin', 'skin soft love love', 'love skin soft',
         'rash skin rash', 'rash rash skin', 'skin rash rash', 'rash skin skin rash']


class ThemeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = ReviewStore(Path(tmp.name) / 'r.sqlite')
        rows = [{'review_id': f'r{i}', 'product': 'P', 'date': f'2024-0{1 + i % 2}-10',
                 'rating': 5 if 'love' in t else 1, 'text': t} for i, t in enumerate(TEXTS)]
        rows.append({'review_id': 'other', 'product': 'Q', 'date': '2024-01-10',
                     'rating': 1, 'text': 'leak'})
        self.ds, _ = self.store.ingest(pd.DataFrame(rows), 't')
        self.search = ReviewSearch(FakeModel(), Path(tmp.name) / 'chroma')
        self.search.build(self.store, self.ds)

    def test_all_reviews_counted_once(self):
        out = summarize_themes(self.store, self.search, self.ds, 'P', None, 'all')
        self.assertEqual(out['reviews_analyzed'], 12)
        self.assertEqual(sum(t['reviews'] for t in out['themes']), 12)

    def test_focus_filters_by_rating_and_product(self):
        out = summarize_themes(self.store, self.search, self.ds, 'P', None, 'problems')
        self.assertEqual(out['reviews_analyzed'], 8)        # 1-star only, product P only
        ids = {e['review_id'] for t in out['themes'] for e in t['examples']}
        self.assertNotIn('other', ids)

    def test_month_scope(self):
        out = summarize_themes(self.store, self.search, self.ds, 'P', '2024-01', 'all')
        self.assertEqual(out['reviews_analyzed'], 6)


if __name__ == '__main__':
    unittest.main()
