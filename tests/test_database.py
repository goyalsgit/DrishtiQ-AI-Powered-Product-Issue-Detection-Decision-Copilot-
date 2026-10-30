import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd
from src.database import ReviewStore


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'reviews.sqlite'
        self.store = ReviewStore(self.path)
        self.raw = pd.DataFrame([
            ['001', 'P', '2026-01-01', 1, 'broken lid'],
            ['002', 'P', '2026-01-02', 5, 'good'],
            ['003', 'Q', '2026-02-01', 2, 'leaked jar'],
        ], columns=['review_id', 'product', 'date', 'rating', 'text'])

    def tearDown(self):
        self.tmp.cleanup()

    def test_persistence_and_reordered_idempotent_import(self):
        first, removed = self.store.ingest(self.raw, 'source.csv')
        second, _ = self.store.ingest(self.raw.iloc[::-1], 'renamed.csv')
        self.assertEqual(first, second)
        reopened = ReviewStore(self.path)
        self.assertEqual(len(reopened.read(first)), 3)
        self.assertEqual(len(reopened.datasets()), 1)
        self.assertEqual(reopened.read(first).iloc[0].review_id, '001')
        self.assertEqual(removed, 0)

    def test_scope_metrics_and_injection(self):
        dataset, _ = self.store.ingest(self.raw, 'source.csv')
        self.assertEqual(self.store.metrics(dataset, 'P', '2026-01'), (2, 3.0, 1, 50.0))
        self.assertEqual(self.store.metrics(dataset, "P' OR 1=1 --", '2026-01'), (0, None, 0, None))
        self.assertTrue(self.store.read(dataset, 'P', '2026-02').empty)

    def test_changed_data_creates_separate_snapshot(self):
        first, _ = self.store.ingest(self.raw, 'source.csv')
        self.raw.loc[0, 'text'] = 'changed'
        second, _ = self.store.ingest(self.raw, 'source.csv')
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.read(first).iloc[0].text, 'broken lid')

    def test_invalid_import_does_not_create_partial_dataset(self):
        self.raw.loc[1, 'review_id'] = '001'
        with self.assertRaises(ValueError):
            self.store.ingest(self.raw, 'bad.csv')
        self.assertTrue(self.store.datasets().empty)

    def test_catalog_matches_sql_metrics_and_dataset_scope(self):
        dataset, _ = self.store.ingest(self.raw, 'source.csv')
        changed = self.raw.copy()
        changed.loc[0, 'rating'] = 4
        self.store.ingest(changed, 'other.csv')
        catalog = self.store.catalog(dataset)
        self.assertEqual(int(catalog.total_reviews.sum()), 3)
        row = catalog[catalog['product'].eq('P')].iloc[0]
        self.assertEqual(row.total_reviews, 2)
        self.assertEqual(row.average_rating, 3)
        self.assertEqual(row.low_rating_share_percent, 50)

    def test_paging_is_stable_bounded_and_scoped(self):
        dataset, _ = self.store.ingest(self.raw, 'source.csv')
        first = self.store.read_page(dataset, 'P', '2026-01', limit=1)
        second = self.store.read_page(dataset, 'P', '2026-01', limit=1, offset=1)
        self.assertEqual(first.iloc[0].review_id, '001')
        self.assertEqual(second.iloc[0].review_id, '002')
        self.assertTrue(self.store.read_page(dataset, "P' OR 1=1 --", '2026-01').empty)
        for arguments in [{'limit': 101}, {'limit': 0}, {'offset': -1}, {'limit': True}]:
            with self.assertRaises(ValueError):
                self.store.read_page(dataset, 'P', '2026-01', **arguments)

    def test_environment_database_path_survives_reopen(self):
        path = str(Path(self.tmp.name) / 'configured.sqlite')
        with patch.dict('os.environ', {'DRISHTIQ_DB_PATH': path}):
            saved = ReviewStore()
            dataset, _ = saved.ingest(self.raw, 'source.csv')
            reopened = ReviewStore()
            self.assertEqual(reopened.path, Path(path))
            self.assertEqual(len(reopened.read(dataset)), 3)
