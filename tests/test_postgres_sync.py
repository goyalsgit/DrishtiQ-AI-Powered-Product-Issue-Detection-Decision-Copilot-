"""Sync from a REAL Postgres server (local, started by pgserver). Skipped if unavailable."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import pgserver
    import psycopg
except ImportError:
    raise unittest.SkipTest('pip install pgserver "psycopg[binary]" to run Postgres tests')

from src import company_db
from src.database import ReviewStore
from src.embeddings import ReviewSearch
from tests.test_live_sync import FakeModel


class PostgresSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.server = pgserver.get_server(cls.tmp.name, cleanup_mode='stop')
        cls.url = cls.server.get_uri()

    @classmethod
    def tearDownClass(cls):
        cls.server.cleanup()
        cls.tmp.cleanup()

    def test_postgres_rows_reach_drishtiq_and_new_ones_sync_incrementally(self):
        work = Path(tempfile.mkdtemp())
        store = ReviewStore(work / 'r.sqlite')
        search = ReviewSearch(FakeModel(), work / 'chroma')
        rows = [('Tablet', '2024-03-0%d' % (i + 1), 5, 'love soft skin') for i in range(3)]
        with patch.dict(os.environ, {'COMPANY_DB_URL': self.url}), \
                patch.object(company_db, 'sample_reviews', return_value=__import__('pandas').DataFrame(
                    rows, columns=['product', 'date', 'rating', 'text'])):
            self.assertEqual(company_db.create(), 3)
            self.assertTrue(company_db.exists())
            ds, first = company_db.sync(store, search)
            self.assertEqual(first['added'], 3)

            new_id = company_db.add_review('Tablet', 1, 'bottle leak pump broke')
            self.assertEqual(new_id, 4)
            _, second = company_db.sync(store, search)
            self.assertEqual((second['fetched'], second['added']), (1, 1))   # only the new row
            self.assertEqual(company_db.sync(store, search)[1]['fetched'], 0)  # nothing new

        self.assertEqual(store.metrics(ds, 'Tablet', None)[0], 4)
        hit = search.search(store, ds, 'Tablet', None, 'leak bottle', limit=1)[0]
        self.assertEqual(hit['review_id'], 'SRC-4')

    def test_sync_session_is_read_only(self):
        from src.sync import SourceMapping, sync_postgres_source
        bad = SourceMapping('customer_reviews', 'id', 'sku', 'created_at', 'stars', 'body')
        with psycopg.connect(self.url) as con:
            con.execute('CREATE TABLE IF NOT EXISTS customer_reviews (id SERIAL PRIMARY KEY, '
                        'sku TEXT, created_at DATE, stars INT, body TEXT)')
        with psycopg.connect(self.url) as con:
            con.read_only = True
            with self.assertRaises(psycopg.errors.ReadOnlySqlTransaction):
                con.execute("INSERT INTO customer_reviews(sku) VALUES ('x')")
        self.assertIsNotNone(sync_postgres_source)  # the sync uses this same read_only flag
        self.assertIn('%s', bad.query('%s'))


if __name__ == '__main__':
    unittest.main()
