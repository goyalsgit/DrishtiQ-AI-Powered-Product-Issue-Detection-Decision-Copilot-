"""Live ingestion: new database rows reach SQL metrics and search incrementally.

Uses a tiny fake embedding model so the test is fast and offline. The real
MiniLM path is exercised by scripts/demo_live_sync.py.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.database import ReviewStore
from src.embeddings import ReviewSearch
from src.sync import SourceMapping, sync_sqlite_source, watermark

WORDS = ['leak', 'bottle', 'soft', 'skin', 'smell', 'rash', 'pump', 'love']


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return ' '.join(tokens)


class FakeModel:
    """Bag-of-words vectors over a fixed vocabulary; enough to test plumbing."""
    tokenizer = FakeTokenizer()

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for text in texts:
            v = np.array([text.lower().count(w) for w in WORDS], dtype=float) + 1e-3
            rows.append(v / np.linalg.norm(v))
        return np.array(rows)


MAPPING = SourceMapping(table='customer_reviews', id_column='id', product='sku',
                        date='created_at', rating='stars', text='body')


class LiveSyncTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.source = self.dir / 'company.sqlite'
        with sqlite3.connect(self.source) as con:
            con.execute('CREATE TABLE customer_reviews (id INTEGER PRIMARY KEY, sku TEXT, '
                        'created_at TEXT, stars INTEGER, body TEXT)')
            con.executemany('INSERT INTO customer_reviews VALUES (?,?,?,?,?)', [
                (1, 'CREAM', '2024-03-02', 5, 'love it soft skin'),
                (2, 'CREAM', '2024-03-05', 4, 'soft skin nice smell'),
                (3, 'CREAM', '2024-03-09', 5, 'love love love'),
            ])
        self.store = ReviewStore(self.dir / 'drishtiq.sqlite')
        self.search = ReviewSearch(FakeModel(), self.dir / 'chroma')
        self.ds = self.store.create_live_dataset('company reviews')

    def sync(self):
        return sync_sqlite_source(self.source, MAPPING, self.store, self.search, self.ds)

    def add_source_review(self, row):
        with sqlite3.connect(self.source) as con:
            con.execute('INSERT INTO customer_reviews VALUES (?,?,?,?,?)', row)

    def test_first_sync_loads_everything(self):
        result = self.sync()
        self.assertEqual(result['added'], 3)
        self.assertEqual(self.store.metrics(self.ds, 'CREAM', '2024-03')[0], 3)
        self.assertTrue(self.search.ready(self.ds))

    def test_new_review_is_searchable_after_next_sync(self):
        self.sync()
        self.add_source_review((4, 'CREAM', '2024-03-20', 1, 'bottle leak pump broke leak'))
        result = self.sync()
        self.assertEqual(result['fetched'], 1)          # only the new row was read
        self.assertEqual(result['added'], 1)
        n, _, low, _ = self.store.metrics(self.ds, 'CREAM', '2024-03')
        self.assertEqual((n, low), (4, 1))              # SQL sees it immediately
        hits = self.search.search(self.store, self.ds, 'CREAM', '2024-03', 'leak bottle', limit=1)
        self.assertEqual(hits[0]['review_id'], 'SRC-4')  # search finds it
        self.assertTrue(self.search.ready(self.ds))

    def test_repeat_sync_adds_nothing(self):
        self.sync()
        self.assertEqual(self.sync()['fetched'], 0)
        self.assertEqual(self.store.metrics(self.ds, 'CREAM', '2024-03')[0], 3)

    def test_retry_after_crash_does_not_duplicate(self):
        self.sync()
        self.add_source_review((4, 'CREAM', '2024-03-20', 1, 'bottle leak'))
        self.sync()
        chunks = self.search.collection(self.ds).count()
        # Simulate a crash before the watermark was saved: roll it back and rerun.
        with self.store.connect() as con:
            con.execute('UPDATE sync_state SET last_source_id=0')
        result = self.sync()
        self.assertEqual(result['added'], 0)
        self.assertEqual(self.store.metrics(self.ds, 'CREAM', '2024-03')[0], 4)
        self.assertEqual(self.search.collection(self.ds).count(), chunks)
        self.assertTrue(self.search.ready(self.ds))

    def test_changed_content_for_same_id_is_rejected(self):
        self.sync()
        with self.store.connect() as con:
            con.execute('UPDATE sync_state SET last_source_id=0')
        with sqlite3.connect(self.source) as con:
            con.execute("UPDATE customer_reviews SET body='edited text' WHERE id=1")
        with self.assertRaises(ValueError):
            self.sync()

    def test_snapshots_stay_immutable(self):
        snapshot, _ = self.store.ingest(pd.DataFrame([{
            'review_id': 'A', 'product': 'X', 'date': '2024-01-01',
            'rating': 3, 'text': 'ok'}]), 'snapshot')
        with self.assertRaises(ValueError):
            self.store.append(snapshot, pd.DataFrame())

    def test_unsafe_column_names_rejected(self):
        bad = SourceMapping('customer_reviews; DROP TABLE x', 'id', 'sku',
                            'created_at', 'stars', 'body')
        with self.assertRaises(ValueError):
            bad.query()

    def test_watermark_advances(self):
        self.sync()
        self.assertEqual(watermark(self.store, self.ds,
                                   f'sqlite:{self.source}:customer_reviews'), 3)


if __name__ == '__main__':
    unittest.main()


class EvidenceMonthTests(unittest.TestCase):
    """With 'All months', a search result without a month gets it from SQLite."""

    def test_missing_month_is_filled_from_database(self):
        from langchain.messages import AIMessage
        from langchain.tools import tool
        from unittest.mock import patch
        from src.investigation import run_investigation

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ReviewStore(Path(tmp.name) / 'r.sqlite')
        ds, _ = store.ingest(pd.DataFrame([{'review_id': 'r1', 'product': 'P', 'date': '2024-03-05',
                                            'rating': 1, 'text': 'Broke fast'}]), 't')

        @tool
        def search_review_evidence(question: str) -> dict:
            """Old-style search result without a month field."""
            return {'scope': {'dataset_id': ds, 'product': 'P', 'month': None},
                    'reviews': [{'review_id': 'r1', 'rating': 1, 'text': 'Broke fast'}]}

        class Model:
            calls = 0
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                Model.calls += 1
                if Model.calls == 1:
                    return AIMessage(content='', tool_calls=[{'name': 'search_review_evidence',
                                     'args': {'question': 'x'}, 'id': '1'}])
                return AIMessage(content='done')

        with patch('src.investigation.make_tools', return_value=[search_review_evidence]), \
                patch('src.reporting.finalize_report'):
            result = run_investigation(Model(), store, ds, 'P', None, 'problems?')
        self.assertEqual(result['evidence'][0]['month'], '2024-03')
