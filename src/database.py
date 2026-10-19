"""Persistent, immutable dataset snapshots. SQL is fixed and parameterized."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from src.analytics import clean

DEFAULT_DB = Path(__file__).resolve().parents[1] / 'data/reviews.sqlite'


class ReviewStore:
    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    is_demo INTEGER NOT NULL, row_count INTEGER NOT NULL,
                    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    dataset_id TEXT NOT NULL REFERENCES datasets(id),
                    review_id TEXT NOT NULL, product TEXT NOT NULL,
                    date TEXT NOT NULL, month TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    text TEXT NOT NULL CHECK(length(text)>0),
                    PRIMARY KEY(dataset_id, review_id)
                );
                CREATE INDEX IF NOT EXISTS reviews_scope
                    ON reviews(dataset_id, product, month);
            ''')

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.execute('PRAGMA foreign_keys=ON')
        try:
            with con:
                yield con
        finally:
            con.close()

    def ingest(self, raw, name, is_demo=False):
        df, removed = clean(raw)
        if df.empty:
            raise ValueError('No valid reviews remain after validation.')
        records = []
        for row in df.sort_values('review_id').itertuples():
            records.append((row.review_id, row.product, row.date.isoformat(),
                            row.month, int(row.rating), row.text))
        # Row order and source filename do not change the identity.
        identity = json.dumps({'schema': 1, 'demo': bool(is_demo), 'rows': records},
                              ensure_ascii=False, separators=(',', ':'))
        dataset_id = hashlib.sha256(identity.encode()).hexdigest()
        with self.connect() as con:
            con.execute('INSERT OR IGNORE INTO datasets(id,name,is_demo,row_count) VALUES(?,?,?,?)',
                        (dataset_id, name, int(is_demo), len(records)))
            con.executemany('INSERT OR IGNORE INTO reviews VALUES(?,?,?,?,?,?,?)',
                            [(dataset_id,) + r for r in records])
        return dataset_id, removed

    def datasets(self):
        with self.connect() as con:
            return pd.read_sql_query('SELECT * FROM datasets ORDER BY created DESC, id', con)

    def read(self, dataset_id, product=None, month=None):
        sql = 'SELECT review_id, product, date, month, rating, text FROM reviews WHERE dataset_id=?'
        args = [dataset_id]
        for column, value in [('product', product), ('month', month)]:
            if value is not None:
                sql += f' AND {column}=?'
                args.append(value)
        with self.connect() as con:
            df = pd.read_sql_query(sql + ' ORDER BY date,review_id', con, params=args)
        df['date'] = pd.to_datetime(df['date'])
        df['negative'] = df['rating'].le(2)
        return df

    def metrics(self, dataset_id, product, month):
        with self.connect() as con:
            n, avg, neg = con.execute('''SELECT COUNT(*), AVG(rating),
                COALESCE(SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END),0)
                FROM reviews WHERE dataset_id=? AND product=? AND month=?''',
                (dataset_id, product, month)).fetchone()
        return n, avg, neg, 100 * neg / n if n else None
