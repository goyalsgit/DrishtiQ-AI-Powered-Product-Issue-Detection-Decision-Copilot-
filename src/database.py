"""Persistent, immutable dataset snapshots. SQL is fixed and parameterized."""
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from src.analytics import clean

DEFAULT_DB = Path(__file__).resolve().parents[1] / 'data/reviews.sqlite'


class ReviewStore:
    def __init__(self, path=None):
        self.path = Path(path if path is not None else os.getenv('DRISHTIQ_DB_PATH', DEFAULT_DB))
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

    # ---- Live datasets: one stable ID that grows as new reviews arrive. ----
    # Snapshots (above) stay immutable. Live IDs start with "live-" so the two
    # kinds can never be confused.

    def create_live_dataset(self, name):
        """Return a stable ID for a growing dataset. Same name -> same ID."""
        dataset_id = 'live-' + hashlib.sha256(name.encode()).hexdigest()[:24]
        with self.connect() as con:
            con.execute('INSERT OR IGNORE INTO datasets(id,name,is_demo,row_count) '
                        'VALUES(?,?,0,0)', (dataset_id, name))
        return dataset_id

    def append(self, dataset_id, raw):
        """Add new reviews to a live dataset. Returns the IDs actually added.

        Re-sending a review that is already stored is ignored (safe to retry).
        Sending a known review ID with different content is rejected, so a
        review can never silently change after it was indexed."""
        if not dataset_id.startswith('live-'):
            raise ValueError('Only live datasets accept new reviews; snapshots are immutable.')
        df, removed = clean(raw)
        records = [(row.review_id, row.product, row.date.isoformat(), row.month,
                    int(row.rating), row.text) for row in df.itertuples()]
        added = []
        with self.connect() as con:
            if con.execute('SELECT 1 FROM datasets WHERE id=?', (dataset_id,)).fetchone() is None:
                raise ValueError('Unknown live dataset. Call create_live_dataset first.')
            for record in records:
                existing = con.execute(
                    'SELECT product, date, month, rating, text FROM reviews '
                    'WHERE dataset_id=? AND review_id=?', (dataset_id, record[0])).fetchone()
                if existing is None:
                    con.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?)', (dataset_id,) + record)
                    added.append(record[0])
                elif tuple(existing) != record[1:]:
                    raise ValueError(f'Review {record[0]} already exists with different content.')
            con.execute('UPDATE datasets SET row_count=(SELECT COUNT(*) FROM reviews '
                        'WHERE dataset_id=?) WHERE id=?', (dataset_id, dataset_id))
        return added, removed

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

    # ---- Dashboard analytics: always read live from SQLite (no caching), so
    # reviews added by CSV append or database sync appear on the next refresh.

    @staticmethod
    def _filter_sql(dataset_id, products=None, start=None, end=None,
                    ratings=None, text=None):
        """Build a WHERE clause from fixed column names and bound parameters only."""
        clauses, args = ['dataset_id=?'], [dataset_id]
        if products:
            clauses.append(f'product IN ({",".join("?" * len(products))})')
            args += list(products)
        if start:
            clauses.append('date>=?')
            args.append(str(start))
        if end:
            clauses.append('date<=?')
            args.append(str(end))
        if ratings:
            clauses.append(f'rating IN ({",".join("?" * len(ratings))})')
            args += [int(r) for r in ratings]
        if text and text.strip():
            # Escape LIKE wildcards so user text is matched literally.
            needle = text.strip().lower().replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            clauses.append("lower(text) LIKE ? ESCAPE '\\'")
            args.append(f'%{needle}%')
        return ' AND '.join(clauses), args

    def summary(self, dataset_id, **filters):
        """Headline numbers for any filter combination."""
        where, args = self._filter_sql(dataset_id, **filters)
        with self.connect() as con:
            row = con.execute(f'''SELECT COUNT(*), AVG(rating),
                COALESCE(SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END),0),
                COUNT(DISTINCT product), MIN(date), MAX(date)
                FROM reviews WHERE {where}''', args).fetchone()
        n, avg, low, products, first, last = row
        return {'reviews': n, 'average_rating': avg, 'low_rating_reviews': low,
                'low_rating_share_percent': 100 * low / n if n else None,
                'products': products, 'first_date': first, 'last_date': last}

    def product_table(self, dataset_id, **filters):
        """One row per product, worst low-rating share first."""
        where, args = self._filter_sql(dataset_id, **filters)
        with self.connect() as con:
            return pd.read_sql_query(f'''SELECT product, COUNT(*) AS reviews,
                ROUND(AVG(rating), 2) AS average_rating,
                SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END) AS low_rating_reviews,
                ROUND(100.0 * SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END) / COUNT(*), 1)
                    AS low_rating_share_percent,
                MAX(date) AS latest_review
                FROM reviews WHERE {where} GROUP BY product
                ORDER BY low_rating_share_percent DESC, reviews DESC''', con, params=args)

    def monthly_trend(self, dataset_id, **filters):
        where, args = self._filter_sql(dataset_id, **filters)
        with self.connect() as con:
            return pd.read_sql_query(f'''SELECT month, COUNT(*) AS reviews,
                ROUND(AVG(rating), 2) AS average_rating,
                ROUND(100.0 * SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END) / COUNT(*), 1)
                    AS low_rating_share_percent
                FROM reviews WHERE {where} GROUP BY month ORDER BY month''', con, params=args)

    def rating_distribution(self, dataset_id, **filters):
        where, args = self._filter_sql(dataset_id, **filters)
        with self.connect() as con:
            return pd.read_sql_query(f'''SELECT rating, COUNT(*) AS reviews FROM reviews
                WHERE {where} GROUP BY rating ORDER BY rating''', con, params=args)

    def filtered_page(self, dataset_id, *, limit=25, offset=0, **filters):
        if (type(limit) is not int or not 1 <= limit <= 100
                or type(offset) is not int or offset < 0):
            raise ValueError('Use a page size of 1–100 and a nonnegative offset.')
        where, args = self._filter_sql(dataset_id, **filters)
        with self.connect() as con:
            return pd.read_sql_query(f'''SELECT date, product, rating, text, review_id
                FROM reviews WHERE {where} ORDER BY date DESC, review_id
                LIMIT ? OFFSET ?''', con, params=args + [limit, offset])

    def filtered_export(self, dataset_id, max_rows=50000, **filters):
        where, args = self._filter_sql(dataset_id, **filters)
        with self.connect() as con:
            return pd.read_sql_query(f'''SELECT review_id, product, date, rating, text
                FROM reviews WHERE {where} ORDER BY date DESC, review_id LIMIT ?''',
                con, params=args + [max_rows])

    def read_ids(self, dataset_id, review_ids):
        """Read specific reviews (used to index only newly added ones)."""
        ids = list(review_ids)
        frames = []
        with self.connect() as con:
            for start in range(0, len(ids), 500):  # stay under SQLite's variable limit
                part = ids[start:start + 500]
                marks = ','.join('?' * len(part))
                frames.append(pd.read_sql_query(
                    'SELECT review_id, product, date, month, rating, text FROM reviews '
                    f'WHERE dataset_id=? AND review_id IN ({marks})',
                    con, params=[dataset_id] + part))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=['review_id', 'product', 'date', 'month', 'rating', 'text'])

    # Phrases customers use when asking for something. Fixed list: the model
    # cannot change this SQL, it can only call the tool that runs it.
    REQUEST_PHRASES = ('wish', 'would be nice', 'would be great', 'would be better',
                       'would like', 'would love', 'should have', 'should be', 'should come',
                       'needs to', 'need a', 'needs a', 'if only', 'please make',
                       'please add', 'hope they', 'bigger size', 'larger size',
                       'more options', 'instead of', 'i want')

    def request_reviews(self, dataset_id, product, month=None, limit=8):
        """Reviews that contain request/wish language, newest first."""
        likes = ' OR '.join(['lower(text) LIKE ?'] * len(self.REQUEST_PHRASES))
        sql = (f'SELECT review_id, product, month, rating, text FROM reviews '
               f'WHERE dataset_id=? AND ({likes})')
        args = [dataset_id] + [f'%{p}%' for p in self.REQUEST_PHRASES]
        if product is not None:
            sql += ' AND product=?'
            args.append(product)
        if month is not None:
            sql += ' AND month=?'
            args.append(month)
        with self.connect() as con:
            return pd.read_sql_query(sql + ' ORDER BY date DESC LIMIT ?', con,
                                     params=args + [limit])

    def metrics(self, dataset_id, product, month):
        """Count, average rating and low ratings. month=None means all months."""
        sql = '''SELECT COUNT(*), AVG(rating),
            COALESCE(SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END),0)
            FROM reviews WHERE dataset_id=?'''
        args = [dataset_id]
        if product is not None:   # product=None means all products
            sql += ' AND product=?'
            args.append(product)
        if month is not None:
            sql += ' AND month=?'
            args.append(month)
        with self.connect() as con:
            n, avg, neg = con.execute(sql, args).fetchone()
        return n, avg, neg, 100 * neg / n if n else None

    def catalog(self, dataset_id):
        """Small SQL aggregate used by selectors/charts instead of loading all text."""
        with self.connect() as con:
            return pd.read_sql_query('''SELECT product, month, COUNT(*) AS total_reviews,
                AVG(rating) AS average_rating,
                SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END) AS low_rating_reviews,
                100.0 * SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END) / COUNT(*)
                    AS low_rating_share_percent
                FROM reviews WHERE dataset_id=?
                GROUP BY product, month ORDER BY product, month''', con, params=[dataset_id])

    def read_page(self, dataset_id, product, month, *, limit=25, offset=0):
        """Bounded source-review browsing with deterministic ordering."""
        if (type(limit) is not int or not 1 <= limit <= 100
                or type(offset) is not int or offset < 0):
            raise ValueError('Use a page size of 1–100 and a nonnegative offset.')
        with self.connect() as con:
            return pd.read_sql_query('''SELECT review_id, product, date, month, rating, text
                FROM reviews WHERE dataset_id=? AND product=? AND month=?
                ORDER BY date, review_id LIMIT ? OFFSET ?''',
                con, params=[dataset_id, product, month, limit, offset])
