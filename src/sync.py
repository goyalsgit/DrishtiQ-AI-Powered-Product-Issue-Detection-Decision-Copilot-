"""Feed reviews from a company database into DrishtiQ (SQL + search index).

How it works (incremental sync with a watermark):
  1. Remember the highest source row ID we have already copied ("watermark").
  2. Each run, ask the company DB only for rows with a larger ID.
  3. Rename their columns to ours (review_id, product, date, rating, text).
  4. Append them to the live dataset in SQLite  -> SQL metrics see them at once.
  5. Embed only those new reviews into Chroma   -> search can find them.
  6. Save the new watermark only after both steps succeed.

If a run crashes halfway, the watermark is unchanged and the next run retries.
Retrying is safe: already-stored reviews are skipped and vectors are upserted.

Limits: assumes the source has an increasing integer ID column. Edited or
deleted source reviews are not detected (that needs an updated_at column or
change data capture). The source connection is read-only.
"""
import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


@dataclass(frozen=True)
class SourceMapping:
    """Which source table/columns hold each DrishtiQ field."""
    table: str
    id_column: str        # increasing integer, used as the watermark
    product: str
    date: str
    rating: str
    text: str

    def query(self, mark='?'):
        """mark is the driver's placeholder: '?' for SQLite, '%s' for Postgres."""
        names = [self.table, self.id_column, self.product, self.date, self.rating, self.text]
        # Identifiers cannot be SQL parameters, so only allow plain names.
        if not all(_IDENT.match(n) for n in names):
            raise ValueError('Table and column names must be plain identifiers.')
        return (f'SELECT {self.id_column} AS source_id, {self.product} AS product, '
                f'{self.date} AS date, {self.rating} AS rating, {self.text} AS text '
                f'FROM {self.table} WHERE {self.id_column} > {mark} '
                f'ORDER BY {self.id_column} LIMIT {mark}')


def _create_state(con):
    con.execute('''CREATE TABLE IF NOT EXISTS sync_state (
        dataset_id TEXT NOT NULL, source TEXT NOT NULL,
        last_source_id INTEGER NOT NULL, synced_at TEXT NOT NULL,
        PRIMARY KEY(dataset_id, source))''')


def watermark(store, dataset_id, source):
    with store.connect() as con:
        _create_state(con)
        row = con.execute('SELECT last_source_id FROM sync_state WHERE dataset_id=? AND source=?',
                          (dataset_id, source)).fetchone()
    return row[0] if row else 0


def sync_sqlite_source(source_path, mapping, store, search, dataset_id, *, batch_size=1000):
    """Copy new rows from a SQLite file (read-only: mode=ro)."""
    src = sqlite3.connect(f'file:{source_path}?mode=ro', uri=True)

    def fetch(last, limit):
        return pd.read_sql_query(mapping.query('?'), src, params=(last, limit))
    try:
        return _copy_new_rows(fetch, f'sqlite:{source_path}:{mapping.table}',
                              store, search, dataset_id, batch_size)
    finally:
        src.close()


def sync_postgres_source(url, mapping, store, search, dataset_id, *, batch_size=1000):
    """Copy new rows from Postgres (e.g. Neon). The session is read-only, so
    DrishtiQ can never change the company database."""
    import psycopg
    with psycopg.connect(url, connect_timeout=15) as src:
        src.read_only = True

        def fetch(last, limit):
            with src.cursor() as cur:
                cur.execute(mapping.query('%s'), (last, limit))
                columns = [c.name for c in cur.description]
                return pd.DataFrame(cur.fetchall(), columns=columns)
        # Identify the source without storing the password in sync_state.
        host = url.split('@')[-1].split('?')[0]
        return _copy_new_rows(fetch, f'postgres:{host}:{mapping.table}',
                              store, search, dataset_id, batch_size)


def _copy_new_rows(fetch, source, store, search, dataset_id, batch_size):
    """The shared loop: fetch rows after the watermark, save, index, move the watermark."""
    totals = {'fetched': 0, 'added': 0, 'skipped_invalid': 0, 'passages_indexed': 0}
    while True:
        last = watermark(store, dataset_id, source)
        rows = fetch(last, batch_size)
        if rows.empty:
            break
        rows['date'] = rows['date'].astype(str)   # Postgres returns date objects
        # Prefix with the source name so IDs from different systems never collide.
        rows['review_id'] = 'SRC-' + rows['source_id'].astype(str)
        added, removed = store.append(dataset_id, rows)
        totals['passages_indexed'] += search.add_reviews(store, dataset_id, added)
        # Move the watermark only after the rows are saved and indexed.
        with store.connect() as con:
            _create_state(con)
            con.execute('INSERT OR REPLACE INTO sync_state VALUES(?,?,?,datetime("now"))',
                        (dataset_id, source, int(rows['source_id'].max())))
        totals['fetched'] += len(rows)
        totals['added'] += len(added)
        totals['skipped_invalid'] += removed
    return totals
