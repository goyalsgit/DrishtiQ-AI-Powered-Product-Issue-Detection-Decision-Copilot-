"""The "company database" DrishtiQ syncs reviews from.

Two modes, chosen automatically:
  * COMPANY_DB_URL is set (secrets.toml or environment) -> a real Postgres
    database, e.g. a free Neon project. Data there survives server restarts.
  * not set -> a local demo file, data/company_reviews.sqlite.

The table uses the company's own names (customer_reviews: id, sku, created_at,
stars, body), so the sync has to map them, exactly as with a real system.

  create()      fill the table with the 1,500 most recent real Fire Tablet reviews
  add_review()  insert one new review, like a customer posting on the website
  sync()        copy only new rows into DrishtiQ's live dataset and index them
"""
import os
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from src.sync import SourceMapping, sync_postgres_source, sync_sqlite_source

ROOT = Path(__file__).resolve().parents[1]
COMPANY_DB = ROOT / 'data/company_reviews.sqlite'
LIVE_NAME = 'Company reviews (live)'
MAPPING = SourceMapping(table='customer_reviews', id_column='id', product='sku',
                        date='created_at', rating='stars', text='body')


def database_url():
    """Postgres connection string, or None to use the local demo file."""
    try:
        import streamlit as st
        url = st.secrets.get('COMPANY_DB_URL')
    except Exception:          # not running in Streamlit, or no secrets file
        url = None
    return url or os.getenv('COMPANY_DB_URL')


def description():
    url = database_url()
    return f'Postgres at {url.split("@")[-1].split("/")[0]}' if url else f'local file {COMPANY_DB.name}'


def exists():
    if not database_url():
        return COMPANY_DB.exists()
    return bool(_postgres("SELECT to_regclass('customer_reviews') IS NOT NULL")[0][0])


def _postgres(sql, params=()):
    """Run one statement on the company Postgres; returns rows (if any)."""
    import psycopg
    with psycopg.connect(database_url(), connect_timeout=15) as con:   # commits on exit
        cursor = con.execute(sql, params)
        return cursor.fetchall() if cursor.description else []


def sample_reviews(rows=1500):
    """The newest real Fire Tablet reviews from 1429_1.csv, in company columns."""
    from scripts.import_datafiniti import SOURCE, convert
    frame = convert(pd.read_csv(SOURCE, low_memory=False))
    frame = frame[frame['product'].str.startswith('Fire Tablet [B018Y229OU]')].dropna()
    frame['date'] = pd.to_datetime(frame['date'], utc=True).dt.strftime('%Y-%m-%d')
    return frame.sort_values('date').tail(rows)[['product', 'date', 'rating', 'text']]


def create(rows=1500):
    data = list(sample_reviews(rows).itertuples(index=False))
    if database_url():
        import psycopg
        with psycopg.connect(database_url(), connect_timeout=15) as con:
            con.execute('DROP TABLE IF EXISTS customer_reviews')
            con.execute('CREATE TABLE customer_reviews (id SERIAL PRIMARY KEY, sku TEXT NOT NULL, '
                        'created_at DATE NOT NULL, stars INTEGER NOT NULL, body TEXT NOT NULL)')
            with con.cursor() as cur:
                cur.executemany('INSERT INTO customer_reviews(sku, created_at, stars, body) '
                                'VALUES (%s,%s,%s,%s)', data)
        return len(data)
    with sqlite3.connect(COMPANY_DB) as con:
        con.execute('DROP TABLE IF EXISTS customer_reviews')
        con.execute('CREATE TABLE customer_reviews (id INTEGER PRIMARY KEY, sku TEXT, '
                    'created_at TEXT, stars INTEGER, body TEXT)')
        con.executemany('INSERT INTO customer_reviews(sku, created_at, stars, body) VALUES (?,?,?,?)', data)
    return len(data)


def add_review(sku, stars, body, created_at=None):
    values = (sku, created_at or date.today().isoformat(), int(stars), body)
    if database_url():
        return _postgres('INSERT INTO customer_reviews(sku, created_at, stars, body) '
                         'VALUES (%s,%s,%s,%s) RETURNING id', values)[0][0]
    with sqlite3.connect(COMPANY_DB) as con:
        return con.execute('INSERT INTO customer_reviews(sku, created_at, stars, body) '
                           'VALUES (?,?,?,?)', values).lastrowid


def live_name():
    """Each source gets its own DrishtiQ dataset, so row IDs from the local demo
    file and from Neon can never clash (both start counting at 1)."""
    return 'Company reviews (Neon)' if database_url() else LIVE_NAME


def sync(store, search):
    dataset_id = store.create_live_dataset(live_name())
    if database_url():
        return dataset_id, sync_postgres_source(database_url(), MAPPING, store, search, dataset_id)
    return dataset_id, sync_sqlite_source(COMPANY_DB, MAPPING, store, search, dataset_id)


def products():
    sql = 'SELECT DISTINCT sku FROM customer_reviews ORDER BY sku'
    if database_url():
        return [r[0] for r in _postgres(sql)]
    with sqlite3.connect(f'file:{COMPANY_DB}?mode=ro', uri=True) as con:
        return [r[0] for r in con.execute(sql)]
