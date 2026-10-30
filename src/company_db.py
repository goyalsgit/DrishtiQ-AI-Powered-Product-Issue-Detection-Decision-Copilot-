"""A demo "company database" that DrishtiQ syncs from.

data/company_reviews.sqlite stands in for a company's own review system. It has
its own table and column names (customer_reviews: id, sku, created_at, stars, body)
so the sync has to map them, exactly as with a real database.

  create()      fill it with the 1,500 most recent real Fire Tablet reviews from 1429_1.csv
  add_review()  insert one new review, like a customer posting on the website
  sync()        copy new rows into DrishtiQ's live dataset and index only those
"""
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from src.sync import SourceMapping, sync_sqlite_source

ROOT = Path(__file__).resolve().parents[1]
COMPANY_DB = ROOT / 'data/company_reviews.sqlite'
LIVE_NAME = 'Company reviews (live)'
MAPPING = SourceMapping(table='customer_reviews', id_column='id', product='sku',
                        date='created_at', rating='stars', text='body')


def create(rows=1500):
    from scripts.import_datafiniti import SOURCE, convert
    frame = convert(pd.read_csv(SOURCE, low_memory=False))
    frame = frame[frame['product'].str.startswith('Fire Tablet [B018Y229OU]')].dropna()
    frame['date'] = pd.to_datetime(frame['date'], utc=True).dt.strftime('%Y-%m-%d')
    frame = frame.sort_values('date').tail(rows)
    with sqlite3.connect(COMPANY_DB) as con:
        con.execute('DROP TABLE IF EXISTS customer_reviews')
        con.execute('CREATE TABLE customer_reviews (id INTEGER PRIMARY KEY, sku TEXT, '
                    'created_at TEXT, stars INTEGER, body TEXT)')
        con.executemany('INSERT INTO customer_reviews(sku, created_at, stars, body) VALUES (?,?,?,?)',
                        frame[['product', 'date', 'rating', 'text']].itertuples(index=False))
    return len(frame)


def add_review(sku, stars, body, created_at=None):
    with sqlite3.connect(COMPANY_DB) as con:
        cur = con.execute('INSERT INTO customer_reviews(sku, created_at, stars, body) VALUES (?,?,?,?)',
                          (sku, created_at or date.today().isoformat(), int(stars), body))
        return cur.lastrowid


def sync(store, search):
    dataset_id = store.create_live_dataset(LIVE_NAME)
    return dataset_id, sync_sqlite_source(COMPANY_DB, MAPPING, store, search, dataset_id)


def products():
    with sqlite3.connect(f'file:{COMPANY_DB}?mode=ro', uri=True) as con:
        return [r[0] for r in con.execute('SELECT DISTINCT sku FROM customer_reviews')]
