"""Import 1429_1.csv (Datafiniti "Consumer Reviews of Amazon Products").

Run: .venv-agent/bin/python -m scripts.import_datafiniti
     .venv-agent/bin/python -m scripts.import_datafiniti --index   # also build search

The file has its own column names, so we map them to DrishtiQ's five columns:
  review_id <- hash of (id, asins, date, username, text)   (reviews.id is almost always empty)
  product   <- short product name + ASIN, e.g. "Echo (White) [B00L9EPT8O]"
  date      <- reviews.date
  rating    <- reviews.rating
  text      <- reviews.title + ". " + reviews.text
Usernames are used only inside the hash and are not stored.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.database import ReviewStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '1429_1.csv'


def short_name(names):
    """Most common product name for an ASIN, cut to something readable."""
    names = names.dropna().str.split(',,,').str[0].str.strip()
    if names.empty:
        return None
    name = names.value_counts().index[0]
    name = name.split(' - ')[0].split(',')[0].strip()
    return name[:45]


def convert(raw):
    labels = raw.groupby('asins')['name'].apply(short_name)
    product = raw['asins'].map(lambda a: f'{labels.get(a)} [{a}]' if labels.get(a) else f'[{a}]')
    identity = (raw['id'].astype(str) + '|' + raw['asins'].astype(str) + '|'
                + raw['reviews.date'].astype(str) + '|' + raw['reviews.username'].astype(str)
                + '|' + raw['reviews.text'].astype(str))
    title = raw['reviews.title'].fillna('').str.strip()
    text = raw['reviews.text'].fillna('').str.strip()
    return pd.DataFrame({
        'review_id': 'DF-' + identity.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:24]),
        'product': product,
        'date': raw['reviews.date'],
        'rating': raw['reviews.rating'],
        'text': (title + '. ' + text).where(title.ne(''), text),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', action='store_true', help='also build the search index')
    args = parser.parse_args()
    raw = pd.read_csv(SOURCE, low_memory=False)
    frame = convert(raw)
    store = ReviewStore()
    dataset_id, removed = store.ingest(frame, 'Amazon devices (Datafiniti 1429_1.csv)')
    print(f'Read {len(raw):,} rows, saved {len(raw) - removed:,}, excluded {removed} '
          f'(missing date/rating/text or duplicates).')
    print('Dataset ID:', dataset_id)
    print(store.product_table(dataset_id).head(12).to_string(index=False))
    if args.index:
        from time import perf_counter
        from src.embeddings import ReviewSearch, load_model
        start = perf_counter()
        passages = ReviewSearch(load_model()).build(store, dataset_id)
        print(f'Indexed {passages:,} passages in {perf_counter() - start:.0f}s')


if __name__ == '__main__':
    main()
