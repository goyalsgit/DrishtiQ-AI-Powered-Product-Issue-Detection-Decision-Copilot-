"""Check the Neon connection step by step.

Run: .venv-agent/bin/python -m scripts.check_neon
  1. Is Neon reachable, and what are the newest rows?
  2. Sync: copy new rows into DrishtiQ
  3. Can DrishtiQ find the newest review?
Never prints the password.
"""
import os
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('COMPANY_DB_URL', tomllib.loads(
    (ROOT / '.streamlit/secrets.toml').read_text()).get('COMPANY_DB_URL', ''))

from src import company_db                       # noqa: E402
from src.database import ReviewStore             # noqa: E402
from src.embeddings import ReviewSearch, load_model  # noqa: E402

print('Source:', company_db.description())
total = company_db._postgres('SELECT COUNT(*) FROM customer_reviews')[0][0]
print(f'\n1. Neon has {total} rows. Newest 3:')
newest = company_db._postgres(
    'SELECT id, created_at, stars, body FROM customer_reviews ORDER BY id DESC LIMIT 3')
for row_id, day, stars, body in newest:
    print(f'   id {row_id} | {day} | {stars} stars | {body[:70]}')

store, search = ReviewStore(), ReviewSearch(load_model())
dataset = store.create_live_dataset(company_db.live_name())
before = store.summary(dataset)['reviews']
_, result = company_db.sync(store, search)
print(f'\n2. Sync: fetched {result["fetched"]} new rows, added {result["added"]}. '
      f'DrishtiQ reviews {before} -> {store.summary(dataset)["reviews"]}')

newest_text = newest[0][3]
found = store.summary(dataset, text=newest_text[:30])['reviews']
hits = search.search(store, dataset, None, None, newest_text, limit=3)
print(f'\n3. Newest review in DrishtiQ database: {"yes" if found else "NO"}')
for rank, hit in enumerate(hits, 1):
    print(f'   search #{rank}: {hit["text"][:70]}')
