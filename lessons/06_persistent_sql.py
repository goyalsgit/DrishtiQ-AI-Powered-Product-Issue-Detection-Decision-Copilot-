"""Run: .venv/bin/python -m lessons.06_persistent_sql
Read real saved reviews and compare a SQL metric with a direct calculation.
"""
import json
from pathlib import Path
from src.database import ReviewStore

# 1. Open the application's existing SQLite database.
store = ReviewStore()
provenance_path = Path(__file__).resolve().parents[1] / 'data/amazon_beauty_provenance.json'
if not provenance_path.exists():
    raise SystemExit('Import the public sample first: python -m scripts.import_public_reviews')
dataset_id = json.loads(provenance_path.read_text())['dataset_id']

# 2. Read this snapshot only. Other datasets cannot enter the calculation.
reviews = store.read(dataset_id)
if reviews.empty:
    raise SystemExit('The public dataset is not in this database. Import it first.')
# Pick the busiest product-month for a useful inspection example.
counts = reviews.groupby(['product', 'month']).size()
product, month = counts.idxmax()

# 3. The method below executes fixed SQL with bound parameters.
n, average, negative, share = store.metrics(dataset_id, product, month)
print(f'Dataset: {dataset_id[:12]} | Product: {product} | Month: {month}')
print(f'SQL result: {n} reviews, {negative} low ratings, {share:.2f}% low-rating share')

# 4. Cross-check against the original rows, independently of the SQL aggregate.
selected = reviews[(reviews['product'] == product) & (reviews['month'] == month)]
assert n == len(selected)
assert negative == int((selected['rating'] <= 2).sum())
print('Verified against the source rows.')
print('With fewer than 20 reviews, our default spike alert is withheld.')
print('\nOpen src/database.py and find metrics():')
print('WHERE dataset_id=? AND product=? AND month=?')
print('Each ? receives a value separately. User input is not pasted into SQL code.')
