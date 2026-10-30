"""Demo: a company database receives new reviews; DrishtiQ picks them up.

Run: .venv-agent/bin/python -m scripts.demo_live_sync
Everything happens in a temporary folder; your real data is untouched.
Uses the real MiniLM model and the 932 public reviews as the "company" history.
"""
import json
import sqlite3
import tempfile
from pathlib import Path
from time import perf_counter

from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model
from src.sync import SourceMapping, sync_sqlite_source

ROOT = Path(__file__).resolve().parents[1]
MAPPING = SourceMapping(table='customer_reviews', id_column='id', product='sku',
                        date='created_at', rating='stars', text='body')
NEW_REVIEW = ('B0107QYW14', '2016-04-28', 1,
              'The pump stopped working after two days and the bottle leaked all over my bag.')


def timed(fn):
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start


def main():
    provenance = json.loads((ROOT / 'data/amazon_beauty_provenance.json').read_text())
    history = ReviewStore().read(provenance['dataset_id']).sort_values(['date', 'review_id'])
    report = {}
    with tempfile.TemporaryDirectory(prefix='drishtiq-sync-') as tmp:
        tmp = Path(tmp)
        company = tmp / 'company.sqlite'   # stands in for the company's database
        with sqlite3.connect(company) as con:
            con.execute('CREATE TABLE customer_reviews (id INTEGER PRIMARY KEY, sku TEXT, '
                        'created_at TEXT, stars INTEGER, body TEXT)')
            con.executemany('INSERT INTO customer_reviews(sku, created_at, stars, body) '
                            'VALUES (?,?,?,?)',
                            [(r.product, r.date.date().isoformat(), int(r.rating), r.text)
                             for r in history.itertuples()])
        store = ReviewStore(tmp / 'drishtiq.sqlite')
        search = ReviewSearch(load_model(), tmp / 'chroma')
        dataset = store.create_live_dataset('company reviews')

        first, t_first = timed(lambda: sync_sqlite_source(company, MAPPING, store, search, dataset))
        report['first_sync'] = {**first, 'seconds': round(t_first, 2)}

        question = 'Did the pump break or the bottle leak?'
        before = search.search(store, dataset, NEW_REVIEW[0], '2016-04', question, limit=3)
        metrics_before = store.metrics(dataset, NEW_REVIEW[0], '2016-04')

        with sqlite3.connect(company) as con:   # a customer posts one new review
            con.execute('INSERT INTO customer_reviews(sku, created_at, stars, body) '
                        'VALUES (?,?,?,?)', NEW_REVIEW)
        one, t_one = timed(lambda: sync_sqlite_source(company, MAPPING, store, search, dataset))
        report['one_new_review'] = {**one, 'seconds': round(t_one, 3)}

        after = search.search(store, dataset, NEW_REVIEW[0], '2016-04', question, limit=3)
        metrics_after = store.metrics(dataset, NEW_REVIEW[0], '2016-04')
        scope = store.read(dataset, NEW_REVIEW[0], '2016-04')
        new_id = scope[scope.text == NEW_REVIEW[3]].review_id.tolist()
        report['new_review_found_rank'] = next(
            (i for i, h in enumerate(after, 1) if h['review_id'] in new_id), None)
        report['found_before_sync'] = any(h['review_id'] in new_id for h in before)
        report['sql_reviews_before_after'] = [metrics_before[0], metrics_after[0]]
        report['sql_low_rating_before_after'] = [metrics_before[2], metrics_after[2]]

        _, t_nothing = timed(lambda: sync_sqlite_source(company, MAPPING, store, search, dataset))
        report['sync_with_nothing_new_seconds'] = round(t_nothing, 3)

        # Old approach for comparison: a new snapshot means re-indexing everything.
        full = ReviewSearch(search.model, tmp / 'chroma_full')
        _, t_full = timed(lambda: full.build(store, dataset))
        report['full_reindex_seconds'] = round(t_full, 2)
        report['speedup_incremental_vs_full'] = round(t_full / t_one, 1)

    out = ROOT / 'reports/live_sync_demo.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Saved', out)


if __name__ == '__main__':
    main()
