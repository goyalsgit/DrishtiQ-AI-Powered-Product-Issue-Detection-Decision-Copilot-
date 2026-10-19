"""Integration check with real local embeddings, isolated temporary databases."""
import tempfile
from pathlib import Path
import pandas as pd
from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model

with tempfile.TemporaryDirectory() as tmp:
    store = ReviewStore(Path(tmp) / 'reviews.sqlite')
    raw = pd.DataFrame([
        ['r1', 'P', '2026-01-01', 1, 'The bottle leaked in the shipping box.'],
        ['r2', 'P', '2026-01-02', 5, 'Wonderful fragrance.'],
        ['r3', 'Q', '2026-01-01', 1, 'The bottle leaked in the shipping box.'],
        ['r4', 'P', '2026-02-01', 1, 'The bottle leaked in the shipping box.'],
        ['r5', 'P', '2026-01-01', 1, 'Long review. ' * 250 + 'The lid was broken.'],
    ], columns=['review_id', 'product', 'date', 'rating', 'text'])
    dataset, _ = store.ingest(raw, 'test')
    search = ReviewSearch(load_model(), Path(tmp) / 'vectors')
    count = search.build(store, dataset)
    assert count > 5, 'Long reviews should be split into passages'
    assert search.build(store, dataset) == count
    hits = search.search(store, dataset, 'P', '2026-01', 'leaking bottle')
    assert hits[0]['review_id'] == 'r1'
    ids = [hit['review_id'] for hit in hits]
    assert set(ids) <= {'r1', 'r2', 'r5'} and len(ids) == len(set(ids))
    assert not search.search(store, dataset, 'Q', '2025-01', 'leaking')
    changed = raw.iloc[:1].copy()
    changed.loc[0, 'text'] = 'Completely different dataset.'
    other, _ = store.ingest(changed, 'other')
    assert not search.ready(other)
    search.build(store, other)
    assert search.search(store, other, 'P', '2026-01', 'different')[0]['text'] == 'Completely different dataset.'
    print('PASS: long-review chunking, index reuse, product/month/dataset isolation, unique review results, source lookup.')
