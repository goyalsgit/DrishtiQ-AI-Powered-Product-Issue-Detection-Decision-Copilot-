"""Build a human labelling sheet for the v2 retrieval test.

Run: .venv-agent/bin/python -m scripts.make_label_sheet
Output: reports/labeling/label_sheet_v2.csv  (open in Excel/Numbers/Sheets)

For every question we pool the top results of keyword, semantic, hybrid and
reranked search, then shuffle them so the labeller cannot tell which method found a review.
Fill the "relevant" column with y or n. Leave nothing blank.
"""
import json
import random
from pathlib import Path

import pandas as pd

from src.analytics import retrieve
from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model
from src.retrieval_methods import CANDIDATES, Reranker, hybrid_ids

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / 'tests/retrieval_v2_questions.json'
OUTPUT = ROOT / 'reports/labeling/label_sheet_v2.csv'
POOL_DEPTH = 12  # results taken from each method per question


def main():
    spec = json.loads(QUESTIONS.read_text())
    dataset_id = spec['dataset_id']
    store = ReviewStore()
    search = ReviewSearch(load_model())
    search.build(store, dataset_id)  # no-op when the index is already ready
    reranker = Reranker()
    rng = random.Random(7)
    rows = []
    for q in spec['questions']:
        reviews = store.read(dataset_id, q['product'])
        by_id = reviews.set_index('review_id')
        semantic = search.search(store, dataset_id, q['product'], None,
                                 q['question'], limit=POOL_DEPTH)
        keyword = retrieve(reviews, q['question'], limit=POOL_DEPTH)
        # Pool every method we will compare, so none is scored on unjudged reviews.
        candidates = hybrid_ids(search, store, dataset_id, q['product'], reviews,
                                q['question'], CANDIDATES)
        reranked = reranker.rank(q['question'], candidates, by_id.text.to_dict())
        pool = list(dict.fromkeys(
            [h['review_id'] for h in semantic] + keyword.review_id.tolist()
            + hybrid_ids(search, store, dataset_id, q['product'], reviews,
                         q['question'], POOL_DEPTH)
            + [rid for rid, _ in reranked[:POOL_DEPTH]]))
        rng.shuffle(pool)
        for review_id in pool:
            row = by_id.loc[review_id]
            rows.append({'case_id': q['case_id'], 'question': q['question'],
                         'review_id': review_id, 'rating': int(row.rating),
                         'review_text': row.text.replace('\n', ' '),
                         'relevant': ''})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame(rows)
    sheet.to_csv(OUTPUT, index=False)
    per_q = sheet.groupby('case_id').size()
    print(f'Wrote {len(sheet)} rows for {len(per_q)} questions to {OUTPUT}')
    print(f'Reviews to judge per question: min {per_q.min()}, '
          f'average {per_q.mean():.1f}, max {per_q.max()}')


if __name__ == '__main__':
    main()
