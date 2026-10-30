"""Evaluate retrieval methods on the human-labelled v2 sheet.

Run after labelling:  .venv-agent/bin/python -m scripts.evaluate_retrieval_v2
Final exam (once):    .venv-agent/bin/python -m scripts.evaluate_retrieval_v2 --split test
Before labelling, the script still runs and reports how many results are judged.

Scope is a whole product (98-173 reviews), so random guessing scores low.
Reviews outside the labelled pool count as not relevant ("unjudged").
"""
import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model
from src.evaluation import score_retrieval, aggregate
from src.retrieval_methods import (CANDIDATES, Reranker, hybrid_ids,
                                   keyword_ids, semantic_ids)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / 'tests/retrieval_v2_questions.json'
SHEET = ROOT / 'reports/labeling/label_sheet_v2.csv'


def load_labels():
    sheet = pd.read_csv(SHEET, dtype=str).fillna('')
    sheet['relevant'] = sheet['relevant'].str.strip().str.lower()
    bad = ~sheet['relevant'].isin(['y', 'n', ''])
    if bad.any():
        raise SystemExit(f'Use only y or n in "relevant". Problem rows: {list(sheet.index[bad][:5])}')
    done = sheet['relevant'].ne('').all()
    gold = sheet[sheet.relevant == 'y'].groupby('case_id').review_id.apply(set).to_dict()
    judged = sheet.groupby('case_id').review_id.apply(set).to_dict()
    return done, gold, judged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Reranker score cutoff for "nothing found" (choose on dev).')
    args = parser.parse_args()

    spec = json.loads(QUESTIONS.read_text())
    questions = [q for q in spec['questions'] if args.split == 'all' or q['split'] == args.split]
    done, gold, judged = load_labels()
    dataset_id = spec['dataset_id']
    store = ReviewStore()
    search = ReviewSearch(load_model())
    reranker = Reranker()
    rng = random.Random(1)

    results = {m: [] for m in ['random', 'keyword', 'semantic', 'hybrid', 'rerank']}
    latency = {m: [] for m in results}
    unjudged = {m: 0 for m in results}
    rerank_top_scores = []
    for q in questions:
        reviews = store.read(dataset_id, q['product'])
        scope = set(reviews.review_id)
        texts = reviews.set_index('review_id').text.to_dict()
        relevant = gold.get(q['case_id'], set())
        runs = {}
        t = perf_counter(); runs['random'] = rng.sample(sorted(scope), args.k); latency['random'].append(perf_counter() - t)
        t = perf_counter(); runs['keyword'] = keyword_ids(reviews, q['question'], args.k); latency['keyword'].append(perf_counter() - t)
        t = perf_counter(); runs['semantic'] = semantic_ids(search, store, dataset_id, q['product'], q['question'], args.k); latency['semantic'].append(perf_counter() - t)
        t = perf_counter(); runs['hybrid'] = hybrid_ids(search, store, dataset_id, q['product'], reviews, q['question'], args.k); latency['hybrid'].append(perf_counter() - t)
        t = perf_counter()
        candidates = hybrid_ids(search, store, dataset_id, q['product'], reviews, q['question'], CANDIDATES)
        ranked = reranker.rank(q['question'], candidates, texts)
        if args.threshold is not None:
            ranked = [(rid, s) for rid, s in ranked if s > args.threshold]
        runs['rerank'] = [rid for rid, _ in ranked[:args.k]]
        latency['rerank'].append(perf_counter() - t)
        rerank_top_scores.append({'case_id': q['case_id'], 'answerable': bool(relevant),
                                  'top_score': round(ranked[0][1], 3) if ranked else None})
        for method, ids in runs.items():
            unjudged[method] += len(set(ids) - judged.get(q['case_id'], set()))
            metrics = score_retrieval(ids, relevant, scope, args.k)
            results[method].append({'case_id': q['case_id'], 'returned_ids': ids, 'metrics': metrics})

    print(f'\nSplit: {args.split} | questions: {len(questions)} | k={args.k} | '
          f'labels complete: {done}')
    if not done:
        print('Labels are not finished yet. Only "unjudged" below is meaningful.\n')
    header = f'{"method":9} {"recall":>7} {"precision":>9} {"MRR":>6} {"hit":>6} {"no-answer handled":>17} {"unjudged":>8} {"median ms":>9}'
    print(header)
    summary = {}
    for method, rows in results.items():
        s = aggregate(rows)
        handled = None if s['unsupported_return_rate'] is None else 1 - s['unsupported_return_rate']
        ms = sorted(latency[method])[len(latency[method]) // 2] * 1000
        summary[method] = {**s, 'no_answer_handled': handled, 'unjudged_results': unjudged[method],
                           'median_latency_ms': round(ms, 1)}
        fmt = lambda v, pct=True: '   n/a' if v is None else (f'{v:.1%}' if pct else f'{v:.3f}')
        print(f'{method:9} {fmt(s["macro_recall_at_k"]):>7} {fmt(s["macro_precision_at_k"]):>9} '
              f'{fmt(s["mean_reciprocal_rank_at_k"], False):>6} {fmt(s["hit_rate_at_k"]):>6} '
              f'{fmt(handled):>17} {unjudged[method]:>8} {ms:>9.1f}')
    out = ROOT / f'reports/retrieval_v2_{args.split}.json'
    out.write_text(json.dumps({'split': args.split, 'k': args.k, 'threshold': args.threshold,
                               'labels_complete': bool(done), 'summary': summary,
                               'rerank_top_scores': rerank_top_scores}, indent=2) + '\n')
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
