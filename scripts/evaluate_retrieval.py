"""Run: .venv/bin/python -m scripts.evaluate_retrieval --k 3
Requires the public sample imported and the local model downloaded.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from src.analytics import retrieve
from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model, MODEL_NAME, INDEX_VERSION
from src.evaluation import score_retrieval, validate_cases, aggregate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--cases', type=Path, default=ROOT / 'tests/retrieval_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/retrieval_evaluation.json')
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be positive')
    content = args.cases.read_bytes()
    manifest = json.loads(content)
    store = ReviewStore()
    scopes = validate_cases(manifest, store)
    print('Loading local model; labels are assistant-reviewed, pending human review.', flush=True)
    search = ReviewSearch(load_model())
    search.build(store, manifest['dataset_id'])
    # Warm up before recording single-user per-query time. Not a load benchmark.
    first = manifest['cases'][0]
    search.search(store, manifest['dataset_id'], first['product'], first['month'], first['question'], limit=args.k)
    results = {'semantic': [], 'keyword': []}
    for case in manifest['cases']:
        scope = scopes[(case['product'], case['month'])]
        df = store.read(manifest['dataset_id'], case['product'], case['month'])
        for method in results:
            start = perf_counter()
            if method == 'semantic':
                hits = search.search(store, manifest['dataset_id'], case['product'], case['month'], case['question'], limit=args.k)
                ids = [hit['review_id'] for hit in hits]
                scores = [hit['similarity'] for hit in hits]
            else:
                ids = retrieve(df, case['question'], limit=args.k).review_id.tolist()
                scores = None
            duration = (perf_counter() - start) * 1000
            metrics = score_retrieval(ids, case['relevant_review_ids'], scope, args.k)
            results[method].append({
                'case_id': case['case_id'], 'question': case['question'],
                'product': case['product'], 'month': case['month'],
                'relevant_ids': case['relevant_review_ids'], 'returned_ids': ids,
                'similarities': scores, 'latency_ms': round(duration, 2), 'metrics': metrics,
            })
    summaries = {method: aggregate(rows) for method, rows in results.items()}
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'k': args.k,
        'dataset_id': manifest['dataset_id'], 'cases_sha256': hashlib.sha256(content).hexdigest(),
        'label_status': manifest['label_status'], 'case_count': len(manifest['cases']),
        'model': MODEL_NAME, 'index_version': INDEX_VERSION,
        'packages': {name: version(name) for name in ['sentence-transformers', 'chromadb', 'transformers', 'torch']},
        'limitations': [
            'Assistant-reviewed labels, pending independent human review; 24 development cases in four small scopes.',
            'No held-out evaluation or full RAG answer generation. These are retrieval diagnostics, not an accuracy claim.',
            'Precision@k divides by k, even if fewer results are returned; answerable cases only.',
            'No-evidence cases are excluded from precision/recall and reported separately.',
            'Unsupported return means results for a scope with no labelled evidence; it is not a generated hallucination.',
            'No abstention threshold is implemented. Single-user warm-query timings exclude model load and indexing.',
        ],
        'summary': summaries, 'results': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Retrieval evaluation — development diagnostic', '',
             '**Assistant-reviewed labels; independent human review pending. Not full RAG accuracy.**', '',
             f'Cases: {len(manifest["cases"])}. Requested results per query: {args.k}.', '',
             '| Method | Precision@k | Recall@k | MRR@k | Unsupported returns | Scope violations |',
             '|---|---:|---:|---:|---:|---:|']
    for method, summary in summaries.items():
        lines.append(f'| {method} | {summary["macro_precision_at_k"]:.1%} | {summary["macro_recall_at_k"]:.1%} | {summary["mean_reciprocal_rank_at_k"]:.3f} | {summary["unsupported_return_rate"]:.1%} | {summary["scope_violations"]} |')
    lines += ['', '## Cases to inspect', '', 'Missed relevant reviews and unsupported returns:', '']
    for method, rows in results.items():
        for row in rows:
            if row['metrics']['missing_relevant_ids'] or row['metrics']['unsupported_return']:
                lines.append(f'- {method}, {row["case_id"]}: {row["question"]} — missing={len(row["metrics"]["missing_relevant_ids"])}; unsupported return={row["metrics"]["unsupported_return"]}.')
    lines += ['', '## Interpretation limits', ''] + ['- ' + item for item in report['limitations']]
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summaries, indent=2))
    print('Report:', args.output)


if __name__ == '__main__':
    main()
