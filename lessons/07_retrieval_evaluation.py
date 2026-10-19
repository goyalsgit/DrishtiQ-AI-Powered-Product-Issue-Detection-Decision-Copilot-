"""Run: .venv/bin/python -m lessons.07_retrieval_evaluation
Optional: add --case Q21 to inspect a question with no supporting evidence.
This reads saved results; it does not run an LLM or download a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
from src.database import ReviewStore

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', default='Q19')
args = parser.parse_args()
case_bytes = (ROOT / 'tests/retrieval_cases.json').read_bytes()
manifest = json.loads(case_bytes)
report_path = ROOT / 'reports/retrieval_evaluation.json'
if not report_path.exists():
    raise SystemExit('First run: .venv/bin/python -m scripts.evaluate_retrieval')
report = json.loads(report_path.read_text())
if report['cases_sha256'] != hashlib.sha256(case_bytes).hexdigest():
    raise SystemExit('Labels changed. Rerun scripts.evaluate_retrieval before inspecting results.')
case = next((c for c in manifest['cases'] if c['case_id'] == args.case), None)
if case is None:
    raise SystemExit('Unknown case ID. Use Q01 through Q24.')
result = next(r for r in report['results']['semantic'] if r['case_id'] == args.case)
reviews = ReviewStore().read(manifest['dataset_id'], case['product'], case['month'])

print('QUESTION:', case['question'])
print('SCOPE:', case['product'], case['month'])
print('LABEL RULE:', case['label_rationale'])
print('LABEL STATUS:', manifest['label_status'])
print('\nRead ALL scoped reviews and decide which actually support the question:')
for number, row in enumerate(reviews.itertuples(), 1):
    expected = row.review_id in case['relevant_review_ids']
    rank = result['returned_ids'].index(row.review_id) + 1 if row.review_id in result['returned_ids'] else '-'
    print(f'\nReview {number} | ID {row.review_id} | labelled relevant={expected} | search rank={rank}')
    print(row.text)

print('\nMETRICS:', json.dumps(result['metrics'], indent=2))
print('\nPrecision@3: relevant results returned / 3.')
print('Recall@3: relevant results returned / all labelled relevant reviews in this scope.')
print('No-evidence cases have no recall denominator; inspect unsupported_return instead.')
print('These are assistant labels to review, not a human-certified ground truth.')
