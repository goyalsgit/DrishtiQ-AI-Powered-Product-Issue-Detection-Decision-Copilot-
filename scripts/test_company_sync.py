"""End-to-end proof that a new review in the company database reaches DrishtiQ.

Run: .venv-agent/bin/python -m scripts.test_company_sync           (no Gemini)
     .venv-agent/bin/python -m scripts.test_company_sync --ask     (also asks Gemini)

1. create the demo company database (1,500 real Fire Tablet reviews)
2. sync -> SQL counts, search, themes
3. post ONE new review in the company database
4. check it is NOT in DrishtiQ yet
5. sync again -> check SQL count +1, search finds it, theme grouping includes it
6. (--ask) ask Gemini and check the answer quotes the new review
Results: reports/company_sync_test.json
"""
import argparse
import json
import time
import tomllib
from pathlib import Path

from src import company_db
from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model
from src.themes import summarize_themes

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = 'Fire Tablet [B018Y229OU]'
NEW_TEXT = ('Charger melted. The charger got extremely hot and melted at the plug '
            'after one week, and the tablet now smells like burnt plastic.')
QUESTION = 'Did anyone report the charger overheating or melting?'


def found(search, store, ds, question):
    hits = search.search(store, ds, PRODUCT, None, question, limit=6)
    ranks = [i for i, h in enumerate(hits, 1) if h['text'] == NEW_TEXT]
    return ranks[0] if ranks else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ask', action='store_true')
    args = parser.parse_args()
    store, search = ReviewStore(), ReviewSearch(load_model())
    report = {}

    if not company_db.COMPANY_DB.exists():
        report['company_rows_created'] = company_db.create()
    t = time.perf_counter()
    ds, first = company_db.sync(store, search)
    report['first_sync'] = {**first, 'seconds': round(time.perf_counter() - t, 1)}
    before = store.metrics(ds, PRODUCT, None)
    report['before'] = {'reviews': before[0], 'low_ratings': before[2],
                        'new_review_found_by_search': found(search, store, ds, QUESTION)}

    row = company_db.add_review(PRODUCT, 1, NEW_TEXT)
    report['posted_company_row'] = row
    report['in_drishtiq_before_sync'] = bool(
        store.summary(ds, products=[PRODUCT], text='charger melted')['reviews'])

    t = time.perf_counter()
    _, second = company_db.sync(store, search)
    report['second_sync'] = {**second, 'seconds': round(time.perf_counter() - t, 3)}
    after = store.metrics(ds, PRODUCT, None)
    themes = summarize_themes(store, search, ds, PRODUCT, None, 'problems')
    in_themes = any(e['text'] == NEW_TEXT for t in themes['themes'] for e in t['examples'])
    report['after'] = {'reviews': after[0], 'low_ratings': after[2],
                       'search_rank_of_new_review': found(search, store, ds, QUESTION),
                       'problem_reviews_grouped': themes['reviews_analyzed'],
                       'new_review_is_a_theme_example': in_themes}
    report['passed'] = (after[0] == before[0] + 1 and not report['in_drishtiq_before_sync']
                        and report['after']['search_rank_of_new_review'] is not None)

    if args.ask:
        from src.investigation import run_investigation
        from src.llm import make_llm
        secrets = tomllib.loads((ROOT / '.streamlit/secrets.toml').read_text())
        llm = make_llm(secrets.get('GEMINI_MODEL'), api_key=secrets['GOOGLE_API_KEY'])
        r = run_investigation(llm, store, ds, PRODUCT, None, QUESTION, search=search)
        content = (r.get('report') or {}).get('content', {})
        quotes = [c['quote'] for f in content.get('findings', []) for c in f['citations']]
        report['ask'] = {'status': r['status'], 'summary': content.get('summary'),
                         'quotes_new_review': any(q in NEW_TEXT for q in quotes), 'quotes': quotes}
    out = ROOT / 'reports/company_sync_test.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
