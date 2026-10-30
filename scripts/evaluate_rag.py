"""Measure answer quality end to end: old (search only) vs new (all-review themes).

Run: .venv-agent/bin/python -m scripts.evaluate_rag
Data: Datafiniti Amazon devices, Fire Tablet [B018Y229OU], all months.
Saves reports/rag_evaluation.json. Uses live Gemini; spaced for the free tier.

Metrics per answer
  coverage          reviews the answer is based on / reviews in scope
  reviews_read      reviews passed to Gemini as quotable evidence
  findings          separate points in the answer
  cited_reviews     distinct reviews quoted
  words_per_finding detail level
  citations_valid   quotes/IDs/months match the database (automatic check)
  grounded_share    findings a judge says the quotes really support
  relevance_1to5    judge score: does the answer address the question
  seconds, tokens   cost
The judge is the same Gemini model, so treat its scores as indicative only.
"""
import json
import pathlib
import time
import tomllib

from langchain.messages import HumanMessage

import src.investigation as investigation
from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model
from src.llm import make_llm

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRODUCT = 'Fire Tablet [B018Y229OU]'
QUESTIONS = ['What are the main problems customers face?',
             'What are the positive points of the reviews?',
             'What do customers say about battery life?',
             'Why do people give 1 or 2 stars?']

JUDGE = """You grade an AI answer about customer reviews. Reply with JSON only:
{"relevance": 1-5, "findings": [{"index": i, "supported": true/false}]}
relevance: 5 = fully answers the question, 1 = off topic.
supported: true only if the quoted review text actually supports the statement.
QUESTION: %s
FINDINGS: %s"""


def judge(llm, question, findings):
    if not findings:
        return None, None
    items = [{'index': i, 'statement': f['statement'],
              'quotes': [c['quote'] for c in f['citations']]} for i, f in enumerate(findings)]
    reply = llm.invoke([HumanMessage(content=JUDGE % (question, json.dumps(items)))])
    text = investigation.answer_text(reply.content).strip().strip('`').removeprefix('json')
    data = json.loads(text[text.find('{'):text.rfind('}') + 1])
    supported = [f['supported'] for f in data['findings']]
    return data['relevance'], round(sum(supported) / len(supported), 2) if supported else None


def run(llm, store, search, dataset, question, use_themes):
    original = investigation.make_tools

    def tools(*args, **kwargs):
        found = original(*args, **kwargs)
        return found if use_themes else [t for t in found if t.name != 'summarize_review_themes']

    investigation.make_tools = tools
    try:
        start = time.perf_counter()
        r = investigation.run_investigation(llm, store, dataset, PRODUCT, None, question, search=search)
        seconds = time.perf_counter() - start
    finally:
        investigation.make_tools = original
    in_scope = store.metrics(dataset, PRODUCT, None)[0]
    themes = [t['data'] for t in r['tool_results'] if t['tool'] == 'summarize_review_themes'
              and t['status'] == 'success']
    based_on = max([t['reviews_analyzed'] for t in themes] or [len(r['evidence'])])
    findings = (r.get('report') or {}).get('content', {}).get('findings', [])
    time.sleep(15)
    relevance, grounded = judge(llm, question, findings)
    words = [len(f['statement'].split()) for f in findings]
    return {'question': question, 'mode': 'themes' if use_themes else 'search_only',
            'status': r['status'], 'coverage_percent': round(100 * based_on / in_scope, 2),
            'reviews_read': len(r['evidence']), 'findings': len(findings),
            'cited_reviews': len({c['review_id'] for f in findings for c in f['citations']}),
            'words_per_finding': round(sum(words) / len(words), 1) if words else 0,
            'citations_valid': r.get('validation', {}).get('status') == 'passed',
            'grounded_share': grounded, 'relevance_1to5': relevance,
            'seconds': round(seconds, 1), 'tokens': r['usage']['total_tokens']}


def main():
    secrets = tomllib.loads((ROOT / '.streamlit/secrets.toml').read_text())
    store = ReviewStore()
    dataset = next(d.id for d in store.datasets().itertuples() if 'Datafiniti' in d.name)
    search = ReviewSearch(load_model())
    llm = make_llm(secrets.get('GEMINI_MODEL'), api_key=secrets['GOOGLE_API_KEY'])
    rows = []
    for question in QUESTIONS:
        for use_themes in (False, True):
            for attempt in range(2):
                try:
                    rows.append(run(llm, store, search, dataset, question, use_themes))
                    break
                except Exception as exc:
                    print('  waiting after', type(exc).__name__, flush=True)
                    time.sleep(60)
            print(json.dumps(rows[-1]), flush=True)
            time.sleep(25)
    summary = {}
    for mode in ('search_only', 'themes'):
        part = [r for r in rows if r['mode'] == mode]
        avg = lambda k: round(sum(r[k] or 0 for r in part) / len(part), 2)
        summary[mode] = {k: avg(k) for k in ('coverage_percent', 'reviews_read', 'findings',
                         'cited_reviews', 'words_per_finding', 'grounded_share',
                         'relevance_1to5', 'seconds', 'tokens')}
        summary[mode]['citations_valid_share'] = avg('citations_valid')
    out = ROOT / 'reports/rag_evaluation.json'
    out.write_text(json.dumps({'product': PRODUCT, 'summary': summary, 'rows': rows}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
