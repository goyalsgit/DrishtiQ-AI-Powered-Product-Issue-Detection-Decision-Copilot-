"""Retrieval metrics against explicit relevance labels, not generated answers."""


def score_retrieval(returned_ids, relevant_ids, scope_ids, k):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('k must be a positive integer')
    ranked = list(returned_ids)[:k]
    gold = set(relevant_ids)
    scope = set(scope_ids)
    if not gold <= scope:
        raise ValueError('Relevance labels reference reviews outside the query scope')
    # Duplicates never earn multiple hits; expose them as an integrity failure.
    hits = len(set(ranked) & gold)
    first_rank = next((i for i, review_id in enumerate(ranked, start=1)
                       if review_id in gold), None)
    return {
        'precision_at_k': hits / k if gold else None,
        'recall_at_k': hits / len(gold) if gold else None,
        'reciprocal_rank_at_k': 1 / first_rank if first_rank else (0.0 if gold else None),
        'hit_at_k': int(hits > 0) if gold else None,
        'unsupported_return': bool(ranked) if not gold else None,
        'out_of_scope_ids': sorted(set(ranked) - scope),
        'duplicate_count': len(ranked) - len(set(ranked)),
        'missing_relevant_ids': sorted(gold - set(ranked)),
        'irrelevant_returned_ids': [rid for rid in ranked if rid not in gold],
    }


def validate_cases(manifest, store):
    if manifest.get('schema_version') != 1 or not manifest.get('cases'):
        raise ValueError('Missing cases or unsupported case format')
    dataset = manifest['dataset_id']
    scopes = {}
    for item in manifest['scopes']:
        key = (item['product'], item['month'])
        actual = set(store.read(dataset, *key).review_id)
        declared = item['reviewed_review_ids']
        if not actual or actual != set(declared) or len(declared) != len(set(declared)):
            raise ValueError(f'Scope differs from reviewed source records: {key}')
        scopes[key] = actual
    seen = set()
    for case in manifest['cases']:
        if case['case_id'] in seen:
            raise ValueError('Duplicate case ID')
        seen.add(case['case_id'])
        scope = scopes.get((case['product'], case['month']))
        gold = case['relevant_review_ids']
        if scope is None or not set(gold) <= scope or len(gold) != len(set(gold)):
            raise ValueError(f'Invalid relevance IDs or scope in {case["case_id"]}')
        if not case['question'].strip() or not case['label_rationale'].strip():
            raise ValueError('Each case needs a question and a relevance rationale')
        expected = 'retrieve_evidence' if gold else 'insufficient_evidence'
        if case['expected_behavior'] != expected:
            raise ValueError('Expected behavior conflicts with relevance labels')
    return scopes


def aggregate(results):
    answerable = [r for r in results if r['metrics']['recall_at_k'] is not None]
    unsupported = [r for r in results if r['metrics']['unsupported_return'] is not None]
    def mean(field, items):
        return sum(r['metrics'][field] for r in items) / len(items) if items else None
    return {
        'answerable_cases': len(answerable), 'no_evidence_cases': len(unsupported),
        'macro_precision_at_k': mean('precision_at_k', answerable),
        'macro_recall_at_k': mean('recall_at_k', answerable),
        'mean_reciprocal_rank_at_k': mean('reciprocal_rank_at_k', answerable),
        'hit_rate_at_k': mean('hit_at_k', answerable),
        'unsupported_return_rate': mean('unsupported_return', unsupported),
        'scope_violations': sum(len(r['metrics']['out_of_scope_ids']) for r in results),
        'duplicate_results': sum(r['metrics']['duplicate_count'] for r in results),
    }
