"""Candidate retrieval methods compared in the v2 evaluation.

keyword  - word overlap (baseline)
semantic - MiniLM embeddings in Chroma (current production search)
hybrid   - keyword + semantic merged with Reciprocal Rank Fusion
rerank   - hybrid candidates re-sorted by a cross-encoder that reads
           question and review together (slower, usually more precise)
"""
from src.analytics import retrieve

RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
CANDIDATES = 30   # how many results each first-stage method contributes
RRF_K = 60        # standard Reciprocal Rank Fusion constant


def keyword_ids(reviews, question, limit):
    return retrieve(reviews, question, limit=limit).review_id.tolist()


def semantic_ids(search, store, dataset_id, product, question, limit):
    hits = search.search(store, dataset_id, product, None, question, limit=limit)
    return [h['review_id'] for h in hits]


def rrf_merge(rankings, limit):
    """Each list votes 1/(60 + rank) for its items. Items found high by
    both methods win. No score calibration between methods is needed."""
    scores = {}
    for ranking in rankings:
        for rank, review_id in enumerate(ranking, start=1):
            scores[review_id] = scores.get(review_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)[:limit]


def hybrid_ids(search, store, dataset_id, product, reviews, question, limit):
    semantic = semantic_ids(search, store, dataset_id, product, question, CANDIDATES)
    keyword = keyword_ids(reviews, question, CANDIDATES)
    return rrf_merge([semantic, keyword], limit)


class Reranker:
    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(RERANK_MODEL, device='cpu')

    def rank(self, question, candidate_ids, texts_by_id):
        """Return (review_id, score) pairs, best first. Higher score = more relevant."""
        if not candidate_ids:
            return []
        pairs = [(question, texts_by_id[rid]) for rid in candidate_ids]
        scores = self.model.predict(pairs)
        return sorted(zip(candidate_ids, map(float, scores)),
                      key=lambda item: item[1], reverse=True)
