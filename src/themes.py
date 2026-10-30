"""Group EVERY review in scope into themes, without calling Gemini.

Normal search returns the 6 closest reviews to a question, so a report sees only
a small sample. This module looks at all reviews in the selected scope:
  1. take each review's MiniLM vector from Chroma (already computed at indexing)
  2. group similar reviews with k-means
  3. for each group: exact size, share of scope, average rating, and the
     3 reviews closest to the group centre as examples
Gemini then names the groups and quotes the examples. Counts come from Python.

Limits: a group is reviews with similar wording, not a verified topic. One
review belongs to exactly one group, even if it mentions several things.
"""
import numpy as np

FOCUS_RATINGS = {
    'problems': (1, 2, 3),   # complaints mostly live in 1-3 star reviews
    'praise': (4, 5),
    'all': (1, 2, 3, 4, 5),
}
MAX_THEMES = 8
EXAMPLES_PER_THEME = 3


def review_vectors(search, dataset_id, product, month):
    """One vector per review (average of its passages) from the Chroma index."""
    from src.embeddings import scope_filter
    where = scope_filter(product, month)
    got = search.collection(dataset_id).get(include=['embeddings', 'metadatas'],
                                            **({'where': where} if where else {}))
    by_review = {}
    for vector, meta in zip(got['embeddings'], got['metadatas']):
        by_review.setdefault(meta['review_id'], []).append(vector)
    ids = list(by_review)
    vectors = np.array([np.mean(by_review[r], axis=0) for r in ids]) if ids else np.empty((0, 384))
    if len(ids):
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return ids, vectors


def summarize_themes(store, search, dataset_id, product, month, focus='all',
                     max_themes=MAX_THEMES, examples=EXAMPLES_PER_THEME):
    from sklearn.cluster import KMeans

    reviews = store.read(dataset_id, product, month).set_index('review_id')
    in_scope = len(reviews)
    reviews = reviews[reviews['rating'].isin(FOCUS_RATINGS[focus])]
    ids, vectors = review_vectors(search, dataset_id, product, month)
    keep = [i for i, rid in enumerate(ids) if rid in reviews.index]
    ids, vectors = [ids[i] for i in keep], vectors[keep]
    result = {'focus': focus, 'reviews_in_scope': in_scope,
              'reviews_analyzed': len(ids), 'ratings_included': list(FOCUS_RATINGS[focus]),
              'themes': []}
    if len(ids) < 3:
        return result
    k = min(max_themes, max(2, len(ids) // 15))
    model = KMeans(n_clusters=k, n_init=4, random_state=0).fit(vectors)
    for label in range(k):
        members = np.where(model.labels_ == label)[0]
        distance = np.linalg.norm(vectors[members] - model.cluster_centers_[label], axis=1)
        closest = members[np.argsort(distance)[:examples]]
        group = reviews.loc[[ids[i] for i in members]]
        result['themes'].append({
            'reviews': len(members),
            'share_of_analyzed_percent': round(100 * len(members) / len(ids), 1),
            'average_rating': round(float(group['rating'].mean()), 2),
            # Which products this theme comes from (matters for "all products").
            'top_products': group['product'].value_counts().head(3).to_dict(),
            'examples': [{'review_id': ids[i], 'product': reviews.loc[ids[i], 'product'],
                          'month': reviews.loc[ids[i], 'month'],
                          'rating': int(reviews.loc[ids[i], 'rating']),
                          'text': reviews.loc[ids[i], 'text']} for i in closest],
        })
    result['themes'].sort(key=lambda t: t['reviews'], reverse=True)
    return result


def problems_by_product(store, search, dataset_id, month=None, top=6):
    """For the products with the most 1-2 star reviews, group each one's
    1-3 star reviews into up to 4 problem themes (2 example reviews each)."""
    dates = dict(start=f'{month}-01' if month else None,
                 end=f'{month}-31T23:59:59' if month else None)
    table = store.product_table(dataset_id, **dates).sort_values(
        ['low_rating_reviews', 'reviews'], ascending=False).head(top)
    products = []
    for row in table.itertuples():
        themes = summarize_themes(store, search, dataset_id, row.product, month,
                                  'problems', max_themes=4, examples=2)
        products.append({'product': row.product, 'total_reviews': int(row.reviews),
                         'low_rating_reviews': int(row.low_rating_reviews),
                         'reviews_analyzed': themes['reviews_analyzed'],
                         'themes': themes['themes']})
    return products
