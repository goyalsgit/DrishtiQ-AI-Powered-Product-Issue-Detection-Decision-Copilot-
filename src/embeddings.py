"""Local semantic search with dataset isolation and review-level results."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.model_cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_VERSION = 'minilm-cosine-200tokens-overlap40-v1'


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, cache_folder=str(ROOT / '.model_cache'),
                               device='cpu', local_files_only=True)


def scope_filter(product, month):
    """Chroma filter. None means 'all' for product and/or month."""
    parts = [{k: v} for k, v in (('product', product), ('month', month)) if v is not None]
    return None if not parts else parts[0] if len(parts) == 1 else {'$and': parts}


class ReviewSearch:
    def __init__(self, model, path=None):
        import chromadb
        from chromadb.config import Settings
        self.model = model
        self.client = chromadb.PersistentClient(
            path=str(path or ROOT / 'data/chroma_reviews'),
            settings=Settings(anonymized_telemetry=False))

    def collection(self, dataset_id):
        key = hashlib.sha256((dataset_id + INDEX_VERSION).encode()).hexdigest()[:40]
        return self.client.get_or_create_collection(
            name='reviews-' + key, embedding_function=None,
            metadata={'hnsw:space': 'cosine', 'dataset': dataset_id,
                      'index_version': INDEX_VERSION, 'ready': False})

    def ready(self, dataset_id):
        collection = self.collection(dataset_id)
        metadata = collection.metadata or {}
        return (metadata.get('ready') is True and
                collection.count() == metadata.get('expected_chunks'))

    def chunks(self, text):
        tokens = self.model.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        for start in range(0, len(tokens), 160):
            yield self.model.tokenizer.decode(tokens[start:start + 200], skip_special_tokens=True)
            if start + 200 >= len(tokens):
                break

    def build(self, store, dataset_id):
        if self.ready(dataset_id):
            return self.collection(dataset_id).count()
        df = store.read(dataset_id)
        if df.empty:
            raise ValueError('Cannot index an empty or unknown dataset.')
        collection = self.collection(dataset_id)
        total = self._index_rows(collection, df)
        self._mark_ready(collection, total)
        return total

    def add_reviews(self, store, dataset_id, review_ids):
        """Index only new reviews of a live dataset (incremental update).

        Chunk IDs come from review_id, so re-adding a review overwrites the
        same vectors instead of duplicating them. Returns passages added."""
        if not self.ready(dataset_id):
            self.build(store, dataset_id)  # first time: index everything
            return self.collection(dataset_id).count()
        if not review_ids:
            return 0
        df = store.read_ids(dataset_id, review_ids)
        collection = self.collection(dataset_id)
        before = collection.count()
        self._index_rows(collection, df)
        # Count after upserting, so retries never inflate the expected total.
        self._mark_ready(collection, collection.count())
        return collection.count() - before

    def _index_rows(self, collection, df):
        batch = []
        total = 0

        def flush():
            vectors = self.model.encode([r[1] for r in batch], normalize_embeddings=True)
            collection.upsert(ids=[r[0] for r in batch], documents=[r[1] for r in batch],
                              embeddings=vectors.tolist(), metadatas=[r[2] for r in batch])

        for row in df.itertuples():
            for number, passage in enumerate(self.chunks(row.text)):
                chunk_id = hashlib.sha256(f'{row.review_id}:{number}'.encode()).hexdigest()
                batch.append((chunk_id, passage, {'review_id': row.review_id,
                              'product': row.product, 'month': row.month}))
                total += 1
                if len(batch) == 64:
                    flush()
                    batch = []
        if batch:
            flush()
        return total

    def _mark_ready(self, collection, expected):
        # Chroma rejects hnsw:space in modify(), even if its value is unchanged.
        metadata = {k: v for k, v in collection.metadata.items() if k != 'hnsw:space'}
        collection.modify(metadata={**metadata, 'ready': True, 'expected_chunks': expected})

    def search(self, store, dataset_id, product, month, question, limit=6):
        if not question.strip():
            return []
        if not self.ready(dataset_id):
            raise ValueError('Build the semantic index before searching.')
        # SQLite is authoritative: we never trust vector results to define scope.
        originals = store.read(dataset_id, product, month).set_index('review_id')
        if originals.empty:
            return []
        collection = self.collection(dataset_id)
        vector = self.model.encode([question], normalize_embeddings=True).tolist()
        count = min(collection.count(), max(limit * 4, 24))
        # Increase candidate count when many passages belong to the same review.
        where = scope_filter(product, month)
        while True:
            result = collection.query(query_embeddings=vector, n_results=count,
                **({'where': where} if where else {}),
                include=['documents', 'metadatas', 'distances'])
            hits = []
            seen = set()
            for passage, meta, distance in zip(result['documents'][0], result['metadatas'][0], result['distances'][0]):
                review_id = meta['review_id']
                if review_id in seen or review_id not in originals.index:
                    continue
                seen.add(review_id)
                row = originals.loc[review_id]
                hits.append({'review_id': review_id, 'product': row['product'],
                             'month': row.month,
                             'rating': int(row.rating),
                             'text': row.text, 'matched_passage': passage,
                             'similarity': round(1 - float(distance), 4)})
                if len(hits) == limit:
                    return hits
            if count == collection.count() or len(result['ids'][0]) < count:
                return hits
            count = min(collection.count(), count * 2)
