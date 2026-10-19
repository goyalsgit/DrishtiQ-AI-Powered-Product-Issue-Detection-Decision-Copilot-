"""Run: .venv/bin/python -m lessons.05_chroma_storage

Four fictional reviews; local embeddings and a local persistent Chroma database.
Run twice to see saved records survive between processes without duplication.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.model_cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# STEP 1: Open a database saved inside the project folder.
# Persistent means the data remains after this program finishes.
client = chromadb.PersistentClient(
    path=str(ROOT / 'data/chroma_lesson'),
    settings=Settings(anonymized_telemetry=False),
)

# STEP 2: A collection is a named group of records.
# We supply our own vectors, so disable Chroma's default embedding function.
collection = client.get_or_create_collection(
    name='lesson_reviews_minilm_v1',
    embedding_function=None,
    metadata={'hnsw:space': 'cosine', 'embedding_model': 'all-MiniLM-L6-v2'},
)
print('Records already saved before this run:', collection.count(), flush=True)

model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    cache_folder=str(ROOT / '.model_cache'),
    device='cpu',
    local_files_only=True,  # Lesson 4 already downloaded the model.
)

# STEP 3: Store IDs, original text, embeddings, and metadata together.
ids = ['C101', 'C102', 'C103', 'C104']
texts = [
    'The moisturizer jar leaked inside the delivery box.',
    'My skin became itchy after applying this moisturizer.',
    'The moisturizer has a pleasant fragrance.',
    'The shampoo bottle leaked and arrived nearly empty.',
]
metadata = [
    {'product': 'Moisturizer', 'month': '2026-03'},
    {'product': 'Moisturizer', 'month': '2026-03'},
    {'product': 'Moisturizer', 'month': '2026-03'},
    {'product': 'Shampoo', 'month': '2026-03'},
]
vectors = model.encode(texts, normalize_embeddings=True)
collection.upsert(ids=ids, documents=texts, embeddings=vectors.tolist(), metadatas=metadata)
print('Records after upsert:', collection.count())

# STEP 4: Encode a question and search only within the selected product.
question = 'Any problems with leaking packaging?'
product = 'Moisturizer'
query_vector = model.encode([question], normalize_embeddings=True)
results = collection.query(
    query_embeddings=query_vector.tolist(),
    n_results=2,
    where={'product': product},
    include=['documents', 'metadatas', 'distances'],
)

# STEP 5: Results are grouped by question. [0] selects our one question.
print(f'\nQuestion: {question}\nProduct filter: {product}')
for review_id, text, details, distance in zip(
    results['ids'][0], results['documents'][0],
    results['metadatas'][0], results['distances'][0],
):
    print(f'{review_id} | {details["product"]} | distance={distance:.3f} | {text}')

print('\nFor cosine distance, SMALLER means more similar (not a confidence score).')
print('Top results may still be irrelevant: nearest does not guarantee relevant.')
print('Try product = "Shampoo", then run again. Only shampoo reviews should appear.')
