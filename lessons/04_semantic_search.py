"""Run: .venv/bin/python -m lessons.04_semantic_search
First run downloads a pretrained model; embedding computation runs locally.
"""
import os
from pathlib import Path

# Keep downloaded files inside the project rather than your global cache.
cache = Path(__file__).resolve().parents[1] / '.model_cache'
os.environ.setdefault('HF_HOME', str(cache))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from sentence_transformers import SentenceTransformer

# STEP 1: Load an existing model. We are not training a new model.
print('Loading the local embedding model (first run needs a download)...', flush=True)
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    cache_folder=str(cache),
    device='cpu',
)

# STEP 2: Four fictional reviews. IDs preserve the link to the original text.
reviews = [
    ('R101', 'Liquid spilled inside the delivery box because the cap was loose.'),
    ('R102', 'The fragrance is lovely and lasts all day.'),
    ('R103', 'My skin became itchy and red after applying the cream.'),
    ('R104', 'The bottle leaked and arrived almost empty.'),
]
question = 'Does this cause skin irritation?'

# STEP 3: Convert reviews and question using the SAME model.
texts = [text for review_id, text in reviews]
review_vectors = model.encode(texts, normalize_embeddings=True)
question_vector = model.encode(question, normalize_embeddings=True)
print('\nReview vector shape:', review_vectors.shape)
print('Meaning: four reviews, each represented by 384 numbers.')

# STEP 4: For unit-length vectors, dot product equals cosine similarity.
# @ is the mathematical matrix multiplication operator.
scores = review_vectors @ question_vector
ranked_indices = scores.argsort()[::-1]  # Highest similarity first.

print('\nQuestion:', question)
for index in ranked_indices:
    review_id, text = reviews[index]
    print(f'{review_id} | similarity={scores[index]:.3f} | {text}')

print('\nSimilarity scores are NOT probabilities or percentages of correctness.')
print('This ranks a tiny sample; it does not prove retrieval quality on real data.')
print('Next exercise: change the question to "Does this cause skin irritation?"')
