"""Deterministic analytics: all figures come from data, not a language model."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path
import pandas as pd

DB = Path(__file__).resolve().parents[1] / 'data' / 'dristiq.sqlite'
TOPICS = {'Packaging': r'\b(?:leak\w*|broken|crack\w*|packaging|lid)\b',
          'Skin irritation': r'\b(?:irritat\w*|rash\w*|itch\w*|burn\w*)\b',
          'Fragrance': r'\b(?:smell\w*|scent\w*|fragrance)\b',
          'Delivery': r'\b(?:delivery|late|delay\w*)\b'}


def clean(frame):
    required = ['review_id', 'product', 'date', 'rating', 'text']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    df = frame[required].copy()
    for col in ['review_id', 'product', 'text']:
        df[col] = df[col].fillna('').astype(str).str.strip()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
    valid = df['rating'].isin([1, 2, 3, 4, 5]) & df['date'].notna()
    for col in ['review_id', 'product', 'text']:
        valid &= df[col].ne('')
    df = df[valid].drop_duplicates().copy()
    if df['review_id'].duplicated().any():
        raise ValueError('Conflicting rows share review IDs. Give each review a unique ID.')
    df['negative'] = df['rating'].le(2)
    df['month'] = df['date'].dt.strftime('%Y-%m')
    return df.reset_index(drop=True), len(frame) - len(df)


def metrics(df):
    # Fixed SQL, with no model-generated statements.
    with sqlite3.connect(':memory:') as con:
        df[['rating', 'negative']].to_sql('reviews', con, index=False)
        n, avg, negative = con.execute(
            'SELECT COUNT(*), AVG(rating), COALESCE(SUM(negative), 0) FROM reviews'
        ).fetchone()
    return n, avg, negative, 100 * negative / n if n else None


def topics(df):
    negative = df[df['negative']]
    return {topic: negative[negative['text'].str.contains(pattern, case=False, regex=True)]
            for topic, pattern in TOPICS.items()}


def retrieve(df, query, limit=6):
    words = set(re.findall(r'\w+', query.lower())) - {'the', 'a', 'are', 'is', 'why', 'what', 'customers', 'reviews'}
    scores = df['text'].apply(lambda t: len(words & set(re.findall(r'\w+', t.lower()))))
    return df.loc[scores[scores.gt(0)].sort_values(ascending=False).head(limit).index]


def save_decision(context, action, decision, reviewer):
    DB.parent.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps([context, action], sort_keys=True).encode()).hexdigest()
    with sqlite3.connect(DB) as con:
        con.execute('CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, context TEXT, action TEXT, decision TEXT, reviewer TEXT)')
        con.execute('INSERT OR IGNORE INTO decisions (id, context, action, decision, reviewer) VALUES (?, ?, ?, ?, ?)',
                    (key, json.dumps(context), action, decision, reviewer))
    return key


def history(dataset_id=None):
    if not DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB) as con:
        df = pd.read_sql_query('SELECT created, action, decision, reviewer, context FROM decisions ORDER BY created DESC', con)
    if dataset_id is not None and not df.empty:
        df = df[df['context'].map(lambda value: json.loads(value).get('dataset') == dataset_id)]
    return df
