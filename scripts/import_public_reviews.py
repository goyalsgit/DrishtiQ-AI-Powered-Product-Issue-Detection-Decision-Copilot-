"""Stream a bounded, non-random public sample, retain well-represented products.
Run: .venv/bin/python -m scripts.import_public_reviews
No remote dataset loading code is executed. User identifiers are not retained.
"""
import hashlib
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd
from src.database import ReviewStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023'


def main():
    # Pin this download to a specific upstream revision for reproducibility.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e')
    revision = parser.parse_args().revision
    url = f'{SOURCE}/resolve/{revision}/raw/review_categories/All_Beauty.jsonl'
    rows = []
    scanned = 0
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            scanned += 1
            raw = json.loads(line)
            text = raw.get('text', '').strip()
            if text and raw.get('parent_asin') and raw.get('timestamp'):
                # Hash original identity fields; do not retain reviewer IDs.
                identity = json.dumps([raw.get('user_id'), raw['parent_asin'], raw['timestamp'], text])
                rows.append({
                    'review_id': 'AMZ-' + hashlib.sha256(identity.encode()).hexdigest(),
                    'product': raw['parent_asin'],
                    'date': datetime.fromtimestamp(raw['timestamp'] / 1000, timezone.utc).strftime('%Y-%m-%d'),
                    'rating': raw['rating'], 'text': text,
                })
            if scanned >= 50000:
                break
    df = pd.DataFrame(rows).drop_duplicates('review_id')
    top = df['product'].value_counts().head(8).index
    # Retain all sampled reviews for the selected products, regardless of rating/date.
    df = df[df['product'].isin(top)].sort_values(['product', 'date', 'review_id'])
    path = ROOT / 'data/amazon_beauty_sample.csv'
    df.to_csv(path, index=False)
    store = ReviewStore()
    dataset, excluded = store.ingest(df, 'Amazon All Beauty • public convenience sample')
    provenance = {
        'source': SOURCE, 'project_page': 'https://amazon-reviews-2023.github.io/',
        'revision': revision, 'download_url': url,
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'sampling': 'First 50,000 source records; keep eight most frequent parent_asin products. Not a random or representative sample.',
        'source_rows_scanned': scanned, 'retained_rows': len(df), 'excluded_on_import': excluded,
        'dataset_id': dataset, 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'product_identity': 'Original parent_asin; product names are not inferred.',
        'dates': 'Original timestamps converted from Unix milliseconds to UTC dates.',
        'limits': 'Incomplete historical sample. Metrics describe this saved sample only, not all Amazon customers or a brand. No human-verified topic labels.',
    }
    (ROOT / 'data/amazon_beauty_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: provenance[k] for k in ['source_rows_scanned', 'retained_rows', 'dataset_id']}, indent=2))
    print(df.groupby('product').size().to_string())


if __name__ == '__main__':
    main()
