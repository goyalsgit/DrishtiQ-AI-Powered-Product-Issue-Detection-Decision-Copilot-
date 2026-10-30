"""Restore the public demo on hosts where local files can disappear."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ensure_public_sample(store, root=ROOT, importer=None):
    """Reuse the pinned sample or run its importer; never substitute fictional data."""
    manifest_path = root / 'data/amazon_beauty_provenance.json'
    manifest = json.loads(manifest_path.read_text())
    dataset_id = manifest['dataset_id']
    existing = store.datasets()
    if dataset_id in set(existing['id']):
        return dataset_id
    csv_path = root / 'data/amazon_beauty_sample.csv'
    if csv_path.exists():
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != manifest['csv_sha256']:
            raise ValueError('Public sample checksum differs from its provenance.')
        restored, removed = store.ingest(
            pd.read_csv(csv_path, dtype={'review_id': str, 'product': str}),
            'Amazon All Beauty • public convenience sample')
        if removed or restored != dataset_id:
            raise ValueError('Restored sample differs from the pinned dataset.')
    else:
        if importer is None:
            subprocess.run([sys.executable, '-m', 'scripts.import_public_reviews',
                            '--revision', manifest['revision']],
                           cwd=root, check=True, timeout=300)
        else:
            importer()
    if dataset_id not in set(store.datasets()['id']):
        raise ValueError('Importer did not restore the expected public dataset.')
    return dataset_id


def prepare_cloud_demo():
    """Called once per app process by the cloud entrypoint's Streamlit cache."""
    from src.database import ReviewStore
    from src.dashboard_ai import get_search
    from src.embeddings import MODEL_NAME
    store = ReviewStore()
    dataset_id = ensure_public_sample(store)
    try:
        search = get_search()
    except OSError:
        # Only the free-cloud entrypoint allows a first-run model download.
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_NAME, cache_dir=str(ROOT / '.model_cache'),
                          allow_patterns=['*.json', '*.txt', '*.safetensors'],
                          ignore_patterns=['onnx/*', 'openvino/*'])
        search = get_search()
    passages = search.build(store, dataset_id)
    # The cloud disk is wiped on restart. If a company Postgres is configured,
    # sync it now: the watermark starts at 0 again, so every review comes back.
    from src import company_db
    synced = None
    if company_db.database_url() and company_db.exists():
        synced = company_db.sync(store, search)[1]
    return {'dataset_id': dataset_id, 'passages': passages, 'company_sync': synced}
