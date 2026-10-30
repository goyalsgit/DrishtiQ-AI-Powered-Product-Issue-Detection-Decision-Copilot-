"""One-time setup: put 1,500 real reviews into the company database.

Run on your laptop (it needs 1429_1.csv, which is not on GitHub):
    .venv-agent/bin/python -m scripts.load_company_db
Uses COMPANY_DB_URL from .streamlit/secrets.toml (your Neon Postgres) if present,
otherwise the local demo file. Replaces any existing customer_reviews table.
"""
import pathlib
import tomllib
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
secrets = ROOT / '.streamlit/secrets.toml'
if secrets.exists() and 'COMPANY_DB_URL' in tomllib.loads(secrets.read_text()):
    os.environ.setdefault('COMPANY_DB_URL', tomllib.loads(secrets.read_text())['COMPANY_DB_URL'])

from src import company_db  # noqa: E402  (reads COMPANY_DB_URL)

print('Target:', company_db.description())
print('Rows written:', company_db.create())
print('Products:', company_db.products())
