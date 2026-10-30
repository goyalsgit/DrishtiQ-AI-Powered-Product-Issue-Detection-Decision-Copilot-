"""Check that Gemini works, step by step.

Run: .venv-agent/bin/python -m scripts.diagnose_gemini

Reads your key from .streamlit/secrets.toml (never prints it) and checks:
  1. the key and model work
  2. Gemini can call tools (needed to ask for SQL numbers and reviews)
  3. Gemini returns a report in our format
It stops at the first failed step and shows why.
"""
import pathlib
import tomllib

from src.connection import check_connection
from src.llm import make_llm

SECRETS = pathlib.Path(__file__).resolve().parents[1] / '.streamlit/secrets.toml'


def main():
    if not SECRETS.exists():
        raise SystemExit('Missing .streamlit/secrets.toml. Copy secrets.toml.example and add your key.')
    settings = tomllib.loads(SECRETS.read_text())
    key = settings['GOOGLE_API_KEY']
    llm = make_llm(settings.get('GEMINI_MODEL'), api_key=key)

    print('Model:', llm.model)
    for step in check_connection(llm, key):
        status = 'PASS' if step['ok'] else 'FAIL'
        print(f"{status}  {step['step']}  ({step['seconds']}s)  {step['detail']}")


if __name__ == '__main__':
    main()
