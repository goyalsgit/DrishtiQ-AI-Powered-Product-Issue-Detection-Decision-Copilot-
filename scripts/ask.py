"""Ask DrishtiQ one question from the terminal and print the answer.

Run: .venv-agent/bin/python -m scripts.ask "What are the main problems?"
     .venv-agent/bin/python -m scripts.ask "Is battery good?" --product "Fire Tablet [B018Y229OU]"
Default dataset: the Datafiniti Amazon devices data. Default product: all products.
"""
import argparse
import pathlib
import time
import tomllib

from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model
from src.investigation import run_investigation
from src.llm import make_llm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question')
    parser.add_argument('--product', default=None)
    parser.add_argument('--month', default=None)
    parser.add_argument('--dataset', default='Datafiniti', help='part of the dataset name')
    args = parser.parse_args()

    secrets = tomllib.loads((ROOT / '.streamlit/secrets.toml').read_text())
    store = ReviewStore()
    dataset = next(d.id for d in store.datasets().itertuples() if args.dataset in d.name)
    llm = make_llm(secrets.get('GEMINI_MODEL'), api_key=secrets['GOOGLE_API_KEY'])
    start = time.perf_counter()
    result = run_investigation(llm, store, dataset, args.product, args.month, args.question,
                               search=ReviewSearch(load_model()))
    report = result.get('report') or {}
    content = report.get('content', {})
    print(f"\n{result['status']} · {time.perf_counter() - start:.0f}s · "
          f"tools: {', '.join(t['tool'] for t in result['tool_results'])}\n")
    print('ANSWER:', content.get('summary') or result.get('answer'))
    for number, finding in enumerate(content.get('findings', []), 1):
        print(f"\n{number}. {finding.get('product') or finding['kind']}: {finding['statement']}")
        for point in finding.get('points', []):
            print('   -', point)
        print('   quote: "' + finding['citations'][0]['quote'][:110] + '"')
    for action in report.get('action_plan', []):
        print(f"\nDO [{action['priority']}] {action['owner']}: {action['suggestion']}")


if __name__ == '__main__':
    main()
