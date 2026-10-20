"""Run: .venv-agent/bin/python -m lessons.09_tools_and_graph

Real LangChain tool + real LangGraph workflow + saved SQLite reviews.
No LLM, API key, embeddings, or model-selected routing in this first lesson.
"""
import argparse
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from src.database import ReviewStore
from src.tools import make_tools


class LessonState(TypedDict, total=False):
    metrics: dict
    explanation: str


def build_graph(metrics_tool):
    def fetch_metrics(state):
        # LangChain validates the input and executes our Python function.
        return {"metrics": metrics_tool.invoke({})}

    def route(state):
        # This is an explicit Python rule, not an LLM decision.
        return "explain" if state["metrics"]["total_reviews"] else "no_data"

    def explain(state):
        m = state["metrics"]
        return {"explanation": (
            f'{m["low_rating_reviews"]} of {m["total_reviews"]} reviews '
            f'have 1 or 2 stars ({m["low_rating_share_percent"]:.2f}%). '
            "This is a template, not generated AI text."
        )}

    def no_data(state):
        return {"explanation": "No reviews exist in this scope. No percentage is available."}

    builder = StateGraph(LessonState)
    builder.add_node("fetch_metrics", fetch_metrics)
    builder.add_node("explain", explain)
    builder.add_node("no_data", no_data)
    builder.add_edge(START, "fetch_metrics")
    builder.add_conditional_edges("fetch_metrics", route,
                                  {"explain": "explain", "no_data": "no_data"})
    builder.add_edge("explain", END)
    builder.add_edge("no_data", END)
    return builder.compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty", action="store_true",
                        help="Use an absent month to demonstrate the no-data branch.")
    args = parser.parse_args()
    store = ReviewStore()
    datasets = store.datasets()
    if datasets.empty:
        parser.error("Import reviews using the existing dashboard first.")
    dataset = datasets.iloc[0]["id"]
    rows = store.read(dataset)
    product, month = rows.groupby(["product", "month"]).size().idxmax()
    if args.empty:
        month = "0000-00"
    metrics_tool = make_tools(store, dataset, product, month)[0]
    print("1. TOOL NAME:", metrics_tool.name)
    print("2. TOOL DESCRIPTION:", metrics_tool.description)
    print("3. TOOL INPUT SCHEMA:")
    print(json.dumps(metrics_tool.get_input_schema().model_json_schema(), indent=2))
    print("\n4. GRAPH NODE UPDATES:")
    for update in build_graph(metrics_tool).stream({}, stream_mode="updates"):
        print(json.dumps(update, indent=2))
    print("\nThis is real orchestration, but it is not an autonomous AI agent.")
    print("Next: a model node can request tools; the graph executes and returns results.")


if __name__ == "__main__":
    main()
