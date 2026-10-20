"""Visible GenAI, RAG and agent experiments. See docs/learning/10-agent-lab.md."""
import argparse
import getpass
import json
import os

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from src.agent import SYSTEM_PROMPT, build_agent
from src.comparison import previous_month, validate_comparison
from src.database import ReviewStore
from src.llm import make_llm
from src.tools import make_tools


class ScriptedModel:
    """OFFLINE teaching fixture: predefined tool calls, never real model reasoning."""
    def __init__(self, question):
        self.question = question

    def bind_tools(self, tools):
        self.names = {tool.name for tool in tools}
        return self

    def invoke(self, messages):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            name = ("compare_review_periods" if "compare_review_periods" in self.names
                    else "get_review_metrics")
            return AIMessage(content="", tool_calls=[
                {"name": name, "args": {}, "id": "demo-metrics"}
            ])
        if len(results) == 1 and "search_review_evidence" in self.names:
            periods = ["current", "baseline"] if "compare_review_periods" in self.names else ["current"]
            return AIMessage(content="", tool_calls=[
                {"name": "search_review_evidence",
                 "args": {"question": self.question, "period": period},
                 "id": "demo-search-" + period} for period in periods
            ])
        metrics = json.loads(results[0].content)
        if "current" in metrics:
            return AIMessage(content=(
                "OFFLINE SCRIPTED OUTPUT — not Gemini: "
                f'current={metrics["current"]["total_reviews"]} reviews; '
                f'baseline={metrics["baseline"]["total_reviews"]} reviews; '
                f'change={metrics["low_rating_share_change_percentage_points"]} percentage points; '
                f'status={metrics["status"]}. No cause has been established.'
            ))
        return AIMessage(content=(
            "OFFLINE SCRIPTED OUTPUT — not Gemini: "
            f"SQL returned {metrics.get('total_reviews')} reviews and "
            f"{metrics.get('low_rating_reviews')} low ratings. "
            "Inspect the tool messages for evidence; no semantic interpretation "
            "or generated investigation has been performed."
        ))


def display_message(message):
    if isinstance(message, ToolMessage):
        print(f"TOOL RESULT [{message.name}] id={message.tool_call_id} status={message.status}")
        print(json.dumps(json.loads(message.content), indent=2, ensure_ascii=False))
    elif getattr(message, "tool_calls", None):
        print("MODEL REQUESTS (Python has not executed these yet):")
        print(json.dumps(message.tool_calls, indent=2, ensure_ascii=False))
    else:
        # Providers may return a list of content blocks rather than a string.
        # Display answer text; keep provider metadata on the original message
        # so future model calls still receive the complete conversation.
        content = message.content
        if isinstance(content, list):
            content = "\n".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
                if isinstance(block, str) or (
                    isinstance(block, dict) and block.get("type") == "text"
                )
            )
        print("DRAFT RESPONSE:\n" + (content or "[No text response returned.]"))
    usage = getattr(message, "usage_metadata", None)
    if usage:
        print("TOKEN USAGE:", json.dumps(usage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["chat", "rag", "agent"], default="agent")
    parser.add_argument("--offline", action="store_true", help="Scripted agent, no API call.")
    parser.add_argument("--ask-key", action="store_true", help="Read key privately for this process.")
    parser.add_argument("--semantic", action="store_true", help="Enable real local review search.")
    parser.add_argument("--model", help="Override GEMINI_MODEL / default model.")
    parser.add_argument("--dataset")
    parser.add_argument("--product")
    parser.add_argument("--month")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with the previous calendar month (unless --baseline is supplied).")
    parser.add_argument("--baseline", help="Explicit earlier month (YYYY-MM); enables comparison.")
    parser.add_argument("--minimum-reviews", type=int, default=20,
                        help="Minimum reviews per period for the descriptive sample rule.")
    parser.add_argument("--question", default="How many reviews have low ratings, and what can we conclude?")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("The question cannot be empty.")
    if args.offline and args.mode != "agent":
        parser.error("--offline demonstrates the agent loop only.")
    if args.mode == "rag" and not args.semantic:
        parser.error("Use --semantic for RAG so real review evidence is retrieved.")
    if args.mode == "chat" and args.semantic:
        parser.error("Chat mode has no retrieval; use --mode rag or agent.")
    if args.mode == "chat" and (args.compare or args.baseline):
        parser.error("Chat mode has no database tools; use --mode rag or agent.")
    if args.minimum_reviews < 1:
        parser.error("--minimum-reviews must be positive.")
    if args.ask_key and not args.offline:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Gemini API key (not saved): ").strip()
    try:
        model = ScriptedModel(args.question) if args.offline else make_llm(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    if args.mode == "chat":
        print("CHAT: question only, no access to your database.")
        display_message(model.invoke([
            SystemMessage(content="You are a tutor. You have no access to the user's review database. Do not invent its contents."),
            HumanMessage(content=args.question),
        ]))
        return

    store = ReviewStore()
    datasets = store.datasets()
    if datasets.empty:
        parser.error("Import reviews in the dashboard first.")
    dataset = args.dataset or datasets.iloc[0]["id"]
    rows = store.read(dataset, args.product, args.month)
    if rows.empty:
        parser.error("No saved reviews match this scope.")
    product, month = rows.groupby(["product", "month"]).size().idxmax()
    baseline = args.baseline
    try:
        if args.compare and baseline is None:
            baseline = previous_month(month)
        if baseline is not None:
            validate_comparison(month, baseline, args.minimum_reviews)
    except ValueError as exc:
        parser.error(str(exc))
    print("SCOPE:", json.dumps({"dataset": dataset, "product": product,
                               "month": month, "baseline_month": baseline}))
    search = None
    if args.semantic:
        from src.embeddings import ReviewSearch, load_model
        print("Loading local MiniLM and existing Chroma index...", flush=True)
        search = ReviewSearch(load_model())
        if not search.ready(dataset):
            parser.error("Build this dataset's index in the dashboard first.")
    tools = make_tools(store, dataset, product, month, search=search,
                       baseline_month=baseline, minimum_reviews=args.minimum_reviews)
    by_name = {tool.name: tool for tool in tools}
    for tool in tools:
        print("AVAILABLE TOOL:", tool.name)
        print(json.dumps(tool.get_input_schema().model_json_schema(), indent=2))

    if args.mode == "rag":
        # Fixed retrieval sequence: our Python code, not the model, chooses tools.
        context = {
            "question": args.question,
            "metrics": by_name["get_review_metrics"].invoke({}),
            "evidence": by_name["search_review_evidence"].invoke({"question": args.question}),
        }
        if baseline is not None:
            context["comparison"] = by_name["compare_review_periods"].invoke({})
            context["baseline_evidence"] = by_name["search_review_evidence"].invoke(
                {"question": args.question, "period": "baseline"}
            )
        print("CONTEXT SENT TO GEMINI:")
        print(json.dumps(context, indent=2, ensure_ascii=False))
        display_message(model.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(context, ensure_ascii=False)),
        ]))
    else:
        print("OFFLINE SCRIPTED MODEL" if args.offline else "LIVE GEMINI AGENT")
        graph = build_agent(model, tools)
        for update in graph.stream(
            {"messages": [HumanMessage(content=args.question)],
             "model_calls": 0, "tool_calls": 0},
            config={"recursion_limit": 20}, stream_mode="updates",
        ):
            for node, values in update.items():
                print(f"\nNODE: {node}", flush=True)
                for message in values.get("messages", []):
                    display_message(message)
    print("\nLearning prototype: drafts are not fact-checked or approved. No decision was saved.")


if __name__ == "__main__":
    main()
