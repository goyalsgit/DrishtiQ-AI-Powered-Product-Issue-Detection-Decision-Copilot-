"""UI-neutral, observable execution of the existing agent. No credentials retained."""
import json

from langchain.messages import HumanMessage, ToolMessage
from src.agent import build_agent
from src.tools import make_tools


def answer_text(content):
    """Extract display text without exposing signatures or reasoning blocks."""
    if isinstance(content, str):
        return content
    return "\n".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or (
            isinstance(block, dict) and block.get("type") == "text"
        )
    )


def run_investigation(model, store, dataset_id, product, month, question, *,
                      search=None, baseline_month=None, minimum_reviews=20,
                      on_event=None):
    """Run one independent question; return a JSON-serializable draft and trace."""
    if not question.strip() or len(question) > 4000:
        raise ValueError("Enter a question between 1 and 4000 characters.")
    tools = make_tools(store, dataset_id, product, month, search=search,
                       baseline_month=baseline_month, minimum_reviews=minimum_reviews)
    graph = build_agent(model, tools, collect_only=True)
    result = {
        "question": question.strip(),
        "scope": {"dataset_id": dataset_id, "product": product, "month": month,
                  "baseline_month": baseline_month, "minimum_reviews": minimum_reviews},
        "answer": "", "status": "incomplete", "trace": [],
        "evidence": [], "tool_results": [], "tool_errors": [],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    evidence_keys = set()
    limited = False
    for update in graph.stream(
        {"messages": [HumanMessage(content=question.strip())],
         "model_calls": 0, "tool_calls": 0},
        config={"recursion_limit": 20}, stream_mode="updates",
    ):
        for node, values in update.items():
            limited = limited or node == "limit_reached"
            for message in values.get("messages", []):
                if isinstance(message, ToolMessage):
                    data = json.loads(message.content)
                    event = {"kind": "result", "node": node, "tool": message.name,
                             "call_id": message.tool_call_id, "status": message.status,
                             "data": data}
                    result["tool_results"].append(event)
                    if message.status == "error":
                        result["tool_errors"].append(message.name)
                    else:
                        scope = data.get("scope", {})
                        missing = [r["review_id"] for r in data.get("reviews", [])
                                   if not (r.get("month") or scope.get("month"))]
                        if missing:
                            # Every citation needs a real month. Take it from SQLite,
                            # the source of truth, if a search result lacks one.
                            months = store.read_ids(dataset_id, missing).set_index(
                                "review_id")["month"].to_dict()
                            for r in data["reviews"]:
                                r.setdefault("month", None)
                                r["month"] = r["month"] or months.get(r["review_id"])
                        for review in data.get("reviews", []):
                            key = (scope.get("dataset_id"), scope.get("product"),
                                   scope.get("month"), review["review_id"])
                            if key not in evidence_keys:
                                evidence_keys.add(key)
                                result["evidence"].append({
                                    # Each review keeps its own month (needed for "All months").
                                    **review, "month": review.get("month") or scope.get("month"),
                                    "period": data.get("period", "current"),
                                })
                elif getattr(message, "tool_calls", None):
                    event = {"kind": "requests", "node": node,
                             "calls": message.tool_calls}
                else:
                    result["answer"] = answer_text(message.content)
                    event = {"kind": "answer", "node": node}
                usage = getattr(message, "usage_metadata", None) or {}
                for field in result["usage"]:
                    result["usage"][field] += usage.get(field, 0) or 0
                result["trace"].append(event)
                if on_event:
                    on_event(event)
    # At the call limit, still write a report if evidence was already collected;
    # the report states that collection stopped early.
    if result["answer"] and (not limited or result["evidence"]):
        result["status"] = "draft_with_tool_errors" if result["tool_errors"] else "draft"
        if limited:
            result["tool_errors"].append("call_limit_reached_some_searches_not_run")
        from src.reporting import finalize_report
        finalize_report(model, result, on_event)
    return result
