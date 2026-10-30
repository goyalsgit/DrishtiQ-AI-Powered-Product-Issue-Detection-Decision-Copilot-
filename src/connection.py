"""Step-by-step Gemini connection check. Each step depends on the previous one.

1. key + model reachable   (one tiny plain call)
2. tool calling works      (the agent needs this to request SQL/search)
3. report schema accepted  (the final structured report)
Messages contain error types/codes and provider text, never the key itself.
"""
import json
from time import perf_counter

from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool

REPORT_INPUT = json.dumps({
    "question": "What problems are customers reporting and how can we fix them?",
    "scope": {"dataset_id": "demo", "product": "P1", "month": "2016-04"},
    "statistics": [{"total_reviews": 2, "average_rating": 3.0}],
    "evidence": [{"review_id": "R1", "month": "2016-04",
                  "text": "The bottle leaked all over my bag and the pump broke."},
                 {"review_id": "R2", "month": "2016-04",
                  "text": "Smells nice and feels soft on skin."}],
    "tool_errors": [],
})


def _describe(exc, key):
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).replace(key, "[key]") if key else str(exc)
    return f"{type(exc).__name__}" + (f" (code {code})" if code else "") + f": {text[:400]}"


def _step(name, fn, key):
    start = perf_counter()
    try:
        detail = fn()
        return {"step": name, "ok": True, "seconds": round(perf_counter() - start, 1), "detail": detail}
    except Exception as exc:
        return {"step": name, "ok": False, "seconds": round(perf_counter() - start, 1),
                "detail": _describe(exc, key)}


def check_connection(llm, key=None):
    from src.reporting import FINAL_PROMPT, checked_content, gemini_schema

    def plain():
        from src.investigation import answer_text
        reply = llm.invoke("Reply with the single word OK.")
        return f"model replied: {answer_text(reply.content).strip()[:40]!r}"

    @tool
    def ping(word: str) -> str:
        """Echo a word back."""
        return word

    def tools():
        reply = llm.bind_tools([ping]).invoke("Call the ping tool with word='hello'.")
        if not reply.tool_calls:
            raise RuntimeError("Model answered in text instead of calling the tool.")
        return f"tool requested: {reply.tool_calls[0]['name']}"

    def report():
        structured = llm.with_structured_output(gemini_schema(), method="json_schema",
                                                include_raw=True)
        response = structured.invoke([SystemMessage(content=FINAL_PROMPT),
                                      HumanMessage(content=REPORT_INPUT)])
        meta = getattr(response.get("raw"), "response_metadata", None) or {}
        if response.get("parsing_error") is not None or response.get("parsed") is None:
            raise RuntimeError(f"Output did not match the schema "
                               f"(finish_reason={meta.get('finish_reason')}): "
                               f"{str(response.get('parsing_error'))[:300]}")
        parsed = response["parsed"]
        payload = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
        content = checked_content(payload, json.loads(REPORT_INPUT)["evidence"])
        return (f"report OK: {len(content.findings)} finding(s), "
                f"{len(content.recommended_actions)} action(s), finish={meta.get('finish_reason')}")

    results = []
    for name, fn in [("1. Key and model", plain), ("2. Tool calling", tools),
                     ("3. Report format", report)]:
        outcome = _step(name, fn, key)
        results.append(outcome)
        if not outcome["ok"]:
            break  # later steps cannot pass if an earlier one fails
    return results
