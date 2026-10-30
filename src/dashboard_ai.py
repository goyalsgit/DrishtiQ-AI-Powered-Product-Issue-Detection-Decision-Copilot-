"""The "Ask reviews" screen: question box, progress, and the report.

Report layout (top to bottom):
  1. Answer            - Gemini's short summary
  2. How this was made - one plain line: what was read, counted and checked
  3. What we found     - each point with real customer quotes
  4. What to do        - suggested actions with team and priority
  5. Details           - possible reasons, limits, database numbers, all reviews used
"""
import importlib.util
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

REPORT_FORMAT_VERSION = 3

# Plain-language progress message for each tool Gemini can call.
TOOL_MESSAGES = {
    "get_review_metrics": "Counting reviews and star ratings in the database",
    "compare_products": "Ranking every product by its number of 1-2 star reviews",
    "problems_by_product": "Grouping each top product's low-rated reviews into problems",
    "summarize_review_themes": "Reading every review in scope and grouping them into themes",
    "search_review_evidence": "Searching for reviews about",
    "find_customer_requests": "Looking for reviews where customers ask for something",
    "compare_review_periods": "Comparing the two selected months",
}
FINDING_LABELS = {"reported_problem": "🔴 Problem", "positive_feedback": "🟢 Customers like",
                  "customer_request": "💡 Customers want"}
PRIORITY_ICONS = {"urgent": "🔴 Urgent", "high": "🟠 High", "normal": "🔵 Normal", "low": "⚪ Low"}


@st.cache_resource
def get_search():
    from src.embeddings import ReviewSearch, load_model
    return ReviewSearch(load_model())


def safe(text):
    """Show review/AI text literally: markdown symbols in it must not become links or formatting."""
    return re.sub(r'([\\`*_{}\[\]()#+\-.!|<>~$])', r'\\\1', " ".join(str(text).split()))


def short_name(product):
    """'Fire Tablet [B018Y229OU]' -> 'Fire Tablet'. Unnamed products keep their ID."""
    product = str(product or "")
    return product if product.startswith("[") else product.split(" [")[0]


# ---------------- Settings and errors ----------------

def _secret(name):
    """Server-side setting from .streamlit/secrets.toml or the environment (never sent to the browser)."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file
        value = None
    return value or os.getenv(name)


def server_api_key():
    return _secret("GOOGLE_API_KEY") or _secret("GEMINI_API_KEY")


def server_model():
    return _secret("GEMINI_MODEL") or "gemini-3.5-flash-lite"


def reset_key():
    st.session_state["ai_api_key"] = ""


def render_ai_settings():
    if server_api_key():
        st.session_state.setdefault("ai_model", server_model())
        st.caption(f"✅ Gemini key configured on the server · model: {st.session_state['ai_model']}")
        return
    with st.expander("Gemini connection", expanded=True):
        st.warning("No server key found. Add GOOGLE_API_KEY to .streamlit/secrets.toml "
                   "and restart, or paste a key below for this session only.")
        st.text_input("Gemini API key", type="password", key="ai_api_key",
                      placeholder="Paste your key here",
                      help="Used only for this browser session. Never written to a project file.")
        st.button("Forget entered key", on_click=reset_key, key="ai_forget_key")
        st.text_input("Gemini model", key="ai_model", value=server_model())


def safe_error_message(exc):
    """A clear message without raw provider text, which could echo the key."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (401, 403):
        return "Gemini rejected the key. Check your API key and model permissions."
    if code == 429 or "RateLimit" in type(exc).__name__:
        return ("Gemini's limit was reached (free plan: 15 requests per minute; one question "
                "uses 3-5). Wait a minute and try again.")
    if code == 404:
        return "The Gemini model name is not available. Check GEMINI_MODEL."
    if code == 400:
        return "Gemini rejected the request (400). Run scripts.diagnose_gemini for details."
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "Could not reach Gemini (timeout or no internet). Try again."
    detail = type(exc).__name__ + (f", code {code}" if code else "")
    return ("The investigation could not finish. Check your key, Gemini model access, "
            f"quota and network connection, then retry. (Error type: {detail})")


# ---------------- Report ----------------

def how_this_was_made(result):
    """One plain sentence describing what the answer is based on."""
    parts = []
    for row in result["tool_results"]:
        if row["status"] != "success":
            continue
        data = row["data"]
        if row["tool"] == "summarize_review_themes":
            parts.append(f'grouped all {data["reviews_analyzed"]:,} matching reviews '
                         f'into {len(data["themes"])} themes')
        elif row["tool"] == "compare_products":
            parts.append(f'ranked {data["products_total"]} products by complaints')
        elif row["tool"] == "problems_by_product":
            parts.append(f'grouped {sum(p["reviews_analyzed"] for p in data["products"]):,} '
                         f'low-rated reviews of the top {len(data["products"])} products into problems')
        elif row["tool"] == "get_review_metrics":
            parts.append(f'counted {data["total_reviews"]:,} reviews')
    parts.append(f'quoted from {len(result["evidence"])} real reviews')
    return "Based on: " + ", ".join(parts) + ". Every quote was checked against the database."


def show_quote(cite, by_id):
    review = by_id.get(cite["review_id"], {})
    where = " · ".join(x for x in [short_name(review.get("product")),
                                   f'{review["rating"]}★' if "rating" in review else "",
                                   cite["month"]] if x)
    st.markdown(f'> "{safe(cite["quote"])}"  \n> <small>{safe(where)}</small>',
                unsafe_allow_html=True)


def show_report(report, by_id):
    content = report["content"]
    if content.get("summary"):
        st.info(safe(content["summary"]))

    if content["findings"]:
        st.markdown("#### What we found")
    for number, finding in enumerate(content["findings"], 1):
        with st.container(border=True):
            product = finding.get("product")
            title = f'{number}. {safe(short_name(product))}' if product else \
                f'{number}. {FINDING_LABELS[finding["kind"]]}'
            st.markdown(f'**{title}**  \n{safe(finding["statement"])}')
            for point in finding.get("points", []):
                st.markdown(f'- {safe(point)}')
            with st.expander(f'What customers wrote ({len(finding["citations"])})',
                             expanded=not finding.get("points")):
                for cite in finding["citations"]:
                    show_quote(cite, by_id)

    plan = report.get("action_plan") or []
    if plan:
        st.markdown("#### What to do")
        st.dataframe(pd.DataFrame([{
            "Priority": PRIORITY_ICONS[a["priority"]], "Team": a["owner"],
            "Suggestion": a["suggestion"], "First step": a["standard_first_step"]}
            for a in plan]), hide_index=True, width="stretch")
        st.caption("Suggestions only. Nothing is done automatically; a manager decides.")

    if content["possible_causes"]:
        with st.expander("Possible reasons (not confirmed)"):
            for cause in content["possible_causes"]:
                st.markdown(f'- {safe(cause["hypothesis"])}  \n  *How to check:* '
                            f'{safe(cause["verification_needed"])}')
    with st.expander("Limits of this answer"):
        for item in content["limitations"] + content["next_steps"]:
            st.markdown("- " + safe(item))


def show_numbers(result):
    """SQL numbers and theme sizes. All exact counts, none from the AI."""
    for row in result["tool_results"]:
        if row["status"] != "success":
            continue
        data = row["data"]
        if row["tool"] == "compare_products":
            st.markdown("**Products with the most 1-2 star reviews**")
            table = pd.DataFrame(data["products"])
            st.dataframe(pd.DataFrame({
                "Product": table["product"].map(short_name),
                "1-2★ reviews": table["low_rating_reviews"], "All reviews": table["reviews"],
                "% low": table["low_rating_share_percent"]}), hide_index=True, width="stretch")
        elif row["tool"] == "problems_by_product":
            st.markdown("**Problem themes per product (exact counts)**")
            st.dataframe(pd.DataFrame([{
                "Product": short_name(p["product"]), "1-2★ reviews": p["low_rating_reviews"],
                "Theme size": t["reviews"], "Example": t["examples"][0]["text"][:120]}
                for p in data["products"] for t in p["themes"]]), hide_index=True, width="stretch")
        elif row["tool"] == "summarize_review_themes":
            st.markdown(f'**Themes in {data["reviews_analyzed"]:,} reviews** '
                        f'(ratings {data["ratings_included"]})')
            st.dataframe(pd.DataFrame([{
                "Reviews": t["reviews"], "Share %": t["share_of_analyzed_percent"],
                "Avg ★": t["average_rating"], "Example": t["examples"][0]["text"][:150]}
                for t in data["themes"]]), hide_index=True, width="stretch")
            st.caption("A theme is a group of similarly worded reviews; read the example.")
        elif row["tool"] in ("get_review_metrics", "compare_review_periods"):
            periods = [data["baseline"], data["current"]] if "current" in data else [
                {"month": data["scope"]["month"] or "all months", **data}]
            st.markdown("**Rating numbers**")
            st.dataframe(pd.DataFrame([{
                "Period": p["month"], "Reviews": p["total_reviews"],
                "Avg ★": round(p["average_rating"] or 0, 2), "1-2★": p["low_rating_reviews"],
                "% low": round(p["low_rating_share_percent"] or 0, 1)} for p in periods]),
                hide_index=True, width="stretch")


def show_result(result):
    st.divider()
    st.subheader("Investigation report")
    st.caption(f'Question: {result["question"]}')
    status = result["status"]

    if status == "unsupported_request":
        st.warning("This question needs data DrishtiQ does not have yet.")
        st.text(result["answer"])
        return
    if status == "validation_failed":
        st.error("The answer was withheld: its quotes did not match the real reviews. Please ask again.")
    elif status == "incomplete":
        st.warning("The investigation stopped early and has no answer. Try a more specific question.")
    elif status == "insufficient_evidence":
        st.warning("No review clearly answers this question.")

    by_id = {review["review_id"]: review for review in result["evidence"]}
    report = result.get("report")
    if report:
        st.caption(how_this_was_made(result))
        show_report(report, by_id)
    elif result["answer"] and status != "validation_failed":
        st.text(result["answer"])

    with st.expander("📊 Numbers from the database"):
        show_numbers(result)
    with st.expander(f'📝 All reviews used ({len(result["evidence"])})'):
        for review in result["evidence"]:
            st.markdown(f'**{review["rating"]}★ · {review["month"]} · '
                        f'{safe(short_name(review.get("product")))}**  \n{safe(review["text"])}')
    with st.expander("⚙️ Technical details"):
        st.caption(f'Tokens used: {result["usage"]["total_tokens"]:,}')
        st.json({"validation": result.get("validation"), "trace": result["trace"]}, expanded=False)
    st.download_button("Download report (JSON)", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="drishtiq-report.json", mime="application/json", key="ai_download")


# ---------------- Question box ----------------

def render_investigation(store, dataset_id, product, month, baseline, minimum_reviews):
    st.subheader("Ask a question")
    st.caption("Ask anything about the reviews: problems, what customers like or want, a feature, "
               "or choose 🌐 All products to compare products. Sales and returns are not connected.")
    if importlib.util.find_spec("langchain") is None:
        st.info("Start the dashboard with the Python 3.12 environment: "
                ".venv-agent/bin/python -m streamlit run app.py")
        return

    with st.form("ai_question_form"):
        question = st.text_area("Your question", key="ai_question", height=100, max_chars=4000,
                                placeholder="e.g. What are the main problems customers face?")
        submitted = st.form_submit_button("Investigate reviews", type="primary", width="stretch")

    # Clear an old report when the scope changes.
    signature = (dataset_id, product, month, baseline, minimum_reviews,
                 st.session_state.get("ai_model", "").strip(), REPORT_FORMAT_VERSION)
    previous = st.session_state.get("ai_result")
    if previous and previous["selection"] != signature:
        st.session_state.pop("ai_result", None)
        st.info("The selection changed. Ask again for this scope.")

    if submitted:
        st.session_state.pop("ai_result", None)
        key = server_api_key() or st.session_state.get("ai_api_key", "").strip()
        model_name = st.session_state.get("ai_model", "").strip()
        if not question.strip():
            st.warning("Enter a question first.")
            return
        if not key:
            st.warning("No Gemini API key. Add GOOGLE_API_KEY to .streamlit/secrets.toml and restart the app.")
            return
        if not model_name:
            st.warning("Enter a Gemini model name under Gemini connection.")
            return
        run_question(store, dataset_id, product, month, baseline, minimum_reviews,
                     question, key, model_name, signature)

    saved = st.session_state.get("ai_result")
    if saved:
        show_result(saved["report"])


def run_question(store, dataset_id, product, month, baseline, minimum_reviews,
                 question, key, model_name, signature):
    with st.status("Working on your question…", expanded=True) as progress:
        try:
            search = get_search()
            if not search.ready(dataset_id):
                progress.update(label="Prepare review search first", state="error")
                st.warning("Open the Data tab, click Prepare review search for this dataset, "
                           "then ask again (prepare this dataset first).")
                return
            from src.investigation import run_investigation
            from src.llm import make_llm

            def on_event(event):
                if event["kind"] == "requests":
                    for call in event["calls"]:
                        text = TOOL_MESSAGES.get(call["name"], call["name"])
                        if call["name"] == "search_review_evidence":
                            text += f' "{call["args"].get("question", "")}"'
                        progress.write(f"🔎 {text}…")
                elif event["kind"] == "validation":
                    progress.write("✍️ Writing the answer and checking every quote…")

            result = run_investigation(make_llm(model_name, api_key=key), store, dataset_id,
                                       product, month, question, search=search,
                                       baseline_month=baseline, minimum_reviews=minimum_reviews,
                                       on_event=on_event)
            result["created_utc"] = datetime.now(timezone.utc).isoformat()
            result["model"] = model_name
            st.session_state["ai_result"] = {"selection": signature, "report": result}
            labels = {"unsupported_request": "Cannot answer with current data",
                      "validation_failed": "Answer withheld", "incomplete": "Stopped early",
                      "insufficient_evidence": "No clear evidence"}
            failed = result["status"] in ("unsupported_request", "validation_failed", "incomplete")
            progress.update(label=labels.get(result["status"], "Answer ready"),
                            state="error" if failed else "complete", expanded=False)
        except Exception as exc:
            progress.update(label="Could not finish", state="error")
            st.error(safe_error_message(exc))
