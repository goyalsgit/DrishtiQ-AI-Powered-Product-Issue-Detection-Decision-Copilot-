"""Streamlit interface for asking one scoped review question at a time."""
import importlib.util
import json
import os
from datetime import datetime, timezone

import streamlit as st


@st.cache_resource
def get_search():
    from src.embeddings import ReviewSearch, load_model
    return ReviewSearch(load_model())


def reset_key():
    st.session_state["ai_api_key"] = ""


def select_question(text):
    st.session_state["ai_question"] = text


def safe_error_message(exc):
    """Actionable categories without printing raw provider errors or credentials."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (401, 403):
        return "Gemini rejected access. Check your API key and model permissions."
    if code == 429:
        return "Gemini's quota or rate limit was reached. Check your quota and retry later."
    if code == 404:
        return "The selected Gemini model is unavailable. Check the model name in AI settings."
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "The request could not reach Gemini or timed out. Check your connection and retry."
    return ("The investigation could not finish. Check your key, Gemini model access, "
            "quota and network connection, then retry. No complete report was saved.")


def show_result(result):
    st.divider()
    st.subheader("Investigation report")
    st.caption(f'Question: {result["question"]}')
    if result["status"] == "incomplete":
        st.warning("The agent reached its limit or returned no answer. This investigation is incomplete.")
    elif result["tool_errors"]:
        st.warning("Some tools failed. Read the limitations and tool activity before using this draft.")
    st.caption("AI draft • Check the evidence before acting. Possible causes are not confirmed root causes.")
    st.markdown(result["answer"] or "No answer was returned.")

    evidence = result["evidence"]
    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"Review evidence · {len(evidence)}")
        st.caption("These are retrieved reviews, not a complete count of complaints. Similarity is not confidence.")
        if not evidence:
            st.info("No review evidence was retrieved in this run. Review-based claims are not substantiated by a search result.")
        for number, review in enumerate(evidence, 1):
            with st.expander(f'{number}. {review["rating"]} stars · {review["month"]} · {review["period"]}'):
                st.code(review["review_id"], language=None)
                st.text(review["text"])
                st.caption(f'Semantic similarity: {review.get("similarity", "unavailable")}')
    with right:
        st.subheader("Database results")
        metrics = [r for r in result["tool_results"]
                   if r["status"] == "success" and r["tool"] in
                   ("get_review_metrics", "compare_review_periods")]
        if not metrics:
            st.caption("The agent did not request rating metrics.")
        for row in metrics:
            with st.expander("Period comparison" if row["tool"] == "compare_review_periods" else "Rating statistics"):
                st.json(row["data"])
        st.caption(f'Tokens reported across this run: {result["usage"]["total_tokens"]:,}')

    with st.expander("How the agent investigated"):
        for step, event in enumerate(result["trace"], 1):
            if event["kind"] == "requests":
                st.write(f"{step}. Gemini requested tools")
                st.json(event["calls"])
            elif event["kind"] == "result":
                st.write(f'{step}. {event["tool"]} returned {event["status"]}')
                st.json(event["data"])
            else:
                st.write(f"{step}. A response was produced")
    st.download_button(
        "Download report and evidence", json.dumps(result, ensure_ascii=False, indent=2),
        file_name="drishtiq-investigation.json", mime="application/json",
        key="ai_download",
    )
    st.caption("This download is a draft report, not an approval or an executed action.")


def render_investigation(store, dataset_id, product, month, baseline, minimum_reviews):
    st.subheader("Ask about your customer reviews")
    st.write("Find reported problems, inspect the supporting reviews, and identify what to investigate next.")
    if importlib.util.find_spec("langchain") is None:
        st.info("Start the AI dashboard with the Python 3.12 agent environment.")
        st.code(".venv-agent/bin/python -m streamlit run app.py")
        return

    with st.expander("AI settings · add your Gemini key", expanded=not bool(
        st.session_state.get("ai_api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )):
        st.text_input("Gemini API key", type="password", key="ai_api_key",
                      placeholder="Paste your key here",
                      help="Used only for this browser session. Never written to a project file.")
        st.button("Forget entered key", on_click=reset_key, key="ai_forget_key")
        st.text_input("Gemini model", key="ai_model",
                      value=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
        st.caption("You can also use a GOOGLE_API_KEY or GEMINI_API_KEY environment variable on the server.")

    compare = st.checkbox("Compare with the baseline month", value=False,
                          disabled=baseline is None, key="ai_compare")
    compare_month = baseline if compare else None
    semantic = st.checkbox("Use customer review evidence", value=True, key="ai_semantic")
    st.caption(
        f'Product: {product} · Analysis: {month}'
        + (f" · Baseline: {compare_month}" if compare_month else " · Single period")
        + " · Change the selection in the sidebar."
    )
    if compare and baseline is None:
        st.info("This product has no earlier month available. The investigation will use the analysis month.")

    suggestions = [
        ("Find problems", "What problems are customers reporting? Use rating metrics and relevant reviews. Separate reported problems from possible causes."),
        ("Investigate quality", "Find evidence of product quality problems. Cite review IDs and say what remains uncertain."),
        ("Compare periods", "Compare the selected periods. Search relevant reviews in both and explain what changed, without assuming a cause."),
    ]
    for column, (label, question) in zip(st.columns(3), suggestions):
        column.button(label, key="ai_example_" + label, on_click=select_question,
                      args=(question,), disabled=label == "Compare periods" and not compare_month,
                      use_container_width=True)

    with st.form("ai_question_form"):
        question = st.text_area(
            "Your question", key="ai_question", height=110, max_chars=4000,
            placeholder="What problems are customers reporting, and which reviews support them?",
        )
        submitted = st.form_submit_button("Investigate reviews", type="primary", use_container_width=True)
        st.caption("Sends your question and requested review evidence to Gemini. Each submission starts a new investigation.")

    if semantic:
        with st.expander("Prepare review search"):
            st.caption("For a newly imported dataset, prepare the local search index once. No Gemini request is made.")
            if st.button("Prepare this dataset", key="ai_prepare"):
                try:
                    with st.spinner("Preparing review search…"):
                        count = get_search().build(store, dataset_id)
                    st.success(f"{count:,} passages ready for semantic search.")
                except Exception:
                    st.error("Could not prepare search. Check the local MiniLM cache and install requirements-dashboard.txt.")

    signature = (dataset_id, product, month, compare_month, minimum_reviews, semantic,
                 st.session_state.get("ai_model", "").strip())
    previous = st.session_state.get("ai_result")
    if previous and previous["selection"] != signature:
        st.session_state.pop("ai_result", None)
        st.info("The selection changed. Run a new investigation for this scope.")

    if submitted:
        st.session_state.pop("ai_result", None)
        if not question.strip():
            st.warning("Enter a question first.")
            return
        key = st.session_state.get("ai_api_key", "").strip()
        if not (key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            st.warning("Add your Gemini API key in AI settings above.")
            return
        model_name = st.session_state.get("ai_model", "").strip()
        if not model_name:
            st.warning("Enter a Gemini model name in AI settings.")
            return
        with st.status("Investigating your reviews…", expanded=True) as progress:
            try:
                search = None
                if semantic:
                    progress.write("Loading local review search…")
                    search = get_search()
                    if not search.ready(dataset_id):
                        progress.update(label="Prepare review search first", state="error")
                        st.warning("Open 'Prepare review search' above and prepare this dataset, then submit again.")
                        return
                from src.llm import make_llm
                from src.investigation import run_investigation
                model = make_llm(model_name, api_key=key or None)

                def on_event(event):
                    if event["kind"] == "requests":
                        for call in event["calls"]:
                            progress.write(f'Requesting {call["name"]}…')
                    elif event["kind"] == "result":
                        progress.write(f'{event["tool"]}: {event["status"]}')
                    else:
                        progress.write("Writing the report…")

                result = run_investigation(
                    model, store, dataset_id, product, month, question,
                    search=search, baseline_month=compare_month,
                    minimum_reviews=minimum_reviews, on_event=on_event,
                )
                result["created_utc"] = datetime.now(timezone.utc).isoformat()
                result["model"] = model_name
                st.session_state["ai_result"] = {"selection": signature, "report": result}
                incomplete = result["status"] == "incomplete"
                progress.update(label="Investigation incomplete" if incomplete else "Draft report ready",
                                state="error" if incomplete else "complete", expanded=False)
            except Exception as exc:
                progress.update(label="Investigation could not finish", state="error")
                st.error(safe_error_message(exc))
    saved = st.session_state.get("ai_result")
    if saved:
        show_result(saved["report"])
