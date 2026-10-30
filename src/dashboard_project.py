"""A concise, evidence-based account of project behavior and measured quality."""
import hashlib
import pandas as pd
import streamlit as st
from src.project_status import ROOT, read_report, verification_status


def render_project_status():
    st.subheader("What this project does")
    st.write("DrishtiQ helps a product or support analyst investigate customer-reported problems. "
             "It combines exact rating statistics with relevant reviews, then drafts an explanation.")
    st.code("Question → Gemini requests tools → SQL / review search → Typed report → Source checks → Draft", language=None)
    st.dataframe(pd.DataFrame([
        ("SQLite", "Stores original reviews and calculates counts and ratings."),
        ("MiniLM + Chroma", "Turns search text into vectors and retrieves similar review passages."),
        ("LangChain", "Exposes Python functions as tools the model can request."),
        ("LangGraph", "Runs the model/tool loop with call limits."),
        ("Gemini", "Chooses tool requests and explains their results."),
        ("Pydantic + source checks", "Enforces report fields, cited review IDs/periods and exact source quotes."),
    ], columns=["Component", "Responsibility"]), hide_index=True, width="stretch")
    st.caption("The agent does not write SQL. It cannot prove a root cause from reviews alone. "
               "Reported issues and possible causes must remain distinct.")

    st.subheader("What has been tested")
    checks, current = verification_status()
    if checks:
        if current and checks["passed"] and not checks["skipped"]:
            st.success(f'{checks["tests_run"]} offline tests passed for the current source.')
        else:
            st.info("The saved test run is not a complete passing check of this source. Run verification again.")
        st.caption(f'Run: {checks["created_utc"]} · Failures: {checks["failures"]} · '
                   f'Errors: {checks["errors"]} · Skipped: {checks["skipped"]}')
    else:
        st.info("No saved test run yet.")
    st.write("Checks cover database persistence and scope, metrics, period comparison, tool routing, "
             "call limits, report structure and quotes, session-key handling, errors and stale-report clearing.")
    st.caption("Scripted models are used for automated tests. Passing tests is not a Gemini accuracy score.")

    current_report = ROOT / "reports/retrieval_current.json"
    report = read_report(current_report) or read_report(ROOT / "reports/retrieval_evaluation.json")
    if report:
        st.subheader("Measured retrieval quality")
        st.caption(f'Development diagnostic · {report["case_count"]} questions · '
                   f'Top {report["k"]} results · {report["created_utc"]}')
        st.dataframe(pd.DataFrame([{
            "Search": method,
            f"Precision@{report['k']}": f'{values["macro_precision_at_k"]:.1%}',
            f"Recall@{report['k']}": f'{values["macro_recall_at_k"]:.1%}',
            "No-evidence queries returning results": f'{values["unsupported_return_rate"]:.0%}',
            "Out-of-scope results": values["scope_violations"],
        } for method, values in report["summary"].items()]), hide_index=True, width="stretch")
        st.write("**Precision:** how many result slots contained relevant reviews. "
                 "**Recall:** how many labelled relevant reviews were found.")
        st.warning("Labels still need independent human review. This is a small development set, "
                   "not an unseen test set or end-to-end answer accuracy. "
                   "Returning matches for no-evidence questions is an unresolved weakness.")
        agent_report = read_report(ROOT / "reports/retrieval_agent_k6.json")
        if agent_report:
            values = agent_report["summary"]["semantic"]
            st.caption(f'The agent requests 6 reviews per search. Its top-6 diagnostic measured '
                       f'{values["macro_precision_at_k"]:.1%} precision and '
                       f'{values["macro_recall_at_k"]:.1%} recall on the same development questions. '
                       'This tests direct retrieval, not model-written queries or final answers.')
        manifest = ROOT / "tests/retrieval_cases.json"
        if manifest.exists() and hashlib.sha256(manifest.read_bytes()).hexdigest() != report.get("cases_sha256"):
            st.warning("The question labels changed after this report; rerun retrieval evaluation.")
        with st.expander("Reproduce these checks"):
            st.code(".venv-agent/bin/python -m scripts.verify_project\n"
                    ".venv-agent/bin/python -m scripts.evaluate_retrieval --k 3 "
                    "--output reports/retrieval_current.json")
            st.caption("Retrieval results belong to the benchmark dataset, not necessarily the current sidebar selection.")

    st.subheader("Remaining work")
    st.dataframe(pd.DataFrame([
        ("1", "Evaluate AI report meaning", "Structure/quotes are checked; claim support and narrative numbers still need evaluation."),
        ("2", "Improve evidence quality", "Human-reviewed labels, unseen questions and calibrated abstention."),
        ("3", "Save and approve investigations", "Persistent reports, human approval and audit history."),
        ("4", "Connect operational data", "Read-only PostgreSQL/MySQL adapter, data refresh and index updates."),
        ("5", "Prepare for deployment", "Authentication, backups, monitoring and end-to-end evaluation."),
    ], columns=["Order", "Work", "Completion criterion"]), hide_index=True, width="stretch")
