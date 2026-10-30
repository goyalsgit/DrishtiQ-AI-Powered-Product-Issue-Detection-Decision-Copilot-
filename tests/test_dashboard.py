"""Browser-session workflow tests with a scripted model; no Gemini requests."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
try:
    from streamlit.testing.v1 import AppTest
    from langchain.messages import AIMessage, ToolMessage
except ImportError:
    raise unittest.SkipTest("Run dashboard tests in .venv-agent with requirements-dashboard.txt.")

from src.database import ReviewStore
from src.investigation import answer_text, run_investigation

BaseScriptedModel = importlib.import_module("lessons.10_gemini_agent").ScriptedModel


class ScriptedModel(BaseScriptedModel):
    """Real graph/tools plus a deterministic typed final response, not a quality eval."""
    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            data = json.loads(messages[-1].content)
            reviews = data["evidence"]
            findings = [
                {"kind": "positive_feedback" if review["text"] == "Good." else "reported_problem",
                 "statement": review["text"],
                 "citations": [{"review_id": review["review_id"], "month": review["month"],
                                "quote": review["text"]}]}
                for review in reviews
            ]
            # Like real Gemini with a dict schema: return plain parsed JSON.
            content = dict({
                "request": {"data_needed": ["reviews"], "product_scope": "selected_product",
                            "period_scope": "configured_periods", "focus": "both"},
                "evidence_status": "supported" if reviews else "metrics_only",
                "findings": findings, "possible_causes": [],
                "limitations": ["Small scripted test sample."], "next_steps": [],
            })
            return {"parsed": content, "raw": AIMessage(content=""), "parsing_error": None}
        return Mock(invoke=respond)
APP = Path(__file__).resolve().parents[1] / "app.py"


def submit(app):
    return next(b for b in app.button if b.label == "Investigate reviews").click().run()


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ReviewStore(Path(self.tmp.name) / "reviews.sqlite")
        self.dataset, _ = self.store.ingest(pd.DataFrame([
            {"review_id": "r1", "product": "A", "date": "2026-01-01", "rating": 1, "text": "Leaked."},
            {"review_id": "r2", "product": "A", "date": "2026-02-01", "rating": 5, "text": "Good."},
            {"review_id": "r3", "product": "B", "date": "2026-02-01", "rating": 1, "text": "Other."},
        ]), "Dashboard test", is_demo=True)
        self.search = Mock()
        self.search.ready.return_value = True
        self.search.search.side_effect = lambda store, dataset, product, month, question, limit: [
            {"review_id": row.review_id, "rating": row.rating, "text": row.text,
             "matched_passage": row.text.lower(), "similarity": 0.2}
            for row in store.read_page(dataset, product, month, limit=limit).itertuples()
        ]
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        for target, replacement in [
            ("src.database.ReviewStore", self.store),
            ("src.dashboard_ai.get_search", self.search),
            ("src.llm.make_llm", ScriptedModel("quality")),
            # Never read the developer's real .streamlit/secrets.toml in tests.
            ("src.dashboard_ai._secret", None),
        ]:
            patcher = patch(target, return_value=replacement)
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            if target == "src.llm.make_llm":
                self.llm = mock
        self.app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(self.app.exception)

    def fill(self):
        self.app.text_input(key="ai_api_key").set_value("test-session-credential")
        self.app.text_area(key="ai_question").set_value("What problems are reported?")

    def test_missing_key_and_blank_question_never_call_model(self):
        submit(self.app)
        self.assertTrue(any("Enter a question" in item.value for item in self.app.warning))
        self.app.text_area(key="ai_question").set_value("What is wrong?")
        submit(self.app)
        self.assertTrue(any("API key" in item.value for item in self.app.warning))
        self.llm.assert_not_called()

    def test_submit_retains_report_and_scope_change_clears_it(self):
        self.fill()
        submit(self.app)
        self.assertFalse(self.app.exception)
        saved = self.app.session_state["ai_result"]["report"]
        self.assertEqual(saved["status"], "checked_draft")
        self.assertEqual(saved["validation"]["status"], "passed")
        self.assertTrue(saved["report"]["content"]["findings"])
        self.assertEqual(saved["scope"]["product"], "A")
        self.assertEqual(saved["evidence"][0]["review_id"], "r2")
        self.assertNotIn("test-session-credential", json.dumps(saved))
        self.assertNotIn("GOOGLE_API_KEY", os.environ)
        self.llm.assert_called_once_with("gemini-3.5-flash-lite", api_key="test-session-credential")
        self.app.run()
        self.llm.assert_called_once()  # A rerun is not another paid request.
        self.assertTrue(any("Investigation report" in item.value for item in self.app.subheader))
        next(s for s in self.app.selectbox if s.label == "Product").select("B").run()
        self.assertFalse(self.app.exception)
        self.assertFalse(any("Investigation report" in item.value for item in self.app.subheader))

    def test_missing_index_and_forget_key(self):
        self.search.ready.return_value = False
        self.fill()
        submit(self.app)
        self.assertTrue(any("prepare this dataset" in item.value for item in self.app.warning))
        self.llm.assert_not_called()
        self.app.button(key="ai_forget_key").click().run()
        self.assertEqual(self.app.text_input(key="ai_api_key").value, "")

    def test_comparison_includes_evidence_for_both_periods(self):
        self.fill()
        self.app.checkbox(key="ai_compare").check()
        submit(self.app)
        self.assertFalse(self.app.exception)
        report = self.app.session_state["ai_result"]["report"]
        comparison = next(r for r in report["tool_results"] if r["tool"] == "compare_review_periods")
        self.assertEqual(comparison["data"]["baseline"]["month"], "2026-01")
        self.assertEqual(comparison["data"]["current"]["month"], "2026-02")
        self.assertEqual({r["month"] for r in report["evidence"]}, {"2026-01", "2026-02"})

    def test_simple_navigation_uses_database_without_csv_or_full_text_load(self):
        with patch("pandas.read_csv", side_effect=AssertionError("CSV should not be read")), \
                patch.object(self.store, "read", side_effect=AssertionError("Use SQL catalog/page")):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual([tab.label for tab in self.app.tabs],
                         ["Overview", "Ask reviews", "Reviews", "Data", "Project & quality"])
        self.assertFalse(any(s.label == "Search method" for s in self.app.radio))
        self.assertTrue(any("saved SQLite database" in m.value for m in self.app.markdown))

    def test_empty_database_does_not_silently_create_fictional_reviews(self):
        empty = ReviewStore(Path(self.tmp.name) / "empty.sqlite")
        with patch("src.database.ReviewStore", return_value=empty):
            app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertTrue(empty.datasets().empty)
        self.assertTrue(any("database is empty" in item.value for item in app.info))

    def test_provider_failure_does_not_expose_key(self):
        self.fill()
        self.llm.side_effect = RuntimeError("provider echoed test-session-credential")
        submit(self.app)
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.error)
        self.assertNotIn("test-session-credential", " ".join(e.value for e in self.app.error))
        self.assertFalse(any("Investigation report" in item.value for item in self.app.subheader))

    def test_invalid_report_is_withheld_in_dashboard(self):
        model = ScriptedModel("quality")
        model.with_structured_output = Mock(return_value=Mock(invoke=Mock(return_value={
            "parsed": {"invented_claim": "All products are defective"},
            "raw": AIMessage(content="Do not display this unvalidated answer"),
            "parsing_error": None,
        })))
        self.llm.return_value = model
        self.fill()
        submit(self.app)
        self.assertFalse(self.app.exception)
        report = self.app.session_state["ai_result"]["report"]
        self.assertEqual(report["status"], "validation_failed")
        self.assertIsNone(report["report"])
        self.assertTrue(any("withheld" in item.value for item in self.app.error))
        self.assertNotIn("All products are defective", json.dumps(report))

    def test_report_format_change_clears_an_old_session_report(self):
        self.fill()
        submit(self.app)
        import src.dashboard_ai as screen
        with patch("src.dashboard_ai.REPORT_FORMAT_VERSION", screen.REPORT_FORMAT_VERSION + 1):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertFalse(any("Investigation report" in item.value for item in self.app.subheader))

    def test_sales_question_shows_missing_data_instead_of_successful_review_report(self):
        class SalesModel(ScriptedModel):
            def with_structured_output(self, schema, **kwargs):
                content = {
                    "request": {"data_needed": ["sales", "reviews"],
                                "product_scope": "multiple_or_unresolved_products",
                                "period_scope": "configured_periods", "focus": "problems"},
                    "evidence_status": "insufficient_evidence", "findings": [], "possible_causes": [],
                    "limitations": ["Sales data is unavailable."], "next_steps": [],
                }
                return Mock(invoke=Mock(return_value={"parsed": content, "parsing_error": None,
                                                      "raw": AIMessage(content="")}))
        self.llm.return_value = SalesModel("quality")
        self.fill()
        self.app.text_area(key="ai_question").set_value(
            "which is the most selling product and what are the issue in that")
        submit(self.app)
        self.assertFalse(self.app.exception)
        report = self.app.session_state["ai_result"]["report"]
        self.assertEqual(report["status"], "unsupported_request")
        self.assertIsNone(report["report"])
        self.assertTrue(any("Sales/order quantities" in item.value for item in self.app.text))
        self.assertTrue(any(item.label == "Cannot answer with current data" for item in self.app.status))
        self.assertFalse(any("Positive feedback" in item.value for item in self.app.markdown))

    def test_content_blocks_render_text_only(self):
        self.assertEqual(answer_text([
            {"type": "text", "text": "Answer", "extras": {"signature": "private-metadata"}},
            {"type": "reasoning", "text": "hidden"},
        ]), "Answer")

    def test_limit_results_are_marked_incomplete(self):
        class LoopModel:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                return AIMessage(content="", tool_calls=[
                    {"name": "get_review_metrics", "args": {},
                     "id": "call-" + str(len(messages))}
                ])
        result = run_investigation(LoopModel(), self.store, self.dataset,
                                   "A", "2026-02", "Keep investigating")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("call limit", result["answer"])


if __name__ == "__main__":
    unittest.main()


class ServerKeyTests(unittest.TestCase):
    """With a server-side key, the browser never shows or needs a key field."""

    def test_server_key_hides_key_field_and_reaches_model(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ReviewStore(Path(tmp.name) / "reviews.sqlite")
        store.ingest(pd.DataFrame([{"review_id": "r1", "product": "A", "date": "2026-01-01",
                                    "rating": 1, "text": "Leaked."}]), "t", is_demo=True)
        search = Mock()
        search.ready.return_value = True
        search.search.return_value = []
        secrets = {"GOOGLE_API_KEY": "server-side-test-key"}
        patches = [patch("src.database.ReviewStore", return_value=store),
                   patch("src.dashboard_ai.get_search", return_value=search),
                   patch("src.dashboard_ai._secret", side_effect=secrets.get),
                   patch.dict(os.environ, {}, clear=True)]
        llm = patch("src.llm.make_llm", return_value=ScriptedModel("quality"))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        make_llm = llm.start()
        self.addCleanup(llm.stop)
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertFalse([w for w in app.text_input if w.key == "ai_api_key"])
        app.text_area(key="ai_question").set_value("What problems are reported?")
        submit(app)
        self.assertEqual(make_llm.call_args.kwargs["api_key"], "server-side-test-key")
