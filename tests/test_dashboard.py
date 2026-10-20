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

ScriptedModel = importlib.import_module("lessons.10_gemini_agent").ScriptedModel
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
        self.search.search.return_value = [
            {"review_id": "r2", "rating": 5, "text": "Good.", "matched_passage": "good.",
             "similarity": 0.2}
        ]
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        for target, replacement in [
            ("src.database.ReviewStore", self.store),
            ("src.dashboard_ai.get_search", self.search),
            ("src.llm.make_llm", ScriptedModel("quality")),
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
        self.assertEqual(saved["status"], "draft")
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

    def test_comparison_and_metrics_only(self):
        self.fill()
        self.app.checkbox(key="ai_compare").check()
        self.app.checkbox(key="ai_semantic").uncheck()
        submit(self.app)
        self.assertFalse(self.app.exception)
        report = self.app.session_state["ai_result"]["report"]
        comparison = next(r for r in report["tool_results"] if r["tool"] == "compare_review_periods")
        self.assertEqual(comparison["data"]["baseline"]["month"], "2026-01")
        self.assertEqual(comparison["data"]["current"]["month"], "2026-02")
        self.assertEqual(report["evidence"], [])

    def test_provider_failure_does_not_expose_key(self):
        self.fill()
        self.llm.side_effect = RuntimeError("provider echoed test-session-credential")
        submit(self.app)
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.error)
        self.assertNotIn("test-session-credential", " ".join(e.value for e in self.app.error))
        self.assertFalse(any("Investigation report" in item.value for item in self.app.subheader))

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
