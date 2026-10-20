"""Offline execution checks; no API key or network needed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
try:
    from langchain.messages import AIMessage, HumanMessage, ToolMessage
except ImportError:
    raise unittest.SkipTest("Run agent tests with .venv-agent/bin/python.")
from src.agent import build_agent
from src.database import ReviewStore
from src.llm import make_llm
from src.tools import make_tools


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        return next(self.responses)


def request(name="get_review_metrics", args=None, call_id="call-1"):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args or {}, "id": call_id}
    ])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ReviewStore(Path(self.tmp.name) / "reviews.sqlite")
        self.dataset, _ = self.store.ingest(pd.DataFrame([
            {"review_id": "r1", "product": "A", "date": "2026-01-01", "rating": 1, "text": "Leaked."},
            {"review_id": "r2", "product": "A", "date": "2026-01-02", "rating": 5, "text": "Good."},
            {"review_id": "r3", "product": "B", "date": "2026-01-01", "rating": 1, "text": "Other product."},
        ]), "test")
        self.search = Mock()
        self.search.search.return_value = [{"review_id": "r1", "text": "Leaked."}]
        self.tools = make_tools(self.store, self.dataset, "A", "2026-01", self.search)

    def run_graph(self, model, **kwargs):
        return build_agent(model, self.tools, **kwargs).invoke({
            "messages": [HumanMessage(content="Investigate reviews")],
            "model_calls": 0, "tool_calls": 0,
        })

    def test_real_sql_scope_and_result_returned_to_model(self):
        model = SequenceModel([request(), AIMessage(content="Draft.")])
        result = self.run_graph(model)
        observation = next(m for m in model.inputs[1] if isinstance(m, ToolMessage))
        self.assertEqual(json.loads(observation.content)["total_reviews"], 2)
        self.assertEqual(json.loads(observation.content)["low_rating_share_percent"], 50)
        self.assertEqual(observation.tool_call_id, "call-1")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["tool_calls"], 1)

    def test_search_uses_bound_scope(self):
        model = SequenceModel([request("search_review_evidence", {"question": "leaking"}),
                               AIMessage(content="Draft.")])
        self.run_graph(model)
        self.search.search.assert_called_once_with(
            self.store, self.dataset, "A", "2026-01", "leaking", limit=6)

    def test_comparison_result_reaches_model(self):
        self.tools = make_tools(self.store, self.dataset, "A", "2026-01",
                                self.search, baseline_month="2025-12")
        model = SequenceModel([request("compare_review_periods"), AIMessage(content="Missing baseline.")])
        self.run_graph(model)
        observation = next(m for m in model.inputs[1] if isinstance(m, ToolMessage))
        result = json.loads(observation.content)
        self.assertEqual(result["status"], "missing_period_data")
        self.assertEqual(result["current"]["total_reviews"], 2)
        self.assertIsNone(result["low_rating_share_change_percentage_points"])

    def test_baseline_search_uses_only_configured_month(self):
        self.tools = make_tools(self.store, self.dataset, "A", "2026-01",
                                self.search, baseline_month="2025-12")
        model = SequenceModel([
            request("search_review_evidence", {"question": "leak", "period": "baseline"}),
            AIMessage(content="Draft."),
        ])
        result = self.run_graph(model)
        self.search.search.assert_called_once_with(
            self.store, self.dataset, "A", "2025-12", "leak", limit=6)
        self.assertEqual(json.loads(result["messages"][-2].content)["scope"]["month"], "2025-12")

    def test_unconfigured_or_arbitrary_search_period_is_rejected(self):
        for period in ["baseline", "2026-02"]:
            with self.subTest(period=period):
                model = SequenceModel([
                    request("search_review_evidence", {"question": "leak", "period": period}),
                    AIMessage(content="Unavailable."),
                ])
                result = self.run_graph(model)
                self.assertEqual(result["messages"][-2].status, "error")
        self.search.search.assert_not_called()

    def test_comparison_cannot_override_bound_scope(self):
        self.tools = make_tools(self.store, self.dataset, "A", "2026-01",
                                self.search, baseline_month="2025-12")
        for args in [{"product": "B"}, {"baseline_month": "2020-01"}, {"minimum_reviews": 1}]:
            with self.subTest(args=args):
                model = SequenceModel([request("compare_review_periods", args), AIMessage(content="Invalid.")])
                result = self.run_graph(model)
                self.assertEqual(result["messages"][-2].status, "error")

    def test_unknown_tool_returns_error(self):
        model = SequenceModel([request("delete_reviews"), AIMessage(content="Cannot do that.")])
        result = self.run_graph(model)
        self.assertEqual(result["messages"][-2].status, "error")
        self.assertEqual(len(self.store.read(self.dataset)), 3)

    def test_invalid_search_does_not_execute(self):
        model = SequenceModel([
            request("search_review_evidence", {"question": " ", "product": "B"}),
            AIMessage(content="Need a valid query."),
        ])
        result = self.run_graph(model)
        self.assertEqual(result["messages"][-2].status, "error")
        self.search.search.assert_not_called()

    def test_tool_failure_does_not_expose_raw_exception(self):
        self.search.search.side_effect = RuntimeError("sensitive internal configuration")
        model = SequenceModel([request("search_review_evidence", {"question": "leak"}),
                               AIMessage(content="Unavailable.")])
        result = self.run_graph(model)
        observation = result["messages"][-2]
        self.assertEqual(observation.status, "error")
        self.assertNotIn("sensitive", observation.content)

    def test_model_limit_stops_before_unnecessary_tool_execution(self):
        model = SequenceModel([request("search_review_evidence", {"question": "leak"})])
        result = self.run_graph(model, max_model_calls=1)
        self.search.search.assert_not_called()
        self.assertIn("call limit", result["messages"][-1].content)
        self.assertEqual(result["tool_calls"], 0)

    def test_tool_budget_counts_multiple_requests(self):
        response = AIMessage(content="", tool_calls=[
            {"name": "get_review_metrics", "args": {}, "id": str(i)} for i in range(2)
        ])
        result = self.run_graph(SequenceModel([response]), max_tool_calls=1)
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(len([m for m in result["messages"] if isinstance(m, ToolMessage)]), 2)

    def test_direct_response_finishes(self):
        result = self.run_graph(SequenceModel([AIMessage(content="Please clarify.")]))
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["model_calls"], 1)

    def test_missing_credentials_fail_clearly(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
                make_llm()


if __name__ == "__main__":
    unittest.main()
