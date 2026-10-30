"""Adversarial structure/provenance checks; these do not grade semantic meaning."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

try:
    from langchain.messages import AIMessage
except ImportError:
    raise unittest.SkipTest("Run report tests in .venv-agent.")
from src.reporting import InvestigationContent, checked_content, finalize_report


def payload():
    return {
        "request": {"data_needed": ["reviews"], "product_scope": "selected_product",
                    "period_scope": "configured_periods", "focus": "problems"},
        "evidence_status": "supported",
        "findings": [{"kind": "reported_problem", "statement": "A customer reported leakage.",
                      "citations": [{"review_id": "r1", "month": "2026-01",
                                     "quote": "The bottle leaked."}]}],
        "possible_causes": [], "limitations": ["Only one retrieved review."],
        "next_steps": ["Inspect the returned bottle."],
    }


def result():
    return {
        "question": "What happened?", "scope": {"dataset_id": "d", "product": "P", "month": "2026-01"},
        "answer": "Unvalidated raw draft", "trace": [], "tool_errors": [],
        "evidence": [{"review_id": "r1", "month": "2026-01", "text": "The bottle leaked."}],
        "tool_results": [{"tool": "get_review_metrics", "status": "success", "data": {
            "scope": {"dataset_id": "d", "product": "P", "month": "2026-01"},
            "total_reviews": 11, "average_rating": 3.18,
            "low_rating_reviews": 3, "low_rating_share_percent": 100*3/11}}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


class ReportingTests(unittest.TestCase):
    def model(self, content, parsing_error=None):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = {
            "parsed": content, "parsing_error": parsing_error,
            "raw": AIMessage(content="private raw model text", usage_metadata={
                "input_tokens": 7, "output_tokens": 4, "total_tokens": 11}),
        }
        return model

    def test_valid_report_has_source_quotes_sql_statistics_and_all_usage(self):
        data = result()
        model = self.model(payload())
        finalize_report(model, data)
        self.assertEqual(data["status"], "checked_draft")
        self.assertEqual(data["validation"]["semantic_grounding"], "not_evaluated")
        self.assertEqual(data["report"]["statistics"][0]["total_reviews"], 11)
        self.assertEqual(data["usage"]["total_tokens"], 26)
        self.assertEqual(data["answer"], "")
        self.assertNotIn("Unvalidated raw draft", json.dumps(data))
        self.assertNotIn("private raw", json.dumps(data))
        from src.reporting import gemini_schema
        model.with_structured_output.assert_called_once_with(
            gemini_schema(), method="json_schema", include_raw=True)
        model.with_structured_output.return_value.invoke.assert_called_once()
        prompt = json.loads(model.with_structured_output.return_value.invoke.call_args.args[0][-1].content)
        self.assertEqual(prompt["evidence"][0]["text"], "The bottle leaked.")
        self.assertNotIn("matched_passage", json.dumps(prompt))

    def test_unknown_id_wrong_month_invented_quote_are_rejected(self):
        for field, value in [("review_id", "r999"), ("month", "2026-02"),
                             ("quote", "The factory used cheap plastic.")]:
            with self.subTest(field=field):
                content = payload()
                content["findings"][0]["citations"][0][field] = value
                data = result()
                finalize_report(self.model(content), data)
                self.assertEqual(data["status"], "validation_failed")
                self.assertIsNone(data["report"])
                self.assertNotIn(value, json.dumps(data["validation"]))

    def test_missing_extra_wrong_type_and_empty_citations_fail_closed(self):
        missing = payload()
        del missing["limitations"]
        extra = {**payload(), "total_reviews": 999}
        wrong = {**payload(), "findings": "looks okay"}
        empty = payload()
        empty["findings"][0]["citations"] = []
        for content in [missing, extra, wrong, empty]:
            data = result()
            finalize_report(self.model(content), data)
            self.assertEqual(data["status"], "validation_failed")
            self.assertIsNone(data["report"])

    def test_parser_errors_are_sanitized_and_usage_is_counted(self):
        data = result()
        finalize_report(self.model(None, RuntimeError("secret provider detail")), data)
        self.assertEqual(data["status"], "validation_failed")
        self.assertEqual(data["usage"]["total_tokens"], 26)
        self.assertNotIn("secret provider detail", json.dumps(data))

    def test_no_successful_tools_refuses_after_request_assessment(self):
        data = result()
        data["tool_results"] = [{"tool": "search_review_evidence", "status": "error", "data": {}}]
        model = self.model(payload())
        finalize_report(model, data)
        self.assertEqual(data["status"], "insufficient_evidence")
        model.with_structured_output.assert_called_once()
        self.assertIsNone(data["report"])

    def test_irrelevant_search_can_return_insufficient_evidence(self):
        content = {**payload(), "evidence_status": "insufficient_evidence", "findings": []}
        data = result()
        finalize_report(self.model(content), data)
        self.assertEqual(data["status"], "insufficient_evidence")
        self.assertEqual(data["report"]["content"]["findings"], [])
        self.assertEqual(data["validation"]["status"], "passed")

    def test_metrics_only_needs_sql_and_cannot_assert_review_findings(self):
        content = {**payload(), "evidence_status": "metrics_only", "findings": []}
        content["request"]["focus"] = "metrics"
        data = result()
        finalize_report(self.model(content), data)
        self.assertEqual(data["status"], "checked_draft")
        data = result()
        data["tool_results"][0]["tool"] = "search_review_evidence"
        finalize_report(self.model(content), data)
        self.assertEqual(data["status"], "validation_failed")
        data = result()
        content["findings"] = payload()["findings"]
        finalize_report(self.model(content), data)
        self.assertEqual(data["status"], "validation_failed")

    def test_hypothesis_requires_valid_citation_and_verification_step(self):
        content = payload()
        content["possible_causes"] = [{
            "hypothesis": "The seal may have failed.",
            "citations": deepcopy(content["findings"][0]["citations"]),
            "verification_needed": "Inspect the seal; the review does not prove its condition.",
        }]
        self.assertTrue(checked_content(content, result()["evidence"]).possible_causes)
        content["possible_causes"][0]["verification_needed"] = ""
        with self.assertRaises(ValueError):
            checked_content(content, result()["evidence"])

    def test_matching_quote_does_not_prove_semantic_truth(self):
        # Deliberately show the validator's boundary: entailment is a separate eval.
        content = payload()
        content["findings"][0]["statement"] = "Every product is defective."
        checked = checked_content(content, result()["evidence"])
        self.assertEqual(checked.findings[0].statement, "Every product is defective.")

    def test_sales_request_cannot_be_replaced_with_selected_product_feedback(self):
        content = payload()
        content["request"].update(data_needed=["sales", "reviews"],
                                  product_scope="multiple_or_unresolved_products")
        # Even if the model returns a structurally valid, source-matching finding,
        # the missing sales requirement prevents its presentation as an answer.
        data = result()
        data["question"] = "which is the most selling product and what are the issue in that"
        finalize_report(self.model(content), data)
        self.assertEqual(data["status"], "unsupported_request")
        self.assertIsNone(data["report"])
        self.assertIn("Sales/order quantities", data["answer"])
        self.assertEqual(data["request_check"]["status"], "unsupported")
        self.assertEqual(data["usage"]["total_tokens"], 26)

    def test_unsupported_sales_is_explained_even_without_any_tool_calls(self):
        content = payload()
        content["request"]["data_needed"] = ["sales"]
        data = result()
        data["tool_results"] = []
        data["evidence"] = []
        finalize_report(self.model(content), data)
        self.assertEqual(data["status"], "unsupported_request")
        self.assertIn("do not establish sales", data["answer"])

    def test_missing_scope_and_operational_data_are_blocked(self):
        for field, value in [("product_scope", "multiple_or_unresolved_products"),
                             ("period_scope", "other_or_unresolved_periods"),
                             ("data_needed", ["inventory"]), ("data_needed", ["returns"])]:
            content = payload()
            content["request"][field] = value
            data = result()
            finalize_report(self.model(content), data)
            self.assertEqual(data["status"], "unsupported_request")

    def test_problem_question_does_not_display_only_praise_as_an_answer(self):
        content = payload()
        content["findings"][0]["kind"] = "positive_feedback"
        data = result()
        finalize_report(self.model(content), data)
        self.assertEqual(data["status"], "insufficient_evidence")
        self.assertEqual(data["report"]["content"]["findings"], [])

    def test_problem_focus_keeps_concern_from_mixed_review(self):
        content = payload()
        quote = "It is beautiful but is much darker in person. Great quality."
        data = result()
        data["evidence"][0]["text"] = quote
        content["findings"][0].update(statement="The item appeared darker in person.")
        content["findings"][0]["citations"][0]["quote"] = "much darker in person."
        praise = deepcopy(content["findings"][0])
        praise.update(kind="positive_feedback", statement="The customer praised quality.")
        praise["citations"][0]["quote"] = "Great quality."
        content["findings"].append(praise)
        finalize_report(self.model(content), data)
        self.assertEqual(data["status"], "checked_draft")
        self.assertEqual(len(data["report"]["content"]["findings"]), 1)
        self.assertEqual(data["report"]["content"]["findings"][0]["kind"], "reported_problem")


class ActionPlanTests(unittest.TestCase):
    """Suggested actions: cited, routed by the fixed playbook, never for praise."""

    def run_report(self, content):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = {
            "parsed": content, "parsing_error": None, "raw": AIMessage(content="")}
        data = result()
        finalize_report(model, data)
        return data

    def action(self, problem_type="packaging_or_leakage", quote="The bottle leaked."):
        return {"problem_type": problem_type, "suggestion": "Check the cap seal.",
                "citations": [{"review_id": "r1", "month": "2026-01", "quote": quote}]}

    def test_action_gets_owner_priority_and_first_step_from_playbook(self):
        content = payload()
        content["recommended_actions"] = [self.action()]
        plan = self.run_report(content)["report"]["action_plan"]
        self.assertEqual(plan[0]["owner"], "Packaging")
        self.assertEqual(plan[0]["priority"], "high")
        self.assertEqual(plan[0]["evidence_review_ids"], ["r1"])
        self.assertEqual(plan[0]["status"], "awaiting_manager_decision")

    def test_safety_actions_are_listed_first(self):
        content = payload()
        content["recommended_actions"] = [self.action("price_or_value"),
                                          self.action("skin_reaction_safety")]
        plan = self.run_report(content)["report"]["action_plan"]
        self.assertEqual([p["priority"] for p in plan], ["urgent", "low"])

    def test_action_with_invented_quote_is_removed_not_shown(self):
        content = payload()
        content["recommended_actions"] = [self.action(quote="The cap was cracked.")]
        data = self.run_report(content)
        self.assertEqual(data["status"], "checked_draft")      # real finding survives
        self.assertEqual(data["report"]["action_plan"], [])     # invented quote is gone
        self.assertNotIn("cap was cracked", json.dumps(data["report"]))
        self.assertTrue(any("removed" in x for x in data["report"]["content"]["limitations"]))

    def test_report_with_only_invented_quotes_is_withheld(self):
        content = payload()
        content["findings"][0]["citations"][0]["quote"] = "Totally invented text."
        self.assertEqual(self.run_report(content)["status"], "validation_failed")

    def test_unknown_problem_type_is_rejected(self):
        content = payload()
        content["recommended_actions"] = [self.action("fire_the_supplier")]
        self.assertEqual(self.run_report(content)["status"], "validation_failed")

    def test_no_actions_without_supported_evidence(self):
        content = payload()
        content.update(evidence_status="insufficient_evidence", findings=[],
                       recommended_actions=[self.action()])
        self.assertEqual(self.run_report(content)["status"], "validation_failed")

    def test_praise_question_gets_no_actions(self):
        content = payload()
        content["request"]["focus"] = "praise"
        content["findings"][0]["kind"] = "positive_feedback"
        content["recommended_actions"] = [self.action()]
        plan = self.run_report(content)["report"]["action_plan"]
        self.assertEqual(plan, [])

    def test_old_reports_without_actions_still_work(self):
        data = self.run_report(payload())
        self.assertEqual(data["status"], "checked_draft")
        self.assertEqual(data["report"]["action_plan"], [])


class GeminiSchemaTests(unittest.TestCase):
    """The provider gets a relaxed schema; Python still enforces every limit."""

    def test_provider_schema_has_no_limits_but_keeps_fields_and_values(self):
        from src.reporting import gemini_schema
        text = json.dumps(gemini_schema())
        for word in ("maxItems", "minItems", "maxLength", "minLength", "pattern"):
            self.assertNotIn(f'"{word}"', text)
        self.assertIn("recommended_actions", text)
        self.assertIn("skin_reaction_safety", text)

    def test_limits_removed_from_schema_are_still_enforced_in_python(self):
        content = payload()
        content["limitations"] = [f"limit {i}" for i in range(9)]   # max is 8
        with self.assertRaises(ValueError):
            checked_content(content, result()["evidence"])
        content = payload()
        content["findings"][0]["citations"][0]["month"] = "April"   # pattern YYYY-MM
        with self.assertRaises(ValueError):
            checked_content(content, result()["evidence"])


class QuoteCheckTests(unittest.TestCase):
    """Real failure seen with live Gemini: two parts joined with '...'."""
    SOURCE = ("I picked this model because it has a dip, but it didn't really help much "
              "with my tailbone pain. Comfortable aside from that. This wasn't worth the cost.")

    def test_exact_quote(self):
        from src.reporting import quote_in_source
        self.assertTrue(quote_in_source("Comfortable aside from that.", self.SOURCE))

    def test_ellipsis_joins_real_parts_in_order(self):
        from src.reporting import quote_in_source
        self.assertTrue(quote_in_source(
            "it didn't really help much with my tailbone pain... wasn't worth the cost.", self.SOURCE))
        self.assertTrue(quote_in_source("...tailbone pain…worth the cost", self.SOURCE))

    def test_invented_or_reordered_parts_fail(self):
        from src.reporting import quote_in_source
        self.assertFalse(quote_in_source("tailbone pain... it was dangerous", self.SOURCE))
        self.assertFalse(quote_in_source("worth the cost... tailbone pain", self.SOURCE))
        self.assertFalse(quote_in_source("...", self.SOURCE))
        self.assertFalse(quote_in_source("Tailbone Pain", self.SOURCE))   # case must match


class CustomerRequestTests(unittest.TestCase):
    """'What are customers demanding?' must return requests, not complaints/praise."""

    def run_report(self, content, evidence):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = {
            "parsed": content, "parsing_error": None, "raw": AIMessage(content="")}
        data = result()
        data["evidence"] = evidence
        finalize_report(model, data)
        return data

    def test_requests_focus_keeps_only_requests(self):
        evidence = [{"review_id": "r1", "month": "2026-01", "text": "Wish it came with a strap."},
                    {"review_id": "r2", "month": "2026-02", "text": "Love the smell."}]
        content = payload()
        content["request"]["focus"] = "requests"
        content["findings"] = [
            {"kind": "customer_request", "statement": "Customers want a strap.",
             "citations": [{"review_id": "r1", "month": "2026-01", "quote": "Wish it came with a strap."}]},
            {"kind": "positive_feedback", "statement": "Customers like the smell.",
             "citations": [{"review_id": "r2", "month": "2026-02", "quote": "Love the smell."}]}]
        data = self.run_report(content, evidence)
        kinds = [f["kind"] for f in data["report"]["content"]["findings"]]
        self.assertEqual(kinds, ["customer_request"])

    def test_requests_focus_without_requests_is_insufficient(self):
        content = payload()
        content["request"]["focus"] = "requests"   # only a reported_problem finding
        data = self.run_report(content, result()["evidence"])
        self.assertEqual(data["status"], "insufficient_evidence")
