"""LangChain tools bound to the dataset, product and periods selected by the app."""
from typing import Literal
from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field
from src.comparison import compare_periods, validate_comparison


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)
    period: Literal["current", "baseline"] = Field(
        default="current",
        description="Search current month, or the baseline month if configured by the app.",
    )


def make_tools(store, dataset_id, product, month, search=None, *,
               baseline_month=None, minimum_reviews=20):
    """Always expose metrics; optionally expose an initialized ReviewSearch."""
    scope = {"dataset_id": dataset_id, "product": product, "month": month}
    if baseline_month is not None:
        validate_comparison(month, baseline_month, minimum_reviews)

    @tool(args_schema=NoArguments)
    def get_review_metrics() -> dict:
        """Get exact counts and rating statistics for the selected scope.

        Low ratings mean one or two stars, not a particular complaint category.
        """
        n, average, low, share = store.metrics(dataset_id, product, month)
        return {
            "scope": scope,
            "total_reviews": n,
            "average_rating": average,
            "low_rating_reviews": low,
            "low_rating_share_percent": share,
        }

    tools = [get_review_metrics]
    if search is not None:
        @tool(args_schema=EvidenceQuery)
        def search_review_evidence(question: str, period: str = "current") -> dict:
            """Find reviews in the current or configured baseline period.

            Results may be irrelevant. Similarity is not confidence; top hits
            cannot establish complaint prevalence or causation. Baseline search
            is available only when the application has configured a comparison.
            """
            if period == "baseline" and baseline_month is None:
                raise ValueError("No baseline period is configured.")
            selected_month = baseline_month if period == "baseline" else month
            return {
                "scope": {**scope, "month": selected_month},
                "period": period,
                "reviews": search.search(
                    store, dataset_id, product, selected_month, question, limit=6
                ),
            }
        tools.append(search_review_evidence)
    if baseline_month is not None:
        @tool(args_schema=NoArguments)
        def compare_review_periods() -> dict:
            """Compare exact SQL rating metrics for configured baseline and current months.

            Use this for increase/decrease questions. The change is in percentage
            points, not relative percent. Report missing data and sample warnings.
            This measures low-rating share, not issue-specific complaint rates or causes.
            """
            return compare_periods(
                store, dataset_id, product, month, baseline_month, minimum_reviews
            )
        tools.append(compare_review_periods)
    return tools
