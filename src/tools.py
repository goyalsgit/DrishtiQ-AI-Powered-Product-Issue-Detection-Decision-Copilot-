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


class ThemeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    focus: Literal["problems", "praise", "all"] = Field(
        default="all", description="problems = 1-3 stars, praise = 4-5 stars, all = every review.")


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

    if product is None:
        @tool(args_schema=NoArguments)
        def compare_products() -> dict:
            """Which products have the most problems? Exact SQL ranking of ALL products.

            Use for any 'which product has problems / most complaints / worst' question.
            Ranked by number of 1-2 star reviews. Also returns 3 recent 1-2 star
            reviews for each of the top 5 products, so you can quote what is wrong.
            """
            dates = dict(start=f"{month}-01" if month else None,
                         end=f"{month}-31T23:59:59" if month else None)
            table = store.product_table(dataset_id, **dates).sort_values(
                ["low_rating_reviews", "reviews"], ascending=False)
            examples = []
            for name in table.head(5)["product"]:
                page = store.filtered_page(dataset_id, limit=3, products=[name], ratings=[1, 2], **dates)
                examples += [{"review_id": r.review_id, "product": r.product, "month": r.date[:7],
                              "rating": int(r.rating), "text": r.text} for r in page.itertuples()]
            return {"scope": scope, "ranked_by": "number of 1-2 star reviews",
                    "products": table.head(10).to_dict("records"),
                    "products_total": len(table), "reviews": examples}
        tools.append(compare_products)

        if search is not None:
            @tool(args_schema=NoArguments)
            def problems_by_product() -> dict:
                """Problems of each product, product by product. Use FIRST for any
                'problems per product / list problems with product name' question.

                Takes the 6 products with the most 1-2 star reviews and groups ALL of
                each product's 1-3 star reviews into up to 4 problem themes, with
                exact counts and 2 example reviews per theme.
                """
                from src.themes import problems_by_product as breakdown
                products = breakdown(store, search, dataset_id, month)
                reviews = [e for p in products for t in p["themes"] for e in t["examples"]]
                return {"scope": scope, "products": products, "reviews": reviews}
            tools.append(problems_by_product)

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

        @tool(args_schema=ThemeQuery)
        def summarize_review_themes(focus: str = "all") -> dict:
            """Group EVERY review in the selected scope into up to 8 themes.

            Use first for broad questions: main problems, positive points, what
            customers say overall. focus=problems uses 1-3 star reviews,
            praise uses 4-5 stars, all uses every rating. Each theme has an exact
            review count, share and average rating, plus 3 example reviews.
            Themes are wording similarity groups; name them from the examples.
            """
            from src.themes import summarize_themes
            data = summarize_themes(store, search, dataset_id, product, month, focus)
            # Flatten examples into "reviews" so they can be cited like search results.
            data["reviews"] = [e for t in data["themes"] for e in t["examples"]]
            return {"scope": scope, "period": "current", **data}

        tools.append(summarize_review_themes)

    @tool(args_schema=NoArguments)
    def find_customer_requests() -> dict:
        """Find reviews where customers ask or wish for something.

        Use for questions about demands, wants, wishes, expectations, suggestions
        or feature requests. Matches phrases such as 'wish', 'would be nice',
        'should have', 'needs to', 'bigger size'. Up to 8 reviews, newest first.
        A phrase match is not proof of a request; read each review.
        """
        rows = store.request_reviews(dataset_id, product, month)
        return {"scope": scope, "period": "current",
                "reviews": [{"review_id": r.review_id, "product": r.product, "month": r.month,
                             "rating": int(r.rating), "text": r.text}
                            for r in rows.itertuples()]}

    tools.append(find_customer_requests)
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
