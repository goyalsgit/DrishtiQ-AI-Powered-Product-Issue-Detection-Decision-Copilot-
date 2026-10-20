"""Descriptive period comparisons calculated from fixed SQL, never by an LLM."""
from datetime import date, timedelta
import re


def validate_month(month):
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("Month must use YYYY-MM.")
    date.fromisoformat(month + "-01")
    return month


def previous_month(month):
    """Previous calendar month, not the previous month that happens to have data."""
    validate_month(month)
    return (date.fromisoformat(month + "-01") - timedelta(days=1)).isoformat()[:7]


def validate_comparison(current_month, baseline_month, minimum_reviews):
    validate_month(current_month)
    validate_month(baseline_month)
    if baseline_month >= current_month:
        raise ValueError("Baseline month must be earlier than the analysis month.")
    if (isinstance(minimum_reviews, bool) or not isinstance(minimum_reviews, int)
            or minimum_reviews < 1):
        raise ValueError("Minimum reviews must be a positive integer.")


def compare_periods(store, dataset_id, product, current_month, baseline_month,
                    minimum_reviews=20):
    validate_comparison(current_month, baseline_month, minimum_reviews)

    def period_metrics(month):
        n, average, low, share = store.metrics(dataset_id, product, month)
        return {
            "month": month, "total_reviews": n, "average_rating": average,
            "low_rating_reviews": low, "low_rating_share_percent": share,
        }

    current = period_metrics(current_month)
    baseline = period_metrics(baseline_month)
    both_have_data = current["total_reviews"] > 0 and baseline["total_reviews"] > 0
    meets_minimum = min(current["total_reviews"], baseline["total_reviews"]) >= minimum_reviews
    delta = (current["low_rating_share_percent"] - baseline["low_rating_share_percent"]
             if both_have_data else None)
    status = ("missing_period_data" if not both_have_data else
              "descriptive_comparison" if meets_minimum else "insufficient_sample")
    return {
        "scope": {"dataset_id": dataset_id, "product": product},
        "current": current,
        "baseline": baseline,
        "low_rating_share_change_percentage_points": delta,
        "minimum_reviews_per_period": minimum_reviews,
        "meets_minimum_reviews": meets_minimum,
        "status": status,
        "limitations": [
            "Figures describe saved reviews only, not all customers or sales.",
            "Low ratings mean 1 or 2 stars, not confirmed complaint categories.",
            "The minimum sample rule is not a statistical significance test.",
            "A change in ratings does not establish its cause.",
        ],
    }
