"""Validate recommendation structure and citation membership, not claim truth."""
from src.schemas import IssueSummary


def validate_recommendation(payload, allowed_review_ids):
    # First check the required fields, types, and lengths.
    summary = IssueSummary.model_validate(payload)

    # A set is a collection of unique values. Subtraction finds missing IDs.
    unknown_ids = set(summary.evidence_ids) - set(allowed_review_ids)
    if unknown_ids:
        raise ValueError('Unknown evidence IDs: ' + ', '.join(sorted(unknown_ids)))

    return summary
