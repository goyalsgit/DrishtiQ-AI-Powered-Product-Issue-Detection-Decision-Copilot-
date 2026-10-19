"""Data contracts for DristiQ. Structure validation is not fact checking."""
from typing import List
from pydantic import BaseModel, ConfigDict, Field


class IssueSummary(BaseModel):
    # Reject unexpected fields and incorrect types; trim surrounding whitespace.
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)

    product: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    evidence_ids: List[str] = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
