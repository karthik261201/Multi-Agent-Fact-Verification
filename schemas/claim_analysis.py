from typing import Literal

from pydantic import BaseModel, Field


# One individual assertion that needs verification.
class Subclaim(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


# A search query linked to one assertion.
class SearchQuery(BaseModel):
    subclaim_id: str = Field(min_length=1)
    query: str = Field(min_length=1)


# The detailed internal response returned by the Claim Agent.
class ClaimAnalysis(BaseModel):
    original_claim: str = Field(min_length=1)

    # "ready" means clear enough to investigate, not verified as true.
    status: Literal["ready", "needs_clarification"]

    entities: list[str]

    # Actions, properties, dates, and important qualifiers.
    keywords: list[str]

    subclaims: list[Subclaim]
    ambiguities: list[str]
    search_queries: list[SearchQuery]