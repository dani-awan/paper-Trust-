from typing import Literal
from pydantic import BaseModel, Field


class BtwRouteDecision(BaseModel):
    needs_web_search: bool


class RouterDecision(BaseModel):
    route: Literal["retrieve", "web_search", "verify_claim", "direct_answer"]


class RelevancyDecision(BaseModel):
    is_relevant: bool
    reason: str = ""
    relevant_indices: list[int] = Field(default_factory=list, description="1-based source indices useful for answering")


class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str


class ClaimVerificationResult(BaseModel):
    status: Literal["supported", "challenged", "mixed", "insufficient_evidence"]
    verdict_summary: str
    superseding_papers: list[SupersedingPaper] = Field(default_factory=list)
