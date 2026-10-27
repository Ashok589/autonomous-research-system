from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


def merge_observability(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return (existing + updates)[-100:]


class ResearchSource(BaseModel):
    title: str
    url: str
    content: str
    score: float = 0.0
    source_type: str = "web"


class Finding(BaseModel):
    claim: str
    explanation: str
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SourceAssessment(BaseModel):
    url: str
    authority_score: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class Analysis(BaseModel):
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    source_assessments: list[SourceAssessment] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    decision_impact: Literal["low", "medium", "high"] = "medium"


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    explanation: str
    source_urls: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    checks: list[ClaimCheck] = Field(default_factory=list)
    overall_assessment: str
    unresolved_questions: list[str] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    factuality_score: float = Field(ge=0.0, le=1.0)
    source_reliability_score: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    citation_validity: float = Field(ge=0.0, le=1.0)
    response_quality: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class ReportDraft(BaseModel):
    executive_summary: str
    findings: list[Finding] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    query: str
    executive_summary: str
    findings: list[Finding] = Field(default_factory=list)
    source_assessments: list[SourceAssessment] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    evaluation: EvaluationMetrics | None = None
    approval_status: Literal["not_required", "approved"] = "not_required"
    approval_reason: str = ""


class ResearchState(TypedDict, total=False):
    query: str
    documents: list[str]
    max_results: int
    thread_id: str
    sources: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    analysis: dict[str, Any]
    critique: dict[str, Any]
    report: dict[str, Any]
    memory: list[dict[str, Any]]
    approval_status: str
    approval_reason: str
    approval_required: bool
    observability: Annotated[
        list[dict[str, Any]], merge_observability
    ]