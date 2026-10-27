from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.config import get_settings
from app.graph import build_research_graph
from app.state import ResearchReport
from app.tools.search_tool import SearchProviderError
from app.utils.telemetry import configure_telemetry


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    documents: list[Annotated[str, Field(min_length=1, max_length=50_000)]] = Field(
        default_factory=list,
        max_length=20,
    )
    max_results: int | None = Field(default=None, ge=1, le=20)
    thread_id: UUID | None = None
    require_approval: bool = False


class ApprovalRequest(BaseModel):
    approved: bool
    reason: str = Field(default="", max_length=2000)


class ResearchResponse(BaseModel):
    status: Literal["completed", "pending_approval", "rejected"]
    thread_id: UUID
    report: ResearchReport | None = None
    approval_request: dict | None = None
    observability: list[dict] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_telemetry()
    database_path = Path(settings.checkpoint_db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        application.state.research_graph = build_research_graph(checkpointer)
        yield


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def conduct_research(request: ResearchRequest) -> ResearchResponse:
    thread_id = request.thread_id or uuid4()
    config = {"configurable": {"thread_id": str(thread_id)}}
    try:
        result = app.state.research_graph.invoke(
            {
                "query": request.query,
                "documents": request.documents,
                "max_results": request.max_results or settings.search_max_results,
                "approval_required": request.require_approval,
            },
            config=config,
        )
    except SearchProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    pending = result.get("__interrupt__")
    if pending:
        return ResearchResponse(
            status="pending_approval",
            thread_id=thread_id,
            approval_request=pending[0].value,
            observability=result.get("observability", []),
        )

    report = result.get("report")
    if report:
        return ResearchResponse(
            status="completed",
            thread_id=thread_id,
            report=ResearchReport.model_validate(report),
            observability=result.get("observability", []),
        )
    if result.get("approval_status") == "rejected":
        return ResearchResponse(
            status="rejected",
            thread_id=thread_id,
            observability=result.get("observability", []),
        )
    raise HTTPException(status_code=500, detail="Research workflow returned no report.")


@app.post("/research/{thread_id}/approval", response_model=ResearchResponse)
def approve_research(thread_id: UUID, request: ApprovalRequest) -> ResearchResponse:
    config = {"configurable": {"thread_id": str(thread_id)}}
    try:
        result = app.state.research_graph.invoke(
            Command(resume={"approved": request.approved, "reason": request.reason}),
            config=config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    report = result.get("report")
    if report:
        return ResearchResponse(
            status="completed",
            thread_id=thread_id,
            report=ResearchReport.model_validate(report),
            observability=result.get("observability", []),
        )
    if result.get("approval_status") == "rejected":
        return ResearchResponse(
            status="rejected",
            thread_id=thread_id,
            observability=result.get("observability", []),
        )
    raise HTTPException(
        status_code=409,
        detail="No pending approval exists for this research thread.",
    )