from uuid import uuid4
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app import graph
from app.agents import researcher
from app.agents.reporter import _evaluate_report, assemble_report
from app.config import Settings
from app.graph import build_research_graph
from app.main import app
from app.state import (
    Finding,
    ReportDraft,
    ResearchReport,
    ResearchSource,
)
from app.tools.rag_tool import retrieve_and_rerank


def _stub_nodes(impact: str = "low"):
    return {
        "researcher": lambda _state: {"sources": [], "evidence": []},
        "analyst": lambda state: {
            "analysis": {"decision_impact": impact, "summary": "Evidence summary"}
        },
        "critic": lambda _state: {"critique": {"checks": []}},
        "reporter": lambda state: {"report": {"query": state["query"]}},
    }


def test_hybrid_retrieval_ranks_relevant_sources_without_api_key():
    settings = Settings(openai_api_key=None)
    ranked = retrieve_and_rerank(
        "battery storage costs",
        [
            {
                "title": "Battery storage report",
                "url": "https://example.com/battery",
                "content": "Battery storage costs have declined.",
            },
            {
                "title": "Ocean report",
                "url": "https://example.com/ocean",
                "content": "Ocean temperatures are measured.",
            },
        ],
        top_k=1,
        settings=settings,
    )

    assert [source.url for source in ranked] == ["https://example.com/battery"]


def test_hybrid_retrieval_uses_embedding_similarity(monkeypatch):
    import app.tools.rag_tool as rag_tool

    monkeypatch.setattr(
        rag_tool,
        "_semantic_scores",
        lambda _query, _texts, _settings: [1.0, 0.0],
    )
    ranked = retrieve_and_rerank(
        "battery storage costs",
        [
            {
                "title": "Energy research",
                "url": "https://example.com/semantic",
                "content": "The technology became less expensive.",
            },
            {
                "title": "Ocean report",
                "url": "https://example.com/unrelated",
                "content": "Ocean temperatures are measured.",
            },
        ],
        settings=Settings(
            openai_api_key="test-key",
            rag_top_k=2,
        ),
    )

    assert ranked[0].url == "https://example.com/semantic"


def test_report_evaluation_detects_unverified_citation():
    report = ResearchReport(
        query="battery storage costs",
        executive_summary="Battery storage costs changed.",
        findings=[
            Finding(
                claim="Costs declined",
                explanation="The cited source supports this.",
                source_urls=["https://unretrieved.example/fake"],
                confidence=0.9,
            )
        ],
        sources=[
            ResearchSource(
                title="Source",
                url="https://example.com/source",
                content="Source content",
            )
        ],
    )

    evaluation = _evaluate_report("battery storage costs", report, {"checks": []})

    assert evaluation.citation_coverage == 1
    assert evaluation.citation_validity == 0
    assert any("do not match" in warning for warning in evaluation.warnings)


def test_report_assembly_removes_unsupported_claims_and_citations():
    source_url = "https://example.com/source"
    report = assemble_report(
        query="battery storage costs",
        draft=ReportDraft(
            executive_summary="Battery storage costs declined.",
            findings=[
                Finding(
                    claim="Battery storage costs declined",
                    explanation="Supported by the retrieved source.",
                    source_urls=["https://invented.example/citation"],
                    confidence=0.9,
                ),
                Finding(
                    claim="An unsupported second claim",
                    explanation="Not supported by the evidence.",
                    source_urls=[source_url],
                    confidence=0.8,
                ),
            ],
        ),
        sources=[
            {
                "title": "Battery report",
                "url": source_url,
                "content": "Costs declined.",
            }
        ],
        analysis={
            "source_assessments": [
                {
                    "url": source_url,
                    "authority_score": 0.8,
                    "evidence_quality": 0.9,
                    "relevance_score": 1.0,
                    "rationale": "Primary source.",
                }
            ]
        },
        critique={
            "checks": [
                {
                    "claim": "Battery storage costs declined",
                    "supported": True,
                    "source_urls": [source_url],
                },
                {
                    "claim": "An unsupported second claim",
                    "supported": False,
                    "source_urls": [source_url],
                },
            ]
        },
    )

    assert [finding.claim for finding in report.findings] == [
        "Battery storage costs declined"
    ]
    assert report.findings[0].source_urls == []
    assert report.evaluation is not None
    assert report.evaluation.factuality_score == 1
    assert report.evaluation.citation_validity == 0


def test_high_impact_workflow_waits_for_human_approval(monkeypatch):
    monkeypatch.setattr(
        graph,
        "get_settings",
        lambda: Settings(approval_required_for_high_impact=True),
    )
    workflow = build_research_graph(
        checkpointer=MemorySaver(),
        nodes=_stub_nodes(impact="high"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    pending = workflow.invoke({"query": "Should we acquire this company?"}, config=config)
    assert pending["__interrupt__"][0].value["type"] == "high_impact_decision"

    completed = workflow.invoke(
        Command(resume={"approved": True, "reason": "Reviewed by decision owner"}),
        config=config,
    )
    assert completed["report"]["query"] == "Should we acquire this company?"
    assert completed["approval_status"] == "approved"


def test_rejected_high_impact_workflow_does_not_generate_report(monkeypatch):
    monkeypatch.setattr(
        graph,
        "get_settings",
        lambda: Settings(approval_required_for_high_impact=True),
    )
    workflow = build_research_graph(
        checkpointer=MemorySaver(),
        nodes=_stub_nodes(impact="high"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    workflow.invoke({"query": "Should we acquire this company?"}, config=config)

    rejected = workflow.invoke(
        Command(resume={"approved": False, "reason": "Not approved"}),
        config=config,
    )

    assert rejected["approval_status"] == "rejected"
    assert "report" not in rejected


def test_low_impact_workflow_completes_and_records_agent_telemetry(monkeypatch):
    monkeypatch.setattr(
        graph,
        "get_settings",
        lambda: Settings(approval_required_for_high_impact=True),
    )
    workflow = build_research_graph(
        checkpointer=MemorySaver(),
        nodes=_stub_nodes(impact="low"),
    )

    completed = workflow.invoke(
        {"query": "What is the project timeline?"},
        config={"configurable": {"thread_id": str(uuid4())}},
    )

    assert completed["report"]["query"] == "What is the project timeline?"
    assert completed["approval_status"] == "not_required"
    assert {
        event["agent"] for event in completed["observability"]
    } == {"researcher", "analyst", "critic", "reporter"}


def test_sqlite_checkpoint_can_resume_approval_after_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr(
        graph,
        "get_settings",
        lambda: Settings(approval_required_for_high_impact=True),
    )
    database_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": str(uuid4())}}
    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        workflow = build_research_graph(
            checkpointer=checkpointer,
            nodes=_stub_nodes(impact="high"),
        )
        pending = workflow.invoke(
            {"query": "Should we acquire this company?"}, config=config
        )
        assert pending["__interrupt__"]

    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        resumed_workflow = build_research_graph(
            checkpointer=checkpointer,
            nodes=_stub_nodes(impact="high"),
        )
        resumed = resumed_workflow.invoke(
            Command(resume={"approved": True, "reason": "Reviewed"}),
            config=config,
        )

    assert resumed["approval_status"] == "approved"
    assert resumed["report"]["query"] == "Should we acquire this company?"


def test_researcher_carries_prior_report_into_thread_memory(monkeypatch):
    settings = Settings(openai_api_key=None)
    monkeypatch.setattr(researcher, "get_settings", lambda: settings)
    monkeypatch.setattr(
        researcher,
        "search_web",
        lambda *_args, **_kwargs: [],
    )

    result = researcher.researcher_node(
        {
            "query": "Next question",
            "report": {
                "query": "Previous question",
                "executive_summary": "Previous findings summary",
            },
        }
    )

    assert result["memory"] == [
        {"query": "Previous question", "summary": "Previous findings summary"}
    ]


def test_research_thread_memory_survives_between_runs(monkeypatch):
    monkeypatch.setattr(
        graph,
        "get_settings",
        lambda: Settings(approval_required_for_high_impact=False),
    )
    monkeypatch.setattr(
        researcher,
        "get_settings",
        lambda: Settings(openai_api_key=None),
    )
    monkeypatch.setattr(researcher, "search_web", lambda *_args, **_kwargs: [])
    observed_memory = []
    workflow = build_research_graph(
        checkpointer=MemorySaver(),
        nodes={
            "researcher": researcher.researcher_node,
            "analyst": lambda state: (
                observed_memory.append(state.get("memory", []))
                or {
                    "analysis": {
                        "decision_impact": "low",
                        "summary": "Summary",
                    }
                }
            ),
            "critic": lambda _state: {"critique": {"checks": []}},
            "reporter": lambda state: {
                "report": {
                    "query": state["query"],
                    "executive_summary": f"Summary of {state['query']}",
                }
            },
        },
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    workflow.invoke({"query": "First research question"}, config=config)
    second_run = workflow.invoke(
        {"query": "Related follow-up question"},
        config=config,
    )

    assert observed_memory[0] == []
    assert observed_memory[1] == [
        {
            "query": "First research question",
            "summary": "Summary of First research question",
        }
    ]
    assert second_run["memory"] == observed_memory[1]


def test_health_endpoint_reports_service_status(monkeypatch, tmp_path):
    import app.main as main_module

    monkeypatch.setattr(
        main_module.settings,
        "checkpoint_db_path",
        str(tmp_path / "health-checkpoints.sqlite"),
    )
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_returns_pending_approval_then_resumes_it(monkeypatch, tmp_path):
    import app.main as main_module

    monkeypatch.setattr(
        main_module.settings,
        "checkpoint_db_path",
        str(tmp_path / "api-checkpoints.sqlite"),
    )

    class ApprovalGraphStub:
        def invoke(self, graph_input, config):
            assert config["configurable"]["thread_id"]
            if isinstance(graph_input, dict):
                assert graph_input["approval_required"] is True
                return {
                    "__interrupt__": [
                        SimpleNamespace(
                            value={"type": "high_impact_decision", "message": "Approve"}
                        )
                    ],
                    "observability": [],
                }
            assert isinstance(graph_input, Command)
            assert graph_input.resume == {
                "approved": True,
                "reason": "Decision owner review",
            }
            return {
                "report": ResearchReport(
                    query="Should we proceed?",
                    executive_summary="Reviewed report",
                    approval_status="approved",
                    approval_reason="Decision owner review",
                ).model_dump(),
                "observability": [],
            }

    with TestClient(app) as client:
        app.state.research_graph = ApprovalGraphStub()
        pending = client.post(
            "/research",
            json={"query": "Should we proceed?", "require_approval": True},
        )
        assert pending.status_code == 200
        pending_body = pending.json()
        assert pending_body["status"] == "pending_approval"

        completed = client.post(
            f"/research/{pending_body['thread_id']}/approval",
            json={"approved": True, "reason": "Decision owner review"},
        )

    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["report"]["approval_status"] == "approved"
