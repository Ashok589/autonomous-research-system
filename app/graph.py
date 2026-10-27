from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents.analyst import analyst_node
from app.agents.critic import critic_node
from app.agents.reporter import reporter_node
from app.agents.researcher import researcher_node
from app.config import get_settings
from app.state import ResearchState
from app.utils.telemetry import observe_node


def approval_gate_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    requires_approval = (
        state.get("approval_required", False)
        or (
            settings.approval_required_for_high_impact
            and state.get("analysis", {}).get("decision_impact") == "high"
        )
    )
    if not requires_approval:
        return {"approval_required": False, "approval_status": "not_required"}

    decision = interrupt(
        {
            "type": "high_impact_decision",
            "query": state["query"],
            "impact": state["analysis"]["decision_impact"],
            "summary": state["analysis"].get("summary", ""),
            "message": "A human must approve this high-impact research before a report is generated.",
        }
    )
    if not isinstance(decision, dict) or not isinstance(decision.get("approved"), bool):
        raise ValueError("Approval resume data must include a boolean 'approved' value.")

    return {
        "approval_required": True,
        "approval_status": "approved" if decision["approved"] else "rejected",
        "approval_reason": str(decision.get("reason", "")),
    }


def _route_after_approval(state: ResearchState) -> str:
    if state.get("approval_status") in {"approved", "not_required"}:
        return "reporter"
    return END


def build_research_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    nodes: dict[str, Callable[..., dict[str, Any]]] | None = None,
):
    configured_nodes = nodes or {
        "researcher": researcher_node,
        "analyst": analyst_node,
        "critic": critic_node,
        "reporter": reporter_node,
    }
    workflow = StateGraph(ResearchState)
    for name in ("researcher", "analyst", "critic"):
        workflow.add_node(name, observe_node(name, configured_nodes[name]))
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("reporter", observe_node("reporter", configured_nodes["reporter"]))

    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "critic")
    workflow.add_edge("critic", "approval_gate")
    workflow.add_conditional_edges(
        "approval_gate",
        _route_after_approval,
        {"reporter": "reporter", END: END},
    )
    workflow.add_edge("reporter", END)
    return workflow.compile(checkpointer=checkpointer)