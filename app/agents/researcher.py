from app.config import get_settings
from typing import Any

from app.state import ResearchSource, ResearchState
from app.tools.rag_tool import retrieve_and_rerank
from app.tools.search_tool import search_web


def researcher_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    sources = [
        result.model_dump()
        for result in search_web(
            state["query"],
            max_results=state.get("max_results", settings.search_max_results),
            settings=settings,
        )
    ]
    sources.extend(
        ResearchSource(
            title=f"Provided document {index}",
            url=f"provided://document/{index}",
            content=document,
            source_type="provided_document",
        ).model_dump()
        for index, document in enumerate(state.get("documents", []), start=1)
        if document.strip()
    )

    evidence = retrieve_and_rerank(
        state["query"],
        sources,
        top_k=settings.rag_top_k,
        settings=settings,
    )
    memory = list(state.get("memory", []))
    previous_report = state.get("report")
    if previous_report:
        memory.append(
            {
                "query": previous_report.get("query", ""),
                "summary": previous_report.get("executive_summary", ""),
            }
        )
    return {
        "sources": sources,
        "evidence": [source.model_dump() for source in evidence],
        "memory": memory[-5:],
        "report": {},
    }