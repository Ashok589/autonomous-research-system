from typing import Any

from pydantic import BaseModel
from tavily import TavilyClient
from duckduckgo_search import DDGS

from app.config import Settings, get_settings


class SearchProviderError(RuntimeError):
    """Raised when the configured search provider cannot complete a search."""


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float = 0.0
    source_type: str = "web"


def _search_tavily(query: str, max_results: int, settings: Settings) -> list[SearchResult]:
    if settings.tavily_api_key is None:
        raise SearchProviderError(
            "Tavily search was selected, but TAVILY_API_KEY is not configured."
        )

    try:
        client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
        response: dict[str, Any] = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
    except Exception as exc:
        raise SearchProviderError(f"Tavily search failed: {exc}") from exc

    return [
        SearchResult(
            title=item.get("title") or item.get("url") or "Untitled result",
            url=item.get("url", ""),
            content=item.get("content", ""),
            score=float(item.get("score") or 0.0),
        )
        for item in response.get("results", [])
        if item.get("url")
    ]


def _search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    try:
        with DDGS(timeout=10) as search_client:
            results = search_client.text(query, max_results=max_results)
    except Exception as exc:
        raise SearchProviderError(f"DuckDuckGo search failed: {exc}") from exc

    return [
        SearchResult(
            title=item.get("title") or item.get("href") or "Untitled result",
            url=item.get("href", ""),
            content=item.get("body", ""),
        )
        for item in results
        if item.get("href")
    ]


def search_web(
    query: str,
    max_results: int | None = None,
    settings: Settings | None = None,
) -> list[SearchResult]:
    """Search the configured provider and return normalized, citable results."""
    current_settings = settings or get_settings()
    result_limit = max_results or current_settings.search_max_results
    provider = current_settings.search_provider

    if provider == "tavily":
        return _search_tavily(query, result_limit, current_settings)
    if provider == "duckduckgo":
        return _search_duckduckgo(query, result_limit)
    if current_settings.tavily_api_key is not None:
        return _search_tavily(query, result_limit, current_settings)
    return _search_duckduckgo(query, result_limit)