import re
from collections import Counter
from math import log, sqrt
from typing import Any

from app.config import Settings, get_settings
from app.state import ResearchSource


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(
    query_tokens: list[str],
    documents: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not query_tokens or not documents:
        return [0.0] * len(documents)

    term_frequencies = [Counter(document) for document in documents]
    document_frequencies = Counter(
        token
        for token in set(query_tokens)
        for document in term_frequencies
        if document.get(token, 0) > 0
    )
    average_length = sum(map(len, documents)) / len(documents)
    if average_length == 0:
        return [0.0] * len(documents)

    scores: list[float] = []
    for document, frequencies in zip(documents, term_frequencies):
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            inverse_frequency = log(
                1
                + (len(documents) - document_frequencies[token] + 0.5)
                / (document_frequencies[token] + 0.5)
            )
            length_normalization = k1 * (
                1 - b + b * len(document) / average_length
            )
            score += inverse_frequency * (
                frequency * (k1 + 1) / (frequency + length_normalization)
            )
        scores.append(score)

    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0 if maximum > 0 else 0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _rerank_score(query: str, source: ResearchSource, hybrid_score: float) -> float:
    query_terms = set(_tokens(query))
    title_terms = set(_tokens(source.title))
    title_match = len(query_terms & title_terms) / len(query_terms) if query_terms else 0.0
    return 0.8 * hybrid_score + 0.2 * title_match


def _semantic_scores(
    query: str,
    texts: list[str],
    settings: Settings,
) -> list[float] | None:
    if settings.openai_api_key is None or not texts:
        return None

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    vectors = embeddings.embed_documents(texts)
    query_vector = embeddings.embed_query(query)

    scores: list[float] = []
    for vector in vectors:
        dot_product = sum(left * right for left, right in zip(query_vector, vector))
        query_norm = sqrt(sum(value * value for value in query_vector))
        vector_norm = sqrt(sum(value * value for value in vector))
        scores.append(
            dot_product / (query_norm * vector_norm)
            if query_norm and vector_norm
            else 0.0
        )
    return scores


def retrieve_and_rerank(
    query: str,
    sources: list[ResearchSource | dict[str, Any]],
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[ResearchSource]:
    """Rank source passages using lexical relevance and optional embedding similarity."""
    current_settings = settings or get_settings()
    normalized = [
        source if isinstance(source, ResearchSource) else ResearchSource.model_validate(source)
        for source in sources
    ]
    if not normalized:
        return []

    texts = [f"{source.title}\n{source.content}" for source in normalized]
    query_tokens = _tokens(query)
    tokenized_documents = [_tokens(text) for text in texts]
    lexical_scores = _bm25_scores(query_tokens, tokenized_documents)
    semantic_scores = _semantic_scores(query, texts, current_settings)
    normalized_lexical = _min_max_normalize(lexical_scores)
    normalized_semantic = (
        _min_max_normalize(semantic_scores)
        if semantic_scores is not None
        else normalized_lexical
    )

    ranked: list[ResearchSource] = []
    for index, source in enumerate(normalized):
        hybrid_score = (
            0.55 * normalized_lexical[index] + 0.45 * normalized_semantic[index]
        )
        source.score = _rerank_score(query, source, hybrid_score)
        ranked.append(source)

    limit = top_k or current_settings.rag_top_k
    return sorted(ranked, key=lambda source: source.score, reverse=True)[:limit]