import logging
from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
Node = TypeVar("Node", bound=Callable[..., dict[str, Any]])


def configure_telemetry(settings: Settings | None = None) -> None:
    current_settings = settings or get_settings()
    if not current_settings.telemetry_enabled:
        return

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    tracer_provider = register(endpoint=current_settings.phoenix_endpoint)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


def observe_node(name: str, node: Node) -> Node:
    @wraps(node)
    def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        try:
            updates = node(state)
        except Exception:
            logger.exception(
                "Research agent step failed",
                extra={
                    "agent": name,
                    "status": "failed",
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )
            raise

        token_usage = updates.pop("_token_usage", {})
        event = {
            "agent": name,
            "status": "success",
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "token_usage": token_usage,
        }
        logger.info("Research agent step completed", extra=event)
        return {**updates, "observability": [event]}

    return wrapped  # type: ignore[return-value]


def invoke_with_usage(prompt: Any, model: Any, values: dict[str, Any]) -> tuple[Any, dict]:
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    usage_handler = UsageMetadataCallbackHandler()
    result = (prompt | model).invoke(
        values,
        config={"callbacks": [usage_handler]},
    )
    usage: dict = {}
    for model_name, metadata in usage_handler.usage_metadata.items():
        usage[model_name] = {
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
            "total_tokens": metadata.get("total_tokens", 0),
        }
    return result, usage