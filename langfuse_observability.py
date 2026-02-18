"""
Langfuse Observability Module for CoreStack Agent
===================================================

Provides production-ready tracing, span management, cost monitoring,
feedback tracking, and error logging for the SmolAgents CodeAgent pipeline.

Trace hierarchy (as seen in Langfuse dashboard):
──────────────────────────────────────────────────
  Session (session_id)  ← groups multiple user turns
  └─ Trace: run_hybrid_agent  ← root per user request
       ├─ Span: location_resolution  ← CoreStack API workflow
       ├─ Generation: llm_call  ← each LLM inference
       ├─ Span: tool_execution / fetch_corestack_data
       ├─ Span: agent_reasoning_step
       └─ Span: final_response

Usage:
------
    from langfuse_observability import (
        lf_client, observe, generate_session_id,
        trace_llm_call, trace_tool_call, trace_agent_execution,
        score_trace, log_error_to_trace, flush,
    )
"""

from __future__ import annotations

import os
import time
import uuid
import functools
import traceback as _tb
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 1. SDK INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
# Langfuse v3 reads these env vars automatically:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (or LANGFUSE_BASE_URL)
# We import the decorator-based API (`observe`, `get_client`) which uses OTEL
# under the hood and automatically nests spans inside the current trace.
# ──────────────────────────────────────────────────────────────────────────────

_LANGFUSE_ENABLED: bool = False

try:
    from langfuse import (
        Langfuse,
        observe as _raw_observe,
        get_client as _get_langfuse_client,
    )

    # Eagerly create singleton so auth_check runs once at import time
    lf_client: Langfuse = _get_langfuse_client()

    if lf_client.auth_check():
        _LANGFUSE_ENABLED = True
        print("✅ [langfuse_observability] Langfuse SDK authenticated")
    else:
        print("⚠️  [langfuse_observability] Langfuse auth_check failed — running in no-op mode")

    observe = _raw_observe  # re-export the real decorator

except Exception as _init_err:  # pragma: no cover
    print(f"⚠️  [langfuse_observability] Langfuse unavailable: {_init_err}")

    # ── Fallback stubs so the rest of the codebase never crashes ──

    class _LangfuseNoop:
        """Mimics the Langfuse client with safe no-op methods."""

        def auth_check(self) -> bool:
            return False

        def flush(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

        # Trace / span context helpers
        def update_current_trace(self, **kw: Any) -> None:
            pass

        def update_current_span(self, **kw: Any) -> None:
            pass

        def update_current_generation(self, **kw: Any) -> None:
            pass

        def score_current_trace(self, **kw: Any) -> None:
            pass

        def score_current_span(self, **kw: Any) -> None:
            pass

        def get_current_trace_id(self) -> Optional[str]:
            return None

        def get_current_observation_id(self) -> Optional[str]:
            return None

        def get_trace_url(self, trace_id: str) -> str:
            return ""

        def create_score(self, **kw: Any) -> None:
            pass

        # Context-manager stubs for start_as_current_*
        @contextmanager
        def start_as_current_span(self, **kw: Any):
            yield _NoopSpan()

        @contextmanager
        def start_as_current_generation(self, **kw: Any):
            yield _NoopGeneration()

    class _NoopSpan:
        def end(self) -> None:
            pass

    class _NoopGeneration:
        def end(self) -> None:
            pass

    lf_client = _LangfuseNoop()  # type: ignore[assignment]

    def observe(*args: Any, **kwargs: Any):  # type: ignore[misc]
        """No-op @observe decorator."""
        def _decorator(fn: Callable) -> Callable:
            return fn
        if args and callable(args[0]):
            return args[0]
        return _decorator


def is_enabled() -> bool:
    """Return True when Langfuse is connected and recording."""
    return _LANGFUSE_ENABLED


# ──────────────────────────────────────────────────────────────────────────────
# 2. SESSION / TRACE ID HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def generate_session_id(prefix: str = "corestack") -> str:
    """
    Create a unique session ID.

    Format:  ``corestack-20260218-143022-a1b2c3d4``

    Sessions group multiple traces in the Langfuse dashboard so you can view
    an entire user conversation or batch-run as one logical unit.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}-{ts}-{short}"


def generate_trace_id() -> str:
    """Return a fresh UUID-hex string suitable for a Langfuse trace ID."""
    return uuid.uuid4().hex


# ──────────────────────────────────────────────────────────────────────────────
# 3. TRACE CONTEXT UPDATERS  (call inside an @observe-decorated function)
# ──────────────────────────────────────────────────────────────────────────────

def set_trace_metadata(
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Attach metadata to the *current* Langfuse trace (must be called from
    within an ``@observe``-decorated function).

    Parameters
    ----------
    session_id : str
        Groups traces under one session in the dashboard.
    user_id : str
        Identifies the end-user who triggered the request.
    user_query : str
        The raw user prompt (recorded as ``input`` on the trace).
    tags : list[str]
        Arbitrary tags for filtering in the dashboard.
    version : str
        Application/prompt version string.
    metadata : dict
        Free-form key/value metadata dict.
    """
    update_kwargs: Dict[str, Any] = {}
    if session_id is not None:
        update_kwargs["session_id"] = session_id
    if user_id is not None:
        update_kwargs["user_id"] = user_id
    if user_query is not None:
        update_kwargs["input"] = user_query
    if tags is not None:
        update_kwargs["tags"] = tags
    if version is not None:
        update_kwargs["version"] = version
    if metadata is not None:
        update_kwargs["metadata"] = metadata
    if update_kwargs:
        lf_client.update_current_trace(**update_kwargs)


def set_trace_output(output: Any) -> None:
    """Record the final output on the current trace."""
    lf_client.update_current_trace(output=output)


# ──────────────────────────────────────────────────────────────────────────────
# 4. HELPER WRAPPERS — trace_llm_call / trace_tool_call / trace_agent_execution
# ──────────────────────────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])


def trace_llm_call(
    *,
    name: str = "llm_call",
    model: Optional[str] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator that wraps a function in a Langfuse **generation** span.

    Records input/output, latency, model name, model params, and token usage
    (if the wrapped function returns a dict with a ``usage`` key).

    Example
    -------
    ::

        @trace_llm_call(name="gemini_inference", model="gemini/gemini-2.5-flash-lite")
        def call_model(prompt: str) -> str:
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            input_payload = {"args": args, "kwargs": kwargs}

            with lf_client.start_as_current_generation(
                name=name,
                input=input_payload,
                model=model,
                model_parameters=model_parameters,
            ) as gen:
                try:
                    result = fn(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    # Try to extract token usage if result is a dict
                    usage: Dict[str, int] = {}
                    cost: Dict[str, float] = {}
                    if isinstance(result, dict):
                        if "usage" in result:
                            u = result["usage"]
                            usage = {
                                "input": u.get("prompt_tokens", u.get("input_tokens", 0)),
                                "output": u.get("completion_tokens", u.get("output_tokens", 0)),
                                "total": u.get("total_tokens", 0),
                            }
                        if "cost" in result:
                            cost = {"total": float(result["cost"])}

                    lf_client.update_current_generation(
                        output=result,
                        metadata={"latency_ms": round(elapsed_ms, 2)},
                        usage_details=usage or None,
                        cost_details=cost or None,
                    )
                    return result

                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    lf_client.update_current_generation(
                        output=str(exc),
                        level="ERROR",
                        status_message=str(exc),
                        metadata={
                            "latency_ms": round(elapsed_ms, 2),
                            "traceback": _tb.format_exc(),
                        },
                    )
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def trace_tool_call(
    *,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator that wraps a function in a Langfuse **tool** span.

    Captures input args, output, latency, and errors.

    Example
    -------
    ::

        @trace_tool_call(name="fetch_corestack_data")
        def my_tool(query: str) -> str:
            ...
    """

    def decorator(fn: F) -> F:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()

            with lf_client.start_as_current_span(
                name=span_name,
                input={"args": args, "kwargs": kwargs},
            ):
                try:
                    result = fn(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    lf_client.update_current_span(
                        output=result,
                        metadata={
                            "tool_name": span_name,
                            "latency_ms": round(elapsed_ms, 2),
                        },
                    )
                    return result

                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    lf_client.update_current_span(
                        output=str(exc),
                        level="ERROR",
                        status_message=str(exc),
                        metadata={
                            "tool_name": span_name,
                            "latency_ms": round(elapsed_ms, 2),
                            "traceback": _tb.format_exc(),
                        },
                    )
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def trace_agent_execution(
    *,
    name: str = "agent_execution",
) -> Callable[[F], F]:
    """
    Decorator that wraps the top-level agent execution in a Langfuse **span**.

    Captures the full prompt, final answer, total latency, and any exceptions.

    Example
    -------
    ::

        @trace_agent_execution(name="run_hybrid_agent")
        def run_hybrid_agent(user_query: str) -> str:
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()

            with lf_client.start_as_current_span(
                name=name,
                input={"args": args, "kwargs": kwargs},
            ):
                try:
                    result = fn(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    lf_client.update_current_span(
                        output=result,
                        metadata={
                            "execution_duration_ms": round(elapsed_ms, 2),
                            "status": "success",
                        },
                    )
                    return result

                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    lf_client.update_current_span(
                        output=str(exc),
                        level="ERROR",
                        status_message=str(exc),
                        metadata={
                            "execution_duration_ms": round(elapsed_ms, 2),
                            "status": "error",
                            "traceback": _tb.format_exc(),
                        },
                    )
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# 5. INLINE CONTEXT MANAGERS  (for ad-hoc spans without decorators)
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def span(name: str, *, input: Any = None, metadata: Optional[Dict] = None):
    """
    Open a child span under the current trace.

    Example
    -------
    ::

        with span("location_resolution", input={"query": q}) as s:
            result = _run_corestack_workflow(q)
            lf_client.update_current_span(output=result)
    """
    with lf_client.start_as_current_span(name=name, input=input, metadata=metadata) as s:
        yield s


@contextmanager
def generation(
    name: str,
    *,
    model: Optional[str] = None,
    model_parameters: Optional[Dict] = None,
    input: Any = None,
):
    """
    Open a child **generation** span (for LLM calls).

    Example
    -------
    ::

        with generation("gemini_call", model="gemini/gemini-2.5-flash-lite", input=prompt):
            response = model(prompt)
            lf_client.update_current_generation(output=response, usage_details={...})
    """
    with lf_client.start_as_current_generation(
        name=name, model=model, model_parameters=model_parameters, input=input,
    ) as g:
        yield g


# ──────────────────────────────────────────────────────────────────────────────
# 6. SCORING / USER FEEDBACK
# ──────────────────────────────────────────────────────────────────────────────

def score_trace(
    *,
    name: str = "user_feedback",
    value: Union[float, str],
    data_type: Optional[str] = None,
    comment: Optional[str] = None,
) -> None:
    """
    Score the *current* trace (e.g. thumbs-up / thumbs-down).

    Call from within an ``@observe``-decorated function.

    Parameters
    ----------
    name : str
        Score dimension name, e.g. ``"user_feedback"``, ``"accuracy"``.
    value : float | str
        Numeric (0-1) or categorical ("good"/"bad").
    data_type : str, optional
        ``"NUMERIC"``, ``"CATEGORICAL"``, or ``"BOOLEAN"``.
    comment : str, optional
        Free-form user comment.
    """
    kwargs: Dict[str, Any] = {"name": name, "value": value}
    if data_type:
        kwargs["data_type"] = data_type
    if comment:
        kwargs["comment"] = comment
    lf_client.score_current_trace(**kwargs)


def score_trace_by_id(
    trace_id: str,
    *,
    name: str = "user_feedback",
    value: Union[float, str],
    data_type: Optional[str] = None,
    comment: Optional[str] = None,
) -> None:
    """
    Score a trace by its ID (for async/deferred feedback from a UI).

    Parameters
    ----------
    trace_id : str
        The Langfuse trace ID returned earlier.
    name : str
        Score dimension.
    value : float | str
        Numeric or categorical score.
    """
    kwargs: Dict[str, Any] = {
        "trace_id": trace_id,
        "name": name,
        "value": value,
    }
    if data_type:
        kwargs["data_type"] = data_type
    if comment:
        kwargs["comment"] = comment
    lf_client.create_score(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# 7. ERROR / EXCEPTION LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def log_error_to_trace(exc: BaseException, *, context: Optional[str] = None) -> None:
    """
    Record an exception on the current trace/span as an ERROR-level event.

    Parameters
    ----------
    exc : BaseException
        The caught exception.
    context : str, optional
        Additional context (e.g. which step failed).
    """
    lf_client.update_current_span(
        level="ERROR",
        status_message=f"[{type(exc).__name__}] {exc}",
        metadata={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": _tb.format_exc(),
            "context": context,
        },
    )


def log_error_to_trace_root(exc: BaseException, *, context: Optional[str] = None) -> None:
    """
    Record an exception on the *root trace* (vs the current span).
    """
    lf_client.update_current_trace(
        metadata={
            "error": True,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": _tb.format_exc(),
            "context": context,
        },
        tags=["error"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. PROMPT VERSION TRACKING
# ──────────────────────────────────────────────────────────────────────────────

_PROMPT_VERSION: str = "v1.0.0"


def set_prompt_version(version: str) -> None:
    """Override the global prompt version tag."""
    global _PROMPT_VERSION
    _PROMPT_VERSION = version


def get_prompt_version() -> str:
    return _PROMPT_VERSION


def tag_prompt_version() -> None:
    """Attach the current prompt version to the active trace."""
    lf_client.update_current_trace(
        version=_PROMPT_VERSION,
        tags=[f"prompt:{_PROMPT_VERSION}"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9. COST MONITORING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def record_generation_cost(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: Optional[float] = None,
) -> None:
    """
    Update the current generation span with token/cost data.

    Must be called inside a ``generation()`` context or a function decorated
    with ``@trace_llm_call``.
    """
    usage = {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens or (input_tokens + output_tokens),
    }
    cost_details = {"total": cost_usd} if cost_usd is not None else None

    lf_client.update_current_generation(
        model=model,
        usage_details=usage,
        cost_details=cost_details,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 10. FLUSH / SHUTDOWN
# ──────────────────────────────────────────────────────────────────────────────

def flush() -> None:
    """Flush pending events to the Langfuse backend (blocking)."""
    lf_client.flush()


def shutdown() -> None:
    """Flush and release Langfuse resources."""
    lf_client.shutdown()


# ──────────────────────────────────────────────────────────────────────────────
# 11. CONVENIENCE EXPORTS
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Core
    "lf_client",
    "observe",
    "is_enabled",
    # IDs
    "generate_session_id",
    "generate_trace_id",
    # Trace context
    "set_trace_metadata",
    "set_trace_output",
    # Decorators
    "trace_llm_call",
    "trace_tool_call",
    "trace_agent_execution",
    # Context managers
    "span",
    "generation",
    # Scoring / feedback
    "score_trace",
    "score_trace_by_id",
    # Error logging
    "log_error_to_trace",
    "log_error_to_trace_root",
    # Prompt versioning
    "set_prompt_version",
    "get_prompt_version",
    "tag_prompt_version",
    # Cost
    "record_generation_cost",
    # Lifecycle
    "flush",
    "shutdown",
]
