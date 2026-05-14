"""applyops.obs — OpenTelemetry tracing for the agent pipeline.

The whole pipeline is instrumented at three nesting levels:
- stack.land — the run root span
- layer.<name> / gate.<name> — one span per layer or gate invocation
- llm.parse — one span per structured-output LLM call

If `LANGFUSE_PUBLIC_KEY` is not set, `setup_tracing()` is a no-op and the
`start_as_current_span(...)` calls become non-recording spans — the code
paths are identical whether tracing is configured or not.

Public API:
    from applyops.obs import setup_tracing, tracer
"""

from __future__ import annotations

from applyops.obs.tracing import setup_tracing, tracer

__all__ = ["setup_tracing", "tracer"]
