"""OpenTelemetry setup. Idempotent. No-op when Langfuse keys are absent.

Langfuse accepts OTLP/HTTP with a basic-auth header derived from the
public/secret key pair. We hit `<host>/api/public/otel/v1/traces`. No
SDK lock-in — anything that speaks OTLP works (Honeycomb, Grafana,
Tempo, local Jaeger).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

_log = logging.getLogger(__name__)
_initialized: bool = False


def setup_tracing(service_name: str = "applyops") -> bool:
    """Configure OTel tracing if Langfuse env vars are present.

    Returns True iff tracing is now active. Safe to call multiple times;
    subsequent calls after the first are no-ops.
    """
    global _initialized
    if _initialized:
        return True

    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")

    if not public or not secret:
        _log.debug("tracing disabled — Langfuse env vars not set")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        _log.warning(
            "tracing requested but `opentelemetry` extras not installed — "
            "run `pip install -e '.[obs]'`"
        )
        return False

    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    exporter = OTLPSpanExporter(
        endpoint=f"{host}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}"},
    )
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _initialized = True
    _log.info("tracing active — exporting to %s", host)
    return True


def tracer() -> Any:
    """Return the applyops tracer.

    If OTel isn't installed (the `obs` extra isn't present), we return a
    no-op tracer whose `start_as_current_span` is a context manager that
    does nothing. This means callers can write `with tracer().start_as_current_span(...)`
    unconditionally; instrumentation is free when disabled.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return _NoOpTracer()
    return trace.get_tracer("applyops")


class _NoOpSpan:
    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def set_attribute(self, *_args: object, **_kwargs: object) -> None:
        return None

    def add_event(self, *_args: object, **_kwargs: object) -> None:
        return None

    def record_exception(self, *_args: object, **_kwargs: object) -> None:
        return None

    def set_status(self, *_args: object, **_kwargs: object) -> None:
        return None


class _NoOpTracer:
    def start_as_current_span(self, *_args: object, **_kwargs: object) -> _NoOpSpan:
        return _NoOpSpan()
