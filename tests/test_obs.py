"""Tests for the observability wiring.

These tests focus on the no-op path: when LANGFUSE_* env vars are unset,
tracing must be silently disabled and the orchestrator must run
identically. We do NOT test live span export — that's verified manually
against a real Langfuse instance.
"""

from __future__ import annotations

import pytest

from applyops.obs.tracing import _NoOpSpan, _NoOpTracer, setup_tracing, tracer


def _clear_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


def test_setup_tracing_returns_false_when_keys_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langfuse_env(monkeypatch)
    # Reset the global guard so a prior test doesn't make this one return True.
    import applyops.obs.tracing as obs

    obs._initialized = False
    assert setup_tracing() is False


def test_tracer_span_is_a_context_manager_even_with_no_otel() -> None:
    """The most important contract: `with tracer().start_as_current_span(...)`
    works whether OTel is installed or not."""
    span_cm = tracer().start_as_current_span("test")
    with span_cm as span:
        # Has the methods we use in the stack — all no-ops here.
        span.set_attribute("k", "v")
        span.add_event("e")
        span.record_exception(ValueError("x"))


def test_noop_tracer_explicitly() -> None:
    t = _NoOpTracer()
    with t.start_as_current_span("x") as span:
        assert isinstance(span, _NoOpSpan)
        # All no-ops, no side effects.
        span.set_attribute("a", 1)
        span.add_event("e")
        span.set_status("ok")


def test_stack_runs_unchanged_when_tracing_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-check: the orchestrator landing a tiny stack works whether or
    not OTel is configured. We can't easily strip OTel from the import
    graph in a test, but we can confirm no-op spans don't break anything."""
    _clear_langfuse_env(monkeypatch)

    from applyops.gstack import LayerOutput, Stack, StackContext

    class _Echo(LayerOutput):
        message: str

    class StubLayer:
        def __init__(self) -> None:
            self.name = "stub"

        def run(self, ctx: StackContext) -> LayerOutput:
            return _Echo(layer_name=self.name, message="ok")

    run, ctx = Stack(layers=[StubLayer()]).land()
    from applyops.gstack.run import RunStatus

    assert run.status == RunStatus.COMPLETED
    assert ctx.layers["stub"].output is not None
