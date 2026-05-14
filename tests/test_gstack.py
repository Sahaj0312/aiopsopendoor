"""Tests for the gstack orchestrator.

These tests use deterministic stub layers and gates — no LLM calls — to
pin the orchestrator's behavior. They're the contract the real agents
have to honor.
"""

from __future__ import annotations

from typing import cast

import pytest

from applyops.gstack import (
    Layer,
    LayerOutput,
    RebaseRequest,
    Review,
    ReviewGate,
    Stack,
    StackContext,
)
from applyops.gstack.run import RunStatus


class _Echo(LayerOutput):
    message: str


class StubLayer:
    """Records how many times it ran; emits a different message on rebase."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.run_count = 0

    def run(self, ctx: StackContext) -> LayerOutput:
        self.run_count += 1
        rebase = ctx.pending_rebase(self.name)
        message = f"v{self.run_count}: " + ("rebased: " + rebase.reason if rebase else "initial")
        return _Echo(layer_name=self.name, message=message)


class AlwaysPassGate:
    name = "always-pass"

    def review(self, output: LayerOutput, ctx: StackContext) -> Review:
        return Review(gate_name=self.name, passed=True)


class PassAfterNRebases:
    """Fails for the first N reviews, then passes. Mirrors a strict critic."""

    def __init__(self, n: int) -> None:
        self.name = f"pass-after-{n}"
        self.n = n
        self.calls = 0

    def review(self, output: LayerOutput, ctx: StackContext) -> Review:
        self.calls += 1
        if self.calls > self.n:
            return Review(gate_name=self.name, passed=True)
        return Review(
            gate_name=self.name,
            passed=False,
            rebase_request=RebaseRequest(
                gate_name=self.name,
                reason=f"call {self.calls} of {self.n} required",
                findings=["stub finding"],
            ),
        )


def test_protocols_are_satisfied() -> None:
    assert isinstance(StubLayer("x"), Layer)
    assert isinstance(AlwaysPassGate(), ReviewGate)


def test_stack_rejects_duplicate_layer_names() -> None:
    with pytest.raises(ValueError, match="duplicate layer names"):
        Stack(layers=[StubLayer("a"), StubLayer("a")])


def test_stack_rejects_gate_for_unknown_layer() -> None:
    with pytest.raises(ValueError, match="unknown layers"):
        Stack(layers=[StubLayer("a")], gates={"b": AlwaysPassGate()})


def test_happy_path_lands_completed() -> None:
    a, b = StubLayer("a"), StubLayer("b")
    stack = Stack(layers=[a, b])
    run, ctx = stack.land()
    assert run.status == RunStatus.COMPLETED
    assert a.run_count == 1
    assert b.run_count == 1
    assert cast(_Echo, ctx.output_of("a")).message == "v1: initial"


def test_gate_rebase_loop_passes_after_required_iterations() -> None:
    writer = StubLayer("writer")
    critic = PassAfterNRebases(n=2)
    stack = Stack(layers=[writer], gates={"writer": critic}, max_rebases_per_gate=3)
    run, ctx = stack.land()
    assert run.status == RunStatus.COMPLETED
    assert writer.run_count == 3  # 1 initial + 2 rebases
    assert ctx.layers["writer"].rebases == 2
    assert len(ctx.layers["writer"].gate_reviews) == 3  # 2 fails + 1 pass


def test_gate_blocks_run_when_rebase_budget_exhausted() -> None:
    writer = StubLayer("writer")
    critic = PassAfterNRebases(n=99)  # will never pass within budget
    stack = Stack(layers=[writer], gates={"writer": critic}, max_rebases_per_gate=2)
    run, ctx = stack.land()
    assert run.status == RunStatus.BLOCKED
    assert run.blocked_on is not None
    assert "rebases" in run.blocked_on
    assert ctx.layers["writer"].rebases == 2


def test_up_to_stops_partial() -> None:
    a, b, c = StubLayer("a"), StubLayer("b"), StubLayer("c")
    stack = Stack(layers=[a, b, c])
    run, ctx = stack.land(up_to="b")
    assert run.status == RunStatus.PARTIAL
    assert a.run_count == 1
    assert b.run_count == 1
    assert c.run_count == 0
    assert "c" not in ctx.layers


def test_passing_review_with_rebase_request_is_invalid() -> None:
    with pytest.raises(ValueError, match="passing review must not carry"):
        Review(
            gate_name="x",
            passed=True,
            rebase_request=RebaseRequest(gate_name="x", reason="r"),
        )


def test_failing_review_must_carry_rebase_request() -> None:
    with pytest.raises(ValueError, match="failing review must carry"):
        Review(gate_name="x", passed=False)


def test_rebase_request_renders_as_prompt_fragment() -> None:
    rr = RebaseRequest(
        gate_name="critic",
        reason="claim ungrounded",
        findings=["bullet 2 cites no fact"],
        suggested_changes=["weaken or remove bullet 2"],
    )
    s = rr.as_prompt_fragment()
    assert "REBASE REQUEST" in s
    assert "claim ungrounded" in s
    assert "bullet 2" in s
