"""Protocols for the things a stack composes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from applyops.gstack.types import LayerOutput, Review

if TYPE_CHECKING:
    from applyops.gstack.context import StackContext


@runtime_checkable
class Layer(Protocol):
    """A unit in the stack that transforms inputs into a structured output.

    Implementations are responsible for their own LLM/tool calls. The
    orchestrator only knows about `name` and `run`. A layer can read the
    outputs of any layer below it via `ctx.layers[other_name].output`,
    and it can see any pending rebase request via `ctx.pending_rebase()`.
    """

    name: str

    def run(self, ctx: StackContext) -> LayerOutput: ...


@runtime_checkable
class ReviewGate(Protocol):
    """A review gate inspects a layer's output and decides pass or rebase.

    Gates do not produce content. They emit a `Review` with `passed=True`
    or a `Review` with `passed=False` and a `rebase_request`. The
    orchestrator handles the rebase loop.
    """

    name: str

    def review(self, output: LayerOutput, ctx: StackContext) -> Review: ...
