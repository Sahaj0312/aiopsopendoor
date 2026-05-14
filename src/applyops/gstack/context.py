"""StackContext — the shared state passed to every layer and gate during a run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from applyops.gstack.types import LayerOutput, RebaseRequest, Review

if TYPE_CHECKING:
    from applyops.gstack.run import Run


class LayerState(BaseModel):
    """The recorded state of one layer during a run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    output: LayerOutput | None = None
    rebases: int = 0
    gate_reviews: list[Review] = Field(default_factory=list)
    pending_rebase: RebaseRequest | None = None
    stale: bool = False

    def latest_review(self) -> Review | None:
        return self.gate_reviews[-1] if self.gate_reviews else None


class StackContext(BaseModel):
    """Mutable per-run state. Layers and gates read and write here."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run: Run
    layers: dict[str, LayerState] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Run-level inputs: jd source, facts path, model overrides, etc.",
    )

    def output_of(self, layer_name: str) -> LayerOutput:
        """Return the latest output of a named layer. Raises if absent."""
        state = self.layers.get(layer_name)
        if state is None or state.output is None:
            raise KeyError(f"no output recorded for layer {layer_name!r}")
        return state.output

    def pending_rebase(self, layer_name: str) -> RebaseRequest | None:
        """Return the pending rebase request for a layer, if the orchestrator set one."""
        state = self.layers.get(layer_name)
        return state.pending_rebase if state else None


# Resolve the forward reference now that Run is importable at runtime.
from applyops.gstack.run import Run  # noqa: E402

StackContext.model_rebuild()
