"""Data types passed between layers and gates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LayerOutput(BaseModel):
    """Base class for any layer's structured output.

    Concrete layers subclass this with their own fields. The orchestrator
    only needs the base fields; everything else is the layer's business.
    """

    model_config = ConfigDict(extra="forbid")

    layer_name: str
    produced_at: datetime = Field(default_factory=_now)


class RebaseRequest(BaseModel):
    """A gate's request that a layer re-run with this feedback attached."""

    model_config = ConfigDict(extra="forbid")

    gate_name: str
    reason: str = Field(..., description="One-line summary of why the gate failed.")
    findings: list[str] = Field(
        default_factory=list,
        description="Specific issues the layer should address on rebase.",
    )
    suggested_changes: list[str] = Field(
        default_factory=list,
        description="Concrete edits the gate proposes. The layer is free to ignore.",
    )

    def as_prompt_fragment(self) -> str:
        """Render the rebase request as a string suitable for inclusion in an LLM prompt."""
        lines = [f"REBASE REQUEST from gate `{self.gate_name}`:", f"  reason: {self.reason}"]
        if self.findings:
            lines.append("  findings:")
            lines.extend(f"    - {f}" for f in self.findings)
        if self.suggested_changes:
            lines.append("  suggested changes:")
            lines.extend(f"    - {s}" for s in self.suggested_changes)
        return "\n".join(lines)


class Review(BaseModel):
    """A gate's verdict on a layer's output."""

    model_config = ConfigDict(extra="forbid")

    gate_name: str
    passed: bool
    score: float | None = Field(
        default=None,
        description="Optional rubric score in [0, 1]. None if the gate is binary.",
    )
    notes: str = Field(default="", description="Free-form rationale.")
    rebase_request: RebaseRequest | None = Field(
        default=None,
        description="Required when passed is False. Must be None when passed is True.",
    )
    reviewed_at: datetime = Field(default_factory=_now)

    def model_post_init(self, _: Any) -> None:
        if self.passed and self.rebase_request is not None:
            raise ValueError("a passing review must not carry a rebase_request")
        if not self.passed and self.rebase_request is None:
            raise ValueError("a failing review must carry a rebase_request")
