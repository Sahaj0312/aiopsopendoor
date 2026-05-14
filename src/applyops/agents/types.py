"""Shared data types used across agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RequirementKind = Literal["must_have", "nice_to_have", "implied"]
RequirementCategory = Literal[
    "technical",
    "experience",
    "soft_skill",
    "domain",
    "operational",
]


class Requirement(BaseModel):
    """One thing the role asks for, extracted from the JD."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="The requirement, verbatim or lightly paraphrased.")
    kind: RequirementKind
    importance: int = Field(ge=1, le=5, description="1 = trivial, 5 = blocker.")
    category: RequirementCategory
    evidence_anchor: str = Field(
        ...,
        description=(
            "The kind of candidate evidence that would address this "
            "requirement, written so the writer agent can search facts.json. "
            "E.g. 'production LLM eval harness experience' or 'on-call "
            "rotation exposure'. Does not name any specific candidate fact."
        ),
    )


class JDMeta(BaseModel):
    """Provenance for the JD we built the application against."""

    model_config = ConfigDict(extra="forbid")

    url: str | None
    hash: str = Field(..., description="sha256[:12] of the cleaned JD text.")
    snapshot_path: str
    drift: bool = Field(
        default=False,
        description="True when this hash differs from the prior LATEST snapshot.",
    )
