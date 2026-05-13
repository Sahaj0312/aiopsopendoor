"""facts.py — the candidate's source of truth.

Every factual claim in the generated application traces back to an entry
in here, with a `Provenance` block recording where the claim came from and
who verified it. The fact-checker agent treats `Fact.verified == False` as
a hard block: nothing unverified reaches the form.

The shape is intentionally strict. The cost of one extra required field is
small; the cost of an ungrounded claim in the application is large.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

FactKind = Literal[
    "experience",
    "project",
    "skill",
    "education",
    "achievement",
    "publication",
    "credential",
]


class Provenance(BaseModel):
    """Where a fact came from and how we know it's true."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        ...,
        description="Human-readable source. E.g. 'resume.pdf p.1', 'linkedin.com/in/sahaj', 'github.com/Sahaj0312/foo README'.",
    )
    source_url: HttpUrl | None = Field(
        default=None,
        description="URL the source can be retrieved from, if applicable.",
    )
    retrieved_at: date | None = Field(
        default=None,
        description="When the source was retrieved or last verified.",
    )
    verified_by: Literal["self", "third_party", "ai_extracted_unverified"] = Field(
        default="ai_extracted_unverified",
        description=(
            "Verification level. `self`: candidate hand-attested. "
            "`third_party`: reference letter, employer URL, etc. "
            "`ai_extracted_unverified`: pulled from a doc by the parser, "
            "still needs human attestation before use."
        ),
    )

    @property
    def is_attested(self) -> bool:
        return self.verified_by in ("self", "third_party")


class Fact(BaseModel):
    """One atomic, citable claim about the candidate."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable slug, e.g. 'exp-acme-2024-eng-lead'.")
    kind: FactKind
    title: str = Field(..., description="Short label, e.g. 'Senior Engineer at Acme'.")
    detail: str = Field(..., description="The claim itself, written as it could appear in a CV.")
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for retrieval. E.g. ['python', 'observability', 'on-call'].",
    )
    started_on: date | None = None
    ended_on: date | None = None
    metrics: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Quantitative facts attached to this entry, e.g. "
            "{'p95_latency_reduction': '40%', 'team_size': '5'}. "
            "Writer agent may cite these; it may not invent them."
        ),
    )
    provenance: list[Provenance] = Field(
        ...,
        min_length=1,
        description="At least one source per fact. Multiple sources strengthen the claim.",
    )

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(f"id must be a slug (alnum + - _), got: {v!r}")
        return v

    @property
    def verified(self) -> bool:
        return any(p.is_attested for p in self.provenance)


class Candidate(BaseModel):
    """Top-level container. One per applicant."""

    model_config = ConfigDict(extra="forbid")

    name: str
    headline: str = Field(..., description="One-line positioning statement.")
    location: str
    links: dict[str, HttpUrl] = Field(
        default_factory=dict,
        description="Named external links: github, linkedin, website, etc.",
    )
    facts: list[Fact] = Field(..., min_length=1)

    @field_validator("facts")
    @classmethod
    def _unique_ids(cls, v: list[Fact]) -> list[Fact]:
        ids = [f.id for f in v]
        if len(ids) != len(set(ids)):
            dupes = {x for x in ids if ids.count(x) > 1}
            raise ValueError(f"duplicate fact ids: {sorted(dupes)}")
        return v

    def get(self, fact_id: str) -> Fact:
        for f in self.facts:
            if f.id == fact_id:
                return f
        raise KeyError(fact_id)

    def by_kind(self, kind: FactKind) -> list[Fact]:
        return [f for f in self.facts if f.kind == kind]

    def by_tag(self, tag: str) -> list[Fact]:
        return [f for f in self.facts if tag in f.tags]

    def unverified(self) -> list[Fact]:
        return [f for f in self.facts if not f.verified]


def load(path: str | Path) -> Candidate:
    """Load a Candidate from a JSON file."""
    p = Path(path)
    return Candidate.model_validate_json(p.read_text(encoding="utf-8"))


def dump(candidate: Candidate, path: str | Path) -> None:
    """Write a Candidate to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(candidate.model_dump(mode="json"), indent=2), encoding="utf-8")
