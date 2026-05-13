"""Tests for the facts schema. These define what 'valid grounding data' means."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from applyops.facts import Candidate, Fact, Provenance, load

EXAMPLE = Path(__file__).parent.parent / "inputs" / "facts.example.json"


def _minimal_fact(**overrides: object) -> Fact:
    base = {
        "id": "fact-1",
        "kind": "experience",
        "title": "Test",
        "detail": "Did a thing.",
        "provenance": [Provenance(source="resume.pdf", verified_by="self")],
    }
    base.update(overrides)
    return Fact(**base)  # type: ignore[arg-type]


def test_example_file_validates() -> None:
    c = load(EXAMPLE)
    assert c.name
    assert len(c.facts) >= 1
    for f in c.facts:
        assert f.verified, f"example facts should be self-attested, got {f.id}"


def test_fact_requires_at_least_one_provenance() -> None:
    with pytest.raises(ValidationError):
        Fact(
            id="x",
            kind="skill",
            title="Python",
            detail="I know Python.",
            provenance=[],
        )


def test_fact_id_must_be_slug() -> None:
    with pytest.raises(ValidationError):
        _minimal_fact(id="not a slug")


def test_unverified_fact_is_flagged() -> None:
    f = _minimal_fact(
        id="unverified-1",
        provenance=[Provenance(source="parsed from pdf", verified_by="ai_extracted_unverified")],
    )
    assert not f.verified


def test_candidate_rejects_duplicate_fact_ids() -> None:
    a = _minimal_fact(id="dupe")
    b = _minimal_fact(id="dupe", title="other")
    with pytest.raises(ValidationError, match="duplicate fact ids"):
        Candidate(
            name="X",
            headline="Y",
            location="Z",
            facts=[a, b],
        )


def test_candidate_helpers() -> None:
    c = load(EXAMPLE)
    assert c.by_kind("experience")
    assert c.by_tag("evals")
    assert c.unverified() == []
