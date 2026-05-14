"""Rubric tests for the candidate facts file itself.

Independent of the writer — these grade the candidate's facts.json
quality before the writer ever runs. If the facts file has unverified
entries or thin tags, the application downstream will only be as strong
as that input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from applyops.evals import provenance_completeness
from applyops.facts import load

pytestmark = pytest.mark.eval

EXAMPLE = Path(__file__).parent.parent.parent / "inputs" / "facts.example.json"


def test_example_facts_have_full_provenance() -> None:
    """The committed example file must always show 100% attestation.

    If a contributor lowers this, they're leaking a half-attested fact
    into git. The rubric catches it.
    """
    candidate = load(EXAMPLE)
    assert provenance_completeness(candidate) == 1.0


def test_every_committed_fact_has_at_least_one_tag() -> None:
    """Tags are what the writer agent retrieves on. Untagged facts are
    effectively invisible to it."""
    candidate = load(EXAMPLE)
    untagged = [f.id for f in candidate.facts if not f.tags]
    assert not untagged, f"facts without tags: {untagged}"
