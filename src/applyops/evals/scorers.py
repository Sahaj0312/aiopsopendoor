"""Scorer functions — pure, deterministic, fast.

Each scorer takes structured agent outputs and returns a float (or int)
that a Rubric can compare against a threshold. Scorers are intentionally
not coupled to the eval framework; they're useful as runtime checks too
(the critic gate uses some of these directly).

If a scorer needs an LLM call to compute, it doesn't belong here —
that's the critic gate or the factchecker's job. The eval harness's
scorers are deterministic so the rubric is fast, free, and reproducible
across CI runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from applyops.agents.recruiter import RoleAnalysis
from applyops.agents.writer import WriterOutput
from applyops.facts import Candidate

# Stock filler that AI Ops-grade applications should never carry.
BANNED_PHRASES: tuple[str, ...] = (
    "results-driven",
    "results driven",
    "passionate about technology",
    "passionate about ai",
    "leveraging cutting-edge",
    "synergy",
    "synergies",
    "best of breed",
    "best-in-class",
    "world-class",
    "10x engineer",
    "rockstar",
    "ninja",
    "guru",
    "drive impact",
    "thought leader",
)


def jd_coverage_score(
    writer_output: WriterOutput,
    role_analysis: RoleAnalysis,
    *,
    min_importance: int = 4,
) -> float:
    """Fraction of importance>=min_importance requirements that have at
    least one addressing claim.

    Returns 1.0 if there are no requirements at that importance level
    (vacuously satisfied).
    """
    targets = [r for r in role_analysis.requirements if r.importance >= min_importance]
    if not targets:
        return 1.0
    addressed: set[str] = set()
    for claim in writer_output.grounded_claims():
        addressed.update(claim.addresses)
    return sum(1 for r in targets if r.text in addressed) / len(targets)


def grounding_density(writer_output: WriterOutput) -> float:
    """Fraction of grounded claims that cite at least one fact_id.

    Should be 1.0 — the writer's validation enforces this. We re-check
    here as defense against drift.
    """
    claims = writer_output.grounded_claims()
    if not claims:
        return 1.0
    return sum(1 for c in claims if c.fact_ids) / len(claims)


def fact_concentration(writer_output: WriterOutput) -> int:
    """Maximum number of claims citing the same fact_id.

    A high value (>4) suggests over-reliance on one fact — the writer
    is stretching one experience to cover the whole application.
    """
    counts: dict[str, int] = {}
    for claim in writer_output.grounded_claims():
        for fid in claim.fact_ids:
            counts[fid] = counts.get(fid, 0) + 1
    return max(counts.values(), default=0)


def tone_drift_count(
    writer_output: WriterOutput,
    banned: Iterable[str] = BANNED_PHRASES,
) -> int:
    """Number of banned filler phrases found in the writer's output text."""
    haystack_parts: list[str] = [writer_output.cv.summary.text]
    for entries in (
        writer_output.cv.experience,
        writer_output.cv.projects,
        writer_output.cv.education,
    ):
        for entry in entries:
            for bullet in entry.bullets:
                haystack_parts.append(bullet.text)
    for para in writer_output.cover_letter.paragraphs:
        haystack_parts.append(para.text)
    haystack = " || ".join(haystack_parts).lower()

    hits = 0
    for phrase in banned:
        if re.search(rf"\b{re.escape(phrase.lower())}\b", haystack):
            hits += 1
    return hits


def provenance_completeness(candidate: Candidate) -> float:
    """Fraction of facts with at least one attested provenance entry.

    1.0 means every fact has been hand-attested or third-party verified.
    Below 1.0 means some claims could only be grounded by trusting the
    AI parser's output, which the factchecker treats as a hard block.
    """
    if not candidate.facts:
        return 1.0
    return sum(1 for f in candidate.facts if f.verified) / len(candidate.facts)


def cover_letter_addresses_protocol(
    writer_output: WriterOutput,
    role_analysis: RoleAnalysis,
) -> bool:
    """True if every JD application_protocol_note has a cover-letter paragraph
    that plausibly addresses it (substring match on key tokens).

    Cheap heuristic — the critic does deeper judgment work via LLM. This
    scorer is the deterministic floor.
    """
    notes = role_analysis.application_protocol_notes
    if not notes:
        return True
    body = " ".join(p.text for p in writer_output.cover_letter.paragraphs).lower()
    for note in notes:
        tokens = [t for t in note.lower().split() if len(t) >= 4]
        if not any(t in body for t in tokens):
            return False
    return True
