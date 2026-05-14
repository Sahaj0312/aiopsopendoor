"""Rubric tests for the writer's output.

These are pytest tests marked with `@pytest.mark.eval`. They're skipped
from the default `make test` run (which is for unit tests) and gated
into `make eval` and CI.

Each test:
1. loads a writer-output fixture and the canonical role analysis
2. runs the rubric grade()
3. asserts the scorecard passed (or, for known-bad fixtures, asserts it
   failed on the specific metrics it's designed to fail on)

The "known-bad" cases are negative controls — they prove the rubric
actually catches the regressions it claims to catch.
"""

from __future__ import annotations

import pytest

from applyops.agents.recruiter import RoleAnalysis
from applyops.agents.writer import WriterOutput
from applyops.evals import (
    Rubric,
    fact_concentration,
    grounding_density,
    jd_coverage_score,
    load_fixture,
    tone_drift_count,
)
from applyops.evals.rubrics import RubricMetric, grade
from applyops.evals.scorers import cover_letter_addresses_protocol

pytestmark = pytest.mark.eval


def _writer_rubric() -> Rubric:
    return Rubric(
        name="writer-output-rubric-v1",
        metrics=[
            RubricMetric(
                name="jd_coverage_high_importance",
                scorer=jd_coverage_score,
                threshold=0.75,
                direction=">=",
                needs=["writer_output", "role_analysis"],
            ),
            RubricMetric(
                name="grounding_density",
                scorer=grounding_density,
                threshold=1.0,
                direction=">=",
            ),
            RubricMetric(
                name="fact_concentration",
                scorer=fact_concentration,
                threshold=4,
                direction="<=",
            ),
            RubricMetric(
                name="tone_drift",
                scorer=tone_drift_count,
                threshold=0,
                direction="<=",
            ),
            RubricMetric(
                name="protocol_addressed",
                scorer=cover_letter_addresses_protocol,
                threshold=1.0,
                direction=">=",
                needs=["writer_output", "role_analysis"],
            ),
        ],
    )


@pytest.fixture
def role_analysis() -> RoleAnalysis:
    return load_fixture("role_analysis.opendoor.json", RoleAnalysis)


def test_good_writer_output_passes_full_rubric(role_analysis: RoleAnalysis) -> None:
    wo = load_fixture("writer_output.good.json", WriterOutput)
    card = grade(
        _writer_rubric(),
        case_name="good",
        writer_output=wo,
        role_analysis=role_analysis,
    )
    failures = card.failures()
    assert not failures, f"unexpected failures: {[(s.metric, s.value) for s in failures]}"


def test_bad_coverage_fixture_fails_coverage_and_tone(role_analysis: RoleAnalysis) -> None:
    wo = load_fixture("writer_output.bad_coverage.json", WriterOutput)
    card = grade(
        _writer_rubric(),
        case_name="bad_coverage",
        writer_output=wo,
        role_analysis=role_analysis,
    )
    assert not card.passed
    failed = {s.metric for s in card.failures()}
    assert "jd_coverage_high_importance" in failed
    assert "tone_drift" in failed
    assert "protocol_addressed" in failed


def test_overconcentrated_fixture_fails_fact_concentration(role_analysis: RoleAnalysis) -> None:
    wo = load_fixture("writer_output.fact_overconcentrated.json", WriterOutput)
    card = grade(
        _writer_rubric(),
        case_name="overconcentrated",
        writer_output=wo,
        role_analysis=role_analysis,
    )
    assert not card.passed
    failed = {s.metric for s in card.failures()}
    assert "fact_concentration" in failed


def test_jd_coverage_score_independently() -> None:
    """Spot-check that the coverage scorer math is right outside the rubric framework."""
    role = load_fixture("role_analysis.opendoor.json", RoleAnalysis)
    good = load_fixture("writer_output.good.json", WriterOutput)
    bad = load_fixture("writer_output.bad_coverage.json", WriterOutput)
    assert jd_coverage_score(good, role) >= 0.75
    assert jd_coverage_score(bad, role) <= 0.5


def test_tone_drift_catches_banned_phrases() -> None:
    wo = load_fixture("writer_output.bad_coverage.json", WriterOutput)
    # bad_coverage deliberately includes several banned phrases
    assert tone_drift_count(wo) >= 3
