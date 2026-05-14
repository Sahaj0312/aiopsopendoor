"""applyops.evals — rubric harness for the agent pipeline.

The harness is intentionally framework-free. Scorers are plain functions
that take agent outputs and return floats. A `Scorecard` aggregates a
case's scores against thresholds. Pytest runs rubric tests via the
`eval` marker; `applyops eval` runs them outside pytest for demo /
manual use.

The eval harness is the AI Ops centerpiece. Every prompt change to any
agent must pass these rubrics before it can merge. CI enforces.
"""

from __future__ import annotations

from applyops.evals.fixtures import load_fixture
from applyops.evals.rubrics import Rubric, Score, Scorecard
from applyops.evals.scorers import (
    BANNED_PHRASES,
    fact_concentration,
    grounding_density,
    jd_coverage_score,
    provenance_completeness,
    tone_drift_count,
)

__all__ = [
    "BANNED_PHRASES",
    "Rubric",
    "Score",
    "Scorecard",
    "fact_concentration",
    "grounding_density",
    "jd_coverage_score",
    "load_fixture",
    "provenance_completeness",
    "tone_drift_count",
]
