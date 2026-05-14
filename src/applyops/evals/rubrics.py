"""Rubric primitives — Score, Scorecard, Rubric, grade()."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Score(BaseModel):
    """One scorer's output against its threshold."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float
    threshold: float
    direction: str = Field(
        default=">=",
        description="Comparison operator: '>=', '<=', '==', '>', '<'.",
    )

    @property
    def passed(self) -> bool:
        if self.direction == ">=":
            return self.value >= self.threshold
        if self.direction == "<=":
            return self.value <= self.threshold
        if self.direction == "==":
            return self.value == self.threshold
        if self.direction == ">":
            return self.value > self.threshold
        if self.direction == "<":
            return self.value < self.threshold
        raise ValueError(f"unknown direction {self.direction!r}")


class Scorecard(BaseModel):
    """All scores for one case. Pass/fail rolls up from the individual scores."""

    model_config = ConfigDict(extra="forbid")

    case_name: str
    scores: list[Score]

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores)

    def failures(self) -> list[Score]:
        return [s for s in self.scores if not s.passed]


class RubricMetric(BaseModel):
    """One metric in a rubric: which scorer, with what threshold and direction."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    scorer: Callable[..., float | int | bool]
    threshold: float
    direction: str = ">="
    needs: list[str] = Field(
        default_factory=list,
        description="Names of grade() inputs the scorer needs, in order. Defaults to ['writer_output'].",
    )


class Rubric(BaseModel):
    """A named collection of metrics graded together against one case."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    metrics: list[RubricMetric]


def grade(rubric: Rubric, case_name: str, **inputs: Any) -> Scorecard:
    """Run every metric in the rubric against the provided inputs.

    Each metric pulls its arguments from `inputs` by `RubricMetric.needs`.
    Default `needs` is ('writer_output',) so the most common writer-only
    metrics work without ceremony.
    """
    scores: list[Score] = []
    for m in rubric.metrics:
        keys = m.needs or ["writer_output"]
        try:
            args = [inputs[k] for k in keys]
        except KeyError as exc:
            raise KeyError(f"rubric metric {m.name!r} needs {keys}, missing {exc}") from exc
        raw = m.scorer(*args)
        value: float = float(raw) if not isinstance(raw, bool) else (1.0 if raw else 0.0)
        scores.append(
            Score(
                metric=m.name,
                value=value,
                threshold=m.threshold,
                direction=m.direction,
            )
        )
    return Scorecard(case_name=case_name, scores=scores)
