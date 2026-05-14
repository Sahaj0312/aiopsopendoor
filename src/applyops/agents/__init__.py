"""applyops.agents — the layers and gates that compose the stack."""

from __future__ import annotations

from applyops.agents.critic import CriticGate, RubricFindings
from applyops.agents.recruiter import RecruiterAgent, RoleAnalysis
from applyops.agents.types import JDMeta, Requirement
from applyops.agents.writer import (
    CoverLetter,
    CVDraft,
    CVEntry,
    GroundedClaim,
    WriterAgent,
    WriterOutput,
    WriterValidationError,
)

__all__ = [
    "CoverLetter",
    "CriticGate",
    "CVDraft",
    "CVEntry",
    "GroundedClaim",
    "JDMeta",
    "RecruiterAgent",
    "Requirement",
    "RoleAnalysis",
    "RubricFindings",
    "WriterAgent",
    "WriterOutput",
    "WriterValidationError",
]
