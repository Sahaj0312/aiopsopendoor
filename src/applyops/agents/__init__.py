"""applyops.agents — the layers and gates that compose the stack."""

from __future__ import annotations

from applyops.agents.critic import CriticGate, RubricFindings
from applyops.agents.factchecker import (
    ClaimAudit,
    FactCheckerAgent,
    FactCheckOutput,
    Flag,
)
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
    "ClaimAudit",
    "CoverLetter",
    "CriticGate",
    "CVDraft",
    "CVEntry",
    "FactCheckerAgent",
    "FactCheckOutput",
    "Flag",
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
