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
from applyops.agents.submitter import (
    FormField,
    FormFillPlan,
    SubmitterAgent,
    SubmitterBlocked,
    SubmitterOutput,
)
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
    "CVDraft",
    "CVEntry",
    "ClaimAudit",
    "CoverLetter",
    "CriticGate",
    "FactCheckOutput",
    "FactCheckerAgent",
    "Flag",
    "FormField",
    "FormFillPlan",
    "GroundedClaim",
    "JDMeta",
    "RecruiterAgent",
    "Requirement",
    "RoleAnalysis",
    "RubricFindings",
    "SubmitterAgent",
    "SubmitterBlocked",
    "SubmitterOutput",
    "WriterAgent",
    "WriterOutput",
    "WriterValidationError",
]
