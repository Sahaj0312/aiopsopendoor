"""Critic gate — rubric-based review of the writer's output.

The critic is a `ReviewGate`, not a `Layer`. It does not produce content.
It scores the writer's output against a rubric and emits a `Review`:
either pass (stack continues) or fail with a `RebaseRequest` that the
writer must address on its next run.

The rubric is a structured set of checks:
1. JD coverage — do the high-importance requirements have at least one
   addressing claim?
2. Grounding density — what fraction of bullets cite at least one fact?
3. Claim concentration — too many bullets citing the same fact is a smell.
4. Tone — no banned filler phrases ("results-driven", "synergy", etc.).
5. Application protocol — did the cover letter address protocol notes
   from the JD (e.g., "apply using AI")?

The critic's threshold is configurable. The first three are computed
deterministically; the tone and protocol checks call the LLM with a
strict structured output schema.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from applyops.agents.recruiter import RoleAnalysis
from applyops.agents.writer import WriterOutput
from applyops.gstack.context import StackContext
from applyops.gstack.types import LayerOutput, RebaseRequest, Review

CRITIC_SYSTEM_PROMPT = """You are a strict senior reviewer of an AI Ops Engineer application package.

You receive:
- `role_analysis`: requirements and signals from the JD.
- `writer_output`: the CV draft and cover letter you are reviewing.
- `deterministic_findings`: numeric findings already computed by the orchestrator (coverage, grounding density, etc.). Use them; do not re-derive.

Your job is to do the parts of the review that need judgment:

1. Tone (`tone_findings`): list any sentences that read as stock filler, vague boasting, or AI-generated mush. Examples to flag: "results-driven", "passionate about technology", "leveraging cutting-edge AI to drive impact". Empty list if none.

2. Application-protocol response (`protocol_response`): if `role_analysis.application_protocol_notes` is non-empty, does the cover letter actually address them? Return:
   - status="addressed" if the cover letter addresses each protocol note clearly.
   - status="missing" if any note is not addressed.
   - status="not_required" if there are no protocol notes.

3. Overall verdict: based on the deterministic findings AND your tone/protocol checks, decide pass or request_changes. Be strict: this application is competing with strong candidates.

If you request_changes, your `findings` and `suggested_changes` are sent to the writer as a RebaseRequest. Be specific: vague feedback wastes a rebase cycle."""


class CriticPayload(BaseModel):
    """The LLM-driven portion of the critic's review."""

    model_config = ConfigDict(extra="forbid")

    tone_findings: list[str] = Field(
        default_factory=list,
        description="Sentences flagged as filler or AI-mush. Empty if tone is fine.",
    )
    protocol_response: str = Field(
        ...,
        description="One of: addressed, missing, not_required.",
    )
    verdict: str = Field(
        ...,
        description="One of: pass, request_changes.",
    )
    rationale: str = Field(
        ...,
        description="2-4 sentence justification of the verdict.",
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Specific issues for the writer to fix on rebase. Required when verdict=request_changes.",
    )
    suggested_changes: list[str] = Field(
        default_factory=list,
        description="Concrete edits the writer should consider on rebase.",
    )


class RubricFindings(BaseModel):
    """Deterministic findings computed from writer output + role analysis."""

    model_config = ConfigDict(extra="forbid")

    high_importance_requirements_covered: float = Field(
        ...,
        description="Fraction of importance>=4 requirements that have at least one addressing claim.",
    )
    grounding_density: float = Field(
        ...,
        description="Fraction of CV bullets that cite at least one fact_id. (Should be 1.0; writer validation already enforces this, but we re-check.)",
    )
    max_claims_per_fact: int = Field(
        ...,
        description="Most-cited fact's claim count. Above ~4 suggests over-reliance on one fact.",
    )
    cover_letter_paragraph_count: int
    notes: list[str] = Field(default_factory=list)


class StructuredLLM(Protocol):
    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel: ...


class CriticGate:
    """Rubric-based review gate for the writer's output."""

    name = "critic"

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        model: str = "gpt-4.1-mini",
        high_importance_coverage_threshold: float = 0.75,
        max_claims_per_fact: int = 4,
    ) -> None:
        self.llm = llm
        self.model = model
        self.coverage_threshold = high_importance_coverage_threshold
        self.max_claims_per_fact_threshold = max_claims_per_fact

    def review(self, output: LayerOutput, ctx: StackContext) -> Review:
        if not isinstance(output, WriterOutput):
            raise TypeError(f"critic reviews WriterOutput, got {type(output).__name__}")
        role_analysis = ctx.output_of("recruiter")
        if not isinstance(role_analysis, RoleAnalysis):
            raise TypeError(
                f"critic expects RoleAnalysis upstream, got {type(role_analysis).__name__}"
            )

        deterministic = self._compute_rubric(output, role_analysis)
        payload = self._llm_review(output, role_analysis, deterministic)
        return self._compose_review(deterministic, payload)

    def _compute_rubric(
        self,
        output: WriterOutput,
        role_analysis: RoleAnalysis,
    ) -> RubricFindings:
        high_imp_reqs = [r for r in role_analysis.requirements if r.importance >= 4]
        addressed_texts: set[str] = set()
        all_claims = output.grounded_claims()
        for claim in all_claims:
            addressed_texts.update(claim.addresses)

        coverage = (
            sum(1 for r in high_imp_reqs if r.text in addressed_texts) / len(high_imp_reqs)
            if high_imp_reqs
            else 1.0
        )

        # Grounding density across CV bullets (cover letter paragraphs separate).
        cv_bullets = [
            claim
            for entries in (output.cv.experience, output.cv.projects, output.cv.education)
            for entry in entries
            for claim in entry.bullets
        ]
        if cv_bullets:
            density = sum(1 for c in cv_bullets if c.fact_ids) / len(cv_bullets)
        else:
            density = 1.0

        # Most-cited fact concentration.
        fact_citation_counts: dict[str, int] = {}
        for claim in all_claims:
            for fid in claim.fact_ids:
                fact_citation_counts[fid] = fact_citation_counts.get(fid, 0) + 1
        max_concentration = max(fact_citation_counts.values(), default=0)

        notes: list[str] = []
        if coverage < self.coverage_threshold:
            notes.append(
                f"high-importance requirement coverage {coverage:.0%} is below "
                f"threshold {self.coverage_threshold:.0%}"
            )
        if density < 1.0:
            notes.append(f"grounding density {density:.0%} (target 100%)")
        if max_concentration > self.max_claims_per_fact_threshold:
            notes.append(
                f"one fact is cited {max_concentration} times "
                f"(threshold {self.max_claims_per_fact_threshold})"
            )

        return RubricFindings(
            high_importance_requirements_covered=coverage,
            grounding_density=density,
            max_claims_per_fact=max_concentration,
            cover_letter_paragraph_count=len(output.cover_letter.paragraphs),
            notes=notes,
        )

    def _llm_review(
        self,
        output: WriterOutput,
        role_analysis: RoleAnalysis,
        deterministic: RubricFindings,
    ) -> CriticPayload:
        import json

        user = json.dumps(
            {
                "role_analysis": role_analysis.model_dump(mode="json", exclude={"raw_jd_excerpt"}),
                "writer_output": output.model_dump(mode="json", exclude={"layer_name", "produced_at"}),
                "deterministic_findings": deterministic.model_dump(mode="json"),
            },
            indent=2,
            default=str,
        )
        payload = self.llm.parse(
            model=self.model,
            system=CRITIC_SYSTEM_PROMPT,
            user=user,
            schema=CriticPayload,
        )
        assert isinstance(payload, CriticPayload)
        return payload

    def _compose_review(
        self,
        deterministic: RubricFindings,
        payload: CriticPayload,
    ) -> Review:
        # Deterministic findings can force a fail even if the LLM says pass.
        # The reverse is also true: clean numbers don't override tone issues.
        deterministic_fail = bool(deterministic.notes)
        llm_fail = payload.verdict == "request_changes"
        passed = not (deterministic_fail or llm_fail)

        if passed:
            return Review(
                gate_name=self.name,
                passed=True,
                score=deterministic.high_importance_requirements_covered,
                notes=payload.rationale,
            )

        # Compose findings from both sources.
        combined_findings: list[str] = list(deterministic.notes) + list(payload.findings)
        if payload.tone_findings:
            combined_findings.append(
                "tone: " + "; ".join(payload.tone_findings)
            )
        if payload.protocol_response == "missing":
            combined_findings.append(
                "application protocol notes were not addressed in the cover letter"
            )

        return Review(
            gate_name=self.name,
            passed=False,
            score=deterministic.high_importance_requirements_covered,
            notes=payload.rationale,
            rebase_request=RebaseRequest(
                gate_name=self.name,
                reason=payload.rationale,
                findings=combined_findings,
                suggested_changes=list(payload.suggested_changes),
            ),
        )
