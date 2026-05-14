"""Writer agent — RoleAnalysis + Candidate facts → CVDraft + CoverLetter.

The writer is the centerpiece of the stack. Inputs:
- the upstream `RoleAnalysis` (recruiter's structured read of the JD)
- the candidate's `facts.json` (source of truth — gitignored)
- any pending `RebaseRequest` from the critic on a prior writer run

Output is a `WriterOutput` containing:
- a `CVDraft` with per-section grounded claims (every bullet cites at
  least one fact_id from facts.json)
- a `CoverLetter` whose paragraphs also cite facts

The writer never invents facts. Its prompt forbids claims without
fact_id citations, and the agent itself validates after the LLM call:
any fact_id in any GroundedClaim that doesn't exist in the loaded
`Candidate` is a hard error (signals an LLM hallucination — the
factchecker handles softer grounding issues later).

Application-protocol awareness: if the recruiter extracted any
`application_protocol_notes` (e.g., "apply using AI"), the writer is
instructed to address them in the cover letter naturally.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from applyops.agents.recruiter import RoleAnalysis
from applyops.facts import Candidate
from applyops.gstack.context import StackContext
from applyops.gstack.types import LayerOutput

WRITER_SYSTEM_PROMPT = """You are a senior technical writer producing an honest, JD-tailored application package for a candidate applying to an AI Ops Engineer role.

Inputs you receive:
- `role_analysis`: a structured read of the job description (requirements, signals, application-protocol notes).
- `candidate`: the candidate's facts.json — the ONLY source of factual claims you may use.
- `rebase_request` (optional): feedback from the critic on your previous draft. If present, address it.

Hard rules:

1. Every `GroundedClaim` you produce MUST list at least one `fact_id` from `candidate.facts`. No exceptions. Bullets without grounding are dropped by the orchestrator.

2. Do NOT invent metrics, scope, team sizes, dollar amounts, or impact figures. If a metric is not in `candidate.facts[*].metrics` or `detail`, do not state it. Prefer qualitative phrasing over invented numbers.

3. `addresses` on a GroundedClaim must reference one or more `requirement.text` strings from `role_analysis.requirements`. If a claim doesn't address a specific requirement, leave the list empty — don't fabricate a match.

4. Prefer FEWER strong claims over many weak ones. A CV bullet that cites three facts and addresses a high-importance requirement is worth more than five bullets that each cite one fact and address nothing.

5. The CV's `summary` is a one-sentence positioning line. The cover letter's first paragraph hooks; middle paragraphs map experience to top requirements; the last paragraph addresses any application-protocol notes (e.g., if the JD says "apply using AI", briefly mention how the application was produced — but only if the JD asks).

6. For `experience` and `projects` entries:
   - `heading` is the candidate-facing display (e.g., "Software Engineer, Quickplay — Toronto — Jan 2025 – Present").
   - `date_range` is a separate field for downstream rendering; empty string if not applicable.
   - `primary_fact_id` is the fact representing the entry itself (the experience or project fact).
   - `bullets` are GroundedClaims that drill into specifics.

7. `skills_line` is a single comma-separated line of relevant skills, drawn from `candidate.facts` skill entries. Prioritize what `role_analysis.requirements` highlight. Skip skills that don't address any requirement.

8. `education` entries should each cite the relevant education fact.

9. Tone: confident but not boastful. Specific, concrete, evidence-led. Avoid stock phrases ("results-driven", "passionate about technology"). The reader is a senior AI Ops engineer who can smell filler.

10. If a rebase_request is provided, treat its findings as binding: each finding is something you got wrong on the prior draft and must fix. Suggested_changes are advisory; you may diverge if you have a better solution that addresses the same finding.

Output is a strict Pydantic schema. Fill every field. Empty lists for sections that genuinely don't apply are fine; missing fields are not."""


class GroundedClaim(BaseModel):
    """A factual claim that cites the facts.json entries that back it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    fact_ids: list[str] = Field(
        ...,
        description="At least one fact_id from candidate.facts that grounds this claim.",
    )
    addresses: list[str] = Field(
        default_factory=list,
        description="Requirement.text strings from role_analysis this claim addresses. Empty list = doesn't target a specific requirement.",
    )


class CVEntry(BaseModel):
    """One experience / project / education line."""

    model_config = ConfigDict(extra="forbid")

    heading: str
    date_range: str = Field(
        default="",
        description="Human-readable date range; empty string if none.",
    )
    primary_fact_id: str = Field(
        ...,
        description="The fact_id of the experience/project/education itself.",
    )
    bullets: list[GroundedClaim] = Field(default_factory=list)


class CVDraft(BaseModel):
    """The structured CV the writer emits."""

    model_config = ConfigDict(extra="forbid")

    summary: GroundedClaim
    experience: list[CVEntry] = Field(default_factory=list)
    projects: list[CVEntry] = Field(default_factory=list)
    skills_line: str = ""
    education: list[CVEntry] = Field(default_factory=list)


class CoverLetter(BaseModel):
    """The cover letter — paragraphs, each grounded."""

    model_config = ConfigDict(extra="forbid")

    paragraphs: list[GroundedClaim] = Field(default_factory=list)


class WriterPayload(BaseModel):
    """The LLM's direct output. Composed into WriterOutput with run metadata."""

    model_config = ConfigDict(extra="forbid")

    cv: CVDraft
    cover_letter: CoverLetter


class WriterOutput(LayerOutput):
    """Writer's layer output. Read by critic gate, factchecker, and submitter."""

    cv: CVDraft
    cover_letter: CoverLetter

    def grounded_claims(self) -> list[GroundedClaim]:
        """Every GroundedClaim in the output, flattened. For auditing."""
        claims: list[GroundedClaim] = [self.cv.summary]
        for entry_list in (self.cv.experience, self.cv.projects, self.cv.education):
            for entry in entry_list:
                claims.extend(entry.bullets)
        claims.extend(self.cover_letter.paragraphs)
        return claims

    def all_fact_ids(self) -> set[str]:
        """Every fact_id referenced by any claim, deduplicated."""
        ids: set[str] = set()
        for claim in self.grounded_claims():
            ids.update(claim.fact_ids)
        for entry_list in (self.cv.experience, self.cv.projects, self.cv.education):
            for entry in entry_list:
                ids.add(entry.primary_fact_id)
        return ids


class StructuredLLM(Protocol):
    """Same minimal shape used by other agents."""

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel: ...


class WriterValidationError(RuntimeError):
    """Raised when the LLM returns claims grounded on nonexistent fact_ids.

    This is a structural bug in the LLM's output (the schema was violated
    in spirit even if not in form). Distinct from grounding-quality issues
    which are the factchecker's domain.
    """


class WriterAgent:
    """Drafts a CV + cover letter grounded in the candidate's facts."""

    name = "writer"

    def __init__(
        self,
        candidate: Candidate,
        llm: StructuredLLM,
        *,
        model: str = "gpt-4.1",
    ) -> None:
        self.candidate = candidate
        self.llm = llm
        self.model = model
        self._known_ids = {f.id for f in candidate.facts}

    def run(self, ctx: StackContext) -> WriterOutput:
        role_analysis = ctx.output_of("recruiter")
        if not isinstance(role_analysis, RoleAnalysis):
            raise TypeError(
                f"writer expects RoleAnalysis upstream, got {type(role_analysis).__name__}"
            )

        rebase = ctx.pending_rebase(self.name)
        user_prompt = self._build_user_prompt(role_analysis, rebase)

        payload = self.llm.parse(
            model=self.model,
            system=WRITER_SYSTEM_PROMPT,
            user=user_prompt,
            schema=WriterPayload,
        )
        assert isinstance(payload, WriterPayload)

        output = WriterOutput(
            layer_name=self.name,
            cv=payload.cv,
            cover_letter=payload.cover_letter,
        )
        self._validate_fact_ids_resolve(output)
        return output

    def _validate_fact_ids_resolve(self, output: WriterOutput) -> None:
        referenced = output.all_fact_ids()
        unknown = referenced - self._known_ids
        if unknown:
            raise WriterValidationError(
                f"writer cited fact_ids that are not in candidate.facts: {sorted(unknown)}"
            )

    def _build_user_prompt(
        self,
        role_analysis: RoleAnalysis,
        rebase: object,
    ) -> str:
        # Pretty-print the inputs so the LLM gets stable, readable structure.
        ra_dump = role_analysis.model_dump(mode="json", exclude={"raw_jd_excerpt"})
        cand_dump = self.candidate.model_dump(mode="json")
        parts: list[str] = [
            "ROLE_ANALYSIS:",
            _json(ra_dump),
            "",
            "CANDIDATE_FACTS:",
            _json(cand_dump),
        ]
        if rebase is not None:
            # rebase is a RebaseRequest with as_prompt_fragment()
            parts.extend(["", rebase.as_prompt_fragment()])  # type: ignore[attr-defined]
        return "\n".join(parts)


def _json(obj: object) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)
