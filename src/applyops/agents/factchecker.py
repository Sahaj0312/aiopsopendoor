"""FactChecker — audits every grounded claim against the candidate's facts.

The factchecker is a Layer (not a Gate) because its output is structured
data that downstream layers consume: the submitter uses the audits to
attach citation footnotes; a future render step might use them to
generate a "claim → evidence" appendix.

Its responsibilities, in order of severity:

1. **Hard:** every cited fact_id must exist in candidate.facts. The writer
   already validates this, but defense-in-depth catches regressions.
2. **Hard:** no claim may cite an unverified fact (`verified_by ==
   "ai_extracted_unverified"`). The writer can ground on whatever the
   facts.json contains; the factchecker enforces the attestation floor.
3. **Soft:** every numeric metric appearing in a claim's text must also
   appear in one of the cited facts (`metrics` dict or `detail`). This
   catches the most common hallucination: a believable-looking number
   the LLM rounded or invented.
4. **LLM judgment:** for each claim, does the cited evidence actually
   support the claim, or is it an over-extension?

The factchecker is conservative: when in doubt it flags rather than
clears. The submitter refuses to render if any hard flag is present;
soft flags surface in the audit but don't block.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from applyops.agents.writer import GroundedClaim, WriterOutput
from applyops.facts import Candidate
from applyops.gstack.context import StackContext
from applyops.gstack.types import LayerOutput

FactCheckerSystemPrompt = """You are a strict factchecker for an application package.

You receive a list of `claims`. Each claim has:
- `text`: the assertion as it appears in the application
- `cited_facts`: the candidate's facts that the writer cited as evidence (id, title, detail, metrics, verification level)

For each claim, decide:
- `verdict`: one of
  - "grounded" — the cited facts clearly support the claim as written
  - "needs_review" — the cited facts partially support the claim, or the claim is more confident than the evidence warrants (mild over-extension, missing nuance)
  - "ungrounded" — the cited facts do not support the claim; the claim should not be made
- `rationale`: 1-2 sentence justification

Be strict. The cost of a missed hallucination is a damaged reputation; the cost of a false flag is one rebase cycle.

Specifically watch for:
- Inflated scope ("led X" when evidence says "contributed to X")
- Made-up impact ("reduced cost by $1M" when no number is in the facts)
- Wrong directionality (claim says "increased" when fact says "decreased")
- Fabricated team/scale figures
- Claims about clients/partners/customers not in the facts

Return one audit per claim in the same order as the input."""


FlagSeverity = Literal["hard", "soft"]
FlagKind = Literal[
    "unknown_fact_id",
    "unverified_fact_cited",
    "metric_not_in_facts",
    "ungrounded_claim",
    "needs_review",
]
ClaimVerdict = Literal["grounded", "needs_review", "ungrounded"]


class Flag(BaseModel):
    """One issue detected on a specific claim."""

    model_config = ConfigDict(extra="forbid")

    severity: FlagSeverity
    kind: FlagKind
    claim_text: str
    cited_fact_ids: list[str]
    explanation: str
    suggested_fix: str = ""


class ClaimAudit(BaseModel):
    """The audit verdict and rationale for one claim."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str
    cited_fact_ids: list[str]
    addresses: list[str]
    verdict: ClaimVerdict
    rationale: str
    flags: list[Flag] = Field(default_factory=list)


class FactCheckOutput(LayerOutput):
    """The factchecker's structured audit of the writer's output."""

    audits: list[ClaimAudit]
    hard_flags: list[Flag]
    soft_flags: list[Flag]

    @property
    def safe_to_submit(self) -> bool:
        """True iff no hard flags. The submitter checks this before rendering."""
        return not self.hard_flags


# LLM-side schema (deterministic checks are computed outside the LLM call).
class _ClaimAuditPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_index: int
    verdict: ClaimVerdict
    rationale: str


class _FactCheckPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audits: list[_ClaimAuditPayload]


class StructuredLLM(Protocol):
    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel: ...


_METRIC_RE = re.compile(r"(\d[\d,]*\.?\d*\s*(?:%|x|×|k|m|b|pb|tb|gb)?\+?)", re.IGNORECASE)


def _extract_metrics(text: str) -> set[str]:
    """Pull out tokens that look like numeric metrics from a claim's text."""
    return {m.group(1).strip().lower() for m in _METRIC_RE.finditer(text)}


def _metric_supported(metric: str, fact_metrics: dict[str, str], fact_detail: str) -> bool:
    """Cheap support check: the metric token appears in any fact metric value or detail."""
    metric = metric.lower()
    haystacks = [v.lower() for v in fact_metrics.values()] + [fact_detail.lower()]
    return any(metric in h for h in haystacks)


class FactCheckerAgent:
    """Audits the writer's claims for grounding."""

    name = "factchecker"

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

    def run(self, ctx: StackContext) -> FactCheckOutput:
        writer_output = ctx.output_of("writer")
        if not isinstance(writer_output, WriterOutput):
            raise TypeError(
                f"factchecker expects WriterOutput upstream, got {type(writer_output).__name__}"
            )

        claims = writer_output.grounded_claims()
        deterministic_flags: list[Flag] = []
        for claim in claims:
            deterministic_flags.extend(self._deterministic_flags(claim))

        llm_payload = self._llm_audit(claims)
        audits = self._compose_audits(claims, llm_payload, deterministic_flags)

        all_flags: list[Flag] = []
        for audit in audits:
            all_flags.extend(audit.flags)

        return FactCheckOutput(
            layer_name=self.name,
            audits=audits,
            hard_flags=[f for f in all_flags if f.severity == "hard"],
            soft_flags=[f for f in all_flags if f.severity == "soft"],
        )

    def _deterministic_flags(self, claim: GroundedClaim) -> list[Flag]:
        flags: list[Flag] = []
        cited_facts = []
        for fid in claim.fact_ids:
            try:
                fact = self.candidate.get(fid)
            except KeyError:
                flags.append(
                    Flag(
                        severity="hard",
                        kind="unknown_fact_id",
                        claim_text=claim.text,
                        cited_fact_ids=[fid],
                        explanation=f"fact_id {fid!r} is not in candidate.facts",
                        suggested_fix="cite a real fact_id or remove the claim",
                    )
                )
                continue
            cited_facts.append(fact)
            if not fact.verified:
                flags.append(
                    Flag(
                        severity="hard",
                        kind="unverified_fact_cited",
                        claim_text=claim.text,
                        cited_fact_ids=[fid],
                        explanation=(
                            f"fact_id {fid!r} is not yet attested (verified_by != self/third_party)"
                        ),
                        suggested_fix="attest the fact in facts.json or replace the citation",
                    )
                )

        # Metric check: every numeric token in the claim must trace to some cited fact.
        claim_metrics = _extract_metrics(claim.text)
        for metric in claim_metrics:
            if not any(_metric_supported(metric, f.metrics, f.detail) for f in cited_facts):
                flags.append(
                    Flag(
                        severity="soft",
                        kind="metric_not_in_facts",
                        claim_text=claim.text,
                        cited_fact_ids=list(claim.fact_ids),
                        explanation=(
                            f"metric {metric!r} appears in the claim but not in any "
                            f"cited fact's metrics or detail"
                        ),
                        suggested_fix=(
                            "add the metric to the source fact (with provenance) "
                            "or remove it from the claim"
                        ),
                    )
                )

        return flags

    def _llm_audit(self, claims: list[GroundedClaim]) -> _FactCheckPayload:
        if not claims:
            return _FactCheckPayload(audits=[])

        import json

        rows = []
        for idx, claim in enumerate(claims):
            cited = []
            for fid in claim.fact_ids:
                try:
                    f = self.candidate.get(fid)
                except KeyError:
                    cited.append({"id": fid, "missing": True})
                    continue
                cited.append(
                    {
                        "id": f.id,
                        "title": f.title,
                        "detail": f.detail,
                        "metrics": f.metrics,
                        "verified_by": f.provenance[0].verified_by,
                    }
                )
            rows.append({"claim_index": idx, "text": claim.text, "cited_facts": cited})

        user = json.dumps({"claims": rows}, indent=2)
        payload = self.llm.parse(
            model=self.model,
            system=FactCheckerSystemPrompt,
            user=user,
            schema=_FactCheckPayload,
        )
        assert isinstance(payload, _FactCheckPayload)
        return payload

    def _compose_audits(
        self,
        claims: list[GroundedClaim],
        llm_payload: _FactCheckPayload,
        deterministic_flags: list[Flag],
    ) -> list[ClaimAudit]:
        # Group deterministic flags by claim text for lookup.
        flags_by_claim: dict[str, list[Flag]] = {}
        for flag in deterministic_flags:
            flags_by_claim.setdefault(flag.claim_text, []).append(flag)

        # Map LLM verdicts by claim_index.
        verdict_by_index: dict[int, _ClaimAuditPayload] = {
            a.claim_index: a for a in llm_payload.audits
        }

        audits: list[ClaimAudit] = []
        for idx, claim in enumerate(claims):
            llm_audit = verdict_by_index.get(idx)
            verdict: ClaimVerdict = llm_audit.verdict if llm_audit else "needs_review"
            rationale = llm_audit.rationale if llm_audit else "no LLM audit returned for this claim"
            flags = list(flags_by_claim.get(claim.text, []))

            # The LLM's verdict promotes to a flag if not "grounded".
            if verdict == "ungrounded":
                flags.append(
                    Flag(
                        severity="hard",
                        kind="ungrounded_claim",
                        claim_text=claim.text,
                        cited_fact_ids=list(claim.fact_ids),
                        explanation=rationale,
                        suggested_fix="rewrite the claim to match the evidence or drop it",
                    )
                )
            elif verdict == "needs_review":
                flags.append(
                    Flag(
                        severity="soft",
                        kind="needs_review",
                        claim_text=claim.text,
                        cited_fact_ids=list(claim.fact_ids),
                        explanation=rationale,
                    )
                )

            audits.append(
                ClaimAudit(
                    claim_text=claim.text,
                    cited_fact_ids=list(claim.fact_ids),
                    addresses=list(claim.addresses),
                    verdict=verdict,
                    rationale=rationale,
                    flags=flags,
                )
            )

        return audits
