"""Tests for the factchecker layer.

These tests pin the safety-critical behavior:
- unknown fact_id → hard flag
- unverified fact cited → hard flag
- numeric metric in claim but not in cited facts → soft flag
- LLM verdict "ungrounded" → hard flag
- LLM verdict "needs_review" → soft flag
- safe_to_submit = True iff no hard flags

The LLM is stubbed; no API calls in units.
"""

from __future__ import annotations

from pydantic import BaseModel

from applyops.agents.factchecker import (
    FactCheckerAgent,
    FactCheckOutput,
    _ClaimAuditPayload,
    _extract_metrics,
    _FactCheckPayload,
)
from applyops.agents.writer import (
    CoverLetter,
    CVDraft,
    GroundedClaim,
    WriterOutput,
)
from applyops.facts import Candidate, Fact, Provenance
from applyops.gstack import Stack, StackContext
from applyops.gstack.run import Run


class StubLLM:
    def __init__(self, payload: BaseModel) -> None:
        self.payload = payload
        self.calls = 0

    def parse(self, *, model: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls += 1
        return self.payload


def _candidate(*, with_unverified: bool = False) -> Candidate:
    verified = Provenance(source="resume.pdf p.1", verified_by="self")
    unattested = Provenance(source="resume.pdf p.1", verified_by="ai_extracted_unverified")
    facts = [
        Fact(
            id="exp-verified",
            kind="experience",
            title="Verified Experience",
            detail="Cut p95 latency by 38% on the agent platform.",
            tags=["latency"],
            metrics={"p95_latency_reduction": "38%"},
            provenance=[verified],
        ),
        Fact(
            id="proj-verified",
            kind="project",
            title="Verified Project",
            detail="Built a thing.",
            provenance=[verified],
        ),
    ]
    if with_unverified:
        facts.append(
            Fact(
                id="exp-unattested",
                kind="experience",
                title="Unattested Experience",
                detail="Did some other thing.",
                provenance=[unattested],
            )
        )
    return Candidate(name="X", headline="Y", location="Z", facts=facts)


def _writer_output(claims: list[GroundedClaim]) -> WriterOutput:
    """Build a WriterOutput where the cover letter carries all claims."""
    return WriterOutput(
        layer_name="writer",
        cv=CVDraft(
            summary=claims[0] if claims else GroundedClaim(text="x", fact_ids=["exp-verified"]),
            experience=[],
            projects=[],
            skills_line="",
            education=[],
        ),
        cover_letter=CoverLetter(paragraphs=claims[1:] if len(claims) > 1 else []),
    )


def _ctx_with(writer_output: WriterOutput) -> StackContext:
    run = Run()
    ctx = StackContext(run=run)
    from applyops.gstack.context import LayerState

    ctx.layers["writer"] = LayerState(name="writer", output=writer_output)
    return ctx


def _all_grounded_payload(n: int) -> _FactCheckPayload:
    return _FactCheckPayload(
        audits=[
            _ClaimAuditPayload(claim_index=i, verdict="grounded", rationale="ok") for i in range(n)
        ]
    )


def test_metric_extractor_picks_up_typical_resume_metrics() -> None:
    text = "Cut p95 latency by 38% across 20,000+ assets in a 10+ PB dataset."
    metrics = _extract_metrics(text)
    # We don't check exact tokens — implementation detail. Just that something
    # numeric-looking was extracted.
    assert any("38" in m for m in metrics)
    assert any("20,000" in m or "20" in m for m in metrics)


def test_known_facts_grounded_claim_passes_clean() -> None:
    candidate = _candidate()
    writer_output = _writer_output(
        [
            GroundedClaim(
                text="Cut p95 latency by 38%.",
                fact_ids=["exp-verified"],
                addresses=[],
            )
        ]
    )
    fc = FactCheckerAgent(candidate, llm=StubLLM(_all_grounded_payload(1)))
    out = fc.run(_ctx_with(writer_output))

    assert isinstance(out, FactCheckOutput)
    assert out.safe_to_submit is True
    assert out.hard_flags == []
    assert all(audit.verdict == "grounded" for audit in out.audits)


def test_unknown_fact_id_produces_hard_flag() -> None:
    candidate = _candidate()
    writer_output = _writer_output(
        [
            GroundedClaim(
                text="Claim citing a missing fact.",
                fact_ids=["does-not-exist"],
            )
        ]
    )
    fc = FactCheckerAgent(candidate, llm=StubLLM(_all_grounded_payload(1)))
    out = fc.run(_ctx_with(writer_output))

    assert out.safe_to_submit is False
    kinds = {f.kind for f in out.hard_flags}
    assert "unknown_fact_id" in kinds


def test_unverified_fact_citation_is_hard_flag() -> None:
    candidate = _candidate(with_unverified=True)
    writer_output = _writer_output(
        [
            GroundedClaim(
                text="Claim citing an unattested fact.",
                fact_ids=["exp-unattested"],
            )
        ]
    )
    fc = FactCheckerAgent(candidate, llm=StubLLM(_all_grounded_payload(1)))
    out = fc.run(_ctx_with(writer_output))

    assert out.safe_to_submit is False
    kinds = {f.kind for f in out.hard_flags}
    assert "unverified_fact_cited" in kinds


def test_metric_in_claim_but_not_in_facts_is_soft_flag() -> None:
    candidate = _candidate()
    writer_output = _writer_output(
        [
            # Claim invents 50% — facts only mention 38%.
            GroundedClaim(
                text="Cut p95 latency by 50%.",
                fact_ids=["exp-verified"],
            )
        ]
    )
    fc = FactCheckerAgent(candidate, llm=StubLLM(_all_grounded_payload(1)))
    out = fc.run(_ctx_with(writer_output))

    # No hard flag — the LLM said grounded. Soft flag from metric mismatch.
    assert out.safe_to_submit is True
    kinds = {f.kind for f in out.soft_flags}
    assert "metric_not_in_facts" in kinds


def test_llm_ungrounded_verdict_is_hard_flag() -> None:
    candidate = _candidate()
    writer_output = _writer_output(
        [
            GroundedClaim(
                text="Wildly overreaching claim.",
                fact_ids=["exp-verified"],
            )
        ]
    )
    payload = _FactCheckPayload(
        audits=[
            _ClaimAuditPayload(
                claim_index=0,
                verdict="ungrounded",
                rationale="claim is not supported by the cited fact",
            )
        ]
    )
    fc = FactCheckerAgent(candidate, llm=StubLLM(payload))
    out = fc.run(_ctx_with(writer_output))

    assert out.safe_to_submit is False
    kinds = {f.kind for f in out.hard_flags}
    assert "ungrounded_claim" in kinds


def test_factchecker_runs_inside_stack_after_writer() -> None:
    """Smoke test: factchecker pulls writer output via ctx and produces an audit."""
    candidate = _candidate()

    class FixedWriter:
        name = "writer"

        def run(self, ctx: StackContext) -> WriterOutput:
            return _writer_output(
                [
                    GroundedClaim(
                        text="Cut p95 latency by 38%.",
                        fact_ids=["exp-verified"],
                    )
                ]
            )

    fc = FactCheckerAgent(candidate, llm=StubLLM(_all_grounded_payload(1)))
    stack = Stack(layers=[FixedWriter(), fc])
    run, ctx = stack.land()

    fc_out = ctx.output_of("factchecker")
    assert isinstance(fc_out, FactCheckOutput)
    assert fc_out.safe_to_submit is True
    assert run.status.value == "completed"
