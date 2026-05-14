"""End-to-end test: recruiter → writer → critic rebase loop.

Uses stubbed LLMs that return scripted payloads. The point of this test is
NOT to exercise prompt quality — that's the eval harness's job. The point
is to prove that the gstack rebase loop actually works when wired to real
agent implementations:
- the writer reads the recruiter's RoleAnalysis from ctx
- the critic computes deterministic rubric findings on the writer's output
- when the critic fails, the writer's next run sees a RebaseRequest in ctx
- after the writer addresses the rebase, the critic passes and the stack
  lands

If this test breaks, either the gstack contract or the agent contracts
have regressed.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from applyops.agents import (
    CoverLetter,
    CriticGate,
    CVDraft,
    CVEntry,
    GroundedClaim,
    RecruiterAgent,
    RoleAnalysis,
    WriterAgent,
    WriterOutput,
)
from applyops.agents.critic import CriticPayload
from applyops.agents.jd_source import FileJDSource
from applyops.agents.recruiter import RoleAnalysisPayload
from applyops.agents.types import Requirement
from applyops.agents.writer import WriterPayload
from applyops.facts import Candidate, Fact, Provenance
from applyops.gstack import Stack
from applyops.gstack.run import RunStatus

FIXTURE_JD = Path(__file__).parent / "fixtures" / "jd.fake.md"


class ScriptedLLM:
    """Returns scripted payloads in the order the schema types are requested."""

    def __init__(self, script: dict[type[BaseModel], list[BaseModel]]) -> None:
        # script[ModelType] = [payload1, payload2, ...] — consumed in order.
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[type[BaseModel]] = []

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        self.calls.append(schema)
        queue = self.script.get(schema)
        if not queue:
            raise AssertionError(
                f"no scripted payload remaining for schema {schema.__name__}"
            )
        return queue.pop(0)


def _candidate() -> Candidate:
    return Candidate(
        name="Test Candidate",
        headline="Engineer.",
        location="Toronto, ON",
        facts=[
            Fact(
                id="exp-quickplay-2025",
                kind="experience",
                title="Software Engineer at Quickplay",
                detail="Shipped CV+LLM pipeline.",
                tags=["python", "cv", "llm"],
                provenance=[Provenance(source="resume.pdf p.1", verified_by="self")],
            ),
            Fact(
                id="skill-python",
                kind="skill",
                title="Python",
                detail="Production Python at Quickplay.",
                tags=["python"],
                provenance=[Provenance(source="resume.pdf p.1", verified_by="self")],
            ),
        ],
    )


def _recruiter_payload() -> RoleAnalysisPayload:
    return RoleAnalysisPayload(
        role_title="AI Ops Engineer",
        company="Fakeco",
        location="Toronto",
        requirements=[
            Requirement(
                text="Strong Python",
                kind="must_have",
                importance=5,
                category="technical",
                evidence_anchor="production Python experience",
            ),
            Requirement(
                text="Production AI experience",
                kind="must_have",
                importance=5,
                category="experience",
                evidence_anchor="shipped AI/ML systems",
            ),
        ],
        company_signals=["values eval discipline"],
        application_protocol_notes=["apply using AI"],
    )


def _writer_payload_v1() -> WriterPayload:
    """First writer draft — covers Python but skips the AI requirement."""
    return WriterPayload(
        cv=CVDraft(
            summary=GroundedClaim(
                text="Engineer shipping production AI.",
                fact_ids=["exp-quickplay-2025"],
                addresses=["Strong Python"],
            ),
            experience=[
                CVEntry(
                    heading="Software Engineer at Quickplay",
                    date_range="2025 – present",
                    primary_fact_id="exp-quickplay-2025",
                    bullets=[
                        GroundedClaim(
                            text="Shipped a CV+LLM pipeline.",
                            fact_ids=["exp-quickplay-2025"],
                            addresses=["Strong Python"],
                        )
                    ],
                )
            ],
            projects=[],
            skills_line="Python",
            education=[],
        ),
        cover_letter=CoverLetter(
            paragraphs=[
                GroundedClaim(
                    text="I want this role.",
                    fact_ids=["exp-quickplay-2025"],
                    addresses=[],
                )
            ]
        ),
    )


def _writer_payload_v2_addresses_rebase() -> WriterPayload:
    """Second draft — addresses BOTH requirements and the AI-application protocol."""
    return WriterPayload(
        cv=CVDraft(
            summary=GroundedClaim(
                text="Engineer shipping production AI/ML pipelines.",
                fact_ids=["exp-quickplay-2025"],
                addresses=["Strong Python", "Production AI experience"],
            ),
            experience=[
                CVEntry(
                    heading="Software Engineer at Quickplay",
                    date_range="2025 – present",
                    primary_fact_id="exp-quickplay-2025",
                    bullets=[
                        GroundedClaim(
                            text="Shipped a CV + LLM pipeline in production.",
                            fact_ids=["exp-quickplay-2025"],
                            addresses=["Production AI experience"],
                        ),
                        GroundedClaim(
                            text="Production Python services.",
                            fact_ids=["skill-python", "exp-quickplay-2025"],
                            addresses=["Strong Python"],
                        ),
                    ],
                )
            ],
            projects=[],
            skills_line="Python",
            education=[],
        ),
        cover_letter=CoverLetter(
            paragraphs=[
                GroundedClaim(
                    text="This application was built with an AI pipeline I wrote for it.",
                    fact_ids=["exp-quickplay-2025"],
                    addresses=[],
                )
            ]
        ),
    )


def _critic_request_changes() -> CriticPayload:
    return CriticPayload(
        tone_findings=[],
        protocol_response="missing",
        verdict="request_changes",
        rationale="Coverage of the AI experience requirement is missing; protocol note unaddressed.",
        findings=["address requirement 'Production AI experience'"],
        suggested_changes=["mention the CV+LLM pipeline in cover letter"],
    )


def _critic_pass() -> CriticPayload:
    return CriticPayload(
        tone_findings=[],
        protocol_response="addressed",
        verdict="pass",
        rationale="All high-importance requirements covered; protocol addressed.",
        findings=[],
        suggested_changes=[],
    )


def test_end_to_end_rebase_loop_lands_after_writer_addresses_critic() -> None:
    candidate = _candidate()
    llm = ScriptedLLM(
        {
            RoleAnalysisPayload: [_recruiter_payload()],
            WriterPayload: [_writer_payload_v1(), _writer_payload_v2_addresses_rebase()],
            CriticPayload: [_critic_request_changes(), _critic_pass()],
        }
    )

    stack = Stack(
        layers=[
            RecruiterAgent(FileJDSource(FIXTURE_JD), llm=llm),
            WriterAgent(candidate, llm=llm),
        ],
        gates={"writer": CriticGate(llm=llm)},
    )

    run, ctx = stack.land()

    assert run.status == RunStatus.COMPLETED
    # Writer ran twice (initial + 1 rebase), critic ran twice (fail + pass)
    assert ctx.layers["writer"].rebases == 1
    assert len(ctx.layers["writer"].gate_reviews) == 2
    assert ctx.layers["writer"].gate_reviews[0].passed is False
    assert ctx.layers["writer"].gate_reviews[1].passed is True

    writer_output = ctx.output_of("writer")
    assert isinstance(writer_output, WriterOutput)
    # The second draft addresses both high-importance requirements.
    addressed = {a for c in writer_output.grounded_claims() for a in c.addresses}
    assert "Strong Python" in addressed
    assert "Production AI experience" in addressed


def test_writer_rejects_claims_with_unknown_fact_ids() -> None:
    candidate = _candidate()

    bad_payload = WriterPayload(
        cv=CVDraft(
            summary=GroundedClaim(
                text="Engineer.",
                fact_ids=["does-not-exist"],
                addresses=[],
            ),
            skills_line="",
        ),
        cover_letter=CoverLetter(paragraphs=[]),
    )

    llm = ScriptedLLM(
        {
            RoleAnalysisPayload: [_recruiter_payload()],
            WriterPayload: [bad_payload],
        }
    )

    stack = Stack(
        layers=[
            RecruiterAgent(FileJDSource(FIXTURE_JD), llm=llm),
            WriterAgent(candidate, llm=llm),
        ]
    )

    # Writer's validation raises before the orchestrator can complete.
    import pytest

    from applyops.agents.writer import WriterValidationError

    with pytest.raises(WriterValidationError, match="does-not-exist"):
        stack.land()


def test_deterministic_rubric_blocks_when_coverage_is_low() -> None:
    """Even if the LLM critic says 'pass', deterministic findings can fail."""
    candidate = _candidate()
    llm = ScriptedLLM(
        {
            RoleAnalysisPayload: [_recruiter_payload()],
            WriterPayload: [_writer_payload_v1(), _writer_payload_v1()],
            # LLM critic optimistically passes even though coverage is bad
            CriticPayload: [_critic_pass(), _critic_pass()],
        }
    )
    stack = Stack(
        layers=[
            RecruiterAgent(FileJDSource(FIXTURE_JD), llm=llm),
            WriterAgent(candidate, llm=llm),
        ],
        gates={"writer": CriticGate(llm=llm)},
        max_rebases_per_gate=1,
    )

    run, ctx = stack.land()

    # Deterministic findings (missing AI-experience coverage) force a fail
    # even though the LLM critic said pass.
    assert run.status == RunStatus.BLOCKED
    review = ctx.layers["writer"].gate_reviews[0]
    assert review.passed is False
    assert any("high-importance requirement coverage" in f for f in review.rebase_request.findings)  # type: ignore[union-attr]
