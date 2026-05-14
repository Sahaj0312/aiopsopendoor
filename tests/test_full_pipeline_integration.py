"""Full stack: recruiter → writer → critic → factchecker → submitter.

Scripted LLM payloads; no network or API calls. This is the proof that
all five layers + the critic gate wire up correctly with the gstack
orchestrator. If any contract regresses, this test catches it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from applyops.agents import (
    CoverLetter,
    CriticGate,
    CVDraft,
    CVEntry,
    FactCheckerAgent,
    GroundedClaim,
    RecruiterAgent,
    Requirement,
    SubmitterAgent,
    WriterAgent,
)
from applyops.agents.critic import CriticPayload
from applyops.agents.factchecker import _ClaimAuditPayload, _FactCheckPayload
from applyops.agents.jd_source import FileJDSource
from applyops.agents.recruiter import RoleAnalysisPayload
from applyops.agents.writer import WriterPayload
from applyops.facts import Candidate, Fact, Provenance
from applyops.gstack import Stack
from applyops.gstack.run import RunStatus

FIXTURE_JD = Path(__file__).parent / "fixtures" / "jd.fake.md"


class ScriptedLLM:
    def __init__(self, script: dict[type[BaseModel], list[BaseModel]]) -> None:
        self.script = {k: list(v) for k, v in script.items()}

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        queue = self.script.get(schema)
        if not queue:
            raise AssertionError(f"no scripted payload remaining for schema {schema.__name__}")
        return queue.pop(0)


def _candidate() -> Candidate:
    p = Provenance(source="resume.pdf p.1", verified_by="self")
    return Candidate(
        name="Sahaj Test",
        headline="Engineer with production AI experience.",
        location="Toronto, ON",
        links={"github": "https://github.com/Sahaj0312"},
        facts=[
            Fact(
                id="exp-quickplay",
                kind="experience",
                title="Software Engineer at Quickplay",
                detail="Shipped CV+LLM thumbnail pipeline at production scale.",
                tags=["python", "cv", "llm"],
                provenance=[p],
            ),
            Fact(
                id="skill-python",
                kind="skill",
                title="Python",
                detail="Production Python across multiple roles.",
                tags=["python"],
                provenance=[p],
            ),
        ],
    )


def _scripted_run_payloads() -> dict[type[BaseModel], list[BaseModel]]:
    role = RoleAnalysisPayload(
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

    writer = WriterPayload(
        cv=CVDraft(
            summary=GroundedClaim(
                text="Engineer shipping production AI on video and image pipelines.",
                fact_ids=["exp-quickplay"],
                addresses=["Strong Python", "Production AI experience"],
            ),
            experience=[
                CVEntry(
                    heading="Software Engineer, Quickplay — Toronto",
                    date_range="2025 – present",
                    primary_fact_id="exp-quickplay",
                    bullets=[
                        GroundedClaim(
                            text="Shipped a production CV + LLM thumbnail pipeline.",
                            fact_ids=["exp-quickplay"],
                            addresses=["Production AI experience"],
                        ),
                        GroundedClaim(
                            text="Production Python service work.",
                            fact_ids=["skill-python", "exp-quickplay"],
                            addresses=["Strong Python"],
                        ),
                    ],
                )
            ],
            projects=[],
            skills_line="Python, Go",
            education=[],
        ),
        cover_letter=CoverLetter(
            paragraphs=[
                GroundedClaim(
                    text="This application package was built using a small AI Ops system I wrote for it; here's how I did it.",
                    fact_ids=["exp-quickplay"],
                    addresses=[],
                )
            ]
        ),
    )

    critic = CriticPayload(
        tone_findings=[],
        protocol_response="addressed",
        verdict="pass",
        rationale="All high-importance requirements covered.",
        findings=[],
        suggested_changes=[],
    )

    # 4 grounded claims total (1 summary + 2 bullets + 1 cover paragraph)
    factcheck = _FactCheckPayload(
        audits=[
            _ClaimAuditPayload(claim_index=i, verdict="grounded", rationale="ok") for i in range(4)
        ]
    )

    return {
        RoleAnalysisPayload: [role],
        WriterPayload: [writer],
        CriticPayload: [critic],
        _FactCheckPayload: [factcheck],
    }


def test_full_pipeline_runs_and_writes_artifacts(tmp_path: Path) -> None:
    candidate = _candidate()
    llm = ScriptedLLM(_scripted_run_payloads())

    recruiter = RecruiterAgent(FileJDSource(FIXTURE_JD), llm=llm)
    writer = WriterAgent(candidate, llm=llm)
    critic = CriticGate(llm=llm)
    factchecker = FactCheckerAgent(candidate, llm=llm)
    submitter = SubmitterAgent(
        candidate_name=candidate.name,
        candidate_email="sahaj@example.com",
        candidate_phone="555-0100",
        candidate_links={k: str(v) for k, v in candidate.links.items()},
        output_root=tmp_path,
    )

    stack = Stack(
        layers=[recruiter, writer, factchecker, submitter],
        gates={"writer": critic},
    )

    run, ctx = stack.land()

    assert run.status == RunStatus.COMPLETED
    # Every layer produced output
    for name in ("recruiter", "writer", "factchecker", "submitter"):
        assert ctx.layers[name].output is not None, f"{name} did not produce output"
    # Critic passed on first try
    assert ctx.layers["writer"].rebases == 0
    assert ctx.layers["writer"].gate_reviews[-1].passed is True

    # Submitter wrote four artifacts
    sub = ctx.layers["submitter"].output
    out_dir = Path(sub.output_dir)  # type: ignore[union-attr]
    assert (out_dir / "cv.md").exists()
    assert (out_dir / "cover.md").exists()
    assert (out_dir / "form_plan.json").exists()
    assert (out_dir / "audit.md").exists()

    plan = json.loads((out_dir / "form_plan.json").read_text())
    field_names = {f["name"] for f in plan["fields"]}
    assert "full_name" in field_names
    assert "cover_letter" in field_names

    cv_md = (out_dir / "cv.md").read_text()
    assert "Sahaj Test" in cv_md
    assert "Quickplay" in cv_md


def test_runner_build_jd_source_picks_right_implementation(tmp_path: Path) -> None:
    """The runner's tiny picker function — keep it honest."""
    from applyops.agents.jd_source import FileJDSource, HttpJDSource
    from applyops.runner import build_jd_source

    jd_file = tmp_path / "jd.md"
    jd_file.write_text("# fake jd\n")
    cfg_file = _stub_cfg(jd_file=jd_file)
    cfg_url = _stub_cfg(jd_url="https://example.test/jd")

    assert isinstance(build_jd_source(cfg_file), FileJDSource)
    assert isinstance(build_jd_source(cfg_url), HttpJDSource)


def _stub_cfg(*, jd_url: str | None = None, jd_file: Path | None = None) -> object:
    """Build a minimal RunConfig — only fields used by build_jd_source matter here."""
    from applyops.runner import RunConfig

    return RunConfig(
        jd_url=jd_url,
        jd_file=jd_file,
        facts_path=Path("inputs/facts.example.json"),
        output_root=Path("outputs"),
        snapshot_dir=Path("inputs"),
        candidate_email="x@y.com",
        candidate_phone=None,
        target_url_override=None,
        recruiter_model="m",
        writer_model="m",
        critic_model="m",
        factcheck_model="m",
        max_rebases_per_gate=1,
    )
