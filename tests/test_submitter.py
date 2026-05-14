"""Tests for the submitter.

Pins the safety contract (hard flag → SubmitterBlocked) and the artifact
contract (cv.md, cover.md, audit.md, form_plan.json all written to
outputs/<run_id>/).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from applyops.agents.factchecker import FactCheckOutput, Flag
from applyops.agents.recruiter import RoleAnalysis
from applyops.agents.submitter import (
    SubmitterAgent,
    SubmitterBlocked,
    SubmitterOutput,
)
from applyops.agents.types import JDMeta
from applyops.agents.writer import (
    CoverLetter,
    CVDraft,
    CVEntry,
    GroundedClaim,
    WriterOutput,
)
from applyops.gstack.context import LayerState, StackContext
from applyops.gstack.run import Run


def _ctx_with(
    writer_output: WriterOutput,
    factcheck_output: FactCheckOutput,
    role_analysis: RoleAnalysis,
) -> StackContext:
    run = Run()
    ctx = StackContext(run=run)
    ctx.layers["recruiter"] = LayerState(name="recruiter", output=role_analysis)
    ctx.layers["writer"] = LayerState(name="writer", output=writer_output)
    ctx.layers["factchecker"] = LayerState(name="factchecker", output=factcheck_output)
    return ctx


def _writer_output() -> WriterOutput:
    return WriterOutput(
        layer_name="writer",
        cv=CVDraft(
            summary=GroundedClaim(
                text="Engineer with production AI experience.",
                fact_ids=["exp-x"],
            ),
            experience=[
                CVEntry(
                    heading="Software Engineer at Testco",
                    date_range="2025 – present",
                    primary_fact_id="exp-x",
                    bullets=[
                        GroundedClaim(
                            text="Shipped a CV+LLM pipeline.",
                            fact_ids=["exp-x"],
                        )
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
                    text="I built this application with the same kind of system I'd ship at work.",
                    fact_ids=["exp-x"],
                )
            ]
        ),
    )


def _role_analysis() -> RoleAnalysis:
    return RoleAnalysis(
        layer_name="recruiter",
        role_title="AI Ops Engineer",
        company="Fakeco",
        location="Toronto",
        jd_meta=JDMeta(
            url="https://example.test/jd",
            hash="abc",
            snapshot_path="/tmp/jd.md",
            drift=False,
        ),
        requirements=[],
        company_signals=[],
        application_protocol_notes=["apply using AI"],
        raw_jd_excerpt="...",
    )


def _factcheck_clean() -> FactCheckOutput:
    return FactCheckOutput(
        layer_name="factchecker",
        audits=[],
        hard_flags=[],
        soft_flags=[],
    )


def _factcheck_with_hard_flag() -> FactCheckOutput:
    return FactCheckOutput(
        layer_name="factchecker",
        audits=[],
        hard_flags=[
            Flag(
                severity="hard",
                kind="unknown_fact_id",
                claim_text="bad claim",
                cited_fact_ids=["does-not-exist"],
                explanation="not in candidate.facts",
            )
        ],
        soft_flags=[],
    )


def test_submitter_writes_artifacts_to_run_output_dir(tmp_path: Path) -> None:
    ctx = _ctx_with(_writer_output(), _factcheck_clean(), _role_analysis())
    submitter = SubmitterAgent(
        candidate_name="Test Candidate",
        candidate_email="test@example.com",
        candidate_phone="555-0100",
        candidate_links={"github": "https://github.com/example"},
        output_root=tmp_path,
    )
    out = submitter.run(ctx)

    assert isinstance(out, SubmitterOutput)
    cv = Path(out.cv_md_path).read_text(encoding="utf-8")
    cover = Path(out.cover_md_path).read_text(encoding="utf-8")
    plan = json.loads(Path(out.form_plan_path).read_text(encoding="utf-8"))
    audit = Path(out.audit_md_path).read_text(encoding="utf-8")

    assert "Test Candidate" in cv
    assert "Software Engineer at Testco" in cv
    assert "test@example.com" in cover
    assert plan["target_url"] == "https://example.test/jd"
    field_names = {f["name"] for f in plan["fields"]}
    assert {"full_name", "email", "phone", "github", "cover_letter"} <= field_names
    assert "safe_to_submit: **True**" in audit


def test_submitter_blocks_on_hard_flag(tmp_path: Path) -> None:
    ctx = _ctx_with(_writer_output(), _factcheck_with_hard_flag(), _role_analysis())
    submitter = SubmitterAgent(
        candidate_name="Test Candidate",
        candidate_email="test@example.com",
        output_root=tmp_path,
    )
    with pytest.raises(SubmitterBlocked, match="hard flag"):
        submitter.run(ctx)
    # Nothing should have been written.
    assert not any(tmp_path.rglob("*.md"))


def test_submitter_records_block_on_run_notes(tmp_path: Path) -> None:
    ctx = _ctx_with(_writer_output(), _factcheck_with_hard_flag(), _role_analysis())
    submitter = SubmitterAgent(
        candidate_name="Test",
        candidate_email="t@example.com",
        output_root=tmp_path,
    )
    with pytest.raises(SubmitterBlocked):
        submitter.run(ctx)
    assert any("submitter blocked" in note for note in ctx.run.notes)
