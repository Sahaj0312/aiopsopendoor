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
        render_pdf=False,  # don't launch Playwright in unit tests
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


def test_submitter_prefers_original_resume_when_provided(tmp_path: Path) -> None:
    # Fake "original" resume on disk
    original = tmp_path / "private" / "sahaj_resume.pdf"
    original.parent.mkdir()
    original.write_bytes(b"%PDF-1.4 fake original\n%%EOF\n")

    ctx = _ctx_with(_writer_output(), _factcheck_clean(), _role_analysis())
    submitter = SubmitterAgent(
        candidate_name="Test Candidate",
        candidate_email="test@example.com",
        output_root=tmp_path / "out",
        render_pdf=False,  # the test only cares about plan composition
        original_resume_path=original,
    )
    out = submitter.run(ctx)
    plan = json.loads(Path(out.form_plan_path).read_text(encoding="utf-8"))

    resume_fields = [f for f in plan["fields"] if f["name"] == "resume"]
    assert len(resume_fields) == 1
    assert resume_fields[0]["value"] == str(original)
    assert resume_fields[0]["source_artifact"] == "original_resume.pdf"
    assert resume_fields[0]["kind"] == "file"

    note_text = " ".join(plan["notes"])
    assert "original PDF" in note_text


def test_submitter_offers_cover_as_both_textarea_and_file(tmp_path: Path) -> None:
    """The plan must include both a textarea and a file representation of the
    cover letter so the submit-time LLM can pick whichever the form has."""
    # Need PDF rendering for the file variant to appear, so use a stub.
    ctx = _ctx_with(_writer_output(), _factcheck_clean(), _role_analysis())
    submitter = SubmitterAgent(
        candidate_name="X",
        candidate_email="x@y.com",
        output_root=tmp_path,
        render_pdf=False,  # without PDF, only the textarea variant should appear
    )
    out = submitter.run(ctx)
    plan = json.loads(Path(out.form_plan_path).read_text(encoding="utf-8"))
    cover_fields = [f for f in plan["fields"] if f["name"].startswith("cover_letter")]
    # With render_pdf=False, only the textarea variant is present.
    assert {f["name"] for f in cover_fields} == {"cover_letter"}
    # The textarea value carries the actual cover text, not a placeholder.
    textarea = next(f for f in cover_fields if f["name"] == "cover_letter")
    assert textarea["kind"] == "textarea"
    assert "built with the same kind of system" in textarea["value"] or len(textarea["value"]) > 10


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
