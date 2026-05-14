"""Integration test for `applyops submit` driving a real browser.

The test sets up a synthetic outputs/<run-id>/ with a form_plan.json
pointing at a local fake ATS form, stubs the LLM to return a hand-built
SubmitFieldMap, and calls submit() with auto_confirm=True (no terminal
input). It verifies:

- the fields actually get filled in the browser
- the resume file actually gets uploaded
- the submit button is clicked
- the page state changes (URL gets a ?submitted=1)
- submission.json records the outcome

No network. No OpenAI. Real Chromium against a file:// fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

pytest.importorskip("playwright")

from applyops.agents.submitter import FormField, FormFillPlan
from applyops.submit import FieldLocator, SubmitFieldMap, submit


FIXTURE = Path(__file__).parent / "fixtures" / "fake_ats_form.html"


class StubLLM:
    def __init__(self, payload: BaseModel) -> None:
        self.payload = payload
        self.calls = 0

    def parse(self, *, model: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls += 1
        return self.payload


def _seed_run_dir(tmp_path: Path) -> Path:
    """Set up a synthetic outputs/<run-id>/ with form_plan.json + a dummy resume."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    resume_pdf = run_dir / "cv.pdf"
    resume_pdf.write_bytes(b"%PDF-1.4\n%fake\n%%EOF\n")
    plan = FormFillPlan(
        target_url=f"file://{FIXTURE.resolve()}",
        fields=[
            FormField(name="full_name", label="Full name", value="Test User", kind="text", source_artifact="literal"),
            FormField(name="email", label="Email", value="test@example.com", kind="text", source_artifact="literal"),
            FormField(name="phone", label="Phone", value="555-0100", kind="text", source_artifact="literal"),
            FormField(name="resume", label="Resume", value=str(resume_pdf), kind="file", source_artifact="cv.pdf"),
            FormField(
                name="cover_letter",
                label="Cover letter",
                value="I want this role and here's why.",
                kind="textarea",
                source_artifact="cover.md",
            ),
        ],
        files_to_upload=[str(resume_pdf)],
        notes=[],
    )
    (run_dir / "form_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def _ats_field_map(resume_path: str) -> SubmitFieldMap:
    """The mapping the LLM would produce for the fake ATS fixture.

    Mixed strategies on purpose — text fields via role, file via label
    (since <input type=file> has no clean ARIA role).
    """
    return SubmitFieldMap(
        fields=[
            FieldLocator(
                plan_field_name="full_name", locator_strategy="role",
                role="textbox", name="Full name", fill_kind="type", value="Test User",
            ),
            FieldLocator(
                plan_field_name="email", locator_strategy="role",
                role="textbox", name="Email", fill_kind="type", value="test@example.com",
            ),
            FieldLocator(
                plan_field_name="phone", locator_strategy="role",
                role="textbox", name="Phone", fill_kind="type", value="555-0100",
            ),
            FieldLocator(
                plan_field_name="resume", locator_strategy="label",
                name="Resume", fill_kind="upload", value=resume_path,
            ),
            FieldLocator(
                plan_field_name="cover_letter", locator_strategy="role",
                role="textbox", name="Cover letter", fill_kind="type",
                value="I want this role and here's why.",
            ),
        ],
        submit_button_role="button",
        submit_button_name="Submit application",
    )


def test_submit_drives_form_fill_and_clicks_submit(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    resume_path = str(run_dir / "cv.pdf")
    llm = StubLLM(_ats_field_map(resume_path))

    record = submit(
        run_dir,
        llm=llm,
        headless=True,  # CI-safe; the real CLI runs headed.
        auto_confirm=True,  # bypass the input() prompt
    )

    assert record.outcome == "submitted"
    assert "submitted=1" in (record.submit_url_after or "")
    assert set(record.fields_filled) == {"full_name", "email", "phone", "resume", "cover_letter"}
    assert record.fields_skipped == []
    # submission.json was persisted
    persisted = json.loads((run_dir / "submission.json").read_text())
    assert persisted["outcome"] == "submitted"
    # screenshots exist
    assert (run_dir / "submit.before.png").exists()
    assert (run_dir / "submit.after.png").exists()


def test_submit_records_blocked_when_target_url_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    plan = FormFillPlan(target_url=None, fields=[], files_to_upload=[], notes=[])
    (run_dir / "form_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    llm = StubLLM(SubmitFieldMap(fields=[], submit_button_role="button", submit_button_name="x"))
    record = submit(run_dir, llm=llm, headless=True, auto_confirm=True)

    assert record.outcome == "blocked_no_target_url"
    assert "no target_url" in (record.error or "")


def test_submit_skips_unmappable_fields_without_raising(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    # LLM declares it can't find the cover_letter field on the page.
    bad_map = SubmitFieldMap(
        fields=[
            FieldLocator(
                plan_field_name="full_name", locator_strategy="role",
                role="textbox", name="Full name", fill_kind="type", value="x",
            ),
            FieldLocator(
                plan_field_name="cover_letter", locator_strategy="role",
                role="textbox", name="x", fill_kind="skip", value="",
            ),
        ],
        submit_button_role="button",
        submit_button_name="Submit application",
    )
    record = submit(run_dir, llm=StubLLM(bad_map), headless=True, auto_confirm=True)
    assert record.outcome == "submitted"
    assert "cover_letter" in record.fields_skipped
