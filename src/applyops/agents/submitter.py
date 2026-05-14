"""Submitter — turns reviewed artifacts into a submittable plan.

The submitter is deliberately conservative. It does NOT auto-submit. It
produces:
- a markdown CV (`cv.md`)
- a markdown cover letter (`cover.md`)
- a JSON form-fill plan (`form_plan.json`) describing what fields to fill
  in the ATS form, in what order, with what values
- a human-readable audit appendix (`audit.md`) summarizing the
  factchecker's findings

A human reads the plan, confirms the artifacts, and either pastes them
into the ATS form themselves or runs `applyops submit --confirm` to
trigger a Playwright dry-run + final-confirm-prompt step (added later).

If `FactCheckOutput.safe_to_submit` is False, the submitter aborts the
layer by raising a SubmitterBlocked exception. Hard flags are non-
negotiable; the only path forward is fixing the issue upstream.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from applyops.agents.factchecker import FactCheckOutput
from applyops.agents.recruiter import RoleAnalysis
from applyops.agents.writer import CoverLetter, CVDraft, WriterOutput
from applyops.gstack.context import StackContext
from applyops.gstack.types import LayerOutput


class SubmitterBlocked(RuntimeError):
    """Raised when the submitter refuses to render due to hard factcheck flags."""


class FormField(BaseModel):
    """One field on the target ATS form."""

    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    value: str
    kind: str = Field(
        ...,
        description="text | textarea | file | url | select. 'select' values are the human-readable dropdown option text.",
    )
    source_artifact: str = Field(
        ...,
        description="Where the value came from: 'cv.md', 'cover.md', 'facts.local.json', 'literal'.",
    )


class FormFillPlan(BaseModel):
    """The plan a human reviews before any submission."""

    model_config = ConfigDict(extra="forbid")

    target_url: str | None
    fields: list[FormField]
    files_to_upload: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SubmitterOutput(LayerOutput):
    """Everything written to outputs/<run-id>/ for human review."""

    cv_md_path: str
    cv_pdf_path: str | None = None
    cover_md_path: str
    cover_pdf_path: str | None = None
    form_plan_path: str
    audit_md_path: str
    output_dir: str


class SubmitterAgent:
    """Renders artifacts and writes them to outputs/<run-id>/."""

    name = "submitter"

    def __init__(
        self,
        *,
        target_url: str | None = None,
        candidate_name: str,
        candidate_email: str,
        candidate_phone: str | None = None,
        candidate_links: dict[str, str] | None = None,
        output_root: str | Path = "outputs",
        render_pdf: bool = True,
        original_resume_path: str | Path | None = None,
        voluntary_disclosures: dict[str, str] | None = None,
    ) -> None:
        self.target_url = target_url
        self.candidate_name = candidate_name
        self.candidate_email = candidate_email
        self.candidate_phone = candidate_phone
        self.candidate_links = dict(candidate_links or {})
        self.output_root = Path(output_root)
        self.render_pdf = render_pdf
        self.original_resume_path = Path(original_resume_path) if original_resume_path else None
        self.voluntary_disclosures = dict(voluntary_disclosures or {})

    def run(self, ctx: StackContext) -> SubmitterOutput:
        writer_output = ctx.output_of("writer")
        if not isinstance(writer_output, WriterOutput):
            raise TypeError(
                f"submitter expects WriterOutput from 'writer', got {type(writer_output).__name__}"
            )
        factcheck = ctx.output_of("factchecker")
        if not isinstance(factcheck, FactCheckOutput):
            raise TypeError(
                f"submitter expects FactCheckOutput from 'factchecker', got {type(factcheck).__name__}"
            )

        if not factcheck.safe_to_submit:
            ctx.run.note(
                f"submitter blocked: {len(factcheck.hard_flags)} hard flag(s) in factcheck"
            )
            raise SubmitterBlocked(
                f"refusing to render — {len(factcheck.hard_flags)} hard flag(s); "
                "fix upstream and re-land the writer/factchecker layers"
            )

        role_analysis = ctx.output_of("recruiter")
        assert isinstance(role_analysis, RoleAnalysis)

        out_dir = self.output_root / ctx.run.id
        out_dir.mkdir(parents=True, exist_ok=True)

        cv_md = _render_cv(self.candidate_name, self.candidate_links, writer_output.cv)
        cover_md = _render_cover_letter(
            self.candidate_name, self.candidate_email, writer_output.cover_letter
        )
        cover_text = _render_cover_letter_text(writer_output.cover_letter)
        audit_md = _render_audit(factcheck)

        cv_path = out_dir / "cv.md"
        cv_path.write_text(cv_md, encoding="utf-8")
        cover_path = out_dir / "cover.md"
        cover_path.write_text(cover_md, encoding="utf-8")
        audit_path = out_dir / "audit.md"
        audit_path.write_text(audit_md, encoding="utf-8")

        cv_pdf_path: Path | None = None
        cover_pdf_path: Path | None = None
        if self.render_pdf:
            try:
                from applyops.render import markdown_to_pdf

                cv_pdf_path = markdown_to_pdf(
                    cv_md, out_dir / "cv.pdf", title=f"{self.candidate_name} — CV"
                )
                cover_pdf_path = markdown_to_pdf(
                    cover_md,
                    out_dir / "cover.pdf",
                    title=f"{self.candidate_name} — Cover Letter",
                )
            except RuntimeError as exc:
                # Playwright not installed — markdown still written; PDF skipped.
                ctx.run.note(f"pdf rendering skipped: {exc}")

        # Pick the resume PDF to upload. Original takes precedence when set —
        # the generated cv.pdf stays in outputs/ for review/comparison.
        resume_upload_path: Path | None = None
        resume_source = "cv.pdf"
        if self.original_resume_path and self.original_resume_path.exists():
            resume_upload_path = self.original_resume_path
            resume_source = "original_resume.pdf"
            ctx.run.note(f"using original resume for upload: {resume_upload_path}")
        elif cv_pdf_path and cv_pdf_path.exists():
            resume_upload_path = cv_pdf_path

        plan = self._build_plan(
            role_analysis=role_analysis,
            resume_upload_path=resume_upload_path,
            resume_source=resume_source,
            cover_pdf_path=cover_pdf_path,
            cover_text=cover_text,
        )
        plan_path = out_dir / "form_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        return SubmitterOutput(
            layer_name=self.name,
            cv_md_path=str(cv_path),
            cv_pdf_path=str(cv_pdf_path) if cv_pdf_path else None,
            cover_md_path=str(cover_path),
            cover_pdf_path=str(cover_pdf_path) if cover_pdf_path else None,
            form_plan_path=str(plan_path),
            audit_md_path=str(audit_path),
            output_dir=str(out_dir),
        )

    def _build_plan(
        self,
        *,
        role_analysis: RoleAnalysis,
        resume_upload_path: Path | None,
        resume_source: str,
        cover_pdf_path: Path | None,
        cover_text: str,
    ) -> FormFillPlan:
        fields: list[FormField] = [
            FormField(
                name="full_name",
                label="Full name",
                value=self.candidate_name,
                kind="text",
                source_artifact="literal",
            ),
            FormField(
                name="email",
                label="Email",
                value=self.candidate_email,
                kind="text",
                source_artifact="literal",
            ),
        ]
        if self.candidate_phone:
            fields.append(
                FormField(
                    name="phone",
                    label="Phone",
                    value=self.candidate_phone,
                    kind="text",
                    source_artifact="literal",
                )
            )
        for key, url in self.candidate_links.items():
            fields.append(
                FormField(
                    name=key,
                    label=key.capitalize(),
                    value=url,
                    kind="url",
                    source_artifact="literal",
                )
            )

        # Resume file. Most ATSes have exactly one resume upload; the submit
        # LLM will map this to it.
        if resume_upload_path is not None:
            fields.append(
                FormField(
                    name="resume",
                    label="Resume",
                    value=str(resume_upload_path),
                    kind="file",
                    source_artifact=resume_source,
                )
            )

        # Cover letter — offered as BOTH textarea text and PDF file. The submit
        # LLM picks whichever shape matches the actual form. Forms that only
        # accept text will get cover_letter; forms that only accept files
        # will get cover_letter_file. Forms with both can fill either (or both).
        fields.append(
            FormField(
                name="cover_letter",
                label="Cover letter",
                value=cover_text,
                kind="textarea",
                source_artifact="cover.md",
            )
        )
        if cover_pdf_path is not None:
            fields.append(
                FormField(
                    name="cover_letter_file",
                    label="Cover letter (file)",
                    value=str(cover_pdf_path),
                    kind="file",
                    source_artifact="cover.pdf",
                )
            )

        # Voluntary EEOC / disability / veteran / SMS-consent dropdowns.
        # Values are the human-readable option strings; the submit-time LLM
        # routes them to the matching dropdown on the page.
        _LABELS = {
            "gender": "Gender",
            "race": "Race",
            "hispanic_latino": "Hispanic / Latino",
            "veteran_status": "Veteran Status",
            "disability_status": "Disability Status",
            "sms_consent": "Text message consent",
        }
        for key, value in self.voluntary_disclosures.items():
            if not value:
                continue
            fields.append(
                FormField(
                    name=f"voluntary_{key}",
                    label=_LABELS.get(key, key.replace("_", " ").title()),
                    value=value,
                    kind="select",
                    source_artifact="voluntary.local.json",
                )
            )

        notes: list[str] = []
        if role_analysis.application_protocol_notes:
            notes.append(
                "JD application-protocol notes were addressed in cover.md per the writer's draft: "
                + "; ".join(role_analysis.application_protocol_notes)
            )
        notes.append(
            "Resume upload: "
            + (
                f"original PDF at {resume_upload_path}"
                if resume_source == "original_resume.pdf"
                else f"AI-generated PDF at {resume_upload_path}"
                if resume_upload_path
                else "none — no resume PDF available"
            )
        )
        notes.append(
            "HITL gate: review cv.md, cover.md, and audit.md before submitting. "
            "The applyops submit command pauses for explicit human confirmation."
        )

        files_to_upload: list[str] = []
        if resume_upload_path:
            files_to_upload.append(str(resume_upload_path))
        if cover_pdf_path:
            files_to_upload.append(str(cover_pdf_path))

        return FormFillPlan(
            target_url=role_analysis.jd_meta.url or self.target_url,
            fields=fields,
            files_to_upload=files_to_upload,
            notes=notes,
        )


# --- Rendering ---------------------------------------------------------------


def _render_cv(name: str, links: dict[str, str], cv: CVDraft) -> str:
    lines: list[str] = [f"# {name}", ""]
    if links:
        lines.append(" · ".join(f"[{k}]({v})" for k, v in links.items()))
        lines.append("")
    lines.append(f"> {cv.summary.text}")
    lines.append("")

    if cv.experience:
        lines.extend(["## Experience", ""])
        for entry in cv.experience:
            lines.append(
                f"**{entry.heading}**" + (f" — {entry.date_range}" if entry.date_range else "")
            )
            for bullet in entry.bullets:
                lines.append(f"- {bullet.text}")
            lines.append("")

    if cv.projects:
        lines.extend(["## Projects", ""])
        for entry in cv.projects:
            lines.append(
                f"**{entry.heading}**" + (f" — {entry.date_range}" if entry.date_range else "")
            )
            for bullet in entry.bullets:
                lines.append(f"- {bullet.text}")
            lines.append("")

    if cv.skills_line:
        lines.extend(["## Skills", "", cv.skills_line, ""])

    if cv.education:
        lines.extend(["## Education", ""])
        for entry in cv.education:
            lines.append(
                f"**{entry.heading}**" + (f" — {entry.date_range}" if entry.date_range else "")
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_cover_letter(name: str, email: str, cover: CoverLetter) -> str:
    lines: list[str] = [f"# Cover letter — {name}", f"<{email}>", ""]
    for para in cover.paragraphs:
        lines.append(para.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_cover_letter_text(cover: CoverLetter) -> str:
    """Plain paragraph-separated text suitable for pasting into a textarea."""
    paras = [para.text.strip() for para in cover.paragraphs if para.text.strip()]
    return "\n\n".join(paras)


def _render_audit(factcheck: FactCheckOutput) -> str:
    lines: list[str] = ["# Factcheck audit", ""]
    lines.append(f"- safe_to_submit: **{factcheck.safe_to_submit}**")
    lines.append(f"- hard flags: {len(factcheck.hard_flags)}")
    lines.append(f"- soft flags: {len(factcheck.soft_flags)}")
    lines.append("")

    if factcheck.hard_flags:
        lines.extend(["## Hard flags", ""])
        for f in factcheck.hard_flags:
            lines.extend(
                [
                    f"- **[{f.kind}]** {f.claim_text}",
                    f"  - cited: {', '.join(f.cited_fact_ids)}",
                    f"  - {f.explanation}",
                ]
            )
            if f.suggested_fix:
                lines.append(f"  - suggested fix: {f.suggested_fix}")
        lines.append("")

    if factcheck.soft_flags:
        lines.extend(["## Soft flags", ""])
        for f in factcheck.soft_flags:
            lines.extend(
                [
                    f"- **[{f.kind}]** {f.claim_text}",
                    f"  - cited: {', '.join(f.cited_fact_ids)}",
                    f"  - {f.explanation}",
                ]
            )
        lines.append("")

    lines.extend(["## Per-claim audits", ""])
    for audit in factcheck.audits:
        lines.append(f"- **{audit.verdict}** — {audit.claim_text}")
        lines.append(f"  - cited: {', '.join(audit.cited_fact_ids)}")
        lines.append(f"  - {audit.rationale}")
    lines.append("")
    return "\n".join(lines)


# Kept for export.
__all__ = [
    "FormField",
    "FormFillPlan",
    "SubmitterAgent",
    "SubmitterBlocked",
    "SubmitterOutput",
    "_render_audit",
    "_render_cover_letter",
    "_render_cv",
]
