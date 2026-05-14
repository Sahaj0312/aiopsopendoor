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
    kind: str = Field(..., description="text | textarea | file | select | url")
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
    cover_md_path: str
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
    ) -> None:
        self.target_url = target_url
        self.candidate_name = candidate_name
        self.candidate_email = candidate_email
        self.candidate_phone = candidate_phone
        self.candidate_links = dict(candidate_links or {})
        self.output_root = Path(output_root)

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
        audit_md = _render_audit(factcheck)

        plan = self._build_plan(writer_output, role_analysis, out_dir)

        cv_path = out_dir / "cv.md"
        cv_path.write_text(cv_md, encoding="utf-8")
        cover_path = out_dir / "cover.md"
        cover_path.write_text(cover_md, encoding="utf-8")
        audit_path = out_dir / "audit.md"
        audit_path.write_text(audit_md, encoding="utf-8")
        plan_path = out_dir / "form_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        return SubmitterOutput(
            layer_name=self.name,
            cv_md_path=str(cv_path),
            cover_md_path=str(cover_path),
            form_plan_path=str(plan_path),
            audit_md_path=str(audit_path),
            output_dir=str(out_dir),
        )

    def _build_plan(
        self,
        writer_output: WriterOutput,
        role_analysis: RoleAnalysis,
        out_dir: Path,
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
        fields.append(
            FormField(
                name="cover_letter",
                label="Cover letter",
                value="(see cover.md)",
                kind="textarea",
                source_artifact="cover.md",
            )
        )

        notes: list[str] = []
        if role_analysis.application_protocol_notes:
            notes.append(
                "JD application-protocol notes were addressed in cover.md per the writer's draft: "
                + "; ".join(role_analysis.application_protocol_notes)
            )
        notes.append(
            "HITL gate: review cv.md, cover.md, and audit.md before submitting. "
            "No field below is auto-submitted."
        )

        return FormFillPlan(
            target_url=role_analysis.jd_meta.url or self.target_url,
            fields=fields,
            files_to_upload=[str(out_dir / "cv.md")],
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
