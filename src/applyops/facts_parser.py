"""Resume PDF → Candidate parser.

Extracts text from a resume PDF, calls the LLM with the Candidate schema,
and returns a draft `Candidate`. All facts come back marked
`verified_by="ai_extracted_unverified"` — the candidate must attest each
fact (flip to `verified_by="self"`) before the writer agent can ground
claims on it. This is enforced downstream by `Fact.verified` and the
fact-checker; the parser itself only produces drafts.

The parser is intentionally conservative:
- Numeric metrics (50+ MAU, 14%→97% coverage, 20k+ assets) are only kept
  when they appear verbatim in the PDF text. The LLM is prompted to never
  invent or round metrics it didn't see.
- Facts without a clear source span in the resume are dropped, not
  fabricated.
- Provenance always cites the source file path and page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from applyops.facts import Candidate

PARSER_SYSTEM_PROMPT = """You are a careful resume parser. Convert a candidate's resume into a strict Candidate schema.

Rules (violation = a worse application later, so be conservative):

1. Every Fact's `detail` must be supported by text in the resume. Do not infer responsibilities, scope, or impact that the resume does not state. If the resume says "led testing", do not write "led testing for the entire org" — write "led testing across multiple microservices" (or whatever the resume said).

2. `metrics` are only populated with numbers that appear verbatim in the resume text. Examples that are OK: "50+ MAU", "14% to 97%", "20,000+ video assets", "10+ PB". Examples that are NOT OK: rounding "97%" to "~100%", inventing "$1M ARR", inferring "thousands of users" when the resume only says "many users".

3. `id` is a stable slug. Use the pattern:
   - experiences: `exp-<company-slug>-<year>-<role-keyword>` (e.g. `exp-quickplay-2025-swe`)
   - projects: `proj-<name-slug>` (e.g. `proj-pixitt`)
   - skills: `skill-<topic>` (e.g. `skill-python`)
   - education: `edu-<school-slug>` (e.g. `edu-ubc-bcom-cs`)

4. `tags` are searchable keywords the writer agent will use to match facts to a job description. Be generous with synonyms (e.g., "computer-vision", "cv", "vision"). Lowercase, hyphen-separated.

5. Every Fact must have at least one Provenance entry with `verified_by="ai_extracted_unverified"` and a `source` of the form "resume.pdf p.N" where N is the page number (1-indexed).

6. `Candidate.headline` is ONE line summarizing the candidate's positioning. Derive it from the strongest signals in the resume; do not invent a tagline that isn't supported.

7. If a section of the resume is empty or unparseable, skip it. Better to under-extract than to hallucinate.

Be precise. The downstream fact-checker will flag any claim that doesn't trace to this output."""


class ParserPayload(BaseModel):
    """What the LLM returns directly. Composed into a Candidate by the parser."""

    model_config = ConfigDict(extra="forbid")

    name: str
    headline: str
    location: str
    links: dict[str, str] = Field(default_factory=dict)
    facts: list[dict[str, object]] = Field(
        default_factory=list,
        description="Each entry is a Fact dict ready to be validated.",
    )


class StructuredLLM(Protocol):
    """Same minimal shape used by the recruiter; kept local to avoid coupling."""

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel: ...


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract text from a PDF with page markers preserved for provenance."""
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"--- page {i} ---\n{text.strip()}\n")
    return "\n".join(parts)


def parse_resume(
    pdf_path: str | Path,
    *,
    llm: StructuredLLM,
    model: str = "gpt-4.1",
) -> Candidate:
    """Parse a resume PDF into a draft Candidate.

    All extracted facts come back unverified. The caller is expected to
    persist the draft and attest each fact manually before running the
    writer agent.
    """
    resume_text = extract_pdf_text(pdf_path)
    payload = llm.parse(
        model=model,
        system=PARSER_SYSTEM_PROMPT,
        user=resume_text,
        schema=ParserPayload,
    )
    assert isinstance(payload, ParserPayload)

    candidate_dict: dict[str, object] = {
        "name": payload.name,
        "headline": payload.headline,
        "location": payload.location,
        "links": payload.links,
        "facts": payload.facts,
    }
    return Candidate.model_validate(candidate_dict)


def write_draft(candidate: Candidate, out_path: str | Path) -> Path:
    """Write a Candidate to disk as JSON. Creates parent dirs."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(candidate.model_dump(mode="json"), indent=2), encoding="utf-8")
    return p
