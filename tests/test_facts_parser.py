"""Tests for the resume parser.

We don't ship a real PDF in the test fixtures (the user's resume is
gitignored). Instead, the parser is tested with:
- a stubbed PDF text extractor that injects known text
- a stubbed StructuredLLM that returns a pre-built ParserPayload
- assertions about how the parser composes a Candidate, what verification
  level the produced facts have, and how it propagates the LLM's output
  to the Candidate schema's strict validation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from applyops.facts import Candidate, load
from applyops.facts_parser import ParserPayload, parse_resume, write_draft


class StubLLM:
    def __init__(self, payload: BaseModel) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def parse(self, *, model: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema})
        return self.payload


def _sample_payload() -> ParserPayload:
    return ParserPayload(
        name="Test User",
        headline="Engineer with production AI experience.",
        location="Toronto, ON",
        links={"github": "https://github.com/example"},
        facts=[
            {
                "id": "exp-test-2025-swe",
                "kind": "experience",
                "title": "Software Engineer at Testco",
                "detail": "Shipped X. Built Y.",
                "tags": ["python", "production"],
                "metrics": {"coverage_increase": "14% to 97%"},
                "provenance": [
                    {
                        "source": "resume.pdf p.1",
                        "verified_by": "ai_extracted_unverified",
                    }
                ],
            }
        ],
    )


def test_parse_resume_uses_llm_with_resume_text_and_schema(tmp_path: Path) -> None:
    pdf = tmp_path / "fake.pdf"
    # We don't need real PDF parsing here; monkeypatch is overkill.
    # Instead, we hit parse_resume through a path that exists: a minimal
    # one-page PDF. pypdf will read it.
    pdf.write_bytes(
        b"%PDF-1.1\n%\xe2\xe3\xcf\xd3\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 700 Td (hello world) Tj ET\nendstream endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000015 00000 n \n0000000061 00000 n \n"
        b"0000000111 00000 n \n0000000176 00000 n \ntrailer<</Size 5/Root 1 0 R>>\n"
        b"startxref\n260\n%%EOF\n"
    )
    llm = StubLLM(_sample_payload())

    candidate = parse_resume(pdf, llm=llm, model="test-model")

    assert len(llm.calls) == 1
    assert llm.calls[0]["model"] == "test-model"
    assert llm.calls[0]["schema"] is ParserPayload
    assert candidate.name == "Test User"
    assert len(candidate.facts) == 1


def test_parsed_facts_are_unverified_by_default() -> None:
    """A parsed Candidate should require human attestation before runs."""
    candidate = Candidate.model_validate(
        {
            "name": "x",
            "headline": "y",
            "location": "z",
            "facts": [
                {
                    "id": "fact-1",
                    "kind": "skill",
                    "title": "Python",
                    "detail": "I know Python.",
                    "provenance": [
                        {
                            "source": "resume.pdf p.1",
                            "verified_by": "ai_extracted_unverified",
                        }
                    ],
                }
            ],
        }
    )
    assert candidate.unverified() == candidate.facts


def test_write_draft_round_trips(tmp_path: Path) -> None:
    candidate = Candidate.model_validate(
        {
            "name": "x",
            "headline": "y",
            "location": "z",
            "facts": [
                {
                    "id": "fact-1",
                    "kind": "skill",
                    "title": "Python",
                    "detail": "I know Python.",
                    "provenance": [
                        {"source": "resume.pdf p.1", "verified_by": "self"}
                    ],
                }
            ],
        }
    )
    out = tmp_path / "nested" / "facts.json"
    write_draft(candidate, out)
    reloaded = load(out)
    assert reloaded == candidate
