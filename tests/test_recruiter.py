"""Tests for the Recruiter agent.

LLM calls are stubbed. These tests pin the agent's contract:
- it consumes a JDSource
- it calls the LLM with a strict schema and the system prompt
- it composes the RoleAnalysis with run-side metadata
- it records a 'jd-drift' note on the Run when the source flags drift
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from applyops.agents.jd_source import FileJDSource
from applyops.agents.recruiter import (
    RECRUITER_SYSTEM_PROMPT,
    RecruiterAgent,
    RoleAnalysisPayload,
)
from applyops.agents.types import JDMeta, Requirement
from applyops.gstack import Stack
from applyops.gstack.run import RunStatus

FIXTURE = Path(__file__).parent / "fixtures" / "jd.fake.md"


class CapturingStubLLM:
    """Records the call args; returns a pre-built payload."""

    def __init__(self, payload: BaseModel) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def parse(self, *, model: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema})
        return self.payload


class DriftingFakeJDSource:
    """JDSource stub that always returns drift=True so we can test the note."""

    def fetch(self) -> tuple[str, JDMeta]:
        return (
            "fake jd body",
            JDMeta(url="https://x", hash="abc123abc123", snapshot_path="/tmp/x", drift=True),
        )


def _example_payload() -> RoleAnalysisPayload:
    return RoleAnalysisPayload(
        role_title="AI Ops Engineer",
        company="Fakeco",
        location="Toronto",
        requirements=[
            Requirement(
                text="Strong Python",
                kind="must_have",
                importance=4,
                category="technical",
                evidence_anchor="production Python experience",
            )
        ],
        company_signals=["values eval discipline"],
        application_protocol_notes=["apply using AI"],
    )


def test_recruiter_calls_llm_with_jd_text_and_strict_schema() -> None:
    llm = CapturingStubLLM(_example_payload())
    agent = RecruiterAgent(FileJDSource(FIXTURE), llm=llm, model="test-model")
    stack = Stack(layers=[agent])

    run, _ctx = stack.land()

    assert run.status == RunStatus.COMPLETED
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == "test-model"
    assert call["system"] == RECRUITER_SYSTEM_PROMPT
    assert "AI Ops Engineer" in str(call["user"])
    assert call["schema"] is RoleAnalysisPayload


def test_recruiter_composes_role_analysis_with_jd_meta() -> None:
    llm = CapturingStubLLM(_example_payload())
    agent = RecruiterAgent(FileJDSource(FIXTURE), llm=llm)
    stack = Stack(layers=[agent])

    _, ctx = stack.land()

    output = ctx.output_of("recruiter")
    assert output.layer_name == "recruiter"
    from applyops.agents.recruiter import RoleAnalysis

    assert isinstance(output, RoleAnalysis)
    assert output.role_title == "AI Ops Engineer"
    assert output.jd_meta.snapshot_path.endswith("jd.fake.md")
    assert len(output.raw_jd_excerpt) <= 600
    assert "AI Ops Engineer" in output.raw_jd_excerpt


def test_recruiter_notes_jd_drift_on_run() -> None:
    llm = CapturingStubLLM(_example_payload())
    agent = RecruiterAgent(DriftingFakeJDSource(), llm=llm)
    stack = Stack(layers=[agent])

    run, _ = stack.land()

    assert any("jd-drift" in note for note in run.notes)
