"""Recruiter agent — JD → RoleAnalysis.

The recruiter sits at the trunk of the stack. It fetches the JD from a
`JDSource`, snapshots it, then calls the LLM with a strict structured-
output schema to extract requirements, company signals, and application-
protocol notes (special instructions in the posting that change how you
should apply).

The recruiter never invents requirements. Its prompt forbids guessing,
and the `Requirement.text` field must be either verbatim from the JD or
a faithful paraphrase. The downstream writer is the one allowed to make
calls about *how* to address a requirement; the recruiter only says
*what* the requirement is.
"""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from applyops.agents.jd_source import JDSource
from applyops.agents.types import JDMeta, Requirement
from applyops.gstack.context import StackContext
from applyops.gstack.types import LayerOutput

RECRUITER_SYSTEM_PROMPT = """You are a hiring-side recruiter analyzing a job description for an AI Ops Engineer role.

Your job is to extract a structured analysis. You are NOT writing an application. You are NOT matching against any candidate. You are reading the job description and producing a faithful structured summary.

Rules:
1. Every `requirement.text` must be verbatim from the JD or a tight paraphrase that preserves meaning. Do not invent requirements.
2. Classify each requirement as one of:
   - `must_have`: explicitly stated as required.
   - `nice_to_have`: explicitly stated as preferred/bonus.
   - `implied`: not stated directly but obvious from the rest of the JD (e.g., "you'll be on-call" implies on-call rotation experience).
3. `importance` is 1-5. 5 means "the role cannot be done without this." Be conservative; most things are 3.
4. `evidence_anchor` describes the KIND of evidence that would address the requirement, written generically. Examples: "production LLM eval harness experience", "Python + observability tooling", "on-call rotation history". It should NOT name a specific company, project, or candidate.
5. `company_signals` capture culture / values / tech stack hints worth surfacing to the writer agent. Two-to-five short bullets.
6. `application_protocol_notes` capture any UNUSUAL application instructions in the JD itself (e.g., "apply using AI", "include a writeup of how you did X"). Empty list if there are none.

If the JD looks truncated or the text appears to be wrapper UI rather than a job posting, return an empty `requirements` list and set `application_protocol_notes` to ["jd content not detected"]. Do not hallucinate a JD."""


class RoleAnalysisPayload(BaseModel):
    """What the LLM returns. Composed into RoleAnalysis with run-side metadata."""

    model_config = ConfigDict(extra="forbid")

    role_title: str
    company: str
    location: str | None
    requirements: list[Requirement] = Field(default_factory=list)
    company_signals: list[str] = Field(default_factory=list)
    application_protocol_notes: list[str] = Field(default_factory=list)


class RoleAnalysis(LayerOutput):
    """Recruiter's output. Read by every downstream layer."""

    role_title: str
    company: str
    location: str | None
    jd_meta: JDMeta
    requirements: list[Requirement]
    company_signals: list[str]
    application_protocol_notes: list[str]
    raw_jd_excerpt: str = Field(
        ...,
        description="First ~600 chars of the cleaned JD. For trace readability.",
    )


class StructuredLLM(Protocol):
    """Minimal interface the recruiter needs from the LLM client.

    Production impl wraps `openai.OpenAI`. Tests substitute a stub.
    """

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel: ...


class OpenAIStructuredLLM:
    """Thin adapter over the OpenAI Python SDK's structured-output `parse` call."""

    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        from applyops.obs import tracer

        with tracer().start_as_current_span("llm.parse") as span:
            span.set_attribute("llm.provider", "openai")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.schema", schema.__name__)
            span.set_attribute("llm.input.system_chars", len(system))
            span.set_attribute("llm.input.user_chars", len(user))
            completion = self.client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=schema,
            )
            if completion.usage is not None:
                span.set_attribute("llm.usage.input_tokens", completion.usage.prompt_tokens)
                span.set_attribute("llm.usage.output_tokens", completion.usage.completion_tokens)
                span.set_attribute("llm.usage.total_tokens", completion.usage.total_tokens)
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("LLM returned no parsed output (refusal or schema mismatch)")
            return parsed


class RecruiterAgent:
    """The trunk layer of the stack. Implements the `Layer` protocol."""

    name = "recruiter"

    def __init__(
        self,
        jd_source: JDSource,
        llm: StructuredLLM,
        *,
        model: str = "gpt-4.1-mini",
    ) -> None:
        self.jd_source = jd_source
        self.llm = llm
        self.model = model

    def run(self, ctx: StackContext) -> RoleAnalysis:
        jd_text, jd_meta = self.jd_source.fetch()
        if jd_meta.drift:
            ctx.run.note(f"jd-drift detected on hash={jd_meta.hash}")

        payload = self.llm.parse(
            model=self.model,
            system=RECRUITER_SYSTEM_PROMPT,
            user=jd_text,
            schema=RoleAnalysisPayload,
        )
        assert isinstance(payload, RoleAnalysisPayload)

        return RoleAnalysis(
            layer_name=self.name,
            role_title=payload.role_title,
            company=payload.company,
            location=payload.location,
            jd_meta=jd_meta,
            requirements=payload.requirements,
            company_signals=payload.company_signals,
            application_protocol_notes=payload.application_protocol_notes,
            raw_jd_excerpt=jd_text[:600],
        )
