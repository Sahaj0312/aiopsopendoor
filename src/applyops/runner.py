"""Wiring — construct the full stack from configuration and run it.

Lives outside the CLI module so it's testable on its own and can be
imported by future entry points (a web UI, a CI job, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from applyops.agents.critic import CriticGate
from applyops.agents.factchecker import FactCheckerAgent
from applyops.agents.jd_source import (
    FileJDSource,
    HttpJDSource,
    JDSource,
    PlaywrightJDSource,
)
from applyops.agents.recruiter import OpenAIStructuredLLM, RecruiterAgent
from applyops.agents.submitter import SubmitterAgent
from applyops.agents.writer import WriterAgent
from applyops.facts import Candidate
from applyops.facts import load as load_candidate
from applyops.gstack import Stack
from applyops.gstack.context import StackContext
from applyops.gstack.run import Run


@dataclass
class RunConfig:
    """Everything the runner needs. Defaults wired in CLI; this module is dumb."""

    jd_url: str | None
    jd_file: Path | None
    facts_path: Path
    output_root: Path
    snapshot_dir: Path
    candidate_email: str
    candidate_phone: str | None
    target_url_override: str | None
    recruiter_model: str
    writer_model: str
    critic_model: str
    factcheck_model: str
    max_rebases_per_gate: int
    render_jd: bool = False
    jd_wait_selector: str | None = None


def build_jd_source(cfg: RunConfig) -> JDSource:
    if cfg.jd_file is not None:
        return FileJDSource(cfg.jd_file)
    if cfg.jd_url is None:
        raise ValueError("either jd_url or jd_file is required")
    if cfg.render_jd:
        return PlaywrightJDSource(
            cfg.jd_url,
            snapshot_dir=cfg.snapshot_dir,
            slug="jd.opendoor",
            wait_for_selector=cfg.jd_wait_selector,
        )
    return HttpJDSource(cfg.jd_url, snapshot_dir=cfg.snapshot_dir, slug="jd.opendoor")


def build_stack(cfg: RunConfig, candidate: Candidate) -> Stack:
    """Construct the stack. Pure function of config + candidate — no I/O."""
    from openai import OpenAI

    llm = OpenAIStructuredLLM(OpenAI())

    recruiter = RecruiterAgent(
        jd_source=build_jd_source(cfg),
        llm=llm,
        model=cfg.recruiter_model,
    )
    writer = WriterAgent(candidate=candidate, llm=llm, model=cfg.writer_model)
    critic = CriticGate(llm=llm, model=cfg.critic_model)
    factchecker = FactCheckerAgent(candidate=candidate, llm=llm, model=cfg.factcheck_model)
    submitter = SubmitterAgent(
        target_url=cfg.target_url_override,
        candidate_name=candidate.name,
        candidate_email=cfg.candidate_email,
        candidate_phone=cfg.candidate_phone,
        candidate_links={k: str(v) for k, v in candidate.links.items()},
        output_root=cfg.output_root,
    )

    return Stack(
        layers=[recruiter, writer, factchecker, submitter],
        gates={"writer": critic},
        max_rebases_per_gate=cfg.max_rebases_per_gate,
    )


def execute(cfg: RunConfig) -> tuple[Run, StackContext]:
    """Load candidate, build stack, land. Caller handles persistence + reporting."""
    candidate = load_candidate(cfg.facts_path)
    stack = build_stack(cfg, candidate)
    return stack.land()
