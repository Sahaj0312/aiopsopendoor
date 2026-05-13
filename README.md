# applyops — application as production AI Ops

> **Incident:** `INC-2026-05-13-001` — Opendoor is hiring AI Ops Engineers. I want the role.
> **Severity:** Sev-2 (career-impacting, time-bound).
> **Response:** Treat the application as the system. Stack agents. Run evals. Ship traces. Keep humans on the gate.

This repository is my application to Opendoor's AI Ops Engineer role (Toronto), built in response to [Kaz Nejatian's challenge](https://x.com/nejatian) to apply *only* using AI.

Most "AI-first application" attempts produce a tailored resume and a writeup of the workflow. This one is the workflow — a running system with stacked agents, a rubric eval harness, OpenTelemetry traces, and a human-in-the-loop submitter. The application materials it produces are downstream of the system, not the point of it.

If you're hiring for an AI Ops Engineer, the artifact you should care about is the system, not the resume. The resume is just one of its outputs.

---

## The gstack mental model

The agent pipeline is modeled after [Graphite](https://graphite.dev)-style stacked PRs. Each agent is a **layer** in the stack. Each layer's output is a reviewable diff that the next layer builds on. The critic is a reviewer that can request changes and force an upstream **rebase**. You can `land` layers incrementally instead of running the whole pipeline blind.

```
recruiter @ trunk          # parse JD, extract requirements, build evidence map
   └─ writer @ 1           # draft CV + cover letter, grounded in facts.json
        └─ critic @ 2      # rubric eval; can request-changes → rebase writer
             └─ factchk @ 3 # every claim → source citation; flags exaggeration
                  └─ submitter @ 4   # form-fill plan; HITL gate before real send
```

Cascading rebase: if `recruiter` re-runs and the JD analysis changes, every layer above it is marked stale and re-derives in order. No silent drift.

Critic is implemented as a **review gate**, not a step — it sits between layers and blocks the stack until its rubric passes or a human overrides with a logged exception.

---

## Why this shape

A normal application is a one-shot artifact. A production AI system is a pipeline with:

- **Evals** — every prompt change runs against a rubric suite before it can land.
- **Observability** — every LLM call and tool call has a trace; the dashboard is public.
- **Grounding** — every factual claim in the output traces back to `facts.json` with a citation.
- **Safety boundaries** — personal data is gitignored; the submitter pauses for human consent before any real form submission.
- **Postmortems** — failed runs get a writeup, not a retry-until-green loop.

The role description for an AI Ops Engineer is "make AI systems reliable in production." This repo is what that looks like applied to itself.

---

## Repo map

```
src/applyops/        # the system
  agents/            # recruiter, writer, critic, factchecker, submitter
  gstack/            # Stack, Layer, ReviewGate — the orchestrator
  evals/             # rubric definitions and scoring
  obs/               # OTel + Langfuse wiring
  cli.py             # `applyops` CLI (Typer)

inputs/              # JD, facts.json, application-data (personal data gitignored)
outputs/             # generated CV / cover letter / form-fill plans
runs/                # per-run traces, eval scores, decision logs (gitignored)
evidence/            # redacted screenshots from form inspection
docs/                # incident doc, runbooks, decision log, retrospective
tests/               # unit + eval suites (eval marker, run separately)
```

---

## Status

This is being built live with frequent commits — the git history is itself part of the deliverable. A reviewer can read it top-to-bottom and watch the system come together.

Current phase: **bootstrap**. See [the task list in the commits](https://github.com/Sahaj0312/aiopsopendoor/commits/main) for what's landed and what's next.

---

## Running it

```bash
# install
make install                    # uv preferred; falls back to pip

# fast path: end-to-end with HITL gate before submission
applyops run --jd inputs/jd.opendoor.md

# stacked workflow: land one layer at a time
applyops layer recruiter
applyops layer writer
applyops critic --on writer     # blocks until rubric passes
applyops layer factcheck
applyops submit --dry-run       # shows form-fill plan, does not submit
```

---

## Safety and honesty

- `facts.json` is the only source of factual claims. Anything that doesn't trace back to it is flagged by the fact-checker and blocks the stack.
- `.env`, personal application data, and ATS confirmation details are gitignored.
- The submitter never auto-submits. The final form requires explicit human consent in the CLI.
- AI drafts, critiques, and automates. Human approves, signs, sends.

---

## Reading order for a reviewer

1. [`docs/00-incident.md`](docs/00-incident.md) — what we're treating as the incident and why.
2. [`docs/01-architecture.md`](docs/01-architecture.md) — gstack model in detail.
3. [`src/applyops/gstack/`](src/applyops/gstack/) — the orchestrator. Small, no framework.
4. [`src/applyops/agents/`](src/applyops/agents/) — agents and their prompts.
5. [`src/applyops/evals/`](src/applyops/evals/) — what "good" means, defined as code.
6. `runs/<latest>/` — the trace, the eval scores, the decision log of the run that produced the submitted application.
7. [`docs/99-retrospective.md`](docs/99-retrospective.md) — what worked, what didn't, what I'd change.
