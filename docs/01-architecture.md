# Architecture — the gstack model

The agent pipeline is modeled as a stack of layers, in the style of [Graphite](https://graphite.dev)'s stacked PRs. Each layer is a small, focused agent. Each layer's output is a reviewable diff that the next layer builds on. A **review gate** sits between consecutive layers and can block the stack until its rubric passes or a human overrides.

This document is the design rationale. The code is in [`src/applyops/gstack/`](../src/applyops/gstack/) and [`src/applyops/agents/`](../src/applyops/agents/).

## Why a stack and not a graph or a chain

- **A chain** (linear pipeline) is what most agent demos look like. It hides the review structure — you only see the final output, not the per-layer diffs. When something is wrong it's hard to localize.
- **A graph** (DAG with fan-out / fan-in) is what production agent frameworks reach for. It's expressive but expensive: cycle handling, partial-failure semantics, and re-execution policy all become first-class concerns. Overkill for a 5-agent pipeline.
- **A stack** is the right shape for this problem. Layers are totally ordered. Each layer transforms the state and emits a diff. Review gates are local — they only inspect the layer immediately below them. Rebases cascade in one direction. The semantics fit on a napkin.

The stack is also the *right rhetorical shape* for a code reviewer: it reads like a series of dependent PRs. Each layer is a unit you can `land` independently, audit, or roll back.

## Primitives

```python
class Layer(Protocol):
    name: str
    def run(self, ctx: StackContext) -> LayerOutput: ...

class ReviewGate(Protocol):
    name: str
    def review(self, output: LayerOutput, ctx: StackContext) -> Review: ...

class Stack:
    layers: list[Layer]
    gates: dict[str, ReviewGate]  # gate-after-layer-name
    def land(self, up_to: str | None = None) -> Run: ...
    def rebase(self, from_layer: str) -> Run: ...
```

A `Run` is the per-execution record: inputs, every layer's output, every gate's review, every LLM/tool span, and the final state. Runs are append-only; nothing is mutated after a layer lands.

## The five layers

| Layer | Input | Output | Why this is its own layer |
|-------|-------|--------|---------------------------|
| **recruiter** | JD (text or URL) | `RoleAnalysis`, `EvidenceMap` | The JD analysis is the source of truth the writer is graded against. Keeping it separate lets us re-run the writer against an updated analysis without re-doing the writer's work from scratch — the diff is local. |
| **writer** | `RoleAnalysis`, `EvidenceMap`, `facts.json` | `CVDraft`, `CoverLetter` | The writer is *only* a writer — no fact-checking, no rubric scoring. Tight scope = better prompt = better output. |
| **critic** *(gate)* | writer output | `Review` (pass / request-changes) | Critic is a gate, not a layer. It does not produce content. It scores against a rubric and either passes the stack or sends the writer a request-changes diff. Implemented as an LLM call with a structured-output schema. |
| **factchecker** | writer output, `facts.json` | `ClaimCitations`, `Flags` | Every factual claim in the CV/cover gets a citation back to `facts.json`. Unsourceable claims are flagged. This layer is conservative — if it can't find a citation, it asks the writer to weaken the claim, it does not invent one. |
| **submitter** | finalized outputs, ATS form schema | `FormFillPlan`, `DryRunArtifacts` | The submitter never auto-submits. It produces a plan: rendered fields, file uploads, screenshots from a headed Playwright dry run. The human reads the plan and types `yes` to send. |

## The critic loop and rebase

When the critic returns `request-changes` on the writer's output, the stack does **not** continue downstream. Instead:

1. The critic's review is attached to the writer's layer output as a `RebaseRequest`.
2. The writer re-runs with the original inputs *plus* the rebase request as additional context.
3. The new writer output replaces the old one. The critic re-runs.
4. If the critic passes, the stack continues. If it fails again, the rebase counter increments. After N rebases (configurable, default 3), the stack halts and asks for human intervention.

This is exactly the Graphite "land blocked on review" → "push fixup" → "re-request review" loop. The cap on rebases is the equivalent of "if you can't pass code review in 3 rounds, escalate."

## Cascading rebase

If a layer below the writer changes (e.g., the recruiter re-runs with a new JD), every layer above it is marked **stale**. Stale layers are not re-run automatically; the CLI shows them and the human chooses what to land. This is intentional — silent re-execution is how AI pipelines burn through tokens and end up with mysterious output drift.

## What lives outside the stack

- **Evals.** Eval rubrics live in `src/applyops/evals/` and are run by the critic gate *and* by CI on every prompt change. The critic uses a subset (fast); CI runs the full suite (slow, costlier).
- **Observability.** Every LLM call and tool invocation is wrapped in an OTel span. Spans carry `layer.name`, `gate.name`, `rebase.count`, and `run.id` attributes. The trace tree mirrors the stack.
- **Decision log.** Free-form human-written notes on why a particular layer was re-run, why a critic override was used, etc. Lives in `runs/<run-id>/decisions.md`. Not generated.

## What this design buys us

- **Localized failure.** If the cover letter is bad, the diff is on the writer layer. The recruiter's evidence map is untouched.
- **Independent re-runs.** We can iterate on the writer's prompt by replaying the writer layer against a frozen recruiter output.
- **Honest grounding.** The fact-checker is structurally downstream of the writer and structurally upstream of the submitter — so nothing ungrounded can reach the form.
- **A reviewer-friendly artifact.** The trace of a run is a sequence of small, named, scoped diffs. A hiring reviewer can read it without context.

## What this design intentionally gives up

- **Parallelism within a run.** Layers are sequential. We could parallelize the writer's CV and cover letter sub-tasks, but that complicates the diff model and is not worth it at this scale.
- **Cross-layer learning.** The factchecker does not back-propagate findings to the recruiter. If the recruiter's evidence map was wrong, the human fixes it and re-runs. We are not building an end-to-end trainable system.

These are the right tradeoffs for an application pipeline. They might not be the right tradeoffs for a different workload, and that's fine — `gstack` is not a framework. It's a small orchestrator that fits this problem.
