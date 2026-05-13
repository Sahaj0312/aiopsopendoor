# Decision log

Append-only. Each entry: date, decision, alternatives considered, reasoning.

---

## 2026-05-13 — Model split per agent

**Decision:** Use `gpt-4.1` for the writer and fact-checker, `gpt-4.1-mini` for the critic and recruiter. Models wired through env vars (`APPLYOPS_*_MODEL`) so the split is overridable per run.

**Alternatives considered:**
- Single model everywhere. Simpler, but ~3× the cost on critic-heavy runs (the critic is called on every writer rebase).
- gpt-5 across the board if accessible. Deferred until we have a baseline — can't tell if the gains justify cost without runs to compare.

**Reasoning:**
- The **writer** produces grounded prose under tight constraints. Quality is the entire output, so it gets the strong model.
- The **fact-checker** does structured citation lookup and exaggeration detection. Failures here are P0 (a hallucinated claim reaches the form). Strong model, conservative prompt.
- The **critic** runs a rubric and emits a structured score. This is a classification-shaped task and the mini model handles it well. Critic runs on every writer rebase, so cost matters.
- The **recruiter** parses a JD into a structured `RoleAnalysis`. Structured extraction is something the mini model is good at. Runs once per JD, so cost is low either way, but the mini is the right shape.
- The **submitter** doesn't run an LLM as a primary step — it produces a deterministic form-fill plan from prior outputs. No model assigned.

**How to revisit:** After the first 10 runs, look at the eval scores grouped by model and decide whether any agent should be upgraded. If the critic starts under-blocking (false negatives), promote it to the larger model before the rubric, not after.

---

## 2026-05-13 — Live JD fetch with snapshot + diff

**Decision:** The recruiter agent fetches the JD live from `ats.rippling.com/en-CA/opendoor` on each run, hashes the content, and writes a snapshot to `inputs/jd.opendoor.<hash>.md`. If the hash differs from the prior snapshot, the stack flags a JD drift event and marks all downstream layers stale.

**Alternatives considered:**
- Hand-paste the JD into `inputs/jd.opendoor.md`. Simpler, no network dep at run time, but loses drift detection and looks less production-like.
- Fetch live every time, no snapshot. Loses reviewability — the JD that produced the application is not in the repo.

**Reasoning:**
- Production AI Ops systems treat their inputs as data that can change. Snapshotting + diffing is how you avoid silent input drift, which is the single most common failure mode in deployed agent pipelines.
- The snapshot in git means a reviewer can read the exact JD the application was built against, even after the role posting changes or is taken down.
- The drift event is a real signal — if Opendoor edits the JD mid-build, the pipeline should re-derive, not produce a stale application.

**How to revisit:** If Rippling blocks the fetch or the JD URL changes, fall back to manual snapshot. The drift detection still works on hand-pasted updates.

---

## 2026-05-13 — gstack model over LangGraph / Pydantic AI

**Decision:** Hand-roll a small orchestrator (`src/applyops/gstack/`) modeled on stacked PRs. No agent framework dependency.

**Alternatives considered:**
- LangGraph. Mature, expressive, but the graph abstraction is heavier than this problem needs and adds dependency surface.
- Pydantic AI. Lightweight and pleasant, but couples the agent definition to the framework's tool/output model. Future-portability suffers.
- OpenAI Agents SDK. Tight integration with the provider we're already using, but lock-in.

**Reasoning:**
- Five sequential agents do not need a graph. They need a stack with a review-gate primitive, which is ~150 lines of Python.
- "I can build the orchestration myself" is a hiring signal for an AI Ops role. Reaching for a framework on a 5-agent pipeline reads as not-yet-senior.
- We can still use Pydantic for *data models* (RoleAnalysis, CVDraft, Review) without using Pydantic AI for the orchestration. Best of both.

**How to revisit:** If the stack ever needs cross-layer learning, conditional branching beyond pass/request-changes, or parallelism within a run, that's the signal to graduate to a real graph framework — not before.
