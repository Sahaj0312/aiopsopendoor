# INC-2026-05-13-001 — Opendoor AI Ops role open, candidate unhired

**Status:** open
**Severity:** 2 (career-impacting, time-bound)
**Opened:** 2026-05-13 19:01 PT
**Commander:** Sahaj (also: affected user, sole on-call)
**Detection:** [Public tweet from @nejatian](https://x.com/nejatian) announcing the role with an unusual application protocol — "apply ONLY using AI… extra points for creativity."

## Summary

Opendoor opened an AI Ops Engineer role in Toronto with an application protocol that turns the application channel itself into a hiring signal. The standard application path (write a tailored resume, fill the ATS form, hope) is dominated in expected value by **building a small AI Ops system that produces and submits the application** and showing the system to the reviewer.

This document treats that build as a Sev-2 incident response with explicit SLOs, error budget, and runbook discipline. The rest of this repo is the runbook in action.

## SLOs

| SLO | Target | Why |
|-----|--------|-----|
| Time to first interview | ≤ 21 days from incident open | Toronto market is hot for AI Ops; openings close fast. |
| Application materials grounded | 100% of factual claims trace to `facts.json` | One hallucinated claim caught in interview is a P0. |
| Eval-gated prompt changes | 100% pass rubric before merge | Drift in writer/critic prompts silently degrades all future runs. |
| Human consent before send | 100% | Auto-submit a flawed application = lost shot, no rollback. |

## Error budget

- 2 application rejections allowed before the strategy is reviewed (not the candidate).
- 1 hallucinated claim allowed total. After that, the fact-checker gets tighter rubrics.
- Stack rebuild count is unbudgeted — rebuilds are healthy.

## Affected systems

- **Candidate.career** — degraded availability; cannot serve current preferred role until incident closes.
- **applyops/** — under active build; will graduate from "pre-production" to "production" the moment it submits a real application.

## Mitigation strategy

1. Build the system before writing the application. Forces the application to be an output, not the goal.
2. Ground every claim. The fact-checker is the only thing standing between an honest application and one that gets caught at reference-check time.
3. Run rubric evals on every prompt change. Treat prompt regressions like code regressions.
4. Ship traces. The hiring reviewer should be able to read the trace of the run that produced the submitted application.
5. Keep the human on the final gate. The submitter assembles the form-fill plan; a human reviews the rendered fields and types `yes` before anything is sent.

## What we are deliberately *not* doing

- **Not optimizing for surface polish.** A clean LaTeX resume from a prompt is table stakes. The differentiator is the system, not the PDF.
- **Not auto-submitting.** Even if the form-fill works on the dry run, a human approves the final state. Lost shots have no rollback.
- **Not hiding the failures.** The decision log and retrospective record the dead ends. AI Ops is honest about what didn't work.
- **Not stuffing keywords for ATS.** The fact-checker actively penalizes claim-padding. Real AI Ops engineers can tell.

## Open questions

- Final model selection per agent (cost vs. quality tradeoff for the critic specifically).
- Whether the submitter does Playwright form-fill or just generates a copy-paste plan. (Defaulting to plan until we confirm Rippling's ToS.)
- How much of `facts.json` lives in git vs. a local-only file. (Default: schema + redacted example in git, real facts gitignored.)

## Postmortem

Will be written in [`docs/99-retrospective.md`](99-retrospective.md) after the incident closes (offer accepted, offer declined, or 60 days without progress).
