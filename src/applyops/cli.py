"""applyops CLI — entry point.

Subcommands:
- version
- facts parse / status — manage the candidate's source of truth
- run — execute the full stack end-to-end (recruiter → writer → critic
  → factchecker → submitter)
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from applyops import __version__
from applyops.facts import load

app = typer.Typer(
    name="applyops",
    help="Stacked agent pipeline for AI-first job applications.",
    no_args_is_help=True,
    add_completion=False,
)
facts_app = typer.Typer(
    help="Manage the candidate's source of truth (facts.local.json).",
    no_args_is_help=True,
)
app.add_typer(facts_app, name="facts")

console = Console()


def _load_dotenv_if_present() -> None:
    """Best-effort .env load. Silently ignores absence."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


@app.callback()
def _main() -> None:
    """applyops — application as production AI Ops."""
    _load_dotenv_if_present()


@app.command()
def version() -> None:
    """Print version."""
    console.print(f"applyops [bold]{__version__}[/bold]")


@facts_app.command("parse")
def facts_parse(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    out: Path = typer.Option(
        Path("inputs/facts.local.json"),
        "--out",
        "-o",
        help="Where to write the draft facts file.",
    ),
    model: str = typer.Option(
        os.getenv("APPLYOPS_DEFAULT_MODEL", "gpt-4.1"),
        "--model",
        "-m",
        help="OpenAI model for the parsing pass.",
    ),
) -> None:
    """Parse a resume PDF into a draft facts file. All facts come back unverified."""
    from openai import OpenAI

    from applyops.agents.recruiter import OpenAIStructuredLLM
    from applyops.facts_parser import parse_resume, write_draft

    llm = OpenAIStructuredLLM(OpenAI())
    console.print(f"[dim]parsing[/dim] {pdf} [dim]with[/dim] {model}")
    candidate = parse_resume(pdf, llm=llm, model=model)
    write_draft(candidate, out)
    console.print(
        f"[green]ok[/green] wrote {len(candidate.facts)} facts to {out}; "
        f"all marked [yellow]ai_extracted_unverified[/yellow]. "
        "Review and flip provenance to `self` before running the writer."
    )


@facts_app.command("status")
def facts_status(
    path: Path = typer.Argument(Path("inputs/facts.local.json"), exists=True),
) -> None:
    """Show verification status of facts in a facts file."""
    candidate = load(path)
    table = Table(title=f"{candidate.name} — {len(candidate.facts)} facts")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("kind")
    table.add_column("verified", justify="center")
    table.add_column("title")
    for fact in candidate.facts:
        verified_mark = "[green]yes[/green]" if fact.verified else "[red]no[/red]"
        table.add_row(fact.id, fact.kind, verified_mark, fact.title)
    console.print(table)

    n_unverified = len(candidate.unverified())
    if n_unverified == 0:
        console.print("[green]all facts attested.[/green]")
    else:
        console.print(
            f"[yellow]{n_unverified} fact(s) still unverified.[/yellow] "
            "Edit the file and set `verified_by: self` on entries you attest."
        )


@app.command(name="eval")
def eval_cmd(
    case: str = typer.Option(
        "all",
        "--case",
        help="Which fixture to grade: good | bad_coverage | overconcentrated | all.",
    ),
) -> None:
    """Run the writer rubric against fixture cases and print scorecards."""
    from applyops.agents.recruiter import RoleAnalysis
    from applyops.agents.writer import WriterOutput
    from applyops.evals import (
        Rubric,
        fact_concentration,
        grounding_density,
        jd_coverage_score,
        load_fixture,
        tone_drift_count,
    )
    from applyops.evals.rubrics import RubricMetric, grade
    from applyops.evals.scorers import cover_letter_addresses_protocol

    rubric = Rubric(
        name="writer-output-rubric-v1",
        metrics=[
            RubricMetric(
                name="jd_coverage_high_importance",
                scorer=jd_coverage_score,
                threshold=0.75,
                direction=">=",
                needs=["writer_output", "role_analysis"],
            ),
            RubricMetric(name="grounding_density", scorer=grounding_density, threshold=1.0),
            RubricMetric(
                name="fact_concentration", scorer=fact_concentration, threshold=4, direction="<="
            ),
            RubricMetric(name="tone_drift", scorer=tone_drift_count, threshold=0, direction="<="),
            RubricMetric(
                name="protocol_addressed",
                scorer=cover_letter_addresses_protocol,
                threshold=1.0,
                needs=["writer_output", "role_analysis"],
            ),
        ],
    )

    cases = ["good", "bad_coverage", "overconcentrated"] if case == "all" else [case]
    role = load_fixture("role_analysis.opendoor.json", RoleAnalysis)

    overall_passed = True
    for name in cases:
        wo = load_fixture(f"writer_output.{name}.json", WriterOutput)
        card = grade(rubric, case_name=name, writer_output=wo, role_analysis=role)
        table = Table(title=f"case: {name}")
        table.add_column("metric", style="cyan")
        table.add_column("value", justify="right")
        table.add_column("threshold", justify="right")
        table.add_column("result", justify="center")
        for s in card.scores:
            verdict = "[green]pass[/green]" if s.passed else "[red]fail[/red]"
            table.add_row(s.metric, f"{s.value:.2f}", f"{s.direction} {s.threshold}", verdict)
        console.print(table)
        if not card.passed:
            overall_passed = False
            console.print(f"  [red]case {name} failed[/red]")
        else:
            console.print(f"  [green]case {name} passed[/green]")
        console.print()

    if not overall_passed:
        raise typer.Exit(code=1)


@app.command()
def submit(
    run_id: str = typer.Argument(
        ...,
        help="Run directory name under --output-root (e.g. run_20260513T193000Z_abc123).",
    ),
    output_root: Path = typer.Option(
        Path("outputs"),
        "--output-root",
        envvar="APPLYOPS_OUTPUT_ROOT",
    ),
    model: str = typer.Option(
        os.getenv("APPLYOPS_DEFAULT_MODEL", "gpt-4.1"),
        "--model",
        "-m",
        help="OpenAI model for the LLM field-mapping pass.",
    ),
    headless: bool = typer.Option(
        False,
        "--headless/--headed",
        help="Run the browser headless. Default headed so the human can watch the form fill.",
    ),
) -> None:
    """Drive the ATS form for a completed run. Pauses for explicit human SUBMIT."""
    from openai import OpenAI

    from applyops.agents.recruiter import OpenAIStructuredLLM
    from applyops.obs import setup_tracing
    from applyops.submit import submit as run_submit

    setup_tracing()
    run_dir = output_root / run_id
    if not run_dir.exists():
        console.print(f"[red]error[/red]: run directory not found: {run_dir}")
        raise typer.Exit(code=2)

    llm = OpenAIStructuredLLM(OpenAI())
    console.print(f"[bold]applyops submit[/bold] — {run_dir}")
    record = run_submit(run_dir, llm=llm, model=model, headless=headless, console=console)

    color = {
        "submitted": "green",
        "cancelled_by_human": "yellow",
        "blocked_no_target_url": "red",
        "error": "red",
    }[record.outcome]
    console.print(f"\n[{color}]outcome: {record.outcome}[/{color}]")
    console.print(f"  fields filled:  {record.fields_filled}")
    console.print(f"  fields skipped: {record.fields_skipped}")
    if record.submit_url_after:
        console.print(f"  url after:      {record.submit_url_after}")
    if record.error:
        console.print(f"  error:          {record.error}")


@app.command()
def run(
    jd_url: str = typer.Option(
        None,
        "--jd-url",
        envvar="APPLYOPS_JD_URL",
        help="ATS URL to fetch the JD from. Mutually exclusive with --jd-file.",
    ),
    jd_file: Path = typer.Option(
        None,
        "--jd-file",
        exists=True,
        help="Local JD markdown file. Used instead of --jd-url for offline runs.",
    ),
    facts: Path = typer.Option(
        Path("inputs/facts.local.json"),
        "--facts",
        exists=True,
        envvar="APPLYOPS_FACTS_PATH",
        help="Path to the candidate's facts file.",
    ),
    candidate_email: str = typer.Option(
        ...,
        "--email",
        envvar="APPLYOPS_CANDIDATE_EMAIL",
        help="Email to put on the application.",
    ),
    candidate_phone: str = typer.Option(
        None,
        "--phone",
        envvar="APPLYOPS_CANDIDATE_PHONE",
        help="Phone (optional).",
    ),
    output_root: Path = typer.Option(
        Path("outputs"),
        "--output-root",
        envvar="APPLYOPS_OUTPUT_ROOT",
    ),
    snapshot_dir: Path = typer.Option(
        Path("inputs"),
        "--snapshot-dir",
        help="Where the recruiter snapshots fetched JDs.",
    ),
    target_url_override: str = typer.Option(
        None,
        "--target-url",
        help="Override the form target URL (used in form_plan.json).",
    ),
    recruiter_model: str = typer.Option(
        os.getenv("APPLYOPS_RECRUITER_MODEL", "gpt-4.1-mini"),
        "--recruiter-model",
    ),
    writer_model: str = typer.Option(
        os.getenv("APPLYOPS_DEFAULT_MODEL", "gpt-4.1"),
        "--writer-model",
    ),
    critic_model: str = typer.Option(
        os.getenv("APPLYOPS_CRITIC_MODEL", "gpt-4.1-mini"),
        "--critic-model",
    ),
    factcheck_model: str = typer.Option(
        os.getenv("APPLYOPS_FACTCHECK_MODEL", "gpt-4.1"),
        "--factcheck-model",
    ),
    max_rebases: int = typer.Option(
        3,
        "--max-rebases",
        help="Per-gate rebase budget before the run blocks.",
    ),
    render_jd: bool = typer.Option(
        False,
        "--render-jd/--no-render-jd",
        help=(
            "Use a headless browser to fetch the JD. Required for JS-rendered ATS "
            "pages (Rippling, Greenhouse, Lever). Needs the `submit` extras and "
            "`playwright install chromium`."
        ),
    ),
    jd_wait_selector: str = typer.Option(
        None,
        "--jd-wait-selector",
        help="CSS selector to wait for after page load (rendered fetch only).",
    ),
) -> None:
    """Run the full applyops pipeline end-to-end."""
    from applyops.runner import RunConfig, execute

    if jd_url is None and jd_file is None:
        console.print(
            "[red]error[/red]: provide --jd-url or --jd-file (or set APPLYOPS_JD_URL in .env)"
        )
        raise typer.Exit(code=2)

    cfg = RunConfig(
        jd_url=jd_url,
        jd_file=jd_file,
        facts_path=facts,
        output_root=output_root,
        snapshot_dir=snapshot_dir,
        candidate_email=candidate_email,
        candidate_phone=candidate_phone,
        target_url_override=target_url_override,
        recruiter_model=recruiter_model,
        writer_model=writer_model,
        critic_model=critic_model,
        factcheck_model=factcheck_model,
        max_rebases_per_gate=max_rebases,
        render_jd=render_jd,
        jd_wait_selector=jd_wait_selector,
    )

    from applyops.obs import setup_tracing

    tracing_active = setup_tracing()
    console.print("[bold]applyops[/bold] running…")
    console.print(f"  facts:   {facts}")
    console.print(f"  jd:      {jd_url or jd_file}")
    console.print(f"  out:     {output_root}/")
    console.print(
        f"  tracing: [{'green' if tracing_active else 'yellow'}]"
        f"{'active (Langfuse)' if tracing_active else 'disabled (no LANGFUSE_* keys)'}"
        f"[/{'green' if tracing_active else 'yellow'}]"
    )
    console.print()

    run_record, ctx = execute(cfg)
    run_record.persist(output_root)
    _print_summary(run_record, ctx)


def _print_summary(run_record, ctx) -> None:  # type: ignore[no-untyped-def]
    """Render a compact end-of-run summary."""
    from applyops.gstack.run import RunStatus

    status_color = {
        RunStatus.COMPLETED: "green",
        RunStatus.PARTIAL: "yellow",
        RunStatus.BLOCKED: "red",
        RunStatus.FAILED: "red",
        RunStatus.RUNNING: "yellow",
    }[run_record.status]
    console.print(
        f"[{status_color}]run {run_record.id}: {run_record.status.value}[/{status_color}]"
    )
    if run_record.blocked_on:
        console.print(f"  blocked_on: {run_record.blocked_on}")
    if run_record.error:
        console.print(f"  error: {run_record.error}")
    table = Table(title="Layers")
    table.add_column("layer", style="cyan")
    table.add_column("ran", justify="center")
    table.add_column("rebases", justify="right")
    table.add_column("gate verdict")
    for name, state in ctx.layers.items():
        ran = "[green]yes[/green]" if state.output is not None else "[red]no[/red]"
        gate = "—"
        if state.gate_reviews:
            last = state.gate_reviews[-1]
            gate = "[green]pass[/green]" if last.passed else "[red]fail[/red]"
        table.add_row(name, ran, str(state.rebases), gate)
    console.print(table)

    if "submitter" in ctx.layers and ctx.layers["submitter"].output is not None:
        sub = ctx.layers["submitter"].output
        console.print(f"\n[bold]artifacts[/bold] written to {sub.output_dir}/")
        console.print(f"  cv:    {sub.cv_md_path}")
        console.print(f"  cover: {sub.cover_md_path}")
        console.print(f"  plan:  {sub.form_plan_path}")
        console.print(f"  audit: {sub.audit_md_path}")
    elif run_record.notes:
        console.print("\n[bold]run notes[/bold]")
        for note in run_record.notes:
            console.print(f"  - {note}")


if __name__ == "__main__":
    app()
