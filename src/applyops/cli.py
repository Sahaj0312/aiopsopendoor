"""applyops CLI — entry point.

Subcommands grow as the system gains capability. Today:
- version
- facts parse — PDF → facts.local.json draft (all unverified)
- facts status — show what's verified vs not in a facts file
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


@app.callback()
def _main() -> None:
    """applyops — application as production AI Ops."""


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
    # Lazy import so `applyops version` doesn't pay the openai-client init cost.
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


if __name__ == "__main__":
    app()
