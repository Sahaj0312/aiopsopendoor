"""applyops CLI — entry point. Real commands wired up in later layers."""

from __future__ import annotations

import typer
from rich.console import Console

from applyops import __version__

app = typer.Typer(
    name="applyops",
    help="Stacked agent pipeline for AI-first job applications.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print version."""
    console.print(f"applyops [bold]{__version__}[/bold]")


if __name__ == "__main__":
    app()
