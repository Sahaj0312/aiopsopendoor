"""Smoke tests — the package imports and the CLI runs."""

from __future__ import annotations

from typer.testing import CliRunner

from applyops import __version__
from applyops.cli import app


def test_version_is_set() -> None:
    assert __version__


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_facts_status_on_example_file() -> None:
    result = CliRunner().invoke(app, ["facts", "status", "inputs/facts.example.json"])
    assert result.exit_code == 0
    assert "Ada Example" in result.stdout
    assert "all facts attested" in result.stdout
