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
