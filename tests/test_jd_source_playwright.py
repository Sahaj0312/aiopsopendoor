"""Integration tests for PlaywrightJDSource.

These tests launch a real Chromium and drive it against a local file:// URL
that simulates a JS-rendered ATS page. No network. Skipped automatically
if Playwright isn't installed (the `submit` extras aren't required for
unit tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from applyops.agents.jd_source import PlaywrightJDSource

pytest.importorskip("playwright")

FIXTURE = Path(__file__).parent / "fixtures" / "fake_ats_jd.html"


def test_playwright_source_extracts_js_rendered_content(tmp_path: Path) -> None:
    """A bare httpx.get on this fixture would only see 'Loading...'.

    Playwright must wait for the inject and capture the real JD.
    """
    url = f"file://{FIXTURE.resolve()}"
    src = PlaywrightJDSource(
        url,
        snapshot_dir=tmp_path,
        slug="jd",
        wait_for_selector="article[data-testid='job-posting']",
    )
    text, meta = src.fetch()

    assert "AI Ops Engineer" in text
    assert "Strong Python" in text
    assert "Production LLM/ML experience" in text
    # The pre-render placeholder should not survive into the cleaned output.
    assert "Loading..." not in text
    assert meta.url == url
    assert meta.drift is False  # first fetch
    assert (tmp_path / f"jd.{meta.hash}.md").exists()


def test_playwright_source_detects_drift(tmp_path: Path) -> None:
    """Re-fetching the SAME URL with no content change must not flip drift."""
    url = f"file://{FIXTURE.resolve()}"
    src = PlaywrightJDSource(
        url,
        snapshot_dir=tmp_path,
        slug="jd",
        wait_for_selector="article[data-testid='job-posting']",
    )
    _, m1 = src.fetch()
    _, m2 = src.fetch()
    assert m1.hash == m2.hash
    assert m2.drift is False
