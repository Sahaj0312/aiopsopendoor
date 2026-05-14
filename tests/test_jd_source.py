"""Tests for the JD sources — drift detection and HTML cleaning."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from applyops.agents.jd_source import FileJDSource, HttpJDSource, _clean_html_to_md

FIXTURE = Path(__file__).parent / "fixtures" / "jd.fake.md"


def test_file_source_round_trip() -> None:
    text, meta = FileJDSource(FIXTURE).fetch()
    assert "AI Ops Engineer" in text
    assert meta.url is None
    assert len(meta.hash) == 12
    assert meta.drift is False


def test_clean_html_strips_scripts_and_converts_to_md() -> None:
    html = """
    <html><head><style>x{}</style><script>alert(1)</script></head>
    <body>
      <nav>nav stuff</nav>
      <h1>AI Ops Engineer</h1>
      <p>We need <strong>Python</strong>.</p>
      <footer>company &copy; 2026</footer>
    </body></html>
    """
    md = _clean_html_to_md(html)
    assert "alert(1)" not in md
    assert "nav stuff" not in md
    assert "company" not in md
    assert "AI Ops Engineer" in md
    assert "Python" in md


def test_http_source_snapshots_and_detects_drift(tmp_path: Path) -> None:
    pages = iter(
        [
            "<html><body><h1>Role A</h1><p>v1</p></body></html>",
            "<html><body><h1>Role A</h1><p>v2 — edited</p></body></html>",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=next(pages))

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        src = HttpJDSource(
            "https://example.test/jd",
            snapshot_dir=tmp_path,
            slug="jd",
            client=client,
        )
        _, m1 = src.fetch()
        _, m2 = src.fetch()

    assert m1.drift is False  # first fetch ever, no prior LATEST
    assert m2.drift is True
    assert m1.hash != m2.hash
    snapshots = sorted(tmp_path.glob("jd.*.md"))
    assert len(snapshots) == 2  # one per unique hash
    assert (tmp_path / "jd.LATEST").read_text().strip() == m2.hash


def test_http_source_no_drift_on_identical_content(tmp_path: Path) -> None:
    body = "<html><body><h1>Stable</h1></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        src = HttpJDSource(
            "https://example.test/jd",
            snapshot_dir=tmp_path,
            slug="jd",
            client=client,
        )
        _, m1 = src.fetch()
        _, m2 = src.fetch()

    assert m1.hash == m2.hash
    assert m2.drift is False


def test_http_source_raises_on_http_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        src = HttpJDSource("https://example.test/jd", snapshot_dir=tmp_path, client=client)
        with pytest.raises(httpx.HTTPStatusError):
            src.fetch()
