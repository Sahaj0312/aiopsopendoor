"""Sources for the job description.

Two implementations:
- `HttpJDSource` fetches the JD live, converts HTML to clean markdown, and
  snapshots it under inputs/. Diffs against the prior LATEST snapshot to
  detect drift.
- `FileJDSource` reads a markdown file from disk. Useful for tests and for
  reviewers reading the repo without network access.

Both return `(text, JDMeta)`. The recruiter agent doesn't care which one
it got.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_md

from applyops.agents.types import JDMeta


class JDSource(Protocol):
    """Anything that can produce a JD as cleaned markdown plus provenance."""

    def fetch(self) -> tuple[str, JDMeta]: ...


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _clean_html_to_md(html: str) -> str:
    """Strip scripts/styles/nav and convert the main content to markdown.

    Conservative: we don't try to pinpoint a `main` element across every ATS
    layout. We strip the obvious non-content tags and let markdownify produce
    a flat representation. The LLM is robust to extra whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ("script", "style", "nav", "footer", "header", "noscript", "svg"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    md: str = html_to_md(str(soup), heading_style="ATX")
    # Collapse runs of blank lines so the prompt stays tight.
    lines = [line.rstrip() for line in md.splitlines()]
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip() + "\n"


class HttpJDSource:
    """Fetch a JD over HTTP, clean it, snapshot it, detect drift."""

    def __init__(
        self,
        url: str,
        snapshot_dir: str | Path = "inputs",
        slug: str = "jd",
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.snapshot_dir = Path(snapshot_dir)
        self.slug = slug
        self.timeout = timeout
        self._client = client

    def _http_get(self) -> str:
        if self._client is not None:
            r = self._client.get(self.url, timeout=self.timeout)
        else:
            r = httpx.get(self.url, timeout=self.timeout, follow_redirects=True)
        r.raise_for_status()
        return r.text

    def fetch(self) -> tuple[str, JDMeta]:
        html = self._http_get()
        text = _clean_html_to_md(html)
        h = _content_hash(text)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.snapshot_dir / f"{self.slug}.{h}.md"
        if not snapshot_path.exists():
            snapshot_path.write_text(text, encoding="utf-8")

        latest_marker = self.snapshot_dir / f"{self.slug}.LATEST"
        prior_hash: str | None = None
        if latest_marker.exists():
            prior_hash = latest_marker.read_text(encoding="utf-8").strip() or None
        drift = prior_hash is not None and prior_hash != h
        latest_marker.write_text(h, encoding="utf-8")

        return text, JDMeta(
            url=self.url,
            hash=h,
            snapshot_path=str(snapshot_path),
            drift=drift,
        )


class FileJDSource:
    """Load a JD from a local markdown file. Hash is computed; drift always False."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self) -> tuple[str, JDMeta]:
        text = self.path.read_text(encoding="utf-8")
        h = _content_hash(text)
        return text, JDMeta(
            url=None,
            hash=h,
            snapshot_path=str(self.path),
            drift=False,
        )
