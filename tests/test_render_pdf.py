"""Tests for markdown → PDF rendering.

Real Playwright run; verifies the PDF is produced, has a sensible size,
and contains the expected text when re-extracted with pypdf.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("playwright")

from applyops.render import markdown_to_pdf


SAMPLE_MD = """# Sahaj Chhabra

> Engineer shipping production AI on video and image pipelines.

## Experience

**Software Engineer, Quickplay — Toronto** — 2025 – present
- Shipped a CV + LLM thumbnail pipeline for the AMG CMS.
- Verticalization API: 16:9 → 9:16 with GPU-optimized YOLO.

## Skills

Python, Go, LLMs, RAG, computer vision, observability.
"""


def test_markdown_to_pdf_writes_a_readable_pdf(tmp_path: Path) -> None:
    out = tmp_path / "cv.pdf"
    result = markdown_to_pdf(SAMPLE_MD, out, title="Test CV")
    assert result == out
    assert out.exists()
    # Sanity: PDFs are non-trivial in size — under 1KB means rendering failed silently.
    assert out.stat().st_size > 1024

    from pypdf import PdfReader

    reader = PdfReader(str(out))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Sahaj Chhabra" in text
    assert "Quickplay" in text
    assert "verticalization" in text.lower()


def test_markdown_to_pdf_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "doc.pdf"
    markdown_to_pdf("# hi", nested)
    assert nested.exists()
