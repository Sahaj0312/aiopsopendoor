"""Markdown → PDF rendering via Playwright.

ATS systems want a PDF resume, not markdown. The submitter writes
cv.md / cover.md as the source of truth; this module renders matching
cv.pdf / cover.pdf for upload. No additional PDF library — we already
ship Playwright for the form-fill flow.

The styling is deliberately minimal: a single sans-serif typeface, A4
geometry, generous margins, clean section headings. The point is that
the PDF be cleanly readable, not novel.
"""

from __future__ import annotations

from pathlib import Path

import markdown as md_lib

_CSS = """
@page {
  size: Letter;
  margin: 0.75in 0.85in;
}
* { box-sizing: border-box; }
html, body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #111;
}
h1 {
  font-size: 22pt;
  margin: 0 0 4pt 0;
  font-weight: 600;
  letter-spacing: -0.01em;
}
h2 {
  font-size: 12pt;
  margin: 14pt 0 4pt 0;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 0.5pt solid #ccc;
  padding-bottom: 2pt;
}
h3 { font-size: 11pt; margin: 8pt 0 2pt 0; font-weight: 600; }
p, ul, ol { margin: 4pt 0; }
ul { padding-left: 18pt; }
li { margin: 1pt 0; }
blockquote {
  margin: 6pt 0;
  padding: 0 0 0 8pt;
  border-left: 2pt solid #999;
  color: #444;
  font-style: italic;
}
a { color: inherit; text-decoration: underline; }
strong { font-weight: 600; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 9.5pt;
  background: #f3f3f3;
  padding: 1pt 3pt;
  border-radius: 2pt;
}
"""


def _wrap_html(body_html: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>{body_html}</body>
</html>"""


def markdown_to_pdf(markdown_text: str, out_path: str | Path, *, title: str = "") -> Path:
    """Render `markdown_text` to a PDF at `out_path`.

    Returns the resolved Path. Raises if Playwright isn't available.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "markdown_to_pdf requires the `submit` extras. "
            "Run `pip install -e '.[submit]'` and `playwright install chromium`."
        ) from exc

    body_html = md_lib.markdown(
        markdown_text,
        extensions=["extra", "sane_lists", "smarty"],
    )
    html = _wrap_html(body_html, title or "Document")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            page.pdf(path=str(out_path), format="Letter", print_background=True)
        finally:
            browser.close()

    return out_path
