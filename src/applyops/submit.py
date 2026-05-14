"""applyops.submit — drive a headed browser through an ATS form, with a HITL gate.

This is intentionally NOT a gstack Layer. The pipeline produces a
form_plan.json + cv.pdf + cover.md / cover.pdf and stops. The user (or
this module, called via `applyops submit <run-id>`) takes those
artifacts and drives the ATS submission as a separate step. The
separation matters: a form submission is irreversible and deserves its
own command, its own confirmation, and its own audit record.

Flow:

1. Load form_plan.json from outputs/<run-id>/.
2. Launch a HEADED Chromium so the human can see what's happening.
3. Navigate to form_plan.target_url.
4. Snapshot the accessibility tree.
5. Call the LLM with (form_plan.fields + a11y tree) → SubmitFieldMap
   that maps each abstract field to a Playwright role+name locator.
6. Execute the fill on each mapped field. Skip unmapped ones with a
   note in the audit.
7. Take a "before-submit" screenshot to outputs/<run-id>/submit.before.png.
8. Print a HUMAN-REVIEW block in the terminal. Wait for the user to
   type exactly `SUBMIT` (case-sensitive). Any other input cancels.
9. On SUBMIT: locate the submit button via the same LLM-provided
   locator, click it, wait briefly, capture submit.after.png, and
   record outputs/<run-id>/submission.json with the final URL and a
   pass/fail flag.
10. On cancel: keep the browser open with the filled state so the
    human can finish manually. Record cancellation in submission.json.

No state in this module survives the function call. Reads from
outputs/<run-id>/, writes to outputs/<run-id>/, exits.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from applyops.agents.submitter import FormFillPlan

SUBMIT_FIELD_MAP_SYSTEM_PROMPT = """You are mapping abstract application-form fields to a concrete ATS page.

You receive:
- `field_plan`: a list of fields the candidate wants to fill. Each entry has a logical `name` (e.g. "full_name"), a human-readable `label` (e.g. "Full name"), a `value` to enter, and a `kind` (text / textarea / url / file).
- `accessibility_tree_yaml`: the rendered ARIA structure of the actual page (Playwright's aria_snapshot YAML format), showing roles, accessible names, and descendants.

Your job is to produce a `SubmitFieldMap`.

For each entry in `field_plan`, pick the best locator strategy:

- `locator_strategy="role"` (PREFERRED for text, textarea, button, link): use
  Playwright's `get_by_role(role, name=name)`. Fill `role` (textbox, button,
  combobox, checkbox, etc.) and `name` (the accessible name). Leave `selector` empty.

- `locator_strategy="label"` (use for file inputs and complex form rows): use
  `get_by_label(name)`. Fill `name` with the label text. `<input type="file">`
  has no clean ARIA role, so file uploads almost always need this strategy.
  Leave `role` and `selector` empty.

- `locator_strategy="placeholder"` (use when a field has a placeholder but no
  visible label): use `get_by_placeholder(name)`. Fill `name` with the
  placeholder text.

- `locator_strategy="selector"` (LAST RESORT — only when role/label/placeholder
  all fail): use a CSS selector or XPath via `page.locator(selector)`. Fill
  `selector` and leave role/name empty. Example: `"input[type='file']"`,
  `"#email"`, `"xpath=//button[contains(., 'Submit')]"`.

Set `fill_kind`:
- "type" — `locator.fill(value)` (text inputs, textareas, combobox typing)
- "upload" — `locator.set_input_files(value)` (file inputs; value is a local path)
- "click" — `locator.click()` (checkboxes you want checked, radio buttons)
- "select" — for dropdown / "Select..." pickers: locator points at the dropdown trigger; the framework will click it open, then click the option whose accessible name matches `value`. Use this for any custom combobox (Rippling EEO self-ID dropdowns are all this kind).
- "skip" — no plausible match on the page; leave it for the human

`value` is the value to fill / upload (carry from field_plan, or the file path
for uploads). `plan_field_name` mirrors the entry's `name` in field_plan.

Then identify the submit button. If a form has multiple buttons (e.g. "Save
draft" and "Submit application"), pick the one that finalizes submission.
`submit_button_role` is typically "button"; `submit_button_name` is the
visible text on it.

Rules:
- If a field has no good match anywhere on the page, use `fill_kind="skip"`.
  Do NOT invent a selector or guess an accessible name.
- Do not add extra fields beyond what's in `field_plan` — only map what was
  asked for.
- Prefer role/label over selector when possible; selectors are brittle."""


class FieldLocator(BaseModel):
    """One mapping from a form_plan field to a concrete page element."""

    model_config = ConfigDict(extra="forbid")

    plan_field_name: str
    locator_strategy: Literal["role", "label", "placeholder", "selector"] = Field(
        ...,
        description=(
            "How to locate the element. 'role' uses get_by_role + accessible name; "
            "'label' uses get_by_label; 'placeholder' uses get_by_placeholder; "
            "'selector' uses CSS or XPath via page.locator(). Prefer role/label "
            "over selector when possible — they survive UI refactors."
        ),
    )
    role: str = Field(
        default="",
        description="Playwright role (e.g. 'textbox', 'button'). Used only when locator_strategy=='role'.",
    )
    name: str = Field(
        default="",
        description="Accessible name / label text / placeholder. Used by role, label, placeholder strategies.",
    )
    selector: str = Field(
        default="",
        description="CSS selector or XPath. Used only when locator_strategy=='selector'. Examples: \"input[type='file']\", \"#email\", \"xpath=//button[contains(., 'Submit')]\".",
    )
    fill_kind: Literal["type", "upload", "click", "select", "skip"]
    value: str = Field(default="")


class SubmitFieldMap(BaseModel):
    """Full mapping: every field locator + the submit button locator."""

    model_config = ConfigDict(extra="forbid")

    fields: list[FieldLocator]
    submit_button_role: str
    submit_button_name: str


class SubmissionRecord(BaseModel):
    """What we write to outputs/<run-id>/submission.json after a submit attempt."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    target_url: str
    attempted_at: str
    outcome: Literal["submitted", "cancelled_by_human", "blocked_no_target_url", "error"]
    submit_url_after: str | None = None
    error: str | None = None
    fields_filled: list[str] = Field(default_factory=list)
    fields_skipped: list[str] = Field(default_factory=list)


class StructuredLLM(Protocol):
    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel: ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def submit(
    run_dir: str | Path,
    *,
    llm: StructuredLLM,
    model: str = "gpt-4.1",
    headless: bool = False,
    confirm_token: str = "SUBMIT",
    console: Console | None = None,
    auto_confirm: bool = False,
) -> SubmissionRecord:
    """Drive a headed browser through the ATS form for the given run.

    `auto_confirm=True` bypasses the HITL gate and is intended only for
    tests against a local fixture. The CLI never passes it.
    """
    console = console or Console()
    run_dir = Path(run_dir)
    plan_path = run_dir / "form_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"form_plan.json not found in {run_dir}")
    plan = FormFillPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))

    run_id = run_dir.name
    target_url = plan.target_url

    if not target_url:
        rec = SubmissionRecord(
            run_id=run_id,
            target_url="",
            attempted_at=_now_iso(),
            outcome="blocked_no_target_url",
            error="form_plan.json has no target_url",
        )
        _persist_record(rec, run_dir)
        return rec

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "applyops submit requires the `submit` extras. "
            "Run `pip install -e '.[submit]'` and `playwright install chromium`."
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            console.print(f"[dim]navigating to[/dim] {target_url}")
            page.goto(target_url, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(500)

            # Playwright >=1.43 exposes ARIA structure via aria_snapshot (YAML).
            # This replaces the deprecated `page.accessibility.snapshot()`.
            a11y_yaml = page.locator("body").aria_snapshot()
            field_map = _request_field_map(plan, a11y_yaml, llm=llm, model=model)

            filled, skipped = _execute_fills(page, field_map, console=console)

            before_shot = run_dir / "submit.before.png"
            page.screenshot(path=str(before_shot), full_page=True)
            console.print(f"[dim]wrote[/dim] {before_shot}")

            if auto_confirm:
                _click_submit(page, field_map, console=console)
                outcome: Literal["submitted", "cancelled_by_human"] = "submitted"
            else:
                outcome = _human_review_gate(
                    page, field_map, run_id, target_url, confirm_token, console
                )

            after_url = page.url
            after_shot = run_dir / "submit.after.png"
            page.screenshot(path=str(after_shot), full_page=True)

            rec = SubmissionRecord(
                run_id=run_id,
                target_url=target_url,
                attempted_at=_now_iso(),
                outcome=outcome,
                submit_url_after=after_url,
                fields_filled=filled,
                fields_skipped=skipped,
            )
            _persist_record(rec, run_dir)
            return rec
        except Exception as exc:
            rec = SubmissionRecord(
                run_id=run_id,
                target_url=target_url,
                attempted_at=_now_iso(),
                outcome="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            _persist_record(rec, run_dir)
            raise
        finally:
            if not auto_confirm and not headless:
                # Leave the browser open for a moment so the human can
                # eyeball the post-submit state.
                browser.close()
            else:
                browser.close()


def _request_field_map(
    plan: FormFillPlan,
    a11y_tree: str,
    *,
    llm: StructuredLLM,
    model: str,
) -> SubmitFieldMap:
    user = json.dumps(
        {
            "field_plan": [f.model_dump() for f in plan.fields],
            "accessibility_tree_yaml": a11y_tree,
        },
        indent=2,
        default=str,
    )
    payload = llm.parse(
        model=model,
        system=SUBMIT_FIELD_MAP_SYSTEM_PROMPT,
        user=user,
        schema=SubmitFieldMap,
    )
    assert isinstance(payload, SubmitFieldMap)
    return payload


def _execute_fills(
    page: Any,
    field_map: SubmitFieldMap,
    *,
    console: Console,
) -> tuple[list[str], list[str]]:
    filled: list[str] = []
    skipped: list[str] = []
    for fl in field_map.fields:
        if fl.fill_kind == "skip":
            skipped.append(fl.plan_field_name)
            console.print(f"  [yellow]skip[/yellow]    {fl.plan_field_name}: no match")
            continue
        try:
            locator = _build_locator(page, fl)
            if fl.fill_kind == "type":
                locator.fill(fl.value)
            elif fl.fill_kind == "upload":
                locator.set_input_files(fl.value)
            elif fl.fill_kind == "click":
                locator.click()
            elif fl.fill_kind == "select":
                _select_dropdown_option(page, locator, fl.value)
            filled.append(fl.plan_field_name)
            console.print(
                f"  [green]ok[/green]      {fl.plan_field_name}: "
                f"{fl.fill_kind} via {fl.locator_strategy} {_locator_summary(fl)}"
            )
        except Exception as exc:
            skipped.append(fl.plan_field_name)
            console.print(f"  [red]fail[/red]    {fl.plan_field_name}: {type(exc).__name__}: {exc}")
    return filled, skipped


def _select_dropdown_option(page: Any, trigger: Any, option_text: str) -> None:
    """Click a custom-dropdown trigger, then click the option matching `option_text`.

    First tries Playwright's native `select_option` (works for real <select>
    elements). Falls back to the open-then-click pattern that Rippling /
    Greenhouse-style custom React combos need.
    """
    try:
        trigger.select_option(label=option_text, timeout=2000)
        return
    except Exception:
        pass
    trigger.click()
    page.wait_for_timeout(200)
    page.get_by_role("option", name=option_text).first.click()


def _build_locator(page: Any, fl: FieldLocator) -> Any:
    """Return a Playwright Locator for `fl` using the requested strategy.

    `page` is typed as `Any` because the real type
    (playwright.sync_api.Page) is an optional dependency — the package
    works without playwright installed for everything except the live
    submit path. The locator object returned is similarly opaque to mypy.
    """
    if fl.locator_strategy == "role":
        return page.get_by_role(fl.role, name=fl.name)
    if fl.locator_strategy == "label":
        return page.get_by_label(fl.name)
    if fl.locator_strategy == "placeholder":
        return page.get_by_placeholder(fl.name)
    if fl.locator_strategy == "selector":
        return page.locator(fl.selector)
    raise ValueError(f"unknown locator_strategy: {fl.locator_strategy}")


def _locator_summary(fl: FieldLocator) -> str:
    if fl.locator_strategy == "role":
        return f"role={fl.role}, name={fl.name!r}"
    if fl.locator_strategy in ("label", "placeholder"):
        return f"name={fl.name!r}"
    return f"selector={fl.selector!r}"


def _human_review_gate(
    page: Any,
    field_map: SubmitFieldMap,
    run_id: str,
    target_url: str,
    confirm_token: str,
    console: Console,
) -> Literal["submitted", "cancelled_by_human"]:
    console.print()
    console.print("[bold]=== HUMAN REVIEW ===[/bold]")
    console.print(f"  run:    {run_id}")
    console.print(f"  url:    {target_url}")
    console.print(
        f"  submit: role={field_map.submit_button_role}, name={field_map.submit_button_name!r}"
    )
    console.print()
    console.print(
        f"Inspect the browser. To submit, type exactly [bold]{confirm_token}[/bold] "
        "and press enter. Anything else cancels."
    )
    answer = input("> ").strip()
    if answer != confirm_token:
        console.print(
            "[yellow]cancelled.[/yellow] browser will close; form remains filled until then."
        )
        return "cancelled_by_human"

    _click_submit(page, field_map, console=console)
    return "submitted"


def _click_submit(page: Any, field_map: SubmitFieldMap, *, console: Console) -> None:
    submit_loc = page.get_by_role(field_map.submit_button_role, name=field_map.submit_button_name)
    console.print("[bold]submitting…[/bold]")
    submit_loc.click()
    page.wait_for_load_state("networkidle", timeout=30_000)


def _persist_record(rec: SubmissionRecord, run_dir: Path) -> None:
    (run_dir / "submission.json").write_text(rec.model_dump_json(indent=2), encoding="utf-8")
