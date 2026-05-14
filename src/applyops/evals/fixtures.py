"""Eval fixture loader.

Fixtures live under `tests/evals/fixtures/`. Each fixture is a JSON file
matching one of our Pydantic models. The loader returns the parsed model
so eval tests don't repeat boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "evals" / "fixtures"

T = TypeVar("T", bound=BaseModel)


def load_fixture(name: str, model: type[T]) -> T:
    """Load `tests/evals/fixtures/<name>` and validate against `model`."""
    path = FIXTURES_DIR / name
    raw = json.loads(path.read_text(encoding="utf-8"))
    return model.model_validate(raw)
