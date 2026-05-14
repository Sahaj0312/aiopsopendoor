"""Run — the append-only record of one stack execution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(UTC)


def _new_run_id() -> str:
    return f"run_{_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"


class RunStatus(StrEnum):
    """Lifecycle states of a Run."""

    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


class Run(BaseModel):
    """One execution of a Stack. Persisted under runs/<id>/."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_run_id)
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    status: RunStatus = RunStatus.RUNNING
    blocked_on: str | None = None
    error: str | None = None
    notes: list[str] = Field(default_factory=list)

    def mark(
        self, status: RunStatus, *, blocked_on: str | None = None, error: str | None = None
    ) -> None:
        self.status = status
        self.ended_at = _now()
        if blocked_on is not None:
            self.blocked_on = blocked_on
        if error is not None:
            self.error = error

    def note(self, msg: str) -> None:
        self.notes.append(f"{_now().isoformat()} {msg}")

    def dir(self, root: str | Path = "runs") -> Path:
        return Path(root) / self.id

    def persist(self, root: str | Path = "runs") -> Path:
        d = self.dir(root)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "run.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path
