"""gstack — the orchestrator.

A small, framework-free stacked-agent runner. Layers are sequential.
Each layer can have a review gate attached after it; the gate can request
changes and force the layer to rebase (re-run with the review attached
as extra context), up to a configurable cap.

Public API:
    from applyops.gstack import Stack, Layer, ReviewGate, LayerOutput, Review
"""

from __future__ import annotations

from applyops.gstack.context import LayerState, StackContext
from applyops.gstack.protocols import Layer, ReviewGate
from applyops.gstack.run import Run, RunStatus
from applyops.gstack.stack import Stack, StackBlocked
from applyops.gstack.types import LayerOutput, RebaseRequest, Review

__all__ = [
    "Layer",
    "LayerOutput",
    "LayerState",
    "RebaseRequest",
    "Review",
    "ReviewGate",
    "Run",
    "RunStatus",
    "Stack",
    "StackBlocked",
    "StackContext",
]
