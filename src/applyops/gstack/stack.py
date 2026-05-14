"""Stack — the orchestrator. Runs layers in order, threads rebase loops."""

from __future__ import annotations

from collections.abc import Mapping

from applyops.gstack.context import LayerState, StackContext
from applyops.gstack.protocols import Layer, ReviewGate
from applyops.gstack.run import Run, RunStatus
from applyops.obs import tracer


class StackBlocked(RuntimeError):
    """Raised when a gate cannot pass a layer within the rebase budget."""


class Stack:
    """A sequence of layers with optional review gates between them.

    Gates are looked up by the *layer name they review*, i.e.
    `gates["writer"]` is the gate that inspects the writer's output and
    can request a writer-rebase.
    """

    def __init__(
        self,
        layers: list[Layer],
        gates: Mapping[str, ReviewGate] | None = None,
        *,
        max_rebases_per_gate: int = 3,
    ) -> None:
        if not layers:
            raise ValueError("Stack requires at least one layer")
        names = [layer.name for layer in layers]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate layer names: {names}")
        self.layers: list[Layer] = layers
        self.gates: dict[str, ReviewGate] = dict(gates or {})
        self.max_rebases_per_gate = max_rebases_per_gate
        unknown = set(self.gates) - set(names)
        if unknown:
            raise ValueError(f"gates reference unknown layers: {sorted(unknown)}")

    def land(
        self,
        *,
        up_to: str | None = None,
        inputs: Mapping[str, object] | None = None,
    ) -> tuple[Run, StackContext]:
        """Run the stack top-to-bottom. Stop after `up_to` if given.

        Returns the Run record and the final StackContext. If a gate
        exhausts its rebase budget, the Run is marked BLOCKED and the
        partial context is returned — no exception is raised; the caller
        decides how to handle it.
        """
        run = Run()
        ctx = StackContext(run=run, inputs=dict(inputs or {}))
        with tracer().start_as_current_span("stack.land") as root_span:
            root_span.set_attribute("run.id", run.id)
            root_span.set_attribute("stack.layers", ",".join(la.name for la in self.layers))
            root_span.set_attribute("stack.gates", ",".join(self.gates.keys()))
            for layer in self.layers:
                ctx.layers[layer.name] = LayerState(name=layer.name)
                self._run_layer_with_gate(layer, ctx)
                if run.status == RunStatus.BLOCKED:
                    root_span.set_attribute("run.status", run.status.value)
                    return run, ctx
                if up_to is not None and layer.name == up_to:
                    run.mark(RunStatus.PARTIAL)
                    root_span.set_attribute("run.status", run.status.value)
                    return run, ctx
            run.mark(RunStatus.COMPLETED)
            root_span.set_attribute("run.status", run.status.value)
            return run, ctx

    def _run_layer_with_gate(self, layer: Layer, ctx: StackContext) -> None:
        state = ctx.layers[layer.name]
        gate = self.gates.get(layer.name)
        with tracer().start_as_current_span(f"layer.{layer.name}") as span:
            span.set_attribute("layer.name", layer.name)
            output = layer.run(ctx)
            state.output = output

        if gate is None:
            return

        while True:
            with tracer().start_as_current_span(f"gate.{gate.name}") as gate_span:
                gate_span.set_attribute("gate.name", gate.name)
                gate_span.set_attribute("gate.target_layer", layer.name)
                gate_span.set_attribute("gate.rebase_iteration", state.rebases)
                review = gate.review(output, ctx)
                gate_span.set_attribute("gate.passed", review.passed)
                if review.score is not None:
                    gate_span.set_attribute("gate.score", review.score)
            state.gate_reviews.append(review)
            if review.passed:
                state.pending_rebase = None
                return
            if state.rebases >= self.max_rebases_per_gate:
                ctx.run.mark(
                    RunStatus.BLOCKED,
                    blocked_on=f"gate.{gate.name} after {state.rebases} rebases",
                )
                return
            assert review.rebase_request is not None  # invariant from Review
            state.pending_rebase = review.rebase_request
            state.rebases += 1
            with tracer().start_as_current_span(f"layer.{layer.name}.rebase") as rebase_span:
                rebase_span.set_attribute("layer.name", layer.name)
                rebase_span.set_attribute("rebase.iteration", state.rebases)
                output = layer.run(ctx)
                state.output = output
