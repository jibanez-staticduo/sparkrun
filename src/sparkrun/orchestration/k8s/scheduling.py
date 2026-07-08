"""Feasibility precheck for k8s-native launches.

Kueue does the real admission (gang, quota), but a workload that asks for
more GPUs of a class than the cluster can offer would just sit ``Pending``
forever.  This module aggregates the node inventory by GPU class and
answers "can this demand be placed?" *before* submitting — turning a
silent pending into a clear error.

The demand is expressed in sparkrun's own vocabulary (model tokens like
``"gb10"`` / ``"rtx-pro-6000-blackwell"``), which is what the layout
decision produces; :mod:`sparkrun.orchestration.k8s.placement` (step 3)
turns the same per-class split into JobSet podsets + ResourceFlavor
selectors.

Allocatable counts only *schedulable* nodes — a cordoned node's GPUs
can't take new pods, so they don't count toward feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .inventory import GPU_PRODUCT_LABEL, NodeInfo


@dataclass
class GpuClass:
    """Aggregated capacity for one GPU model across the cluster."""

    model: str
    product: str | None  # nvidia.com/gpu.product — the ResourceFlavor selector
    allocatable_gpus: int  # summed over schedulable nodes only
    capacity_gpus: int  # summed over all nodes
    node_count: int
    schedulable_node_count: int


@dataclass
class GpuRequest:
    """A demand for *gpus* accelerators of one model (a rank-group)."""

    model: str
    gpus: int
    label: str = ""


@dataclass
class ClassFeasibility:
    model: str
    required: int
    allocatable: int
    capacity: int
    schedulable_nodes: int
    ok: bool
    known: bool = True

    @property
    def shortfall(self) -> int:
        return max(0, self.required - self.allocatable)


@dataclass
class FeasibilityReport:
    feasible: bool
    classes: list[ClassFeasibility] = field(default_factory=list)

    @property
    def unknown_models(self) -> list[str]:
        return [c.model for c in self.classes if not c.known]

    def summary(self) -> str:
        """One line per class explaining the verdict."""
        lines: list[str] = []
        for c in self.classes:
            if not c.known:
                lines.append("%s: no such GPU class in cluster (need %d)" % (c.model, c.required))
            elif c.ok:
                lines.append("%s: OK (%d/%d allocatable)" % (c.model, c.required, c.allocatable))
            else:
                lines.append(
                    "%s: short by %d (%d needed, %d allocatable across %d schedulable node(s))"
                    % (c.model, c.shortfall, c.required, c.allocatable, c.schedulable_nodes)
                )
        return "\n".join(lines)


def aggregate_gpu_classes(nodes: list[NodeInfo]) -> dict[str, GpuClass]:
    """Aggregate node inventory into ``{model: GpuClass}``.

    Allocatable is summed over schedulable nodes only; capacity over all.
    """
    classes: dict[str, GpuClass] = {}
    for node in nodes:
        if not node.has_accelerators:
            continue
        model = node.hardware.accelerators[0].model
        existing = classes.get(model)
        if existing is None:
            existing = GpuClass(
                model=model,
                product=node.labels.get(GPU_PRODUCT_LABEL),
                allocatable_gpus=0,
                capacity_gpus=0,
                node_count=0,
                schedulable_node_count=0,
            )
            classes[model] = existing
        existing.capacity_gpus += node.capacity_gpus or node.hardware.total_gpus
        existing.node_count += 1
        if node.schedulable:
            existing.allocatable_gpus += node.allocatable_gpus
            existing.schedulable_node_count += 1
        if existing.product is None and node.labels.get(GPU_PRODUCT_LABEL):
            existing.product = node.labels.get(GPU_PRODUCT_LABEL)
    return classes


def check_feasibility(nodes: list[NodeInfo], requests: list[GpuRequest]) -> FeasibilityReport:
    """Check whether *requests* (per-model GPU demand) fit the cluster."""
    inventory = aggregate_gpu_classes(nodes)

    demand: dict[str, int] = {}
    for request in requests:
        demand[request.model] = demand.get(request.model, 0) + request.gpus

    classes: list[ClassFeasibility] = []
    feasible = True
    for model, required in sorted(demand.items()):
        gpu_class = inventory.get(model)
        if gpu_class is None:
            feasible = False
            classes.append(
                ClassFeasibility(model=model, required=required, allocatable=0, capacity=0, schedulable_nodes=0, ok=False, known=False)
            )
            continue
        ok = gpu_class.allocatable_gpus >= required
        if not ok:
            feasible = False
        classes.append(
            ClassFeasibility(
                model=model,
                required=required,
                allocatable=gpu_class.allocatable_gpus,
                capacity=gpu_class.capacity_gpus,
                schedulable_nodes=gpu_class.schedulable_node_count,
                ok=ok,
            )
        )
    return FeasibilityReport(feasible=feasible, classes=classes)


__all__ = [
    "GpuClass",
    "GpuRequest",
    "ClassFeasibility",
    "FeasibilityReport",
    "aggregate_gpu_classes",
    "check_feasibility",
]
