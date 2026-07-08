"""Kubernetes node inventory — synthesize :class:`HostHardware` from Node labels.

The k8s-native analog of :func:`sparkrun.core.hardware_probe.probe_hosts`
(SSH): instead of running ``nvidia-smi``/``ibstat`` over SSH, it reads the
labels the NVIDIA GPU Operator (GPU Feature Discovery) and Node Feature
Discovery publish on each Node, plus the Node's ``allocatable``/``capacity``,
and builds the same :class:`~sparkrun.core.hardware.HostHardware` /
:class:`~sparkrun.core.hardware.AcceleratorSpec` the rest of sparkrun's
scheduler, platform-resolution, and backend-selection stack already
consumes.  This is the spine of k8s-native scheduling: everything
downstream (``resolve_platform``, ``select_backends``,
``resolve_effective_hosts``, layout, VRAM) is unchanged.

Model tokens and memory values come through the *same* normalizers as the
SSH fingerprint path (:mod:`sparkrun.core.fingerprint`), so a hybrid
cluster mixing SSH-probed and k8s-probed hosts resolves to identical
``AcceleratorSpec`` models — a DGX Spark node is ``model="gb10"`` whether
detected via ``nvidia-smi`` or via ``nvidia.com/gpu.product``.

Detection needs **no elevated privilege**: GFD/NFD already did the
privileged probing cluster-side; reading Node objects only needs the
``nodes: get,list`` RBAC the sparkrun service account already carries.
When these labels are absent, a privileged fallback probe (a later
increment) fills the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sparkrun.core.fingerprint import mib_to_gb, normalize_nvidia_model
from sparkrun.core.hardware import AcceleratorSpec, HostHardware

from .client import KubectlClient

# --- GPU Feature Discovery (NVIDIA GPU Operator) node labels ---
GPU_PRODUCT_LABEL = "nvidia.com/gpu.product"
GPU_MACHINE_LABEL = "nvidia.com/gpu.machine"
GPU_COUNT_LABEL = "nvidia.com/gpu.count"
GPU_MEMORY_LABEL = "nvidia.com/gpu.memory"  # MiB, per GPU
GPU_FAMILY_LABEL = "nvidia.com/gpu.family"
GPU_PRESENT_LABEL = "nvidia.com/gpu.present"
MIG_STRATEGY_LABEL = "nvidia.com/mig.strategy"
SHARING_STRATEGY_LABEL = "nvidia.com/gpu.sharing-strategy"

# --- Node Feature Discovery — Mellanox/NVIDIA-Networking PCI vendor id 15b3 ---
NFD_MELLANOX_PCI_LABEL = "feature.node.kubernetes.io/pci-15b3.present"

NVIDIA_GPU_RESOURCE = "nvidia.com/gpu"


@dataclass
class NodeInfo:
    """A single k8s Node's synthesized hardware + scheduling-relevant state."""

    name: str
    hardware: HostHardware
    labels: dict[str, str] = field(default_factory=dict)
    allocatable_gpus: int = 0
    capacity_gpus: int = 0
    schedulable: bool = True

    @property
    def has_accelerators(self) -> bool:
        return bool(self.hardware.accelerators)


# ---------------------------------------------------------------------------
# Label → AcceleratorSpec
# ---------------------------------------------------------------------------


def _quantity_to_int(value) -> int:
    """Parse a k8s quantity string (``"8"``) to int; 0 on absence/parse error."""
    if value is None:
        return 0
    try:
        return int(str(value))
    except ValueError:
        return 0


def _mellanox_present(labels: dict) -> bool:
    return str(labels.get(NFD_MELLANOX_PCI_LABEL, "")).lower() == "true"


def _nvidia_present(labels: dict, capacity: dict) -> bool:
    if str(labels.get(GPU_PRESENT_LABEL, "")).lower() == "true":
        return True
    if labels.get(GPU_PRODUCT_LABEL):
        return True
    return NVIDIA_GPU_RESOURCE in capacity


def _nvidia_model(labels: dict) -> str:
    """Resolve the sparkrun model token from GFD labels.

    Normalizes ``nvidia.com/gpu.product`` through the same path as the SSH
    fingerprint (dashes → spaces first, since GFD sanitizes spaces to
    dashes).  Falls back to the DGX-Spark machine label, then ``"unknown"``.
    """
    product = labels.get(GPU_PRODUCT_LABEL, "")
    if product:
        model = normalize_nvidia_model(product.replace("-", " ").replace("_", " "))
        if model:
            return model
    if "spark" in labels.get(GPU_MACHINE_LABEL, "").lower():
        return "gb10"
    return "unknown"


def _nvidia_caps(labels: dict) -> frozenset[str]:
    """Capability tags, mirroring the SSH path (``cuda`` + RDMA) plus sharing hints."""
    caps = {"cuda"}
    if _mellanox_present(labels):
        caps.add("rdma:roce-v2")
    strategy = labels.get(MIG_STRATEGY_LABEL)
    if strategy and strategy not in ("", "none"):
        caps.add("mig")
    sharing = labels.get(SHARING_STRATEGY_LABEL)
    if sharing and sharing not in ("", "none"):
        caps.add("time-sliced")
    return frozenset(caps)


def build_node_hardware(labels: dict, capacity: dict) -> HostHardware:
    """Build :class:`HostHardware` from a Node's labels + capacity map."""
    accelerators: list[AcceleratorSpec] = []
    if _nvidia_present(labels, capacity):
        count = _quantity_to_int(labels.get(GPU_COUNT_LABEL)) or _quantity_to_int(capacity.get(NVIDIA_GPU_RESOURCE))
        if count > 0:
            accelerators.append(
                AcceleratorSpec(
                    vendor="nvidia",
                    model=_nvidia_model(labels),
                    count=count,
                    memory_gb=mib_to_gb(labels.get(GPU_MEMORY_LABEL)),
                    capabilities=_nvidia_caps(labels),
                )
            )
    return HostHardware(accelerators=accelerators, notes="detected from k8s node labels")


def build_node_info(node: dict) -> NodeInfo:
    """Build a :class:`NodeInfo` from one Node object (``kubectl get node -o json`` item)."""
    meta = node.get("metadata") or {}
    status = node.get("status") or {}
    spec = node.get("spec") or {}
    labels = meta.get("labels") or {}
    capacity = status.get("capacity") or {}
    allocatable = status.get("allocatable") or {}
    return NodeInfo(
        name=str(meta.get("name", "")),
        hardware=build_node_hardware(labels, capacity),
        labels=labels,
        allocatable_gpus=_quantity_to_int(allocatable.get(NVIDIA_GPU_RESOURCE)),
        capacity_gpus=_quantity_to_int(capacity.get(NVIDIA_GPU_RESOURCE)),
        schedulable=not bool(spec.get("unschedulable", False)),
    )


def parse_nodes(payload: dict) -> list[NodeInfo]:
    """Parse a ``kubectl get nodes -o json`` payload into :class:`NodeInfo` list."""
    return [build_node_info(n) for n in (payload.get("items") or [])]


# ---------------------------------------------------------------------------
# Probe (live)
# ---------------------------------------------------------------------------


def probe_nodes(
    client: KubectlClient,
    *,
    selector: str | None = None,
    gpu_only: bool = False,
) -> list[NodeInfo]:
    """Query cluster Nodes and synthesize :class:`NodeInfo` for each.

    *selector* is a ``kubectl -l`` label selector; *gpu_only* filters to
    nodes with at least one detected accelerator.
    """
    args = ["get", "nodes", "-o", "json"]
    if selector:
        args += ["-l", selector]
    nodes = parse_nodes(client.run_json(args))
    if gpu_only:
        nodes = [n for n in nodes if n.has_accelerators]
    return nodes


def probe_node_hardware(
    client: KubectlClient,
    *,
    selector: str | None = None,
    gpu_only: bool = True,
) -> dict[str, HostHardware]:
    """Return ``{node_name: HostHardware}`` — the k8s analog of ``probe_hosts``."""
    return {n.name: n.hardware for n in probe_nodes(client, selector=selector, gpu_only=gpu_only)}


__all__ = [
    "NodeInfo",
    "GPU_PRODUCT_LABEL",
    "GPU_MACHINE_LABEL",
    "GPU_COUNT_LABEL",
    "GPU_MEMORY_LABEL",
    "NFD_MELLANOX_PCI_LABEL",
    "NVIDIA_GPU_RESOURCE",
    "build_node_hardware",
    "build_node_info",
    "parse_nodes",
    "probe_nodes",
    "probe_node_hardware",
]
