"""Translate a placement decision into a Kueue-admitted JobSet.

This is the k8s actuator: sparkrun's layout engine decides *which rank
needs which GPU class* (step (C) — sparkrun owns layout), and this module
turns that into a JobSet where

- each GPU node-class becomes one ``replicatedJob`` (podset) with
  ``replicas`` = the number of ranks on that class;
- each pod carries a ``nodeSelector`` on ``nvidia.com/gpu.product`` (the
  ResourceFlavor selector) so Kueue/k8s bind it to the right class;
- each pod requests ``nvidia.com/gpu: <gpus_per_pod>``;
- the JobSet is labeled with the sparkrun LocalQueue, so Kueue admits the
  whole thing all-or-nothing (gang).

JobSet also provisions a headless Service giving pods stable DNS
(``<jobset>-<replicatedjob>-<jobidx>-<podidx>``) — the address the TCP
NCCL wiring (a later step) uses for the rendezvous master.

The per-pod ``command`` / ``env`` are filled by the runtime layer (the
serve command per rank); this module only assembles the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .inventory import GPU_PRODUCT_LABEL, NodeInfo
from .kueue import DEFAULT_QUEUE_NAME
from .manifests import MANAGED_BY_LABEL, render_manifests
from .scheduling import GpuRequest

JOBSET_API_VERSION = "jobset.x-k8s.io/v1alpha2"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"

DEFAULT_NAMESPACE = "sparkrun"
DEFAULT_SERVICE_ACCOUNT = "sparkrun"


@dataclass
class PodSetPlan:
    """One replicatedJob = all the ranks that land on one GPU node-class."""

    name: str  # replicatedJob name (RFC1123 label — the model token works)
    replicas: int  # number of pods (== ranks on this class)
    gpus_per_pod: int
    image: str
    model: str = ""
    node_selector: dict[str, str] = field(default_factory=dict)
    command: list[str] | None = None
    args: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    field_ref_env: dict[str, str] = field(default_factory=dict)
    """Downward-API env: name -> fieldPath (e.g. the JobSet job-index annotation)."""
    extra_resources: dict[str, str] = field(default_factory=dict)
    """Extra resource limits beyond GPUs (e.g. ``{"rdma/rdma_shared_device_a": "1"}``)."""
    capabilities: list[str] = field(default_factory=list)
    """Linux capabilities to add (e.g. ``["IPC_LOCK"]`` for RDMA)."""


@dataclass
class JobSetPlan:
    """A full workload: JobSet metadata + its podsets."""

    name: str
    pod_sets: list[PodSetPlan]
    namespace: str = DEFAULT_NAMESPACE
    queue: str = DEFAULT_QUEUE_NAME
    service_account: str = DEFAULT_SERVICE_ACCOUNT
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def total_pods(self) -> int:
        return sum(ps.replicas for ps in self.pod_sets)

    @property
    def gpu_requests(self) -> list[GpuRequest]:
        """Per-class GPU demand — feeds the feasibility precheck."""
        return [GpuRequest(model=ps.model or ps.name, gpus=ps.replicas * ps.gpus_per_pod, label=ps.name) for ps in self.pod_sets]


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def _env_list(env: dict[str, str]) -> list[dict]:
    return [{"name": k, "value": str(v)} for k, v in sorted(env.items())]


def _container(pod_set: PodSetPlan) -> dict:
    limits = {"nvidia.com/gpu": str(pod_set.gpus_per_pod)}
    limits.update(pod_set.extra_resources)
    container: dict = {
        "name": "worker",
        "image": pod_set.image,
        "resources": {"limits": limits},
    }
    if pod_set.capabilities:
        container["securityContext"] = {"capabilities": {"add": list(pod_set.capabilities)}}
    if pod_set.command:
        container["command"] = list(pod_set.command)
    if pod_set.args:
        container["args"] = list(pod_set.args)
    env_entries = _env_list(pod_set.env)
    for name, field_path in sorted(pod_set.field_ref_env.items()):
        env_entries.append({"name": name, "valueFrom": {"fieldRef": {"fieldPath": field_path}}})
    if env_entries:
        container["env"] = env_entries
    return container


def _pod_spec(pod_set: PodSetPlan, service_account: str) -> dict:
    spec: dict = {
        "serviceAccountName": service_account,
        "restartPolicy": "Never",
        "containers": [_container(pod_set)],
    }
    if pod_set.node_selector:
        spec["nodeSelector"] = dict(pod_set.node_selector)
    return spec


def replicated_job(pod_set: PodSetPlan, *, service_account: str) -> dict:
    """A JobSet replicatedJob wrapping *pod_set*."""
    return {
        "name": pod_set.name,
        "replicas": pod_set.replicas,
        "template": {
            "spec": {
                "parallelism": 1,
                "completions": 1,
                "backoffLimit": 0,
                "template": {"spec": _pod_spec(pod_set, service_account)},
            }
        },
    }


def build_jobset(plan: JobSetPlan) -> dict:
    """Assemble the JobSet manifest for *plan* (Kueue-labeled, gang-admitted)."""
    labels = dict(MANAGED_BY_LABEL)
    labels[QUEUE_LABEL] = plan.queue
    labels.update(plan.labels)
    return {
        "apiVersion": JOBSET_API_VERSION,
        "kind": "JobSet",
        "metadata": {"name": plan.name, "namespace": plan.namespace, "labels": labels},
        "spec": {"replicatedJobs": [replicated_job(ps, service_account=plan.service_account) for ps in plan.pod_sets]},
    }


def render_jobset(plan: JobSetPlan) -> str:
    return render_manifests([build_jobset(plan)])


# ---------------------------------------------------------------------------
# Placement → plan translation
# ---------------------------------------------------------------------------


def node_selectors_from_nodes(nodes: list[NodeInfo]) -> dict[str, dict[str, str]]:
    """Map ``model -> {nvidia.com/gpu.product: <value>}`` from the inventory.

    The selector is what pins a podset to its GPU class (== ResourceFlavor).
    """
    selectors: dict[str, dict[str, str]] = {}
    for node in nodes:
        if not node.has_accelerators:
            continue
        product = node.labels.get(GPU_PRODUCT_LABEL)
        if not product:
            continue
        model = node.hardware.accelerators[0].model
        selectors.setdefault(model, {GPU_PRODUCT_LABEL: product})
    return selectors


def plan_from_rank_models(
    name: str,
    rank_models: list[str],
    *,
    image: str,
    node_selectors: dict[str, dict[str, str]] | None = None,
    gpus_per_pod: int | dict[str, int] = 1,
    namespace: str = DEFAULT_NAMESPACE,
    queue: str = DEFAULT_QUEUE_NAME,
    service_account: str = DEFAULT_SERVICE_ACCOUNT,
    labels: dict[str, str] | None = None,
) -> JobSetPlan:
    """Group ranks by GPU model into one podset per class.

    *rank_models* is model-per-rank (index = global rank).  Consecutive or
    scattered ranks of the same model collapse into a single replicatedJob
    with ``replicas`` = that model's rank count.  *node_selectors* maps a
    model to its ``nvidia.com/gpu.product`` selector (from
    :func:`node_selectors_from_nodes`); a missing model gets no selector
    (Kueue picks any matching flavor).  *gpus_per_pod* may be a flat int or
    a per-model mapping.
    """
    node_selectors = node_selectors or {}
    counts: dict[str, int] = {}
    order: list[str] = []
    for model in rank_models:
        if model not in counts:
            order.append(model)
        counts[model] = counts.get(model, 0) + 1

    pod_sets: list[PodSetPlan] = []
    for model in order:
        per_pod = gpus_per_pod.get(model, 1) if isinstance(gpus_per_pod, dict) else gpus_per_pod
        pod_sets.append(
            PodSetPlan(
                name=model,
                replicas=counts[model],
                gpus_per_pod=per_pod,
                image=image,
                model=model,
                node_selector=node_selectors.get(model, {}),
            )
        )
    return JobSetPlan(
        name=name,
        pod_sets=pod_sets,
        namespace=namespace,
        queue=queue,
        service_account=service_account,
        labels=labels or {},
    )


__all__ = [
    "JOBSET_API_VERSION",
    "QUEUE_LABEL",
    "PodSetPlan",
    "JobSetPlan",
    "replicated_job",
    "build_jobset",
    "render_jobset",
    "node_selectors_from_nodes",
    "plan_from_rank_models",
]
