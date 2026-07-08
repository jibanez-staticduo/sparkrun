"""Assemble and submit a runnable k8s launch (JobSet + NCCL wiring).

This is the glue on top of the JobSet actuator (:mod:`.jobset`): it takes
a per-rank GPU-class assignment plus a serve command and produces a
fully-populated :class:`~sparkrun.orchestration.k8s.jobset.JobSetPlan` —
each podset carrying its base-rank offset, the TCP NCCL env, a rank-
resolving entrypoint prelude, and the serve command — then submits it and
exposes stop / status / logs over the JobSet.

Rank layout: ranks of the same GPU class must be contiguous in
*rank_models* (the layout engine assigns contiguous per-class ranges);
each podset's base rank is the start of its block, and per-pod RANK is
``base + job_index`` resolved in-pod (see :mod:`.nccl`).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import nccl
from .client import KubectlClient
from .jobset import DEFAULT_NAMESPACE, DEFAULT_SERVICE_ACCOUNT, JobSetPlan, PodSetPlan, render_jobset
from .kueue import DEFAULT_QUEUE_NAME
from .scheduling import FeasibilityReport, check_feasibility

JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"


@dataclass
class RankGroup:
    model: str
    base_rank: int
    count: int


@dataclass
class LaunchJobsetResult:
    """Outcome of a k8s JobSet launch."""

    name: str
    namespace: str
    manifests_yaml: str
    feasible: bool
    dry_run: bool
    feasibility_summary: str = ""
    submitted: bool = False


def group_contiguous_ranks(rank_models: list[str]) -> list[RankGroup]:
    """Collapse a per-rank model list into contiguous per-class blocks.

    Raises ``ValueError`` if a model appears in two non-adjacent blocks —
    ranks for one GPU class must be contiguous so base-rank + job-index
    yields the correct global rank.
    """
    groups: list[RankGroup] = []
    i, n = 0, len(rank_models)
    while i < n:
        model = rank_models[i]
        j = i
        while j < n and rank_models[j] == model:
            j += 1
        groups.append(RankGroup(model=model, base_rank=i, count=j - i))
        i = j
    seen: set[str] = set()
    for group in groups:
        if group.model in seen:
            raise ValueError("ranks for GPU class %r are not contiguous in the layout" % group.model)
        seen.add(group.model)
    return groups


def build_launch_jobset(
    name: str,
    rank_models: list[str],
    *,
    image: str,
    serve_command: str,
    node_selectors: dict[str, dict[str, str]] | None = None,
    gpus_per_pod: int | dict[str, int] = 1,
    env: dict[str, str] | None = None,
    master_port: int = nccl.DEFAULT_MASTER_PORT,
    socket_ifname: str = "eth0",
    namespace: str = DEFAULT_NAMESPACE,
    queue: str = DEFAULT_QUEUE_NAME,
    service_account: str = DEFAULT_SERVICE_ACCOUNT,
    labels: dict[str, str] | None = None,
) -> JobSetPlan:
    """Build a fully-populated JobSetPlan ready to submit.

    *serve_command* is the workload shell command; it runs after a per-pod
    prelude that exports ``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` etc.
    """
    if not rank_models:
        raise ValueError("rank_models must be non-empty")
    node_selectors = node_selectors or {}
    base_env = dict(env or {})
    world_size = len(rank_models)

    groups = group_contiguous_ranks(rank_models)
    head_model = groups[0].model  # rank 0 lives in the first podset
    master = nccl.master_addr(name, head_model)
    tcp_env = nccl.base_tcp_nccl_env(world_size, master, master_port=master_port, socket_ifname=socket_ifname)

    pod_sets: list[PodSetPlan] = []
    for group in groups:
        per_pod = gpus_per_pod.get(group.model, 1) if isinstance(gpus_per_pod, dict) else gpus_per_pod
        pod_env = {**base_env, **tcp_env}
        prelude = nccl.rank_prelude(group.base_rank)
        pod_sets.append(
            PodSetPlan(
                name=group.model,
                replicas=group.count,
                gpus_per_pod=per_pod,
                image=image,
                model=group.model,
                node_selector=node_selectors.get(group.model, {}),
                command=["bash", "-lc", prelude + "\n" + serve_command],
                env=pod_env,
                field_ref_env=nccl.field_ref_env(),
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


# ---------------------------------------------------------------------------
# Submit / lifecycle (live)
# ---------------------------------------------------------------------------


def precheck(plan: JobSetPlan, nodes) -> FeasibilityReport:
    """Feasibility of *plan* against the node inventory (per-class GPU demand)."""
    return check_feasibility(nodes, plan.gpu_requests)


def submit_jobset(client: KubectlClient, plan: JobSetPlan, *, timeout: int | None = 60):
    """Apply the JobSet manifest (Kueue admits it)."""
    return client.apply(render_jobset(plan), timeout=timeout)


def stop_jobset(client: KubectlClient, name: str):
    """Delete a JobSet (cascades to its Jobs/pods); idempotent."""
    return client.run(["delete", "jobset", name, "--ignore-not-found"])


def jobset_status(client: KubectlClient, name: str) -> dict:
    """Return the JobSet object as a dict (``kubectl get jobset -o json``)."""
    return client.run_json(["get", "jobset", name, "-o", "json"])


def jobset_logs(client: KubectlClient, name: str, *, follow: bool = False) -> int:
    """Stream logs from all of a JobSet's pods (by label selector)."""
    args = ["logs", "-l", "%s=%s" % (JOBSET_NAME_LABEL, name), "--all-containers", "--prefix"]
    if follow:
        args.append("-f")
    if client.dry_run:
        return 0
    import subprocess

    return subprocess.call(client.base_args() + args)  # noqa: S603 — argv list, inherits stdio


__all__ = [
    "RankGroup",
    "LaunchJobsetResult",
    "group_contiguous_ranks",
    "build_launch_jobset",
    "precheck",
    "submit_jobset",
    "stop_jobset",
    "jobset_status",
    "jobset_logs",
]
