"""Centralized Kubernetes orchestration primitives.

This package holds everything sparkrun needs to talk to a Kubernetes
cluster via ``kubectl``:

- :mod:`kubectl` — binary acquisition + version cache management.
- :mod:`client` — :class:`KubectlClient`, a subprocess wrapper shaped
  like the SSH transport (a future ``K8sTransport`` seam).
- :mod:`context` — resolve the effective kubeconfig / context / namespace.
- :mod:`connect` — cluster connectivity probe → :class:`ClusterInfo`.
- :mod:`manifests` / :mod:`serviceaccount` — RBAC + service-account setup.

The console-free :mod:`sparkrun.api.k8s` surface wraps these; the CLI
(``sparkrun setup k8s ...``) sits on top of the api.
"""

from __future__ import annotations

from .client import KubectlClient
from .connect import ClusterInfo, probe_cluster, require_reachable
from .context import KubeTarget, resolve_kube_target
from .errors import (
    ClusterUnreachableError,
    K8sError,
    KubectlDownloadError,
    KubectlNotFoundError,
    ServiceAccountSetupError,
)
from .inventory import (
    NodeInfo,
    build_node_info,
    parse_nodes,
    probe_node_hardware,
    probe_nodes,
)
from .job import (
    LauncherJobResult,
    LauncherJobSpec,
    build_launcher_manifests,
    render_launcher_manifests,
)
from .kubectl import KubectlBinary, ensure_kubectl, list_cached
from .kueue import (
    FlavorSpec,
    KueueError,
    KueueSetupResult,
    KueueStatus,
    build_provision_manifests,
    derive_flavors,
)
from .jobset import (
    JobSetPlan,
    PodSetPlan,
    build_jobset,
    node_selectors_from_nodes,
    plan_from_rank_models,
    render_jobset,
)
from .launch import (
    build_launch_jobset,
    group_contiguous_ranks,
    jobset_logs,
    jobset_status,
    precheck,
    stop_jobset,
    submit_jobset,
)
from .scheduling import (
    ClassFeasibility,
    FeasibilityReport,
    GpuClass,
    GpuRequest,
    aggregate_gpu_classes,
    check_feasibility,
)
from .serviceaccount import (
    ServiceAccountResult,
    ServiceAccountSpec,
    configure_service_account,
)

__all__ = [
    "KubectlBinary",
    "ensure_kubectl",
    "list_cached",
    "KubectlClient",
    "KubeTarget",
    "resolve_kube_target",
    "ClusterInfo",
    "probe_cluster",
    "require_reachable",
    "ServiceAccountSpec",
    "ServiceAccountResult",
    "configure_service_account",
    "LauncherJobSpec",
    "LauncherJobResult",
    "build_launcher_manifests",
    "render_launcher_manifests",
    "NodeInfo",
    "probe_nodes",
    "probe_node_hardware",
    "parse_nodes",
    "build_node_info",
    "FlavorSpec",
    "KueueStatus",
    "KueueSetupResult",
    "KueueError",
    "derive_flavors",
    "build_provision_manifests",
    "GpuClass",
    "GpuRequest",
    "ClassFeasibility",
    "FeasibilityReport",
    "aggregate_gpu_classes",
    "check_feasibility",
    "PodSetPlan",
    "JobSetPlan",
    "build_jobset",
    "render_jobset",
    "node_selectors_from_nodes",
    "plan_from_rank_models",
    "build_launch_jobset",
    "group_contiguous_ranks",
    "precheck",
    "submit_jobset",
    "stop_jobset",
    "jobset_status",
    "jobset_logs",
    "K8sError",
    "KubectlNotFoundError",
    "KubectlDownloadError",
    "ClusterUnreachableError",
    "ServiceAccountSetupError",
]
