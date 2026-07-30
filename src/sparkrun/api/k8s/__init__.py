"""Public library API for Kubernetes operations.

Console-free surface over :mod:`sparkrun.orchestration.k8s`.  Mirrors the
top-level :mod:`sparkrun.api` conventions: optional ``sctx``, dataclass
returns, typed :class:`~sparkrun.api._errors.SparkrunError` subclasses,
never writes to stdout/stderr.

Functions:

- :func:`ensure_kubectl` — resolve / download a ``kubectl`` binary.
- :func:`cluster_info` — probe a cluster (and pin its version).
- :func:`configure_service_account` — create the sparkrun SA + RBAC and a
  scoped kubeconfig.
"""

from __future__ import annotations

# Data models (re-exported from the orchestration layer — the dataclass
# shapes are the stable public contract).
from sparkrun.orchestration.k8s import (
    ClusterInfo,
    KubectlBinary,
    KubeTarget,
    LauncherJobResult,
    LauncherJobSpec,
    NodeInfo,
    ServiceAccountResult,
    ServiceAccountSpec,
)
from sparkrun.orchestration.k8s.kueue import KueueSetupResult, KueueStatus
from sparkrun.orchestration.k8s.launch import LaunchJobsetResult
from sparkrun.orchestration.k8s.scheduling import FeasibilityReport, GpuRequest

from ._errors import (
    ClusterUnreachable,
    JobSetLaunchError,
    KubectlUnavailable,
    KueueSetupError,
    LauncherJobError,
    ServiceAccountError,
)
from ._ops import (
    check_feasibility,
    cluster_info,
    configure_service_account,
    ensure_kubectl,
    jobset_status,
    kueue_status,
    launch_jobset,
    list_nodes,
    make_client,
    probe_nodes_fallback,
    run_launcher_job,
    setup_kueue,
    stop_jobset,
)

__all__ = [
    # Functions
    "ensure_kubectl",
    "cluster_info",
    "configure_service_account",
    "list_nodes",
    "probe_nodes_fallback",
    "check_feasibility",
    "kueue_status",
    "setup_kueue",
    "launch_jobset",
    "stop_jobset",
    "jobset_status",
    "run_launcher_job",
    "make_client",
    # Data models
    "KubectlBinary",
    "ClusterInfo",
    "KubeTarget",
    "NodeInfo",
    "ServiceAccountSpec",
    "ServiceAccountResult",
    "LauncherJobSpec",
    "LauncherJobResult",
    "KueueStatus",
    "KueueSetupResult",
    "GpuRequest",
    "FeasibilityReport",
    "LaunchJobsetResult",
    # Errors
    "KubectlUnavailable",
    "ClusterUnreachable",
    "ServiceAccountError",
    "LauncherJobError",
    "KueueSetupError",
    "JobSetLaunchError",
]
