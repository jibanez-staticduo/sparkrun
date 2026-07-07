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
    ServiceAccountResult,
    ServiceAccountSpec,
)

from ._errors import ClusterUnreachable, KubectlUnavailable, LauncherJobError, ServiceAccountError
from ._ops import (
    cluster_info,
    configure_service_account,
    ensure_kubectl,
    make_client,
    run_launcher_job,
)

__all__ = [
    # Functions
    "ensure_kubectl",
    "cluster_info",
    "configure_service_account",
    "run_launcher_job",
    "make_client",
    # Data models
    "KubectlBinary",
    "ClusterInfo",
    "KubeTarget",
    "ServiceAccountSpec",
    "ServiceAccountResult",
    "LauncherJobSpec",
    "LauncherJobResult",
    # Errors
    "KubectlUnavailable",
    "ClusterUnreachable",
    "ServiceAccountError",
    "LauncherJobError",
]
