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
from .kubectl import KubectlBinary, ensure_kubectl, list_cached
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
    "K8sError",
    "KubectlNotFoundError",
    "KubectlDownloadError",
    "ClusterUnreachableError",
    "ServiceAccountSetupError",
]
