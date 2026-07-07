"""Public Kubernetes errors for :mod:`sparkrun.api.k8s`.

Orchestration-level failures (:mod:`sparkrun.orchestration.k8s.errors`)
are translated into these :class:`~sparkrun.api._errors.SparkrunError`
subclasses at the api boundary, so callers can ``except SparkrunError``
uniformly.
"""

from __future__ import annotations

from sparkrun.api._errors import SparkrunError


class KubectlUnavailable(SparkrunError):
    """No usable ``kubectl`` binary could be resolved or downloaded."""


class ClusterUnreachable(SparkrunError):
    """The target cluster's API server did not respond."""


class ServiceAccountError(SparkrunError):
    """Configuring the sparkrun service account failed."""


class LauncherJobError(SparkrunError):
    """Building or applying an in-cluster launcher Job failed."""


__all__ = ["KubectlUnavailable", "ClusterUnreachable", "ServiceAccountError", "LauncherJobError"]
