"""Internal exceptions raised by the ``orchestration.k8s`` layer.

These are *orchestration-level* errors — the public :mod:`sparkrun.api.k8s`
surface translates them into :class:`~sparkrun.api._errors.SparkrunError`
subclasses at the boundary, mirroring how core scheduler errors are
translated (see :mod:`sparkrun.api._errors`).  Keeping them here avoids an
inverted dependency (orchestration must not import the api facade).
"""

from __future__ import annotations


class K8sError(Exception):
    """Base class for all ``orchestration.k8s`` failures."""


class KubectlNotFoundError(K8sError):
    """No usable ``kubectl`` binary could be resolved or downloaded."""


class KubectlDownloadError(K8sError):
    """Downloading / verifying a ``kubectl`` binary failed."""


class ClusterUnreachableError(K8sError):
    """The target cluster could not be reached (no API response)."""


class ServiceAccountSetupError(K8sError):
    """Creating or configuring the sparkrun service account failed."""


__all__ = [
    "K8sError",
    "KubectlNotFoundError",
    "KubectlDownloadError",
    "ClusterUnreachableError",
    "ServiceAccountSetupError",
]
