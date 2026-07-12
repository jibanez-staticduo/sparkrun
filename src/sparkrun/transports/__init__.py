"""Cluster transports — the connectivity seam.

A transport owns how sparkrun *reaches / prepares* a cluster's hosts before the
generic SSH machinery runs.  ``ssh`` (the default) is a no-op; provider-backed
transports (``thunder``) refresh ephemeral connection details out-of-band.

Resolution uses a plain in-process registry keyed by
:attr:`ClusterDefinition.transport` — mirroring :mod:`sparkrun.platforms`.  The
set is tiny and closed, so SAF discovery would add indirection without benefit;
``EXT_TRANSPORT`` is reserved for a future entry-point wiring if that changes.

Layering: ``cli/api → transports → {core, orchestration}``; ``orchestration``
never imports this package.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.transports.base import Transport, TransportError
from sparkrun.transports.ssh import SshTransport
from sparkrun.transports.thunder.transport import ThunderTransport

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition

logger = logging.getLogger(__name__)

# Reserved for future SAF entry-point discovery (see module docstring).
EXT_TRANSPORT = "sparkrun.transport"

DEFAULT_TRANSPORT = "ssh"

_REGISTRY: dict[str, type[Transport]] = {
    "ssh": SshTransport,
    "thunder": ThunderTransport,
}

# Provider transports gated behind a feature flag (off by default).  A cluster
# declaring a gated transport fails closed unless the flag is enabled — the
# transport is never silently downgraded to plain SSH.
_TRANSPORT_FEATURE: dict[str, str] = {
    "thunder": "transports.thunder",
}


def _require_transport_enabled(name: str) -> None:
    """Raise :class:`TransportError` if *name* is feature-gated and disabled.

    Resolves the flag context-free (env → config.yaml → channel default) via
    :func:`sparkrun.core.features.feature_gate_enabled`, so it works from the
    run/status/logs/stop hook where no ``SparkrunContext`` is threaded.
    """
    flag = _TRANSPORT_FEATURE.get(name)
    if flag is None:
        return
    from sparkrun.core.features import feature_gate_enabled

    if not feature_gate_enabled(flag):
        raise TransportError(
            "The %r transport is experimental and disabled. Enable it with: sparkrun setup features enable %s" % (name, flag)
        )


def register_transport(name: str, cls: type[Transport]) -> None:
    """Register a transport class under *name* (external providers)."""
    _REGISTRY[name] = cls
    logger.debug("Registered transport %r -> %s", name, cls.__name__)


def list_transports() -> list[str]:
    """Return the registered transport selector names."""
    return sorted(_REGISTRY)


def resolve_transport(name: str | None) -> Transport:
    """Return a :class:`Transport` instance for selector *name*.

    ``None`` / empty resolves to the default ``ssh`` transport.  An unknown
    selector raises :class:`TransportError` (never a silent fallback — a cluster
    that declares ``transport: foo`` must not silently run over plain SSH).
    """
    key = name or DEFAULT_TRANSPORT
    cls = _REGISTRY.get(key)
    if cls is None:
        raise TransportError("Unknown transport %r (known: %s)" % (key, ", ".join(list_transports())))
    return cls()


def prepare_cluster_transport(cluster: "ClusterDefinition | None", *, dry_run: bool = False) -> None:
    """Run the transport ``prepare`` step for *cluster* before any SSH.

    The single call site helper: resolve the cluster's transport and invoke
    :meth:`Transport.prepare`.  Short-circuits the default ``ssh`` transport
    without constructing anything so existing clusters pay zero cost.  A
    ``None`` cluster (or one with no ``transport`` attribute) is treated as
    plain SSH.
    """
    if cluster is None:
        return
    name = getattr(cluster, "transport", None) or DEFAULT_TRANSPORT
    if name == DEFAULT_TRANSPORT:
        return
    _require_transport_enabled(name)
    resolve_transport(name).prepare(cluster, dry_run=dry_run)


__all__ = [
    "EXT_TRANSPORT",
    "DEFAULT_TRANSPORT",
    "Transport",
    "TransportError",
    "SshTransport",
    "ThunderTransport",
    "register_transport",
    "list_transports",
    "resolve_transport",
    "prepare_cluster_transport",
]
