"""Telemetry providers — substrate resource sampling, keyed by status scope.

Public facade over the telemetry seam (see :mod:`._base`).  Providers are
SAF-discovered and selected by :attr:`TelemetryProvider.scope`, which matches a
cluster's status scope (``host`` / ``k8s`` / ``modal``).  Core ships the
``host`` provider; k8s / modal providers ship in their plugins.

``get_telemetry_provider(scope)`` returns the stateless SAF singleton (or
``None`` when no provider covers that substrate — the caller then reports "no
telemetry available" rather than failing).
"""

from __future__ import annotations

import logging

from scitrera_app_framework import Variables, get_extensions

from sparkrun.core.monitoring import HostTelemetry
from sparkrun.orchestration.telemetry._base import (
    EXT_TELEMETRY,
    TelemetryProvider,
    TelemetrySession,
)

logger = logging.getLogger(__name__)


def _telemetry_extensions(v: Variables | None = None) -> dict:
    if v is None:
        from sparkrun.core.bootstrap import get_variables

        v = get_variables()
    return get_extensions(EXT_TELEMETRY, v=v)


def list_telemetry_scopes(v: Variables | None = None) -> list[str]:
    """Return the scopes with a registered (enabled) telemetry provider."""
    try:
        exts = _telemetry_extensions(v)
    except Exception:
        logger.debug("Telemetry provider enumeration failed", exc_info=True)
        return []
    return sorted({p.scope for p in exts.values() if getattr(p, "scope", "")})


def get_telemetry_provider(scope: str, v: Variables | None = None) -> TelemetryProvider | None:
    """Return the telemetry provider for *scope*, or ``None`` if none is registered.

    Returns the stateless SAF singleton — a live collection's state lives on the
    :class:`TelemetrySession` from :meth:`TelemetryProvider.open`.
    """
    try:
        exts = _telemetry_extensions(v)
    except Exception:
        logger.debug("Telemetry provider lookup failed for scope %r", scope, exc_info=True)
        return None
    for plugin in exts.values():
        if getattr(plugin, "scope", "") == scope:
            return plugin
    return None


__all__ = [
    "EXT_TELEMETRY",
    "HostTelemetry",
    "TelemetryProvider",
    "TelemetrySession",
    "get_telemetry_provider",
    "list_telemetry_scopes",
]
