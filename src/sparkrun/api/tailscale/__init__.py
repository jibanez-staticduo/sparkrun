"""Public library API for Tailscale operations.

Console-free surface over :mod:`sparkrun.orchestration.tailscale`. Mirrors the
:mod:`sparkrun.api.k8s` conventions: dataclass returns, typed
:class:`~sparkrun.api._errors.SparkrunError` subclasses, never writes to
stdout/stderr.

Functions:

- :func:`join` — install Tailscale + join hosts to the tailnet (OAuth JIT key).
- :func:`status` — read each host's local tailnet state.
- :func:`expose` — compute (and optionally wire) a tailnet URL for the endpoint.
- :func:`down` — log hosts out of the tailnet (optionally remove their devices).
"""

from __future__ import annotations

from ._errors import (
    TailscaleAuthFailed,
    TailscaleExposeError,
    TailscaleNotConfigured,
    TailscaleSetupError,
)
from ._ops import (
    DEFAULT_ENDPOINT_PORT,
    DownResult,
    ExposeResult,
    HostDownState,
    HostJoinState,
    HostTailscaleStatus,
    JoinResult,
    StatusResult,
    down,
    expose,
    join,
    status,
)

__all__ = [
    # Functions
    "join",
    "status",
    "expose",
    "down",
    # Data models
    "JoinResult",
    "HostJoinState",
    "StatusResult",
    "HostTailscaleStatus",
    "ExposeResult",
    "DownResult",
    "HostDownState",
    "DEFAULT_ENDPOINT_PORT",
    # Errors
    "TailscaleNotConfigured",
    "TailscaleAuthFailed",
    "TailscaleSetupError",
    "TailscaleExposeError",
]
