"""Public Tailscale errors for :mod:`sparkrun.api.tailscale`.

Orchestration-level failures (:mod:`sparkrun.orchestration.tailscale`) are
translated into these :class:`~sparkrun.api._errors.SparkrunError` subclasses at
the api boundary, so callers can ``except SparkrunError`` uniformly.
"""

from __future__ import annotations

from sparkrun.api._errors import SparkrunError


class TailscaleNotConfigured(SparkrunError):
    """No Tailscale OAuth client credentials are available."""


class TailscaleAuthFailed(SparkrunError):
    """Tailscale rejected the OAuth credentials or a derived token."""


class TailscaleSetupError(SparkrunError):
    """A Tailscale join / teardown / API operation failed."""


class TailscaleExposeError(SparkrunError):
    """The inference endpoint could not be resolved for tailnet exposure."""


__all__ = [
    "TailscaleNotConfigured",
    "TailscaleAuthFailed",
    "TailscaleSetupError",
    "TailscaleExposeError",
]
