"""Telemetry seam: how sparkrun samples a cluster's *resource utilization*.

A :class:`TelemetryProvider` owns the *telemetry* concern — per-host (or
per-node) GPU/CPU/memory/thermal sampling — for one **substrate**, keyed by the
same :attr:`~sparkrun.orchestration.executors._base.Executor.status_scope` used
for occupancy status.  It is the telemetry peer of ``Executor.query_status``:

- occupancy ("what runs where") → ``api.status`` → ``Executor.query_status``
- telemetry ("how loaded is each host") → ``api.telemetry`` → ``TelemetryProvider``

Telemetry is a property of the *substrate*, not of any single executor: on the
host substrate docker and local share one telemetry source (the host's
nvidia-smi / proc), so a provider is selected per scope (``host`` / ``k8s`` /
``modal``), not per executor.  The host provider streams over SSH; a k8s/modal
provider would poll its own control plane — the client sees the same
:class:`HostTelemetry` shape regardless.

Discovery mirrors executors/transports: SAF :class:`~scitrera_app_framework.Plugin`
classes registered at :data:`EXT_TELEMETRY` and discovered via
``find_types_in_modules("sparkrun.orchestration.telemetry", TelemetryProvider)``
in :func:`sparkrun.core.bootstrap.init_sparkrun`.  Selection is by :attr:`scope`
(see :func:`sparkrun.orchestration.telemetry.get_telemetry_provider`).

Providers are **stateless** (the SAF singleton is reused); a live collection's
state lives on the :class:`TelemetrySession` returned by :meth:`open`.
"""

from __future__ import annotations

import logging
from typing import ClassVar, TYPE_CHECKING

from scitrera_app_framework import Plugin, Variables

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.monitoring import HostTelemetry

logger = logging.getLogger(__name__)

EXT_TELEMETRY = "sparkrun.telemetry"


class TelemetrySession:
    """A live telemetry collection over a host set.

    Opened by :meth:`TelemetryProvider.open`; callers poll :meth:`snapshot`
    on their own cadence (the Textual TUI ticks it; the ``api.live_monitor``
    generator ticks it on an interval) and call :meth:`close` when done.
    Implementations own whatever background machinery collection needs (SSH
    reader threads for the host substrate, a control-plane poller for k8s).

    Usable as a context manager so ``close`` is guaranteed.
    """

    def snapshot(self) -> dict[str, "HostTelemetry"]:
        """Return the latest per-host telemetry, keyed by host.

        Non-blocking: returns whatever has been collected so far (a host with
        no reading yet has ``sample=None``).  Every input host is present.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Stop collection and release resources (idempotent)."""
        raise NotImplementedError

    def __enter__(self) -> "TelemetrySession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TelemetryProvider(Plugin):
    """Base class for substrate telemetry providers (one per status scope).

    A concrete provider is a SAF plugin selected by :attr:`scope` (matched
    against a cluster's status scope).  The registered SAF singleton is returned
    directly by :func:`sparkrun.orchestration.telemetry.get_telemetry_provider`
    since providers are stateless — live state is on the :class:`TelemetrySession`.
    """

    eager = False  # don't initialize until requested

    # --- Subclass must define ---
    scope: ClassVar[str] = ""
    """Substrate selector; matches ``Executor.status_scope`` (e.g. ``"host"``).

    The blank base is skipped by bootstrap discovery — only named providers
    register.
    """

    # --- Optional feature gating (mirrors Executor/Transport) ---
    required_feature_flag: ClassVar[str | None] = None

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.telemetry.%s" % self.scope

    def extension_point_name(self, v: Variables) -> str:
        return EXT_TELEMETRY

    def is_enabled(self, v: Variables) -> bool:
        # False for multi-extension plugins — prevents SAF's single-extension
        # cache from short-circuiting subsequent registrations under this point.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        if self.required_feature_flag:
            from sparkrun.core.features import feature_gate_enabled

            return feature_gate_enabled(self.required_feature_flag, v)
        return True

    def initialize(self, v: Variables, logger=None) -> "TelemetryProvider":
        return self

    # --- Provider contract ---

    def open(
        self,
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
        interval: int = 2,
        config: "SparkrunConfig | None" = None,
        backend: str | None = None,
    ) -> TelemetrySession:
        """Start a telemetry :class:`TelemetrySession` over *hosts*.

        *interval* is the desired sampling cadence in seconds.  *backend* is an
        optional collection-backend hint a provider may honor or ignore (the
        host provider maps it to bash / nv-monitor; others ignore it).
        Implementations must return promptly (defer any slow setup to background
        work) so a client can start rendering immediately.
        """
        raise NotImplementedError("TelemetryProvider subclasses must implement open()")
