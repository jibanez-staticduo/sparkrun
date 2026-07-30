"""Host-substrate telemetry provider (scope ``"host"``).

Wraps the existing :class:`~sparkrun.core.monitoring.ClusterMonitor` (bash
``host_monitor.sh`` over SSH) / :class:`NvMonitorClusterMonitor` (nv-monitor +
Prometheus) — the same streaming machinery the ``cluster monitor`` TUI has
always used — behind the substrate-neutral :class:`TelemetrySession` contract.
Docker and local clusters both resolve to the ``"host"`` scope and share this
one telemetry source (the host's nvidia-smi / proc), which is exactly why
telemetry is keyed by scope rather than by executor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.orchestration.telemetry._base import TelemetryProvider, TelemetrySession

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.monitoring import HostTelemetry

logger = logging.getLogger(__name__)


class _HostTelemetrySession(TelemetrySession):
    """A :class:`TelemetrySession` backed by a running ``ClusterMonitor``."""

    def __init__(self, monitor) -> None:
        self._monitor = monitor

    def snapshot(self) -> dict[str, "HostTelemetry"]:
        from sparkrun.core.monitoring import HostTelemetry

        return {host: HostTelemetry(host=host, sample=state.latest, error=state.error) for host, state in self._monitor.states.items()}

    def close(self) -> None:
        try:
            self._monitor.stop()
        except Exception:  # noqa: BLE001 - teardown must never raise
            logger.debug("Telemetry session close failed", exc_info=True)


class HostTelemetryProvider(TelemetryProvider):
    """Telemetry for SSH-host clusters (docker / local) — the default substrate."""

    scope = "host"

    def open(
        self,
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
        interval: int = 2,
        config: "SparkrunConfig | None" = None,
        backend: str | None = None,
    ) -> TelemetrySession:
        from sparkrun.core.monitoring import ClusterMonitor

        ssh_kwargs = ssh_kwargs or {}
        # Backend resolution mirrors the ``cluster monitor`` CLI: explicit hint,
        # then the persisted ``monitor_backend`` config, then bash (always
        # available — nv-monitor needs deployed binaries).
        chosen = (backend or getattr(config, "monitor_backend", None) or "bash").lower()
        if chosen == "nv-monitor":
            from sparkrun.core.monitoring import NvMonitorClusterMonitor

            monitor = NvMonitorClusterMonitor(list(hosts), ssh_kwargs, interval=interval)
        else:
            monitor = ClusterMonitor(list(hosts), ssh_kwargs, interval=interval)
        monitor.start()
        return _HostTelemetrySession(monitor)
