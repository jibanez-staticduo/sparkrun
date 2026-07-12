"""Thunder Compute transport.

``prepare`` refreshes each host's connection details out-of-band (fresh IP/port
from the API, SSH key provisioned via ``add_key``, managed ssh alias rewritten)
so that once it returns, every host in the cluster is a plain ``ssh`` target.

Each Thunder instance is its own single-host cluster.  The cluster stores the
stable instance uuid in ``provider_ref`` and the ssh alias ``tnr-<uuid>`` in
``hosts``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.transports.base import Transport, TransportError
from sparkrun.transports.thunder import api as thunder_api
from sparkrun.transports.thunder import ssh_alias

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import HostHardware
    from sparkrun.transports.thunder.api import ThunderInstance

logger = logging.getLogger(__name__)


def _uuid_for_host(cluster: "ClusterDefinition", host: str) -> str:
    """Resolve the Thunder instance uuid backing *host*.

    Prefers the cluster's ``provider_ref`` (the canonical stable uuid); falls
    back to parsing the ``tnr-<uuid>`` alias when the host isn't the primary.
    """
    provider_ref = getattr(cluster, "provider_ref", None)
    if provider_ref:
        return str(provider_ref)
    if host.startswith(ssh_alias.ALIAS_PREFIX):
        return host[len(ssh_alias.ALIAS_PREFIX) :]
    return host


class ThunderTransport(Transport):
    """Transport backing clusters whose host is a Thunder Compute instance."""

    name = "thunder"

    def prepare(self, cluster: "ClusterDefinition", *, dry_run: bool = False) -> None:
        token, base = thunder_api.load_token()
        wanted = {_uuid_for_host(cluster, h) for h in cluster.hosts}
        instances = {i.uuid: i for i in thunder_api.list_instances(token, base)}

        selected: list["ThunderInstance"] = []
        for uuid in sorted(wanted):
            inst = instances.get(uuid)
            if inst is None:
                raise TransportError("Thunder instance %s (cluster '%s') was not found — was it deleted?" % (uuid, cluster.name))
            if not inst.is_running:
                raise TransportError(
                    "Thunder instance %s (cluster '%s') is %s, not RUNNING." % (uuid, cluster.name, inst.status or "unknown")
                )
            if not inst.ip:
                raise TransportError("Thunder instance %s has no IP yet; try again shortly." % uuid)
            selected.append(inst)

        if dry_run:
            for inst in selected:
                logger.info("[dry-run] Would refresh Thunder alias %s -> %s:%d", ssh_alias.alias_for(inst), inst.ip, inst.port)
            return

        entries = [(inst, ssh_alias.ensure_key(token, base, inst)) for inst in selected]
        ssh_alias.write_aliases(entries)
        logger.debug("Prepared Thunder transport for cluster '%s' (%d host(s))", cluster.name, len(entries))


# ---------------------------------------------------------------------------
# Import-time helpers (used by `sparkrun cluster import thunder`)
# ---------------------------------------------------------------------------

# System-RAM → nominal GPU VRAM is not carried in the list endpoint; a probe is
# the source of truth.  This seed only records vendor/model/count so an
# ``--no-probe`` import still yields sane placement metadata.
_VENDOR = "nvidia"


def seed_hardware(inst: "ThunderInstance") -> "HostHardware":
    """Best-effort :class:`HostHardware` from API fields (no SSH probe).

    Records vendor/model/count only — ``memory_gb`` is left ``None`` (unknown
    from the list endpoint); a probe fills it in.  Used for ``--no-probe``.
    """
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware

    model = (inst.gpu_type or "").strip().lower().replace(" ", "-") or "unknown"
    return HostHardware(
        accelerators=[
            AcceleratorSpec(
                vendor=_VENDOR,
                model=model,
                count=max(1, inst.num_gpus),
                memory_gb=None,
                capabilities=frozenset({"cuda"}),
            )
        ],
        notes="thunder seed (unprobed): %s x%d" % (inst.gpu_type or "?", inst.num_gpus),
    )


def list_running_instances() -> tuple[str, str, list["ThunderInstance"]]:
    """Return ``(token, base, running_instances)`` for the import command."""
    token, base = thunder_api.load_token()
    thunder_api.validate(token, base)
    running = [i for i in thunder_api.list_instances(token, base) if i.is_running]
    return token, base, running
