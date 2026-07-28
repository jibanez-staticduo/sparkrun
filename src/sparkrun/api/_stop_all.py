"""``sparkrun.api.stop_all`` — discover and stop every sparkrun workload.

The discovery-driven counterpart to :func:`sparkrun.api.stop`: instead of
naming a workload, sweep a set of hosts for whatever sparkrun launched
and tear all of it down.

This is console-free like the rest of ``sparkrun.api`` — it returns a
:class:`~sparkrun.api._models.StopAllResult` describing what was found,
what was removed, and what failed; rendering and exit codes are the
caller's business.  (It previously lived inside the CLI's ``--all``
branch, which meant the GUI sidecar and any other library caller had no
way to do it and no way to inherit its fixes.)

Two failure modes are distinguished, and neither is ever silently
reported as success:

- **discovery errors** — a host that could not be queried.  It is *not*
  "nothing to stop"; it may be running containers we never saw.
- **teardown failures** — a host whose containers did not confirm gone.
  Job metadata for anything with a container there is deliberately
  retained, because the workload is still live and the metadata is how
  it is found again.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.api._models import StopAllResult

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterStatusResult
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def stop_all(
    hosts: list[str] | tuple[str, ...],
    *,
    cluster: "str | ClusterDefinition | None" = None,
    cache_dir: str | None = None,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    discovered: "ClusterStatusResult | None" = None,
    sctx: "SparkrunContext | None" = None,
) -> StopAllResult:
    """Discover and stop every sparkrun workload on *hosts*.

    Args:
        hosts: Hosts to sweep.
        cluster: Optional cluster name/definition for executor + transport
            resolution (forwarded to :func:`sparkrun.api.status_report`).
        cache_dir: Cache directory holding job metadata.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log the teardown without executing it.  Discovery still
            runs for real — there is nothing to preview otherwise.
        discovered: A snapshot from :func:`sparkrun.api.status_report` to
            act on instead of re-querying.  Lets a caller that already
            displayed the discovery (the CLI) avoid a second sweep.
        sctx: Optional shared :class:`SparkrunContext`.

    Returns:
        :class:`~sparkrun.api._models.StopAllResult`.
    """
    from sparkrun.api._status import status_report
    from sparkrun.orchestration.docker import parse_teardown_removed
    from sparkrun.orchestration.primitives import cleanup_containers_by_host

    host_list = list(hosts)
    result = discovered
    if result is None:
        result = status_report(host_list, cluster=cluster, ssh_kwargs=ssh_kwargs, cache_dir=cache_dir, sctx=sctx)

    discovery_errors = dict(result.errors)

    if result.total_containers == 0:
        return StopAllResult(
            discovered=result,
            jobs_stopped=0,
            containers_removed=0,
            discovery_errors=discovery_errors,
        )

    host_containers = _containers_by_host(result)

    results = cleanup_containers_by_host(host_containers, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    failed_hosts: dict[str, str] = {}
    for host in host_containers:
        r = results.get(host)
        if r is None:
            failed_hosts[host] = "teardown did not run"
        elif not r.success:
            failed_hosts[host] = (r.stderr or r.stdout).strip() or ("exit code %d" % r.returncode)

    if dry_run:
        # Nothing was executed, so nothing failed and nothing was removed;
        # report the discovered shape as what *would* be stopped.
        return StopAllResult(
            discovered=result,
            jobs_stopped=len(result.groups) + len(result.solo_entries),
            containers_removed=result.total_containers,
            hosts_stopped=tuple(host_containers),
        )

    containers_removed = sum(parse_teardown_removed(r.stdout) for r in results.values())

    # Drop job metadata only for jobs with no container left behind.
    jobs_stopped = 0
    for cid, group in result.groups.items():
        if any(member_host in failed_hosts for member_host, _role, _status, _image in group.members):
            continue
        _forget_job(cid, cache_dir)
        jobs_stopped += 1
    for entry in result.solo_entries:
        if entry.host in failed_hosts:
            continue
        solo_cid = entry.name.removesuffix("_solo") if entry.name.endswith("_solo") else entry.name
        _forget_job(solo_cid, cache_dir)
        jobs_stopped += 1

    return StopAllResult(
        discovered=result,
        jobs_stopped=jobs_stopped,
        containers_removed=containers_removed,
        hosts_stopped=tuple(h for h in host_containers if h not in failed_hosts),
        hosts_failed=failed_hosts,
        discovery_errors=discovery_errors,
    )


def _containers_by_host(result: "ClusterStatusResult") -> dict[str, list[str]]:
    """Map host → container names to tear down, from a discovery snapshot."""
    host_containers: dict[str, list[str]] = {}
    for cid, group in result.groups.items():
        for host, role, _status, _image in group.members:
            host_containers.setdefault(host, []).append("%s_%s" % (cid, role))
    for entry in result.solo_entries:
        host_containers.setdefault(entry.host, []).append(entry.name)
    return host_containers


def _forget_job(cluster_id: str, cache_dir: str | None) -> None:
    """Remove job metadata, tolerating an already-absent record."""
    from sparkrun.orchestration.job_metadata import remove_job_metadata

    try:
        remove_job_metadata(cluster_id, cache_dir=cache_dir)
    except Exception:
        logger.debug("Failed to remove job metadata for %s", cluster_id, exc_info=True)
