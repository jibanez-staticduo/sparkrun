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
    from sparkrun.orchestration.primitives import cleanup_containers_by_host, merge_teardown_results
    from sparkrun.orchestration.teardown import parse_teardown_removed

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

    # Discovery is cross-executor (docker + local share the "host" scope), so
    # teardown must be too: each executor is dispatched with only the
    # workloads *it* reported, then the per-host verdicts are recombined.
    # Sending the whole set to one executor is what let a `local` workload
    # survive a "successful" stop --all.
    results: dict = {}
    for executor_name, grouped in _group_by_executor(host_containers, result).items():
        results = merge_teardown_results(
            results,
            cleanup_containers_by_host(
                grouped,
                ssh_kwargs=ssh_kwargs,
                dry_run=dry_run,
                executor=_resolve_teardown_executor(executor_name, cluster, host_list, sctx),
            ),
        )

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


def _group_by_executor(
    host_containers: dict[str, list[str]],
    result: "ClusterStatusResult",
) -> dict[str, dict[str, list[str]]]:
    """Split a host→containers map into one such map per reporting executor.

    Uses :attr:`ClusterStatusResult.container_executors`, stamped during
    discovery.  Containers with no attribution (a snapshot from an executor
    that predates the field, or a hand-built one in a test) group under ``""``
    and are torn down with the cluster's default executor — the historical
    behaviour, and the safe reading of "we don't know".
    """
    grouped: dict[str, dict[str, list[str]]] = {}
    for host, names in host_containers.items():
        for name in names:
            executor_name = result.container_executors.get((host, name), "")
            grouped.setdefault(executor_name, {}).setdefault(host, []).append(name)
    return grouped


def _resolve_teardown_executor(
    executor_name: str,
    cluster: "str | ClusterDefinition | None",
    hosts: list[str],
    sctx: "SparkrunContext | None",
):
    """Build the :class:`Executor` that tears down *executor_name*'s workloads.

    Resolved through the same chain the status sweep used
    (``resolve_executor`` with the name as a CLI-level override), so the
    executor's config matches the one that reported the workload — that
    matters for substrates whose teardown depends on it, e.g. the ``local``
    executor's ``pid_dir``, which is where the pidfile it must signal lives.

    Returns ``None`` — meaning "let the primitive use its default" — for an
    unattributed group or when resolution fails.  A teardown that can't
    identify its substrate should still attempt the historical one rather than
    skip the host entirely.
    """
    from sparkrun.orchestration.executor import resolve_executor

    if not executor_name:
        return None
    try:
        from sparkrun.api._resolve import resolve_cluster

        cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
        return resolve_executor(
            cluster=cluster_def,
            cli_overrides={"executor": executor_name},
            rootless=False,
            auto_user=False,
            config=sctx.config if sctx is not None else None,
            v=sctx.variables if sctx is not None else None,
        )
    except Exception:
        logger.warning(
            "Could not resolve executor %r for teardown; falling back to the default executor",
            executor_name,
            exc_info=True,
        )
        return None


def _forget_job(cluster_id: str, cache_dir: str | None) -> None:
    """Remove job metadata, tolerating an already-absent record."""
    from sparkrun.orchestration.job_metadata import remove_job_metadata

    try:
        remove_job_metadata(cluster_id, cache_dir=cache_dir)
    except Exception:
        logger.debug("Failed to remove job metadata for %s", cluster_id, exc_info=True)
