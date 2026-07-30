"""``sparkrun.api.stop`` — stop a running sparkrun workload.

Two modes:

- **By cluster_id**: provide the literal ``cluster_id`` (as returned
  by :func:`sparkrun.api.run`); the API loads the job metadata, picks
  the executor that originally launched it, and runs ``stop_cmd``
  against each candidate container name on every host.
- **By recipe+hosts+overrides**: derive the same ``cluster_id`` the
  launcher would have produced and dispatch identically.  Useful for
  ``sparkrun stop <recipe>`` semantics.

Returns a :class:`StopResult` summarizing how many containers were
removed and any per-host errors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import JobNotFound, SparkrunError
from sparkrun.api._models import StopResult

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


def stop(
    *,
    cluster_id: str | None = None,
    recipe: "str | Recipe | None" = None,
    hosts: list[str] | tuple[str, ...] | None = None,
    overrides: dict | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    cache_dir: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> StopResult:
    """Stop a running sparkrun workload.

    Either ``cluster_id`` *or* (``recipe`` + a host source) is required.
    When both are provided, ``cluster_id`` wins.

    Args:
        sctx: Optional shared :class:`SparkrunContext` for chained
            api calls (registry/cluster manager + config sharing).
    """
    from sparkrun.api._resolve import (
        discover_cluster_id_by_intent,
        resolve_cluster,
        resolve_recipe,
    )
    from sparkrun.orchestration.docker import parse_teardown_removed
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.orchestration.job_metadata import (
        generate_intent_id,
        load_job_metadata,
        remove_job_metadata,
    )

    # Derive cluster_id from recipe+hosts when not given explicitly.
    if not cluster_id:
        if recipe is None:
            raise SparkrunError("api.stop requires cluster_id or recipe+hosts")
        cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
        resolved_recipe = resolve_recipe(recipe, sctx=sctx)
        intent_id = generate_intent_id(resolved_recipe, overrides=overrides)
        # Default cache_dir from sctx.config when not explicitly passed.
        if cache_dir is None and sctx is not None:
            try:
                cache_dir = str(sctx.config.cache_dir)
            except Exception:
                cache_dir = None
        target_hosts = list(cluster_def.hosts)

        # Status-driven discovery: ask the executor what's running on
        # the supplied hosts and filter for cluster_ids matching the
        # computed intent.  Load-aware schedulers may have placed the
        # workload on a different host set than ``hosts`` — that's the
        # whole point of separating intent from placement.  Here we
        # accept that the *user's* host scope is the authoritative
        # discovery range.
        cluster_id = discover_cluster_id_by_intent(
            intent_id,
            target_hosts,
            cluster_def=cluster_def,
            cache_dir=cache_dir,
            sctx=sctx,
        )
        meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
    else:
        # cluster_id given — load metadata to recover hosts/executor.
        if cache_dir is None and sctx is not None:
            try:
                cache_dir = str(sctx.config.cache_dir)
            except Exception:
                cache_dir = None
        meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
        if meta is None and not hosts and cluster is None:
            raise JobNotFound("No job metadata found for cluster_id %r and no hosts provided" % cluster_id)

        # Determine target hosts: explicit > metadata > error.
        if hosts:
            target_hosts = list(hosts)
        elif meta and meta.get("hosts"):
            target_hosts = list(meta["hosts"])
        else:
            target_hosts = []

        if not target_hosts:
            raise JobNotFound("No hosts known for cluster_id %r" % cluster_id)

        # Now that we know the hosts, build a cluster definition for
        # downstream consumers.  `resolve_cluster` synthesizes an
        # anonymous one when no explicit cluster is given.
        cluster_def = resolve_cluster(cluster, target_hosts, sctx=sctx)

    # Refresh provider-backed connection details before any SSH (no-op for ssh).
    from sparkrun.api._resolve import prepare_transport

    prepare_transport(cluster_def)

    # Resolve the executor — prefer recipe-encoded selection from metadata
    # so we use the same executor that launched the workload.
    cli_overrides: dict | None = None
    if meta:
        meta_exec = meta.get("executor")
        meta_exec_cfg = meta.get("executor_config")
        cli_overrides = {}
        if meta_exec:
            cli_overrides["executor"] = meta_exec
        if isinstance(meta_exec_cfg, dict):
            cli_overrides.update(meta_exec_cfg)
        if not cli_overrides:
            cli_overrides = None

    executor = resolve_executor(
        cluster=cluster_def,
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
        v=sctx.variables if sctx is not None else None,
    )

    container_names = executor.enumerate_containers(cluster_id, len(target_hosts))

    # ``cleanup_containers_by_host`` is the shared teardown primitive: it
    # dispatches local-vs-SSH per host, verifies the containers are
    # actually gone, and reports what it removed per host.
    from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers_by_host

    config = sctx.config if sctx is not None else _maybe_load_config()
    if config is not None and cluster_def.user:
        # Apply cluster SSH user so downstream ssh_kwargs picks it up.
        try:
            config.ssh_user = cluster_def.user
        except Exception:
            logger.debug("Failed to apply cluster SSH user", exc_info=True)
    ssh_kwargs = build_ssh_kwargs(config) if config else {}

    errors: list[str] = []
    try:
        results = cleanup_containers_by_host(
            {host: list(container_names) for host in target_hosts},
            ssh_kwargs=ssh_kwargs,
        )
    except Exception as e:  # pragma: no cover - defensive; the primitive absorbs per-host failures
        errors.append(str(e))
        results = {}

    hosts_failed = tuple(host for host in target_hosts if not (results.get(host) and results[host].success))
    removed_count = sum(parse_teardown_removed(r.stdout) for r in results.values())
    for host in hosts_failed:
        r = results.get(host)
        detail = ((r.stderr or r.stdout).strip() if r else "") or "teardown did not confirm"
        errors.append("%s: %s" % (host, detail))

    # Cleanup persistent metadata only once teardown is confirmed
    # everywhere: while a container survives, the metadata still describes
    # a live workload and is what ``stop``/``status`` use to find it again.
    if hosts_failed:
        logger.warning(
            "Keeping job metadata for %s — teardown did not confirm on: %s",
            cluster_id,
            ", ".join(hosts_failed),
        )
    else:
        try:
            remove_job_metadata(cluster_id, cache_dir=cache_dir)
        except Exception:
            logger.debug("Failed to remove job metadata for %s", cluster_id, exc_info=True)

    return StopResult(
        cluster_id=cluster_id,
        hosts_targeted=tuple(target_hosts),
        containers_removed=removed_count,
        errors=tuple(errors),
        hosts_failed=hosts_failed,
    )


def _maybe_load_config():
    """Load SparkrunConfig once for the SSH kwargs, returning ``None`` on failure."""
    try:
        from sparkrun.core.config import SparkrunConfig

        return SparkrunConfig()
    except Exception:  # pragma: no cover - defensive
        return None


__all__ = ["stop"]
