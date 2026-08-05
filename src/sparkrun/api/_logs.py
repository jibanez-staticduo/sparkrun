"""``sparkrun.api.logs`` — stream logs from a running sparkrun workload.

The composition layer of the log path.  The runtime says *what* to read
(:meth:`~sparkrun.runtimes.base.RuntimePlugin.log_sources`), the executor
says *how* to read it on its substrate
(:meth:`~sparkrun.orchestration.executors._base.Executor.read_logs_cmd`),
:mod:`sparkrun.orchestration.logs` runs the commands and merges the output,
and this module resolves the workload and wires the three together.

Returns a lazy :class:`Iterator` of :class:`LogLine` records: the CLI
renders them, the desktop sidecar streams them, tests consume them.
Resolution and validation happen eagerly at call time (so a bad target
raises here rather than on the first ``next()``); only the reading is lazy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

from sparkrun.api._errors import JobNotFound, SparkrunError
from sparkrun.core.log_source import SCOPE_ALL, SCOPE_HEAD, LogLine

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


def logs(
    cluster_id: str | None = None,
    *,
    recipe: "str | Recipe | None" = None,
    hosts: list[str] | tuple[str, ...] | None = None,
    overrides: dict | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    scope: str = SCOPE_HEAD,
    follow: bool = False,
    tail: int | None = None,
    cache_dir: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> Iterator[LogLine]:
    """Yield :class:`LogLine` records from a running workload.

    Either ``cluster_id`` *or* (``recipe`` + a host source) is required.
    When both are given, ``cluster_id`` wins — the same contract as
    :func:`sparkrun.api.stop`.  The recipe form resolves through live intent
    discovery (:func:`~sparkrun.api._resolve.discover_cluster_id_by_intent`)
    rather than deriving a cluster_id, so it finds the workload regardless of
    which placement token the scheduler assigned it.

    Args:
        cluster_id: The cluster ID returned by :func:`sparkrun.api.run`.
        recipe: Recipe name or object, when addressing the workload by
            recipe instead of by id.
        hosts: Explicit host list.  Required for the recipe form; for the
            cluster_id form it defaults to the hosts recorded in
            ``~/.cache/sparkrun/jobs/``.
        overrides: Recipe overrides used at launch.  They participate in the
            intent, so port / parallelism overrides must match for the
            recipe form to resolve.
        cluster: Optional cluster name or definition.
        scope: :data:`SCOPE_HEAD` (default) reads only the primary log;
            :data:`SCOPE_ALL` reads every worker/rank too.
        follow: Stream new lines as they arrive.  With several sources this
            interleaves them in arrival (i.e. time) order; without it,
            sources are read rank-grouped.  See
            :mod:`sparkrun.orchestration.logs` for the ordering contract.
        tail: Start this many lines from the end of each source; ``None``
            reads the whole log.
        cache_dir: Override for the sparkrun cache root.  Defaults to
            ``sctx.config.cache_dir`` when *sctx* is provided.
        sctx: Optional shared :class:`SparkrunContext`.

    Raises:
        JobNotFound: No hosts can be determined, or no running workload
            matches the recipe.
        AmbiguousWorkload: The recipe matches several running workloads.
        SparkrunError: Neither ``cluster_id`` nor ``recipe`` was given, or
            *scope* is not a valid value.
    """
    from sparkrun.api._resolve import (
        discover_cluster_id_by_intent,
        prepare_transport,
        resolve_cluster,
        resolve_recipe,
    )
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.orchestration.job_metadata import generate_intent_id, load_job_metadata
    from sparkrun.orchestration.logs import read_log_sources
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    if scope not in (SCOPE_HEAD, SCOPE_ALL):
        raise SparkrunError("Invalid log scope %r: expected %r or %r" % (scope, SCOPE_HEAD, SCOPE_ALL))

    if cache_dir is None and sctx is not None:
        try:
            cache_dir = str(sctx.config.cache_dir)
        except Exception:
            cache_dir = None

    resolved_recipe = None
    if not cluster_id:
        if recipe is None:
            raise SparkrunError("api.logs requires cluster_id or recipe+hosts")
        cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
        prepare_transport(cluster_def)
        resolved_recipe = resolve_recipe(recipe, sctx=sctx)
        target_hosts = list(cluster_def.hosts)
        cluster_id = discover_cluster_id_by_intent(
            generate_intent_id(resolved_recipe, overrides=overrides),
            target_hosts,
            cluster_def=cluster_def,
            cache_dir=cache_dir,
            sctx=sctx,
        )
        meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
    else:
        meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
        if hosts:
            target_hosts = list(hosts)
        elif meta and meta.get("hosts"):
            target_hosts = list(meta["hosts"])
        else:
            raise JobNotFound("No hosts known for cluster_id %r" % cluster_id)
        cluster_def = resolve_cluster(cluster, target_hosts, sctx=sctx)
        prepare_transport(cluster_def)

    runtime = _resolve_runtime_for_job(meta, cluster_id, recipe=resolved_recipe, sctx=sctx)
    executor = resolve_executor(
        cluster=cluster_def,
        cli_overrides=_executor_overrides_from_meta(meta),
        rootless=False,
        auto_user=False,
        v=sctx.variables if sctx is not None else None,
    )

    config = sctx.config if sctx is not None else None
    if config is not None and getattr(cluster_def, "user", None):
        try:
            config.ssh_user = cluster_def.user
        except Exception:
            logger.debug("Failed to apply cluster SSH user", exc_info=True)

    sources = runtime.log_sources(
        cluster_id,
        target_hosts,
        is_solo=len(target_hosts) <= 1,
        scope=scope,
    )

    ssh_kwargs = build_ssh_kwargs(config) if config else {}

    # Liveness precheck: check ALL nodes (not just the head log source) —
    # in a multi-node job the head may have crashed while workers are still
    # running, and the user should still be able to read logs from surviving
    # nodes.  Uses ``executor.query_status`` (one parallel SSH sweep) — the
    # same source of truth as ``api.status`` and ``check_job_running``.
    # See ``_verify_log_source_alive`` for the full decision tree.
    all_sources = (
        sources
        if scope == SCOPE_ALL
        else runtime.log_sources(
            cluster_id,
            target_hosts,
            is_solo=len(target_hosts) <= 1,
            scope=SCOPE_ALL,
        )
    )
    _verify_log_source_alive(executor, all_sources, ssh_kwargs, cluster_id, cache_dir, scope=scope)

    return read_log_sources(
        executor,
        sources,
        follow=follow,
        tail=tail,
        ssh_kwargs=ssh_kwargs,
    )


def _verify_log_source_alive(
    executor, sources, ssh_kwargs: dict, cluster_id: str, cache_dir: str | None, *, scope: str = SCOPE_HEAD
) -> None:
    """Raise :class:`JobNotFound` if the workload isn't running.

    Uses :meth:`Executor.query_status` — the same source of truth as
    ``api.status``, ``check_job_running``, and the monitor TUI — to do
    one parallel SSH sweep across all hosts, rather than probing each
    source sequentially.

    Three outcomes:

    1. **Head alive** (container in the status snapshot) → proceed.
    2. **Head dead, some workers alive** → raise with a pointer to
       ``--all-sources`` so the user can read from surviving nodes.
       With ``scope=SCOPE_ALL`` the reader can handle this, so proceed.
    3. **All sources dead** (none in the snapshot) → follow-up
       ``docker ps -a`` on the head distinguishes **stopped-but-exists**
       (preserve metadata, investigation hints) from **fully removed**
       (clean up stale metadata).  The status API uses ``docker ps``
       (running only), so it can't make this distinction itself.

    Best-effort: if the status query fails or a host is unreachable
    (in ``ClusterStatus.errors``), the precheck is skipped so a network
    blip never causes a false "not running" verdict.
    """
    if not sources:
        return
    from sparkrun.orchestration.primitives import run_command_on_host

    head = sources[0]

    # One parallel SSH sweep via the status API — same source of truth
    # as `api.status`, `check_job_running`, and the monitor TUI.
    all_hosts = list(dict.fromkeys(s.host for s in sources))
    try:
        snapshot = executor.query_status(all_hosts, ssh_kwargs=ssh_kwargs)
    except Exception:  # noqa: BLE001 — best-effort; let log reader surface its own error
        return

    def _container_status(host: str, container_name: str) -> bool | None:
        """True if running, False if confirmed absent, None if host unreachable."""
        occ = snapshot.for_host(host)
        if occ is None:
            return None  # host in errors / unreachable → inconclusive
        for w in occ.workloads:
            if w.cluster_id == cluster_id:
                for c in w.containers:
                    if c.name == container_name:
                        return True
                # Cluster_id present but this container not listed —
                # could be a different container (head vs worker) or
                # the executor didn't populate `containers`.  Check
                # workload presence as a fallback: if the cluster_id
                # has *any* workload on this host, the container might
                # be running under a name we don't recognize.
                if not w.containers:
                    return True  # containers not populated → assume alive
        return False  # host reachable, container not in snapshot

    head_status = _container_status(head.host, head.container)
    if head_status is None:
        return  # inconclusive → skip precheck
    if head_status:
        return  # head is alive → proceed normally

    # Head is confirmed dead.  Check workers — if any are still alive,
    # point the user at ``--all-sources`` rather than letting the reader
    # fail on the dead head container with a raw docker error.
    alive_worker_hosts = []
    for source in sources[1:]:
        status = _container_status(source.host, source.container)
        if status is None:
            return  # inconclusive → skip precheck
        if status:
            alive_worker_hosts.append(source.host)

    if alive_worker_hosts:
        # With ``--all-sources`` the reader can read from the surviving
        # workers, so proceed.  With the default (head-only) scope the
        # reader would fail on the dead head container — raise a helpful
        # error pointing at ``--all-sources`` instead.
        if scope == SCOPE_ALL:
            return
        raise JobNotFound(
            "Workload %s is partially running — the head container on %s has stopped, "
            "but worker containers are still alive on %s.\n"
            "Try `sparkrun logs %s --all-sources` to read from surviving nodes, "
            "or `sparkrun stop %s` to clean up." % (cluster_id, head.host, ", ".join(alive_worker_hosts), cluster_id, cluster_id)
        )

    # Every source is confirmed dead (none in the status snapshot).
    # Determine whether the head container still exists (stopped) or is
    # fully gone.  The status API uses ``docker ps`` (running only), so
    # it can't make this distinction — a follow-up ``docker ps -a`` is
    # the only way.
    #
    # Only rc=0 is actionable: empty stdout = gone, non-empty = stopped.
    # Any other rc (255 SSH failure, 127 docker missing, -1 timeout) is
    # inconclusive — assume the container exists (safer: preserves
    # metadata for investigation rather than deleting it on a network blip).
    exists_cmd = "docker ps -a --filter name=^%s$ --format '{{.ID}}'" % head.container
    try:
        exists_result = run_command_on_host(head.host, exists_cmd, ssh_kwargs=ssh_kwargs, timeout=10, quiet=True)
        if exists_result.returncode == 0:
            container_exists = bool(exists_result.stdout.strip())
        else:
            container_exists = True  # inconclusive → preserve metadata
    except Exception:  # noqa: BLE001 — if the follow-up fails, assume stopped (safer for investigation)
        container_exists = True

    if container_exists:
        raise JobNotFound(
            "Workload %s is not currently running on %s (container %r has stopped).\n"
            "The job metadata has been preserved so you can investigate:\n"
            "  docker logs %s       # check why it stopped\n"
            "  docker inspect %s    # exit code, OOM-killed, etc.\n"
            "Run `sparkrun stop %s` to clean up when ready."
            % (cluster_id, head.host, head.container, head.container, head.container, cluster_id)
        )

    # Container is fully gone — metadata is stale.  Remove it so
    # ``logs <TAB>`` stops suggesting this dead workload.
    try:
        from sparkrun.orchestration.job_metadata import remove_job_metadata

        remove_job_metadata(cluster_id, cache_dir=cache_dir)
    except Exception:
        logger.debug("Failed to remove stale metadata for %s", cluster_id, exc_info=True)

    raise JobNotFound(
        "Workload %s is not currently running on %s (container %r no longer exists).\n"
        "The stale job metadata has been removed. Run `sparkrun status` to see running workloads." % (cluster_id, head.host, head.container)
    )


def _resolve_runtime_for_job(meta: dict | None, cluster_id: str, *, recipe=None, sctx: "SparkrunContext | None"):
    """Resolve the runtime that owns this workload's logs.

    The runtime is what knows where its logs live, so getting this right is
    what keeps ``api.logs`` from reading a container that doesn't exist: a
    Ray job's head is ``{cid}_head`` with its serve output in an
    in-container file, while a native job's head is ``{cid}_node_0``.
    Guessing wrong yields "No such container" or an empty stream — which is
    exactly what the previous hardcoded implementation did.

    Prefers the *recipe* when the caller addressed the workload by one: the
    recipe carries the runtime directly, so the recipe form keeps working
    even when the job-metadata cache is missing (a job launched from another
    control machine, or a cleared cache).  Falls back to metadata for the
    cluster_id form, where the recipe isn't known.
    """
    from sparkrun.core.bootstrap import get_runtime

    runtime_name = getattr(recipe, "runtime", None) or (meta or {}).get("runtime")
    if not runtime_name:
        raise JobNotFound(
            "No job metadata (or no runtime recorded) for cluster_id %r, so sparkrun can't tell where "
            "this workload's logs live. Address it by recipe instead: api.logs(recipe=..., hosts=...)." % cluster_id
        )
    try:
        return get_runtime(runtime_name, sctx.variables if sctx is not None else None)
    except ValueError as e:
        raise SparkrunError("Cannot resolve runtime %r for cluster_id %r: %s" % (runtime_name, cluster_id, e)) from e


def _executor_overrides_from_meta(meta: dict | None) -> dict | None:
    """Recover the launching executor's selector + config from job metadata.

    Reading logs must go through the executor that *launched* the workload —
    a ``local``-executor job has no container to ``docker exec`` into.
    """
    if not meta:
        return None
    overrides: dict = {}
    if meta.get("executor"):
        overrides["executor"] = meta["executor"]
    if isinstance(meta.get("executor_config"), dict):
        overrides.update(meta["executor_config"])
    return overrides or None


__all__ = ["logs"]
