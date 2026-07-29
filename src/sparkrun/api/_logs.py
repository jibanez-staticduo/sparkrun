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
    return read_log_sources(
        executor,
        sources,
        follow=follow,
        tail=tail,
        ssh_kwargs=build_ssh_kwargs(config) if config else {},
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
