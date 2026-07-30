"""Experimental ``sparkrun run`` path for the Kubernetes JobSet launcher.

Gated behind the ``api.run.k8s`` feature flag (off by default).  When the
resolved executor is ``k8s`` *and* the flag is on, :func:`sparkrun.api.run`
delegates here instead of the SSH-oriented ``launch_inference`` path.

This first integration is intentionally narrow: **solo (single-pod)**
launches on a **homogeneous** cluster.  It reuses the runtime's own
``resolve_container`` / ``generate_command`` to produce the image + serve
command, resolves the single GPU class from the node inventory, and submits
a Kueue-admitted JobSet via :func:`sparkrun.api.k8s.launch_jobset`.

Multi-node and hybrid runs raise a clear error pointing at
``sparkrun setup k8s launch`` (the lower-level surface that already handles
per-rank GPU-class layout).  Kueue + JobSet must be installed first
(``sparkrun setup k8s kueue --install``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import SparkrunError
from sparkrun.api._models import RunOptions, RunResult

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def run_k8s(
    options: RunOptions,
    sctx: "SparkrunContext",
    *,
    recipe,
    runtime,
    cluster_def,
    host_list,
    placement,
    is_solo: bool,
    cluster_id: str,
    intent_id: str,
    placement_token: str,
    effective_scheduler,
    started_at: float,
) -> RunResult:
    """Submit a solo k8s JobSet launch and return a :class:`RunResult`."""
    from sparkrun import api
    from sparkrun.api._run import _resolve_scheduler_name

    world_size = 1 if is_solo else (placement.total_ranks if placement is not None else (len(host_list) or 1))
    if world_size != 1:
        raise SparkrunError(
            "The experimental k8s run path (api.run.k8s) currently supports solo "
            "single-pod launches only; use `sparkrun setup k8s launch` for multi-node "
            "(requested world_size=%d)." % world_size
        )

    overrides = dict(options.overrides)
    image = runtime.resolve_container(recipe, overrides)
    serve_command = runtime.generate_command(recipe, overrides, is_cluster=False, num_nodes=1)

    # Kube target: prefer executor_config overrides, else the config k8s block
    # (resolved inside make_client / launch_jobset).
    exec_cfg = options.executor_config or {}
    kubeconfig = exec_cfg.get("kubeconfig")
    kube_context = exec_cfg.get("k8s_context")
    namespace = exec_cfg.get("k8s_namespace")

    model = _resolve_single_gpu_class(sctx, kubeconfig=kubeconfig, context=kube_context)

    env = {str(k): str(v) for k, v in (getattr(recipe, "env", {}) or {}).items()}

    result = api.k8s.launch_jobset(
        sctx,
        name=cluster_id,
        rank_models=[model],
        image=image,
        serve_command=serve_command,
        env=env,
        namespace=namespace,
        kubeconfig=kubeconfig,
        context=kube_context,
        dry_run=options.dry_run,
        follow=options.follow,
    )

    serve_port = 0
    try:
        serve_port = int(overrides.get("port") or 0)
    except (TypeError, ValueError):
        serve_port = 0

    metadata = {
        "recipe": getattr(recipe, "qualified_name", None) or getattr(recipe, "name", None),
        "model": getattr(recipe, "model", None),
        "container_image": image,
        "k8s_jobset": result.name,
        "k8s_gpu_class": model,
        "k8s_feasible": result.feasible,
    }
    return RunResult(
        cluster_id=cluster_id,
        host_list=tuple(host_list),
        placement=placement,
        scheduler=_resolve_scheduler_name(effective_scheduler, sctx),
        runtime=runtime.runtime_name,
        executor="k8s",
        started_at=started_at,
        dry_run=options.dry_run,
        is_solo=True,
        rc=0,
        serve_command=serve_command,
        container_image=image,
        serve_port=serve_port,
        metadata=metadata,
        intent_id=intent_id,
        placement_token=placement_token,
    )


def _resolve_single_gpu_class(sctx, *, kubeconfig, context) -> str:
    """Return the cluster's single GPU model, or raise if 0 / >1 classes."""
    from sparkrun.api.k8s._errors import ClusterUnreachable
    from sparkrun.api.k8s._ops import make_client
    from sparkrun.orchestration.k8s.errors import K8sError
    from sparkrun.orchestration.k8s.inventory import probe_nodes

    client = make_client(sctx, kubeconfig=kubeconfig, context=context)
    try:
        nodes = probe_nodes(client, gpu_only=True)
    except K8sError as exc:
        raise ClusterUnreachable(str(exc)) from exc

    classes = sorted({n.hardware.accelerators[0].model for n in nodes if n.has_accelerators})
    if len(classes) != 1:
        raise SparkrunError(
            "The k8s run path needs exactly one GPU class in the cluster (found %s); "
            "use `sparkrun setup k8s launch` with an explicit layout for hybrid clusters." % (classes or "none")
        )
    return classes[0]


__all__ = ["run_k8s"]
