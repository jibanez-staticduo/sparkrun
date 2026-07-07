"""Console-free Kubernetes operations for the sparkrun library API.

Every function accepts an optional ``sctx`` (a
:class:`~sparkrun.core.context.SparkrunContext`); when ``None`` a fresh
default session is built.  Nothing here writes to stdout/stderr or calls
``sys.exit`` — failures raise :class:`~sparkrun.api._errors.SparkrunError`
subclasses (see :mod:`._errors`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.api._context import resolve_sctx
from sparkrun.orchestration.k8s import (
    KubectlClient,
    configure_service_account as _configure_sa,
    ensure_kubectl as _ensure_kubectl,
    probe_cluster,
    require_reachable,
    resolve_kube_target,
)
from sparkrun.orchestration.k8s.errors import (
    ClusterUnreachableError,
    KubectlDownloadError,
    KubectlNotFoundError,
    ServiceAccountSetupError,
)
from sparkrun.orchestration.k8s.kubectl import normalize_release_version
from sparkrun.orchestration.k8s.serviceaccount import ServiceAccountSpec, DEFAULT_SA_NAME

from ._errors import ClusterUnreachable, KubectlUnavailable, ServiceAccountError

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.orchestration.k8s import ClusterInfo, KubectlBinary, ServiceAccountResult


def ensure_kubectl(
    sctx: "SparkrunContext | None" = None,
    *,
    version: str | None = None,
    context: str | None = None,
    download: bool = True,
) -> "KubectlBinary":
    """Resolve (and, if needed, download) a usable ``kubectl`` binary.

    *version* overrides everything; otherwise the config pin
    (``k8s.kubectl.version``) and any per-context server-matched pin are
    consulted.  Raises :class:`KubectlUnavailable` when nothing resolves.
    """
    sctx = resolve_sctx(sctx)
    config = sctx.config
    resolved_version = version or config.kubectl_version or config.kubectl_pinned_version(context)
    try:
        return _ensure_kubectl(
            config.cache_dir,
            version=resolved_version,
            explicit_path=config.kubectl_path,
            allow_download=download,
        )
    except (KubectlNotFoundError, KubectlDownloadError) as exc:
        raise KubectlUnavailable(str(exc)) from exc


def make_client(
    sctx: "SparkrunContext",
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
    namespace: str | None = None,
    version: str | None = None,
    dry_run: bool = False,
) -> KubectlClient:
    """Build a :class:`KubectlClient` for a resolved kube target."""
    target = resolve_kube_target(sctx.config, kubeconfig=kubeconfig, context=context, namespace=namespace)
    binary = ensure_kubectl(sctx, version=version, context=target.context)
    return KubectlClient(
        binary,
        kubeconfig=target.kubeconfig,
        context=target.context,
        namespace=target.namespace,
        dry_run=dry_run,
    )


def cluster_info(
    sctx: "SparkrunContext | None" = None,
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
    namespace: str | None = None,
    pin: bool = True,
) -> "ClusterInfo":
    """Probe the target cluster and return a :class:`ClusterInfo` snapshot.

    When *pin* is True and the probe succeeds, the server's release version
    is recorded as a per-context ``kubectl`` pin so subsequent calls match
    the client binary to the cluster (downloading it on demand).
    """
    sctx = resolve_sctx(sctx)
    client = make_client(sctx, kubeconfig=kubeconfig, context=context, namespace=namespace)
    info = probe_cluster(client)
    if pin and info.reachable and info.current_context and info.server_version:
        release = normalize_release_version(info.server_version)
        if release and sctx.config.kubectl_pinned_version(info.current_context) != release:
            sctx.config.pin_kubectl_version(info.current_context, release)
            sctx.config.save()
    return info


def configure_service_account(
    sctx: "SparkrunContext | None" = None,
    *,
    name: str = DEFAULT_SA_NAME,
    namespace: str | None = None,
    kubeconfig: str | None = None,
    context: str | None = None,
    create_namespace: bool = True,
    token_duration: str | None = None,
    write_kubeconfig: bool = True,
    dry_run: bool = False,
) -> "ServiceAccountResult":
    """Create the sparkrun service account + RBAC and a derived kubeconfig.

    When *dry_run*, no cluster mutation happens and the returned result
    carries the rendered manifests only.  Otherwise the cluster is
    required to be reachable, the manifests are applied, a token is
    minted, and (when *write_kubeconfig*) a scoped kubeconfig is written
    ``0600`` under the sparkrun config root.
    """
    sctx = resolve_sctx(sctx)
    ns = namespace or DEFAULT_SA_NAME

    spec = ServiceAccountSpec(
        name=name,
        namespace=ns,
        create_namespace=create_namespace,
        token_duration=token_duration,
    )

    # A dry run only renders manifests — don't resolve / download kubectl
    # or touch the cluster.
    if dry_run:
        return _configure_sa(None, spec, kubeconfig_out=None, dry_run=True)

    client = make_client(sctx, kubeconfig=kubeconfig, context=context, namespace=ns)

    kubeconfig_out: Path | None = None
    if write_kubeconfig:
        from sparkrun.core.config import get_config_root

        kubeconfig_out = get_config_root(sctx.variables) / "k8s" / ("%s.kubeconfig" % name)

    try:
        require_reachable(client)
        return _configure_sa(client, spec, kubeconfig_out=kubeconfig_out, dry_run=False)
    except ClusterUnreachableError as exc:
        raise ClusterUnreachable(str(exc)) from exc
    except ServiceAccountSetupError as exc:
        raise ServiceAccountError(str(exc)) from exc


__all__ = ["ensure_kubectl", "make_client", "cluster_info", "configure_service_account"]
