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
    K8sError,
    KubectlDownloadError,
    KubectlNotFoundError,
    ServiceAccountSetupError,
)
from sparkrun.orchestration.k8s.job import LauncherJobResult, LauncherJobSpec
from sparkrun.orchestration.k8s.kubectl import normalize_release_version
from sparkrun.orchestration.k8s.serviceaccount import ServiceAccountSpec, DEFAULT_SA_NAME

from ._errors import (
    ClusterUnreachable,
    KubectlUnavailable,
    KueueSetupError,
    LauncherJobError,
    ServiceAccountError,
)

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.orchestration.k8s import ClusterInfo, KubectlBinary, NodeInfo, ServiceAccountResult
    from sparkrun.orchestration.k8s.kueue import KueueSetupResult, KueueStatus
    from sparkrun.orchestration.k8s.scheduling import FeasibilityReport, GpuRequest


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


def list_nodes(
    sctx: "SparkrunContext | None" = None,
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
    selector: str | None = None,
    gpu_only: bool = False,
) -> "list[NodeInfo]":
    """Return the cluster's nodes as :class:`NodeInfo` (hardware + capacity).

    Synthesizes :class:`~sparkrun.core.hardware.HostHardware` from GPU
    Feature Discovery / Node Feature Discovery labels — the k8s-native
    analog of an SSH hardware probe, needing only ``nodes`` read RBAC.
    """
    from sparkrun.orchestration.k8s.inventory import probe_nodes as _probe_nodes

    sctx = resolve_sctx(sctx)
    client = make_client(sctx, kubeconfig=kubeconfig, context=context)
    try:
        return _probe_nodes(client, selector=selector, gpu_only=gpu_only)
    except K8sError as exc:
        raise ClusterUnreachable(str(exc)) from exc


def check_feasibility(
    sctx: "SparkrunContext | None" = None,
    *,
    requests: "list[GpuRequest]",
    kubeconfig: str | None = None,
    context: str | None = None,
) -> "FeasibilityReport":
    """Check whether *requests* (per-model GPU demand) fit the live cluster.

    Reads the node inventory and returns a :class:`FeasibilityReport`; a
    fast pre-submit check that turns a would-be-Pending workload into a
    clear verdict.  Never raises for an infeasible request — inspect
    ``report.feasible``.
    """
    from sparkrun.orchestration.k8s.inventory import probe_nodes as _probe_nodes
    from sparkrun.orchestration.k8s.scheduling import check_feasibility as _check

    sctx = resolve_sctx(sctx)
    client = make_client(sctx, kubeconfig=kubeconfig, context=context)
    try:
        nodes = _probe_nodes(client, gpu_only=True)
    except K8sError as exc:
        raise ClusterUnreachable(str(exc)) from exc
    return _check(nodes, requests)


def kueue_status(
    sctx: "SparkrunContext | None" = None,
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> "KueueStatus":
    """Report whether Kueue + JobSet CRDs are present on the cluster."""
    from sparkrun.orchestration.k8s import kueue as _kueue

    sctx = resolve_sctx(sctx)
    client = make_client(sctx, kubeconfig=kubeconfig, context=context)
    try:
        return _kueue.detect(client)
    except K8sError as exc:
        raise ClusterUnreachable(str(exc)) from exc


def setup_kueue(
    sctx: "SparkrunContext | None" = None,
    *,
    install: bool = False,
    kueue_version: str | None = None,
    jobset_version: str | None = None,
    namespace: str | None = None,
    kubeconfig: str | None = None,
    context: str | None = None,
    dry_run: bool = False,
) -> "KueueSetupResult":
    """Ensure Kueue + JobSet are present, then provision sparkrun's queues.

    Runs under the admin context.  When *install* and a component is
    missing, its pinned release manifest is applied and the controller
    awaited; without *install*, a missing component raises
    :class:`KueueSetupError` pointing the user to re-run with install.
    ResourceFlavors / ClusterQueue / LocalQueue are derived from the node
    inventory (one flavor per GPU node-class).  *dry_run* renders the
    provisioning manifests without installing or applying anything.
    """
    from sparkrun.orchestration.k8s import kueue as _kueue
    from sparkrun.orchestration.k8s.inventory import probe_nodes as _probe_nodes
    from sparkrun.orchestration.k8s.kueue import KueueSetupResult

    sctx = resolve_sctx(sctx)
    ns = namespace or DEFAULT_SA_NAME
    kv = kueue_version or sctx.config.kueue_version or _kueue.DEFAULT_KUEUE_VERSION
    jv = jobset_version or sctx.config.jobset_version or _kueue.DEFAULT_JOBSET_VERSION

    # A real (non-dry-run) client: reads (detect, node inventory) are safe and
    # required even for a dry run; only install / apply are gated on dry_run.
    client = make_client(sctx, kubeconfig=kubeconfig, context=context, namespace=ns)

    try:
        status = _kueue.detect(client)

        installed_kueue = installed_jobset = False
        if not dry_run:
            if not status.jobset_installed:
                if not install:
                    raise KueueSetupError("JobSet is not installed. Re-run with install=True (CLI: --install).")
                _kueue.install_component(
                    client, url=_kueue.JOBSET_MANIFEST_URL % jv, namespace=_kueue.JOBSET_NAMESPACE, deployment=_kueue.JOBSET_DEPLOYMENT
                )
                installed_jobset = True
            if not status.kueue_installed:
                if not install:
                    raise KueueSetupError("Kueue is not installed. Re-run with install=True (CLI: --install).")
                _kueue.install_component(
                    client, url=_kueue.KUEUE_MANIFEST_URL % kv, namespace=_kueue.KUEUE_NAMESPACE, deployment=_kueue.KUEUE_DEPLOYMENT
                )
                installed_kueue = True

        nodes = _probe_nodes(client, gpu_only=True)
        docs, flavors = _kueue.build_provision_manifests(nodes, namespace=ns)
        manifests_yaml = _kueue.render_manifests(docs)

        result = KueueSetupResult(
            namespace=ns,
            cluster_queue=_kueue.DEFAULT_CLUSTER_QUEUE_NAME,
            local_queue=_kueue.DEFAULT_QUEUE_NAME,
            flavors=flavors,
            manifests_yaml=manifests_yaml,
            dry_run=dry_run,
            kueue_version=kv,
            jobset_version=jv,
            installed_kueue=installed_kueue,
            installed_jobset=installed_jobset,
        )
        if dry_run:
            return result

        apply_res = client.apply(manifests_yaml)
        if not apply_res.success:
            raise KueueSetupError("Failed to apply Kueue provisioning manifests: %s" % apply_res.stderr.strip()[:400])
        result.provisioned = True
        return result
    except _kueue.KueueError as exc:
        raise KueueSetupError(str(exc)) from exc


def run_launcher_job(
    sctx: "SparkrunContext | None" = None,
    *,
    name: str,
    image: str | None = None,
    command: list[str] | None = None,
    script: str | None = None,
    namespace: str | None = None,
    service_account: str = DEFAULT_SA_NAME,
    env: dict[str, str] | None = None,
    ttl_seconds: int | None = None,
    active_deadline_seconds: int | None = None,
    kubeconfig: str | None = None,
    context: str | None = None,
    follow: bool = False,
    dry_run: bool = False,
) -> LauncherJobResult:
    """Apply an in-cluster launcher Job that runs *command* or *script*.

    The Job runs under the sparkrun service account and survives a CLI
    disconnect.  Exactly one of *command* / *script* must be given.  The
    image resolves from *image* or ``config.k8s_launcher_image``; missing
    both raises :class:`LauncherJobError`.  When *follow*, launcher logs
    stream to the terminal until interrupted (the Job keeps running).
    """
    from sparkrun.orchestration.k8s.job import DEFAULT_TTL_SECONDS

    sctx = resolve_sctx(sctx)
    ns = namespace or DEFAULT_SA_NAME
    resolved_image = image or sctx.config.k8s_launcher_image
    if not resolved_image:
        raise LauncherJobError("No launcher image: pass image= or set k8s.launcher_image in config.yaml.")

    try:
        spec = LauncherJobSpec(
            name=name,
            image=resolved_image,
            namespace=ns,
            service_account=service_account,
            command=command,
            script=script,
            env=dict(env or {}),
            ttl_seconds=DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds,
            active_deadline_seconds=active_deadline_seconds,
        )
    except ValueError as exc:
        raise LauncherJobError(str(exc)) from exc

    from sparkrun.orchestration.k8s.job import render_launcher_manifests

    manifests_yaml = render_launcher_manifests(spec)
    result = LauncherJobResult(
        job_name=name,
        namespace=ns,
        image=resolved_image,
        manifests_yaml=manifests_yaml,
        dry_run=dry_run,
    )
    if dry_run:
        return result

    client = make_client(sctx, kubeconfig=kubeconfig, context=context, namespace=ns)
    apply_res = client.run_launcher_job(spec)
    if not apply_res.success:
        raise LauncherJobError("Failed to apply launcher Job %s: %s" % (name, apply_res.stderr.strip()[:400]))
    result.applied = True

    if follow:
        client.follow_job_logs(name)

    return result


__all__ = [
    "ensure_kubectl",
    "make_client",
    "cluster_info",
    "configure_service_account",
    "list_nodes",
    "check_feasibility",
    "kueue_status",
    "setup_kueue",
    "run_launcher_job",
]
