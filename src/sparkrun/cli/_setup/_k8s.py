"""``sparkrun setup k8s`` — Kubernetes setup subcommands.

Thin CLI layer over :mod:`sparkrun.api.k8s`:

- ``setup k8s kubectl`` — resolve / download / list the ``kubectl`` binary.
- ``setup k8s info`` — probe the target cluster.
- ``setup k8s sa`` — configure the sparkrun service account + RBAC.

The api layer holds all logic; these commands only parse flags, call the
api, and render results to the TTY.
"""

from __future__ import annotations

import functools

import click

from .._common import _get_context
from . import setup


def kube_options(func):
    """Shared ``--kubeconfig / --context / --namespace`` decorator."""

    @click.option("--kubeconfig", default=None, help="Path to a kubeconfig file (overrides config / $KUBECONFIG).")
    @click.option("--context", "kube_context", default=None, help="kubeconfig context to target.")
    @click.option("--namespace", "-n", default=None, help="Namespace to target.")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@setup.group("k8s")
def setup_k8s():
    """Kubernetes setup: kubectl acquisition, cluster info, service account.

    sparkrun manages its own ``kubectl`` under ``~/.cache/sparkrun/kubectl/``
    and can configure a scoped service account for driving workloads. These
    commands are the foundation for the (experimental) k8s executor.
    """


# ---------------------------------------------------------------------------
# kubectl
# ---------------------------------------------------------------------------


@setup_k8s.command("kubectl")
@click.option("--version", default=None, help="Download / resolve a specific version (e.g. v1.31.0).")
@click.option("--list", "list_", is_flag=True, help="List cached kubectl binaries and exit.")
@click.option("--path", "show_path", is_flag=True, help="Print only the resolved binary path.")
@click.option("--no-download", is_flag=True, help="Do not download; fail if nothing is cached / on PATH.")
@click.pass_context
def setup_k8s_kubectl(ctx, version, list_, show_path, no_download):
    """Resolve, download, or list the managed kubectl binary."""
    from sparkrun import api
    from sparkrun.orchestration.k8s import list_cached

    sctx = _get_context(ctx)

    if list_:
        cached = list_cached(sctx.config.cache_dir)
        if not cached:
            click.echo("No cached kubectl binaries.")
            return
        for binary in cached:
            click.echo("%-12s %-14s %s" % (binary.version, "%s-%s" % (binary.os_name, binary.arch), binary.path))
        return

    try:
        binary = api.k8s.ensure_kubectl(sctx, version=version, download=not no_download)
    except api.k8s.KubectlUnavailable as exc:
        raise click.ClickException(str(exc))

    if show_path:
        click.echo(str(binary.path))
        return
    click.echo("kubectl %s (%s) [%s]" % (binary.version or "unknown", "%s-%s" % (binary.os_name, binary.arch), binary.source))
    click.echo("  path: %s" % binary.path)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@setup_k8s.command("info")
@kube_options
@click.option("--no-pin", is_flag=True, help="Do not pin the server version for this context.")
@click.pass_context
def setup_k8s_info(ctx, kubeconfig, kube_context, namespace, no_pin):
    """Probe the target cluster and show client / server versions."""
    from sparkrun import api

    sctx = _get_context(ctx)
    try:
        info = api.k8s.cluster_info(
            sctx,
            kubeconfig=kubeconfig,
            context=kube_context,
            namespace=namespace,
            pin=not no_pin,
        )
    except api.k8s.KubectlUnavailable as exc:
        raise click.ClickException(str(exc))

    click.echo("Context:        %s" % (info.current_context or "(default)"))
    click.echo("Namespace:      %s" % (info.namespace or "(default)"))
    click.echo("Client version: %s" % (info.client_version or "unknown"))
    if info.reachable:
        click.echo("Server version: %s" % info.server_version)
        click.secho("Cluster reachable.", fg="green")
    else:
        click.echo("Server version: unreachable")
        click.secho("Cluster unreachable: %s" % (info.message or ""), fg="red")


# ---------------------------------------------------------------------------
# sa
# ---------------------------------------------------------------------------


@setup_k8s.command("sa")
@click.argument("name", default="sparkrun")
@kube_options
@click.option("--no-create-namespace", is_flag=True, help="Do not create the namespace (assume it exists).")
@click.option("--token-duration", default=None, help="Token lifetime for kubectl create token (e.g. 8760h).")
@click.option("--no-kubeconfig", is_flag=True, help="Do not write a derived kubeconfig.")
@click.option("--dry-run", is_flag=True, help="Print the manifests without applying.")
@click.pass_context
def setup_k8s_sa(ctx, name, kubeconfig, kube_context, namespace, no_create_namespace, token_duration, no_kubeconfig, dry_run):
    """Configure the sparkrun service account, RBAC, and a scoped kubeconfig.

    Creates a cluster-wide ClusterRole scoped to only the verbs sparkrun
    needs (pods, pods/log, pods/exec, batch/jobs, services, configmaps,
    secrets, nodes) — not cluster-admin.
    """
    from sparkrun import api

    sctx = _get_context(ctx)
    try:
        result = api.k8s.configure_service_account(
            sctx,
            name=name,
            namespace=namespace,
            kubeconfig=kubeconfig,
            context=kube_context,
            create_namespace=not no_create_namespace,
            token_duration=token_duration,
            write_kubeconfig=not no_kubeconfig,
            dry_run=dry_run,
        )
    except (api.k8s.ClusterUnreachable, api.k8s.ServiceAccountError, api.k8s.KubectlUnavailable) as exc:
        raise click.ClickException(str(exc))

    if dry_run:
        click.echo(result.manifests_yaml)
        click.secho("(dry-run — nothing applied)", fg="yellow")
        return

    click.secho("Service account configured.", fg="green")
    click.echo("  service account: %s/%s" % (result.namespace, result.name))
    click.echo("  cluster role:    %s" % result.cluster_role)
    click.echo("  binding:         %s" % result.binding)
    if result.server:
        click.echo("  server:          %s" % result.server)
    if result.kubeconfig_path:
        click.echo("  kubeconfig:      %s (0600)" % result.kubeconfig_path)
        click.echo("")
        click.echo("Point the k8s executor at it via config.yaml:")
        click.echo("  executor_config:")
        click.echo("    kubeconfig: %s" % result.kubeconfig_path)
