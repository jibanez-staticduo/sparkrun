"""Launcher-Job manifests — run an orchestration payload *inside* the cluster.

The job-driven launch path moves the startup driver loop off the control
machine and into a Kubernetes Job running under the sparkrun service
account.  The CLI applies the manifests and detaches; the Job survives a
CLI disconnect.

Two payload shapes are supported:

- ``command`` — an explicit argv run directly in *image* (the "full
  sparkrun image" path: e.g. ``["sparkrun", "run", ...]`` in a published
  sparkrun container).
- ``script`` — a bash script mounted from a ConfigMap and executed with
  ``bash`` (works with any image that has bash; handy for a kubectl-only
  launcher).

Exactly one of the two must be set.  Job hardening: ``backoffLimit: 0``
(a failed launch is not blindly retried), ``restartPolicy: Never``, an
optional ``activeDeadlineSeconds`` startup ceiling, and
``ttlSecondsAfterFinished`` for automatic cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .manifests import MANAGED_BY_LABEL, render_manifests

DEFAULT_TTL_SECONDS = 3600
DEFAULT_SERVICE_ACCOUNT = "sparkrun"
DEFAULT_NAMESPACE = "sparkrun"
LAUNCHER_COMPONENT_LABEL = {"app.kubernetes.io/component": "launcher"}


@dataclass
class LauncherJobSpec:
    """Inputs describing a launcher Job."""

    name: str
    image: str
    namespace: str = DEFAULT_NAMESPACE
    service_account: str = DEFAULT_SERVICE_ACCOUNT
    command: list[str] | None = None
    script: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    ttl_seconds: int | None = DEFAULT_TTL_SECONDS
    active_deadline_seconds: int | None = None
    backoff_limit: int = 0
    script_mount_path: str = "/sparkrun"
    script_filename: str = "launch.sh"

    def __post_init__(self):
        if bool(self.command) == bool(self.script):
            raise ValueError("LauncherJobSpec requires exactly one of `command` or `script`")

    @property
    def configmap_name(self) -> str:
        return "%s-script" % self.name


@dataclass
class LauncherJobResult:
    """Outcome of applying a launcher Job."""

    job_name: str
    namespace: str
    image: str
    manifests_yaml: str
    dry_run: bool
    applied: bool = False


def _labels(spec: LauncherJobSpec) -> dict[str, str]:
    labels = dict(MANAGED_BY_LABEL)
    labels.update(LAUNCHER_COMPONENT_LABEL)
    labels.update(spec.labels)
    return labels


def _env_list(env: dict[str, str]) -> list[dict]:
    return [{"name": k, "value": str(v)} for k, v in sorted(env.items())]


def configmap_manifest(spec: LauncherJobSpec) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": spec.configmap_name, "namespace": spec.namespace, "labels": _labels(spec)},
        "data": {spec.script_filename: spec.script},
    }


def job_manifest(spec: LauncherJobSpec) -> dict:
    labels = _labels(spec)
    container: dict = {"name": "launcher", "image": spec.image}
    if spec.env:
        container["env"] = _env_list(spec.env)

    pod_spec: dict = {
        "serviceAccountName": spec.service_account,
        "restartPolicy": "Never",
        "containers": [container],
    }

    if spec.script is not None:
        script_path = "%s/%s" % (spec.script_mount_path.rstrip("/"), spec.script_filename)
        container["command"] = ["bash", script_path]
        container["volumeMounts"] = [{"name": "sparkrun-script", "mountPath": spec.script_mount_path}]
        pod_spec["volumes"] = [{"name": "sparkrun-script", "configMap": {"name": spec.configmap_name, "defaultMode": 0o755}}]
    else:
        container["command"] = list(spec.command or [])

    job_spec: dict = {
        "backoffLimit": spec.backoff_limit,
        "template": {"metadata": {"labels": labels}, "spec": pod_spec},
    }
    if spec.ttl_seconds is not None:
        job_spec["ttlSecondsAfterFinished"] = spec.ttl_seconds
    if spec.active_deadline_seconds is not None:
        job_spec["activeDeadlineSeconds"] = spec.active_deadline_seconds

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": spec.name, "namespace": spec.namespace, "labels": labels},
        "spec": job_spec,
    }


def build_launcher_manifests(spec: LauncherJobSpec) -> list[dict]:
    """Return the manifest docs for *spec* (ConfigMap first when script-based)."""
    docs: list[dict] = []
    if spec.script is not None:
        docs.append(configmap_manifest(spec))
    docs.append(job_manifest(spec))
    return docs


def render_launcher_manifests(spec: LauncherJobSpec) -> str:
    return render_manifests(build_launcher_manifests(spec))


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "LauncherJobSpec",
    "LauncherJobResult",
    "configmap_manifest",
    "job_manifest",
    "build_launcher_manifests",
    "render_launcher_manifests",
]
