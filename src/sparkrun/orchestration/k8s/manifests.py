"""Kubernetes manifest builders for sparkrun-managed objects.

Pure dict builders + a multi-document YAML renderer — no cluster access.
Everything sparkrun creates carries the managed-by label so it can be
listed / cleaned up as a set.
"""

from __future__ import annotations

import yaml

MANAGED_BY_LABEL = {"app.kubernetes.io/managed-by": "sparkrun"}


def _labels(extra: dict[str, str] | None = None) -> dict[str, str]:
    labels = dict(MANAGED_BY_LABEL)
    if extra:
        labels.update(extra)
    return labels


def default_runner_rules() -> list[dict]:
    """RBAC policy rules for the sparkrun runner service account.

    A cluster-wide role scoped to exactly the verbs sparkrun needs — not
    ``cluster-admin``.  Includes ``batch/jobs`` so sparkrun can drive pod
    creation via a Job that survives a CLI disconnect during startup, and
    ``pods/exec`` for the kubectl-exec transport trajectory.
    """
    return [
        {
            "apiGroups": [""],
            "resources": ["pods", "pods/log", "pods/status"],
            "verbs": ["get", "list", "watch", "create", "delete"],
        },
        {
            "apiGroups": [""],
            "resources": ["pods/exec"],
            "verbs": ["create"],
        },
        {
            "apiGroups": [""],
            "resources": ["services", "configmaps", "secrets"],
            "verbs": ["get", "list", "watch", "create", "delete"],
        },
        {
            "apiGroups": [""],
            "resources": ["nodes"],
            "verbs": ["get", "list"],
        },
        {
            "apiGroups": [""],
            "resources": ["namespaces"],
            "verbs": ["get", "list"],
        },
        {
            "apiGroups": ["batch"],
            "resources": ["jobs", "jobs/status"],
            "verbs": ["get", "list", "watch", "create", "delete"],
        },
    ]


def namespace_manifest(namespace: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": namespace, "labels": _labels()},
    }


def service_account_manifest(name: str, namespace: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": name, "namespace": namespace, "labels": _labels()},
    }


def cluster_role_manifest(name: str, rules: list[dict]) -> dict:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRole",
        "metadata": {"name": name, "labels": _labels()},
        "rules": rules,
    }


def cluster_role_binding_manifest(name: str, role_name: str, sa_name: str, sa_namespace: str) -> dict:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRoleBinding",
        "metadata": {"name": name, "labels": _labels()},
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "ClusterRole",
            "name": role_name,
        },
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": sa_name,
                "namespace": sa_namespace,
            }
        ],
    }


def render_manifests(docs: list[dict]) -> str:
    """Render manifest dicts as a multi-document YAML stream."""
    return yaml.safe_dump_all(docs, default_flow_style=False, sort_keys=False)


__all__ = [
    "MANAGED_BY_LABEL",
    "default_runner_rules",
    "namespace_manifest",
    "service_account_manifest",
    "cluster_role_manifest",
    "cluster_role_binding_manifest",
    "render_manifests",
]
