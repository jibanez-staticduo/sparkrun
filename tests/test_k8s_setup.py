"""Tests for the Kubernetes setup surface.

Covers kubectl binary acquisition/version-cache, the KubectlClient
wrapper, kube-target resolution, manifest + service-account generation,
config accessors, the api.k8s surface, and the ``setup k8s`` CLI. No real
cluster or network — urllib / subprocess are stubbed.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from sparkrun.orchestration.k8s import kubectl, manifests, serviceaccount
from sparkrun.orchestration.k8s.client import KubectlClient
from sparkrun.orchestration.k8s.connect import ClusterInfo, probe_cluster
from sparkrun.orchestration.k8s.context import resolve_kube_target
from sparkrun.orchestration.k8s.errors import KubectlDownloadError, KubectlNotFoundError
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# Platform detection & cache layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system,expected",
    [("Linux", "linux"), ("Darwin", "darwin"), ("Windows", "windows")],
)
def test_detect_os(monkeypatch, system, expected):
    monkeypatch.setattr(kubectl.platform, "system", lambda: system)
    assert kubectl.detect_os() == expected


@pytest.mark.parametrize(
    "machine,expected",
    [("x86_64", "amd64"), ("amd64", "amd64"), ("aarch64", "arm64"), ("arm64", "arm64")],
)
def test_detect_arch(monkeypatch, machine, expected):
    monkeypatch.setattr(kubectl.platform, "machine", lambda: machine)
    assert kubectl.detect_arch() == expected


def test_binary_name():
    assert kubectl.binary_name("linux") == "kubectl"
    assert kubectl.binary_name("windows") == "kubectl.exe"


def test_cached_binary_path_layout(tmp_path):
    p = kubectl.cached_binary_path(tmp_path, "v1.31.0", "linux", "arm64")
    assert p == tmp_path / "kubectl" / "v1.31.0" / "linux-arm64" / "kubectl"


def test_normalize_release_version():
    assert kubectl.normalize_release_version("v1.31.2") == "v1.31.2"
    assert kubectl.normalize_release_version("v1.31.2+ck1") == "v1.31.2"
    assert kubectl.normalize_release_version("v1.30.0-eks-abc") == "v1.30.0"
    assert kubectl.normalize_release_version("garbage") is None


def test_list_cached_sorted_newest_first(tmp_path):
    for version in ("v1.29.0", "v1.31.0", "v1.30.5"):
        p = kubectl.cached_binary_path(tmp_path, version, "linux", "arm64")
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/sh\n")
    cached = kubectl.list_cached(tmp_path, os_name="linux", arch="arm64")
    assert [b.version for b in cached] == ["v1.31.0", "v1.30.5", "v1.29.0"]


# ---------------------------------------------------------------------------
# Download + integrity
# ---------------------------------------------------------------------------


def _stub_urlopen(mapping):
    """Return a fake urlopen resolving *mapping* (url -> bytes)."""

    class _Resp:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(url, timeout=None):  # noqa: ARG001
        if url not in mapping:
            raise AssertionError("unexpected url: %s" % url)
        return _Resp(mapping[url])

    return _open


def test_download_kubectl_verifies_checksum(tmp_path, monkeypatch):
    payload = b"fake-kubectl-binary"
    digest = hashlib.sha256(payload).hexdigest()
    url = kubectl._release_url("v1.31.0", "linux", "arm64")
    monkeypatch.setattr(
        kubectl.urllib.request,
        "urlopen",
        _stub_urlopen({url: payload, url + ".sha256": ("%s\n" % digest).encode()}),
    )
    dest = kubectl.download_kubectl(tmp_path, "v1.31.0", "linux", "arm64")
    assert dest.read_bytes() == payload
    assert os.access(dest, os.X_OK)


def test_download_kubectl_rejects_bad_checksum(tmp_path, monkeypatch):
    payload = b"fake-kubectl-binary"
    url = kubectl._release_url("v1.31.0", "linux", "arm64")
    monkeypatch.setattr(
        kubectl.urllib.request,
        "urlopen",
        _stub_urlopen({url: payload, url + ".sha256": b"deadbeef\n"}),
    )
    with pytest.raises(KubectlDownloadError, match="checksum mismatch"):
        kubectl.download_kubectl(tmp_path, "v1.31.0", "linux", "arm64")


# ---------------------------------------------------------------------------
# ensure_kubectl resolution order
# ---------------------------------------------------------------------------


def test_ensure_kubectl_explicit_path(tmp_path):
    binary = tmp_path / "mykubectl"
    binary.write_text("#!/bin/sh\n")
    resolved = kubectl.ensure_kubectl(tmp_path, explicit_path=binary)
    assert resolved.source == "config"
    assert resolved.path == binary


def test_ensure_kubectl_explicit_path_missing(tmp_path):
    with pytest.raises(KubectlNotFoundError):
        kubectl.ensure_kubectl(tmp_path, explicit_path=tmp_path / "nope")


def test_ensure_kubectl_prefers_matching_cached_version(tmp_path):
    p = kubectl.cached_binary_path(tmp_path, "v1.31.0", "linux", "arm64")
    p.parent.mkdir(parents=True)
    p.write_text("#!/bin/sh\n")
    resolved = kubectl.ensure_kubectl(tmp_path, version="v1.31.0", os_name="linux", arch="arm64")
    assert resolved.source == "cache"
    assert resolved.version == "v1.31.0"


def test_ensure_kubectl_downloads_when_nothing_cached(tmp_path, monkeypatch):
    called = {}

    def _fake_download(cache_dir, version, os_name, arch, **kw):  # noqa: ARG001
        called["version"] = version
        p = kubectl.cached_binary_path(cache_dir, version, os_name, arch)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/sh\n")
        return p

    monkeypatch.setattr(kubectl, "fetch_stable_version", lambda **kw: "v1.31.0")
    monkeypatch.setattr(kubectl, "download_kubectl", _fake_download)
    resolved = kubectl.ensure_kubectl(tmp_path, allow_path=False, os_name="linux", arch="arm64")
    assert resolved.source == "download"
    assert called["version"] == "v1.31.0"


def test_ensure_kubectl_no_download_raises(tmp_path):
    with pytest.raises(KubectlNotFoundError):
        kubectl.ensure_kubectl(tmp_path, version="v9.9.9", allow_path=False, allow_download=False)


# ---------------------------------------------------------------------------
# KubectlClient
# ---------------------------------------------------------------------------


def test_client_base_args_ordering():
    client = KubectlClient("/usr/bin/kubectl", kubeconfig="/k/cfg", context="ctx", namespace="ns")
    assert client.base_args() == ["/usr/bin/kubectl", "--kubeconfig", "/k/cfg", "--context", "ctx", "-n", "ns"]


def test_client_run_returns_remote_result(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl")

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(
        "sparkrun.orchestration.k8s.client.subprocess.run",
        lambda *a, **k: _Proc(),
    )
    result = client.run(["get", "pods"])
    assert isinstance(result, RemoteResult)
    assert result.success and result.stdout == "ok"


def test_client_run_json_parses(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl")

    class _Proc:
        returncode = 0
        stdout = '{"a": 1}'
        stderr = ""

    monkeypatch.setattr("sparkrun.orchestration.k8s.client.subprocess.run", lambda *a, **k: _Proc())
    assert client.run_json(["version", "-o", "json"]) == {"a": 1}


def test_client_dry_run_short_circuits():
    client = KubectlClient("/usr/bin/kubectl", dry_run=True)
    result = client.run(["apply", "-f", "-"])
    assert result.success and result.stdout == "[dry-run]"


def test_client_exec_builds_kubectl_exec(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl", namespace="ns")
    captured = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, **k):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr("sparkrun.orchestration.k8s.client.subprocess.run", _run)
    client.exec("mypod", "echo hi", container="c1")
    cmd = captured["cmd"]
    assert "exec" in cmd and "mypod" in cmd and "-c" in cmd and "c1" in cmd and "--" in cmd


# ---------------------------------------------------------------------------
# Kube target resolution
# ---------------------------------------------------------------------------


def test_resolve_kube_target_arg_precedence(monkeypatch):
    monkeypatch.delenv("KUBECONFIG", raising=False)
    target = resolve_kube_target(None, kubeconfig="/explicit", context="c", namespace="n")
    assert target.kubeconfig == "/explicit"
    assert target.context == "c"
    assert target.namespace == "n"


def test_resolve_kube_target_env_kubeconfig(monkeypatch):
    monkeypatch.setenv("KUBECONFIG", "/env/cfg")
    target = resolve_kube_target(None)
    assert target.kubeconfig == "/env/cfg"


def test_resolve_kube_target_config_block(monkeypatch, tmp_path):
    from sparkrun.core.config import SparkrunConfig

    monkeypatch.delenv("KUBECONFIG", raising=False)
    cfg = SparkrunConfig(tmp_path / "config.yaml")
    cfg.set("k8s", {"context": "prod", "namespace": "sparkrun"})
    target = resolve_kube_target(cfg)
    assert target.context == "prod"
    assert target.namespace == "sparkrun"


# ---------------------------------------------------------------------------
# Manifests & RBAC
# ---------------------------------------------------------------------------


def test_default_runner_rules_include_jobs_and_exec():
    rules = manifests.default_runner_rules()
    flat = [(g, r, v) for rule in rules for g in rule["apiGroups"] for r in rule["resources"] for v in rule["verbs"]]
    assert ("batch", "jobs", "create") in flat
    assert ("", "pods/exec", "create") in flat
    assert ("", "pods", "watch") in flat


def test_build_manifests_no_cluster_admin(tmp_path):
    spec = serviceaccount.ServiceAccountSpec(name="sparkrun", namespace="sparkrun")
    rendered = serviceaccount.build_manifests(spec)
    assert "kind: ClusterRole" in rendered
    assert "kind: ClusterRoleBinding" in rendered
    assert "cluster-admin" not in rendered
    assert "app.kubernetes.io/managed-by: sparkrun" in rendered


def test_build_manifests_omits_namespace_when_disabled():
    spec = serviceaccount.ServiceAccountSpec(create_namespace=False)
    rendered = serviceaccount.build_manifests(spec)
    assert "kind: Namespace" not in rendered


# ---------------------------------------------------------------------------
# Service account: kubeconfig writing
# ---------------------------------------------------------------------------


def test_build_kubeconfig_binds_token():
    kc = serviceaccount.build_kubeconfig(
        server="https://api:6443",
        cluster={"certificate-authority-data": "CADATA"},
        token="tok123",
        context_name="sparkrun",
        namespace="sparkrun",
    )
    assert kc["users"][0]["user"]["token"] == "tok123"
    assert kc["clusters"][0]["cluster"]["certificate-authority-data"] == "CADATA"
    assert kc["current-context"] == "sparkrun"


def test_write_kubeconfig_is_0600(tmp_path):
    dest = tmp_path / "k8s" / "sparkrun.kubeconfig"
    written = serviceaccount.write_kubeconfig(dest, {"apiVersion": "v1", "kind": "Config"})
    mode = written.stat().st_mode & 0o777
    assert mode == 0o600


def test_configure_service_account_dry_run_does_not_apply():
    spec = serviceaccount.ServiceAccountSpec()

    class _Client:
        def apply(self, *a, **k):
            raise AssertionError("dry-run must not apply")

    result = serviceaccount.configure_service_account(_Client(), spec, dry_run=True)
    assert result.dry_run and not result.applied
    assert "ClusterRole" in result.manifests_yaml


def test_service_account_result_redacts_token():
    r = serviceaccount.ServiceAccountResult(
        name="sparkrun",
        namespace="sparkrun",
        cluster_role="sparkrun-runner",
        binding="sparkrun-runner",
        manifests_yaml="",
        dry_run=False,
        token="secret-token",
    )
    assert r.redacted().token == "***"
    assert r.token == "secret-token"


# ---------------------------------------------------------------------------
# probe_cluster
# ---------------------------------------------------------------------------


def test_probe_cluster_reachable(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl", context="ctx")

    def _run(args, **k):
        if args[:1] == ["version"]:
            return RemoteResult(
                host="ctx",
                returncode=0,
                stdout='{"clientVersion":{"gitVersion":"v1.31.0"},"serverVersion":{"gitVersion":"v1.30.2"}}',
                stderr="",
            )
        return RemoteResult(host="ctx", returncode=0, stdout="ctx", stderr="")

    monkeypatch.setattr(client, "run", _run)
    info = probe_cluster(client)
    assert info.reachable
    assert info.server_version == "v1.30.2"
    assert info.client_version == "v1.31.0"


def test_probe_cluster_unreachable(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl", context="ctx")

    def _run(args, **k):
        if args[:1] == ["version"]:
            return RemoteResult(
                host="ctx",
                returncode=1,
                stdout='{"clientVersion":{"gitVersion":"v1.31.0"}}',
                stderr="dial tcp: connection refused",
            )
        return RemoteResult(host="ctx", returncode=0, stdout="ctx", stderr="")

    monkeypatch.setattr(client, "run", _run)
    info = probe_cluster(client)
    assert not info.reachable
    assert "connection refused" in (info.message or "")


# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------


def test_config_kubectl_accessors_and_pin(tmp_path):
    from sparkrun.core.config import SparkrunConfig

    cfg = SparkrunConfig(tmp_path / "config.yaml")
    assert cfg.k8s_defaults == {}
    assert cfg.kubectl_path is None
    cfg.set("k8s", {"kubectl": {"path": "/x/kubectl", "version": "v1.31.0"}})
    assert cfg.kubectl_path == "/x/kubectl"
    assert cfg.kubectl_version == "v1.31.0"

    cfg.pin_kubectl_version("prod-ctx", "v1.30.2")
    assert cfg.kubectl_pinned_version("prod-ctx") == "v1.30.2"
    assert cfg.kubectl_pinned_version("other") is None
    # existing kubectl settings preserved after pin
    assert cfg.kubectl_path == "/x/kubectl"


# ---------------------------------------------------------------------------
# api.k8s surface
# ---------------------------------------------------------------------------


def _sctx(tmp_path):
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext

    return SparkrunContext(
        variables=init_sparkrun(),
        config=SparkrunConfig(tmp_path / "config.yaml"),
        verbose=False,
    )


def test_api_ensure_kubectl_translates_error(tmp_path):
    from sparkrun import api

    sctx = _sctx(tmp_path)
    with pytest.raises(api.k8s.KubectlUnavailable):
        api.k8s.ensure_kubectl(sctx, version="v9.9.9", download=False)


def test_api_cluster_info_pins_server_version(tmp_path, monkeypatch):
    from sparkrun import api
    from sparkrun.api import k8s as apik8s

    sctx = _sctx(tmp_path)

    fake = ClusterInfo(reachable=True, current_context="prod", server_version="v1.30.2+ck1", client_version="v1.31.0")
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr(apik8s._ops, "probe_cluster", lambda client: fake)

    info = api.k8s.cluster_info(sctx)
    assert info.server_version == "v1.30.2+ck1"
    # normalized (+ck1 stripped) pin persisted
    assert sctx.config.kubectl_pinned_version("prod") == "v1.30.2"


# ---------------------------------------------------------------------------
# K8sExecutor transition onto KubectlClient
# ---------------------------------------------------------------------------


def test_k8s_executor_prefix_uses_resolved_path():
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.k8s import K8sExecutor

    ex = K8sExecutor(ExecutorConfig(kubectl_path="/opt/kubectl", k8s_context="ctx", k8s_namespace="ns"))
    cmd = ex.run_cmd(image="img:tag", command="echo hi", container_name="pod1")
    assert cmd.startswith("/opt/kubectl ")
    assert "--context ctx" in cmd
    assert "-n ns" in cmd


def test_k8s_executor_prefix_falls_back_to_bare_kubectl():
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.k8s import K8sExecutor

    ex = K8sExecutor(ExecutorConfig(executor_type="k8s"))
    cmd = ex.run_cmd(image="img:tag", command="echo hi", container_name="pod1")
    assert cmd.startswith("kubectl ")


def test_k8s_executor_finalize_config_resolves_cached_binary(tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.k8s import K8sExecutor

    os_name, arch = kubectl.detect_os(), kubectl.detect_arch()
    binary = kubectl.cached_binary_path(tmp_path, "v1.31.0", os_name, arch)
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")

    cfg = SparkrunConfig(tmp_path / "config.yaml")
    cfg.set("cache_dir", str(tmp_path))

    ex = K8sExecutor(ExecutorConfig(executor_type="k8s"))
    ex.finalize_config(config=cfg)
    assert ex.config.kubectl_path == str(binary)


def test_k8s_executor_finalize_config_skips_partial_config(tmp_path):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.k8s import K8sExecutor

    class _PartialConfig:
        default_executor = "k8s"
        executor_config: dict = {}

    ex = K8sExecutor(ExecutorConfig(executor_type="k8s"))
    ex.finalize_config(config=_PartialConfig())  # must not raise
    assert ex.config.kubectl_path is None


def test_resolve_executor_wires_kubectl_path(tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.orchestration.executors.k8s import K8sExecutor

    os_name, arch = kubectl.detect_os(), kubectl.detect_arch()
    binary = kubectl.cached_binary_path(tmp_path, "v1.31.0", os_name, arch)
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")

    cfg = SparkrunConfig(tmp_path / "config.yaml")
    cfg.set("cache_dir", str(tmp_path))
    # executor.k8s is force-enabled by the isolate_stateful conftest fixture.

    ex = resolve_executor(cli_overrides={"executor": "k8s"}, config=cfg, rootless=False, auto_user=False)
    assert isinstance(ex, K8sExecutor)
    assert ex.config.kubectl_path == str(binary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_setup_k8s_sa_dry_run(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    # Route the CLI's context at a tmp config so nothing touches real home.
    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    runner = CliRunner()
    result = runner.invoke(main, ["setup", "k8s", "sa", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "ClusterRole" in result.output
    assert "dry-run" in result.output


def test_cli_setup_k8s_kubectl_list_empty(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    runner = CliRunner()
    result = runner.invoke(main, ["setup", "k8s", "kubectl", "--list"])
    assert result.exit_code == 0, result.output
