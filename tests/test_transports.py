"""Tests for the transports seam and the Thunder Compute transport.

All network + filesystem interactions are mocked/redirected — no real Thunder
API calls and no writes outside ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import stat
import urllib.error

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition, ClusterManager
from sparkrun.transports import (
    DEFAULT_TRANSPORT,
    TransportError,
    list_transports,
    prepare_cluster_transport,
    resolve_transport,
)
from sparkrun.transports.thunder import api as tapi
from sparkrun.transports.thunder import ssh_alias
from sparkrun.transports.thunder import transport as ttrans


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def thunder_env(tmp_path, monkeypatch):
    """Redirect HOME / TNR_HOME / sparkrun config root into *tmp_path*."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TNR_HOME", str(tmp_path / "thunder"))
    monkeypatch.delenv("TNR_API_TOKEN", raising=False)
    monkeypatch.delenv("TNR_API_URL", raising=False)
    monkeypatch.setattr("sparkrun.core.config.DEFAULT_CONFIG_DIR", tmp_path / "config")
    return tmp_path


def _inst(**over) -> tapi.ThunderInstance:
    raw = {
        "id": "0",
        "uuid": "ie2pb8eu",
        "ip": "1.2.3.4",
        "port": 30469,
        "status": "RUNNING",
        "gpuType": "A6000",
        "numGpus": "1",
        "memory": "48",
        "storage": 100,
        "cpuCores": "6",
    }
    raw.update(over)
    return tapi.ThunderInstance.from_json(raw)


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Registry / resolution
# ---------------------------------------------------------------------------


def test_registry_lists_builtins():
    assert list_transports() == ["ssh", "thunder"]


def test_resolve_default_is_ssh():
    assert resolve_transport(None).name == "ssh"
    assert resolve_transport("ssh").name == "ssh"
    assert resolve_transport("thunder").name == "thunder"


def test_resolve_unknown_raises():
    with pytest.raises(TransportError):
        resolve_transport("nope")


def test_prepare_ssh_cluster_is_noop(monkeypatch):
    # An ssh cluster must not touch the Thunder code path at all.
    called = []
    monkeypatch.setattr(tapi, "load_token", lambda: called.append(1) or ("t", "b"))
    prepare_cluster_transport(ClusterDefinition(name="c", hosts=["h1"]))
    prepare_cluster_transport(None)
    assert called == []
    assert DEFAULT_TRANSPORT == "ssh"


# ---------------------------------------------------------------------------
# api: token + HTTP
# ---------------------------------------------------------------------------


def test_load_token_env_wins(thunder_env, monkeypatch):
    monkeypatch.setenv("TNR_API_TOKEN", "env-token")
    token, base = tapi.load_token()
    assert token == "env-token"
    assert base == tapi.DEFAULT_API_URL


def test_load_token_from_config_file(thunder_env, monkeypatch):
    cfg = thunder_env / "thunder"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "cli_config.json").write_text(json.dumps({"token": "file-token"}))
    token, base = tapi.load_token()
    assert token == "file-token"
    assert base == tapi.DEFAULT_API_URL


def test_load_token_missing_raises(thunder_env):
    with pytest.raises(tapi.ThunderNotConfigured):
        tapi.load_token()


def test_request_401_maps_to_auth_error(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "unauth", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(tapi.ThunderAuthError):
        tapi._request("GET", "tok", "https://x", "/v1/auth/validate")


def test_request_other_http_error_maps_to_api_error(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(tapi.ThunderApiError):
        tapi._request("GET", "tok", "https://x", "/v1/instances/list")


def test_request_sends_user_agent(monkeypatch):
    captured = {}

    def fake(req, timeout):
        captured["ua"] = req.get_header("User-agent")
        return _FakeResp(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", fake)
    tapi._request("GET", "tok", "https://x", "/v1/auth/validate")
    assert captured["ua"] and "urllib" not in captured["ua"].lower()


def test_list_instances_parses_mapping(monkeypatch):
    payload = {"0": {"uuid": "aa", "ip": "1.1.1.1", "port": 22, "status": "RUNNING", "gpuType": "A6000", "numGpus": "2"}}
    monkeypatch.setattr(tapi, "_request", lambda *a, **k: payload)
    insts = tapi.list_instances("t", "b")
    assert len(insts) == 1
    assert insts[0].id == "0" and insts[0].uuid == "aa" and insts[0].num_gpus == 2


def test_add_key_returns_pem(monkeypatch):
    monkeypatch.setattr(tapi, "_request", lambda *a, **k: {"uuid": "aa", "key": "PEM"})
    assert tapi.add_key("t", "b", "0") == "PEM"


def test_add_key_no_key_raises(monkeypatch):
    monkeypatch.setattr(tapi, "_request", lambda *a, **k: {"uuid": "aa"})
    with pytest.raises(tapi.ThunderApiError):
        tapi.add_key("t", "b", "0")


def test_instance_stopped_has_no_ip():
    inst = _inst(status="STOPPED", ip=None)
    assert not inst.is_running
    assert inst.ip is None


# ---------------------------------------------------------------------------
# ssh_alias
# ---------------------------------------------------------------------------


def test_ensure_key_writes_0600(thunder_env, monkeypatch):
    monkeypatch.setattr(tapi, "add_key", lambda *a, **k: "PRIVATE-KEY-PEM")
    inst = _inst()
    path = ssh_alias.ensure_key("t", "b", inst)
    assert path.read_text() == "PRIVATE-KEY-PEM"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    # Second call is a cache hit (no re-provision) unless forced.
    monkeypatch.setattr(tapi, "add_key", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not re-provision")))
    assert ssh_alias.ensure_key("t", "b", inst) == path


def test_write_aliases_upsert_preserves_others(thunder_env):
    a = _inst(uuid="aaa", ip="1.1.1.1", port=111)
    b = _inst(uuid="bbb", ip="2.2.2.2", port=222)
    ssh_alias.write_aliases([(a, thunder_env / "ka")])
    ssh_alias.write_aliases([(b, thunder_env / "kb")])
    conf = (thunder_env / "config" / "ssh" / "thunder.conf").read_text()
    assert "Host tnr-aaa" in conf and "Host tnr-bbb" in conf
    # Re-writing a with a new port updates it and keeps b.
    a2 = _inst(uuid="aaa", ip="1.1.1.9", port=999)
    ssh_alias.write_aliases([(a2, thunder_env / "ka")])
    conf = (thunder_env / "config" / "ssh" / "thunder.conf").read_text()
    assert "Port 999" in conf and "Host tnr-bbb" in conf
    assert "Port 111" not in conf


def test_ensure_include_idempotent(thunder_env):
    ssh_alias.ensure_include()
    ssh_alias.ensure_include()
    ssh_config = (thunder_env / "home" / ".ssh" / "config").read_text()
    assert ssh_config.count("Include ") == 1
    assert "thunder.conf" in ssh_config


def test_remove_alias(thunder_env):
    a = _inst(uuid="aaa")
    b = _inst(uuid="bbb")
    ssh_alias.write_aliases([(a, thunder_env / "k"), (b, thunder_env / "k")])
    ssh_alias.remove_alias("aaa")
    conf = (thunder_env / "config" / "ssh" / "thunder.conf").read_text()
    assert "Host tnr-aaa" not in conf and "Host tnr-bbb" in conf


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def _patch_api(monkeypatch, instances, *, key="PEM"):
    monkeypatch.setattr(tapi, "load_token", lambda: ("tok", "base"))
    monkeypatch.setattr(tapi, "list_instances", lambda *a, **k: instances)
    monkeypatch.setattr(tapi, "add_key", lambda *a, **k: key)


def test_transport_prepare_writes_alias(thunder_env, monkeypatch):
    inst = _inst()
    _patch_api(monkeypatch, [inst])
    cluster = ClusterDefinition(name="thunder-0", hosts=["tnr-ie2pb8eu"], transport="thunder", provider_ref="ie2pb8eu")
    ttrans.ThunderTransport().prepare(cluster)
    conf = (thunder_env / "config" / "ssh" / "thunder.conf").read_text()
    assert "Host tnr-ie2pb8eu" in conf and "Port 30469" in conf


def test_transport_prepare_dry_run_writes_nothing(thunder_env, monkeypatch):
    inst = _inst()
    _patch_api(monkeypatch, [inst])
    cluster = ClusterDefinition(name="thunder-0", hosts=["tnr-ie2pb8eu"], transport="thunder", provider_ref="ie2pb8eu")
    ttrans.ThunderTransport().prepare(cluster, dry_run=True)
    assert not (thunder_env / "config" / "ssh" / "thunder.conf").exists()


def test_transport_prepare_missing_instance_raises(thunder_env, monkeypatch):
    _patch_api(monkeypatch, [])
    cluster = ClusterDefinition(name="thunder-0", hosts=["tnr-ie2pb8eu"], transport="thunder", provider_ref="ie2pb8eu")
    with pytest.raises(TransportError):
        ttrans.ThunderTransport().prepare(cluster)


def test_transport_prepare_stopped_instance_raises(thunder_env, monkeypatch):
    _patch_api(monkeypatch, [_inst(status="STOPPED", ip=None)])
    cluster = ClusterDefinition(name="thunder-0", hosts=["tnr-ie2pb8eu"], transport="thunder", provider_ref="ie2pb8eu")
    with pytest.raises(TransportError):
        ttrans.ThunderTransport().prepare(cluster)


def test_seed_hardware_from_api():
    hw = ttrans.seed_hardware(_inst(gpuType="A6000", numGpus="2"))
    assert hw.accelerators[0].vendor == "nvidia"
    assert hw.accelerators[0].model == "a6000"
    assert hw.accelerators[0].count == 2
    assert hw.total_gpus == 2


# ---------------------------------------------------------------------------
# Feature gating (transports.thunder, default off)
# ---------------------------------------------------------------------------


def test_prepare_thunder_gate_off_fails_closed(monkeypatch):
    # Flag off (env override 0) → a thunder cluster must NOT silently run over SSH.
    monkeypatch.setenv("SPARKRUN_FEATURE_TRANSPORTS_THUNDER", "0")
    monkeypatch.setattr(tapi, "load_token", lambda: (_ for _ in ()).throw(AssertionError("gate must short-circuit")))
    c = ClusterDefinition(name="thunder-0", hosts=["tnr-x"], transport="thunder", provider_ref="x")
    with pytest.raises(TransportError):
        prepare_cluster_transport(c)


def test_prepare_thunder_gate_on_runs(thunder_env, monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_TRANSPORTS_THUNDER", "1")
    _patch_api(monkeypatch, [_inst()])
    c = ClusterDefinition(name="thunder-0", hosts=["tnr-ie2pb8eu"], transport="thunder", provider_ref="ie2pb8eu")
    prepare_cluster_transport(c)
    assert (thunder_env / "config" / "ssh" / "thunder.conf").exists()


# ---------------------------------------------------------------------------
# ClusterDefinition serialization round-trip
# ---------------------------------------------------------------------------


def test_cluster_transport_fields_roundtrip(tmp_path):
    mgr = ClusterManager(tmp_path)
    mgr.create("thunder-0", ["tnr-abc"], transport="thunder", provider_ref="abc")
    loaded = mgr.get("thunder-0")
    assert loaded.transport == "thunder"
    assert loaded.provider_ref == "abc"
    d = loaded.to_dict()
    assert d["transport"] == "thunder" and d["provider_ref"] == "abc"


def test_thunder_ssh_user_constant_and_alias_block():
    from sparkrun.transports.thunder import ssh_alias
    from sparkrun.transports.thunder.api import ThunderInstance

    assert ssh_alias.THUNDER_SSH_USER == "ubuntu"
    inst = ThunderInstance(
        id="0",
        uuid="abc",
        ip="1.2.3.4",
        port=2222,
        status="RUNNING",
        gpu_type="a6000",
        num_gpus=1,
        memory_gb=48,
        storage_gb=100,
        cpu_cores="8",
    )
    block = ssh_alias._render_block(inst, __import__("pathlib").Path("/tmp/k"))
    assert "    User ubuntu" in block


def test_thunder_imported_cluster_carries_user(tmp_path):
    # Mirrors what `cluster import thunder` persists: the ssh user is stored on
    # the cluster so setup commands don't fall back to $USER (overriding the
    # alias's `User ubuntu`).
    from sparkrun.transports.thunder import ssh_alias

    mgr = ClusterManager(tmp_path)
    mgr.create("thunder-0", ["tnr-abc"], user=ssh_alias.THUNDER_SSH_USER, transport="thunder", provider_ref="abc")
    assert mgr.get("thunder-0").user == "ubuntu"


def test_thunder_imported_cluster_drops_unlimited_memlock(tmp_path):
    # Thunder's zero-capability containers can't raise RLIMIT_MEMLOCK, so the
    # import records an executor_config that replaces the rootless default
    # (memlock=-1 + stack) with stack only.
    from sparkrun.transports.thunder import ssh_alias

    assert ssh_alias.THUNDER_EXECUTOR_CONFIG == {"ulimit": ["stack=67108864"]}
    mgr = ClusterManager(tmp_path)
    mgr.create(
        "thunder-0",
        ["tnr-abc"],
        user=ssh_alias.THUNDER_SSH_USER,
        transport="thunder",
        provider_ref="abc",
        executor_config=dict(ssh_alias.THUNDER_EXECUTOR_CONFIG),
    )
    cfg = mgr.get("thunder-0").executor_config
    assert cfg == {"ulimit": ["stack=67108864"]}
    assert "memlock" not in str(cfg)


def test_ssh_cluster_omits_transport_key(tmp_path):
    mgr = ClusterManager(tmp_path)
    mgr.create("plain", ["h1"])
    loaded = mgr.get("plain")
    assert loaded.transport == "ssh"
    # Default ssh transport is not serialized (keeps existing YAML clean).
    assert "transport" not in loaded.to_dict()
    assert "provider_ref" not in loaded.to_dict()
