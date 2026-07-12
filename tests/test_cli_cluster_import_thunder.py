"""CLI tests for `sparkrun cluster import thunder` and the import group.

Thunder API + SSH probe are mocked; nothing touches the network or the real
~/.config/sparkrun / ~/.ssh.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.cli._common import _get_cluster_manager
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.transports.thunder import ssh_alias
from sparkrun.transports.thunder import transport as thunder_transport
from sparkrun.transports.thunder.api import ThunderInstance


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    import sparkrun.core.config

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", cfg)
    # transports.thunder is off by default; enable it for the import tests
    # (env override short-circuits config reads).
    monkeypatch.setenv("SPARKRUN_FEATURE_TRANSPORTS_THUNDER", "1")


def _inst():
    return ThunderInstance.from_json(
        {"id": "0", "uuid": "ie2pb8eu", "ip": "1.2.3.4", "port": 30469, "status": "RUNNING", "gpuType": "A6000", "numGpus": "1"}
    )


@pytest.fixture
def mock_thunder(monkeypatch, tmp_path):
    """Mock the Thunder API + key/alias writes + SSH probe used by the CLI."""
    inst = _inst()
    monkeypatch.setattr(thunder_transport, "list_running_instances", lambda: ("tok", "base", [inst]))
    monkeypatch.setattr(ssh_alias, "ensure_key", lambda *a, **k: tmp_path / "key")
    monkeypatch.setattr(ssh_alias, "write_aliases", lambda entries: {})

    probed = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="rtx-a6000", count=1, memory_gb=48.0)])
    monkeypatch.setattr("sparkrun.core.hardware_probe.probe_host", lambda *a, **k: probed)
    return inst


def test_import_group_help_lists_svd():
    # ``thunder`` is hidden until the transports.thunder flag is enabled (the
    # hidden state is frozen at import), but ``svd`` is always visible.
    r = CliRunner().invoke(main, ["cluster", "import", "--help"])
    assert r.exit_code == 0
    assert "svd" in r.output


def test_import_thunder_blocked_when_flag_off(monkeypatch):
    # Explicitly disable via env override; the callback gate must reject it.
    monkeypatch.setenv("SPARKRUN_FEATURE_TRANSPORTS_THUNDER", "0")
    r = CliRunner().invoke(main, ["cluster", "import", "thunder"])
    assert r.exit_code != 0
    assert "experimental and disabled" in r.output


def test_import_thunder_creates_cluster(mock_thunder):
    r = CliRunner().invoke(main, ["cluster", "import", "thunder"])
    assert r.exit_code == 0, r.stderr
    assert r.stdout.strip() == "thunder-0"

    c = _get_cluster_manager().get("thunder-0")
    assert c.transport == "thunder"
    assert c.provider_ref == "ie2pb8eu"
    assert c.hosts == ["tnr-ie2pb8eu"]
    # Probed hardware landed on the alias host.
    assert c.hosts_hardware["tnr-ie2pb8eu"].accelerators[0].memory_gb == 48.0


def test_import_thunder_no_probe_seeds_from_api(mock_thunder, monkeypatch):
    # If probe is called we fail loudly — --no-probe must not probe.
    monkeypatch.setattr(
        "sparkrun.core.hardware_probe.probe_host",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not probe")),
    )
    r = CliRunner().invoke(main, ["cluster", "import", "thunder", "--no-probe"])
    assert r.exit_code == 0, r.stderr
    c = _get_cluster_manager().get("thunder-0")
    hw = c.hosts_hardware["tnr-ie2pb8eu"]
    assert hw.accelerators[0].model == "a6000"  # seed from API gpuType
    assert hw.accelerators[0].memory_gb is None  # unknown without probe


def test_import_thunder_dry_run_writes_nothing(mock_thunder):
    r = CliRunner().invoke(main, ["cluster", "import", "thunder", "--dry-run"])
    assert r.exit_code == 0, r.stderr
    assert r.stdout.strip() == "thunder-0"
    names = {c.name for c in _get_cluster_manager().list_clusters()}
    assert "thunder-0" not in names


def test_import_thunder_reimport_syncs_in_place(mock_thunder):
    runner = CliRunner()
    r1 = runner.invoke(main, ["cluster", "import", "thunder"])
    assert r1.exit_code == 0, r1.stderr
    r2 = runner.invoke(main, ["cluster", "import", "thunder", "0"])
    assert r2.exit_code == 0, r2.stderr
    # Still exactly one thunder cluster (matched by uuid, synced in place).
    thunder_clusters = [c for c in _get_cluster_manager().list_clusters() if c.transport == "thunder"]
    assert len(thunder_clusters) == 1


def test_import_thunder_unknown_id_errors(mock_thunder):
    r = CliRunner().invoke(main, ["cluster", "import", "thunder", "99"])
    assert r.exit_code == 1
    assert "no RUNNING Thunder instance matches" in r.stderr


def test_import_svd_subcommand(tmp_path):
    envf = tmp_path / "cluster.env"
    envf.write_text("CLUSTER_NODES=10.0.0.1,10.0.0.2\nETH_IF=enp1s0f1np1\n")
    r = CliRunner().invoke(main, ["cluster", "import", "svd", str(envf)])
    assert r.exit_code == 0, r.stderr
    assert r.stdout.strip() == "cluster"
    assert _get_cluster_manager().get("cluster").hosts == ["10.0.0.1", "10.0.0.2"]


def test_import_eugr_alias(tmp_path):
    envf = tmp_path / "prod.env"
    envf.write_text("CLUSTER_NODES=10.0.0.5\n")
    r = CliRunner().invoke(main, ["cluster", "import", "eugr", str(envf)])
    assert r.exit_code == 0, r.stderr
    assert _get_cluster_manager().get("prod").hosts == ["10.0.0.5"]


def test_show_displays_transport(mock_thunder):
    runner = CliRunner()
    assert runner.invoke(main, ["cluster", "import", "thunder"]).exit_code == 0
    r = runner.invoke(main, ["cluster", "show", "thunder-0"])
    assert r.exit_code == 0, r.stderr
    assert "Transport:" in r.output and "thunder" in r.output
    assert "Provider:" in r.output and "ie2pb8eu" in r.output


def test_resolve_cluster_config_carries_transport(mock_thunder):
    from sparkrun.core.cluster_manager import resolve_cluster_config

    mgr = _get_cluster_manager()
    assert CliRunner().invoke(main, ["cluster", "import", "thunder"]).exit_code == 0
    cfg = resolve_cluster_config("thunder-0", None, None, mgr)
    assert cfg.transport == "thunder"
    assert cfg.provider_ref == "ie2pb8eu"


def test_delete_thunder_cluster_removes_alias(mock_thunder, monkeypatch):
    removed = []
    monkeypatch.setattr(ssh_alias, "remove_alias", lambda uuid: removed.append(uuid))
    runner = CliRunner()
    assert runner.invoke(main, ["cluster", "import", "thunder"]).exit_code == 0
    r = runner.invoke(main, ["cluster", "delete", "thunder-0", "--force"])
    assert r.exit_code == 0, r.stderr
    assert removed == ["ie2pb8eu"]
