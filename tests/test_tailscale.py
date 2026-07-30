"""Tests for the Tailscale setup integration.

Covers the stdlib REST client, join/status/expose script + api logic, and the
CLI feature-flag gating. All HTTP and SSH is mocked — no tailnet is contacted.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from sparkrun.orchestration import tailscale as ts
from sparkrun.orchestration.tailscale import api as tsapi


# ---------------------------------------------------------------------------
# Helpers
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


class _StubConfig:
    """Minimal SparkrunConfig.get(dotted-key) stand-in for settings tests."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


def _settings(**over):
    base = dict(client_id="cid", client_secret="sec")
    base.update(over)
    return ts.TailscaleSettings(**base)


def _remote(host, rc=0, stdout="", stderr=""):
    from sparkrun.orchestration.ssh import RemoteResult

    return RemoteResult(host=host, returncode=rc, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# settings resolution
# ---------------------------------------------------------------------------


def test_load_settings_env_wins(monkeypatch):
    monkeypatch.setenv("TS_API_CLIENT_ID", "env-id")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "env-sec")
    cfg = _StubConfig({"tailscale.oauth_client_id": "cfg-id"})
    s = ts.load_settings(cfg)
    assert s.client_id == "env-id" and s.client_secret == "env-sec"
    assert s.tag == ts.DEFAULT_TAG and s.tailnet == "-"


def test_load_settings_from_config(monkeypatch):
    monkeypatch.delenv("TS_API_CLIENT_ID", raising=False)
    monkeypatch.delenv("TS_API_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TS_TAG", raising=False)
    cfg = _StubConfig(
        {
            "tailscale.oauth_client_id": "cfg-id",
            "tailscale.oauth_client_secret": "cfg-sec",
            "tailscale.tag": "tag:custom",
            "tailscale.ephemeral": True,
        }
    )
    s = ts.load_settings(cfg)
    assert s.client_id == "cfg-id" and s.tag == "tag:custom" and s.ephemeral is True


def test_load_settings_missing_raises(monkeypatch):
    monkeypatch.delenv("TS_API_CLIENT_ID", raising=False)
    monkeypatch.delenv("TS_API_CLIENT_SECRET", raising=False)
    with pytest.raises(ts.TailscaleNotConfigured):
        ts.load_settings(_StubConfig({}))


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------


def test_fetch_access_token(monkeypatch):
    monkeypatch.setattr(tsapi, "_request", lambda *a, **k: {"access_token": "tok", "expires_in": 3600})
    assert tsapi.fetch_access_token(_settings()) == "tok"


def test_fetch_access_token_no_token(monkeypatch):
    monkeypatch.setattr(tsapi, "_request", lambda *a, **k: {})
    with pytest.raises(ts.TailscaleAuthError):
        tsapi.fetch_access_token(_settings())


def test_mint_auth_key_sends_tag(monkeypatch):
    captured = {}

    def fake_request(method, url, *, token=None, data=None, content_type=None):
        captured["body"] = json.loads(data)
        captured["token"] = token
        return {"id": "k1", "key": "tskey-auth-abc"}

    monkeypatch.setattr(tsapi, "_request", fake_request)
    key = tsapi.mint_auth_key(_settings(tag="tag:dgx-spark"), "tok", ephemeral=True)
    assert key == "tskey-auth-abc"
    create = captured["body"]["capabilities"]["devices"]["create"]
    assert create["tags"] == ["tag:dgx-spark"]
    assert create["preauthorized"] is True and create["ephemeral"] is True
    assert captured["token"] == "tok"


def test_mint_auth_key_tag_error(monkeypatch):
    def boom(*a, **k):
        raise ts.TailscaleApiError("Tailscale API POST failed: HTTP 400 requested tags are invalid")

    monkeypatch.setattr(tsapi, "_request", boom)
    with pytest.raises(ts.TailscaleTagError):
        tsapi.mint_auth_key(_settings(), "tok")


def test_request_401_maps_to_auth_error(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.HTTPError("http://x", 401, "unauth", None, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ts.TailscaleAuthError):
        tsapi._request("GET", "https://api.tailscale.com/api/v2/tailnet/-/devices", token="t")


def test_request_500_maps_to_api_error(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.HTTPError("http://x", 500, "boom", None, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ts.TailscaleApiError):
        tsapi._request("GET", "https://api.tailscale.com/api/v2/tailnet/-/devices", token="t")


def test_list_devices_parses_and_ipv4(monkeypatch):
    payload = {
        "devices": [
            {"id": "d1", "hostname": "spark1", "addresses": ["100.64.0.1", "fd7a::1"], "tags": ["tag:dgx-spark"], "online": True},
        ]
    }
    monkeypatch.setattr(tsapi, "_request", lambda *a, **k: payload)
    devs = tsapi.list_devices(_settings(), "tok")
    assert len(devs) == 1
    assert devs[0].ipv4 == "100.64.0.1" and devs[0].hostname == "spark1"


# ---------------------------------------------------------------------------
# script builders
# ---------------------------------------------------------------------------


def test_build_join_scripts_substitution():
    primary, fallback = ts.build_join_scripts("tskey-auth-XYZ", "tag:dgx-spark", enable_ssh=True)
    # Values are interpolated into single-quoted assignments (interpolation-proof).
    assert "AUTHKEY='tskey-auth-XYZ'" in primary
    assert "TAGS='tag:dgx-spark'" in primary and '--advertise-tags="$TAGS"' in primary
    assert "--ssh" in primary
    assert "sudo -n tailscale up" in primary
    assert "tailscale up" in fallback and "sudo -n tailscale up" not in fallback
    # The daemon is ensured up before `tailscale up`: systemd where available,
    # else a manual (userspace-networking) tailscaled for no-systemd containers.
    for script in (primary, fallback):
        assert "systemctl enable --now tailscaled" in script
        assert "--tun=userspace-networking" in script
        assert "nohup tailscaled" in script
    # No leftover format fields (str.format templating must see no bare braces).
    assert "{" not in primary and "}" not in primary
    assert "{" not in fallback and "}" not in fallback


def test_validate_tag_rejects_injection():
    with pytest.raises(ts.TailscaleError):
        ts.validate_tag('tag:x"; rm -rf /; #')
    with pytest.raises(ts.TailscaleError):
        ts.validate_tag("not-a-tag")
    assert ts.validate_tag("tag:dgx-spark") == "tag:dgx-spark"


def test_build_join_scripts_rejects_unsafe_authkey():
    # Single-quote breakout is the only vector after single-quoting the assignments.
    with pytest.raises(ValueError):
        ts.build_join_scripts("tskey'evil", "tag:dgx-spark")
    with pytest.raises(ValueError):
        ts.build_join_scripts("tskey\nevil", "tag:dgx-spark")


def test_load_settings_rejects_http_base_url(monkeypatch):
    monkeypatch.setenv("TS_API_CLIENT_ID", "cid")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "sec")
    monkeypatch.setenv("TS_API_URL", "http://evil.example.com")
    with pytest.raises(ts.TailscaleError):
        ts.load_settings(None)


def test_load_settings_allows_localhost_http(monkeypatch):
    monkeypatch.setenv("TS_API_CLIENT_ID", "cid")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "sec")
    monkeypatch.setenv("TS_API_URL", "http://localhost:8080")
    assert ts.load_settings(None).base_url == "http://localhost:8080"


def test_load_settings_rejects_bad_tag(monkeypatch):
    monkeypatch.setenv("TS_API_CLIENT_ID", "cid")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "sec")
    monkeypatch.setenv("TS_TAG", 'tag:x" evil')
    with pytest.raises(ts.TailscaleError):
        ts.load_settings(None)


def test_build_join_scripts_hostname():
    primary, fallback = ts.build_join_scripts("tskey-auth-x", "tag:thunder", hostname="thunder-0")
    assert "HOSTNAME_ARG='--hostname=thunder-0'" in primary
    assert "HOSTNAME_ARG='--hostname=thunder-0'" in fallback
    # Unsafe characters are dropped (DNS-label sanitization).
    p2, _ = ts.build_join_scripts("k", "tag:x", hostname="bad name!$;")
    assert "HOSTNAME_ARG='--hostname=badname'" in p2
    # No hostname → empty arg.
    p3, _ = ts.build_join_scripts("k", "tag:x")
    assert "HOSTNAME_ARG=''" in p3


def test_cli_join_hostname_defaults_to_cluster_name(monkeypatch):
    from click.testing import CliRunner

    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import get_config_root

    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_TAILSCALE", "1")
    monkeypatch.setenv("TS_API_CLIENT_ID", "cid")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "sec")

    # Created where the CLI actually resolves clusters. `get_config_root()`
    # falls back to DEFAULT_CONFIG_DIR whenever the SAF stateful root is not
    # ready, which is always the case under pytest -- so writing to
    # STATEFUL_ROOT put the fixture somewhere the command never reads. The
    # test then only passed on a machine whose real config happened to define
    # a cluster of this name.
    root = get_config_root()
    root.mkdir(parents=True, exist_ok=True)
    ClusterManager(root).create("ts-fixture-cluster", ["tnr-abc"], user="ubuntu")

    from sparkrun.cli import main

    r = CliRunner().invoke(main, ["setup", "tailscale", "join", "--dry-run", "--cluster", "ts-fixture-cluster"])
    assert r.exit_code == 0, r.output
    assert "as 'ts-fixture-cluster'" in r.output


def test_parse_join_result():
    out = "random noise\nTS_INSTALL=present\nTS_IP=100.64.0.9\nTS_OK=1\n"
    parsed = ts.parse_join_result(out)
    assert parsed == {"TS_INSTALL": "present", "TS_IP": "100.64.0.9", "TS_OK": "1"}


# ---------------------------------------------------------------------------
# api ops — join / status / expose
# ---------------------------------------------------------------------------


def test_api_join_dry_run_no_network(tmp_path, monkeypatch):
    from sparkrun import api

    monkeypatch.setenv("TS_API_CLIENT_ID", "cid")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "sec")

    def fail(*a, **k):  # any network call would blow up the test
        raise AssertionError("dry-run must not touch the network")

    monkeypatch.setattr(ts, "fetch_access_token", fail)
    result = api.tailscale.join(_sctx(tmp_path), ["h1", "h2"], {}, dry_run=True)
    assert result.dry_run and result.ok_count == 0
    assert [h.host for h in result.hosts] == ["h1", "h2"]


def test_api_join_success(tmp_path, monkeypatch):
    from sparkrun import api

    monkeypatch.setenv("TS_API_CLIENT_ID", "cid")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "sec")
    monkeypatch.setattr(ts, "fetch_access_token", lambda s: "tok")
    monkeypatch.setattr(ts, "mint_auth_key", lambda s, t, **k: "tskey-auth-abc")

    def fake_sudo(host_list, script, fallback, ssh_kwargs, **k):
        rmap = {
            "h1": _remote("h1", 0, "TS_INSTALL=present\nTS_IP=100.64.0.1\nTS_OK=1\n"),
            # A non-sudo failure: daemon down. The tailscale error is merged into
            # stdout (2>&1) ahead of the TS_ERROR marker.
            "h2": _remote("h2", 1, "TS_INSTALL=present\nfailed to connect to local tailscaled\nTS_ERROR=up_failed\n"),
        }
        return rmap, ["h2"]

    monkeypatch.setattr("sparkrun.orchestration.sudo.run_with_sudo_fallback", fake_sudo)
    result = api.tailscale.join(_sctx(tmp_path), ["h1", "h2"], {}, tag="tag:dgx-spark")
    assert result.ok_count == 1
    by = {h.host: h for h in result.hosts}
    assert by["h1"].ok and by["h1"].ip == "100.64.0.1"
    # The message surfaces both the marker and the real tailscale error line.
    assert not by["h2"].ok
    assert "up_failed" in by["h2"].message and "tailscaled" in by["h2"].message


def test_api_status(tmp_path, monkeypatch):
    from sparkrun import api

    def fake_parallel(hosts, script, **k):
        return [
            _remote("h1", 0, "TS_STATE=Running\nTS_IP=100.64.0.1\nTS_HOSTNAME=spark1\n"),
            _remote("h2", 0, "TS_STATE=not_installed\n"),
        ]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", fake_parallel)
    result = api.tailscale.status(_sctx(tmp_path), ["h1", "h2"], {})
    by = {h.host: h for h in result.hosts}
    assert by["h1"].joined and by["h1"].ip == "100.64.0.1"
    assert not by["h2"].joined and by["h2"].state == "not_installed"


def test_api_expose_proxy(tmp_path, monkeypatch):
    from sparkrun import api

    monkeypatch.setattr(ts, "local_tailscale_ipv4", lambda: "100.64.0.5")
    monkeypatch.setattr(ts, "local_tailscale_dnsname", lambda: "ctrl.tail1.ts.net")
    result = api.tailscale.expose(_sctx(tmp_path), proxy=True, port=9000)
    assert result.url == "http://ctrl.tail1.ts.net:9000/v1"
    assert result.target == "proxy"


def test_api_expose_proxy_not_on_tailnet(tmp_path, monkeypatch):
    from sparkrun import api

    monkeypatch.setattr(ts, "local_tailscale_ipv4", lambda: None)
    with pytest.raises(api.tailscale.TailscaleExposeError):
        api.tailscale.expose(_sctx(tmp_path), proxy=True)


def test_api_expose_requires_exactly_one(tmp_path):
    from sparkrun import api

    with pytest.raises(api.tailscale.TailscaleExposeError):
        api.tailscale.expose(_sctx(tmp_path), proxy=True, head_host="h1")


def test_build_serve_scripts():
    p, f = ts.build_serve_scripts(8000)
    assert "PORT='8000'" in p and "PORT='8000'" in f
    assert 'tailscale serve --bg --tcp="$PORT" "tcp://127.0.0.1:$PORT"' in p
    assert "sudo -n tailscale serve" in p
    assert "tailscale serve" in f and "sudo -n tailscale serve" not in f
    assert "{" not in p and "}" not in p and "{" not in f and "}" not in f
    # port is coerced to int (no injection surface)
    p2, _ = ts.build_serve_scripts("8000")
    assert "PORT='8000'" in p2


def test_api_expose_head_configures_serve(tmp_path, monkeypatch):
    from sparkrun import api

    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        lambda hosts, script, **k: [_remote("head1", 0, "TS_STATE=Running\nTS_IP=100.64.0.7\nTS_HOSTNAME=head1\n")],
    )
    serve = {}

    def fake_sudo(host_list, primary, fallback, ssh_kwargs, **k):
        serve["primary"] = primary
        return {"head1": _remote("head1", 0, "TS_IP=100.64.0.7\nTS_SERVE_OK=1\n")}, []

    monkeypatch.setattr("sparkrun.orchestration.sudo.run_with_sudo_fallback", fake_sudo)
    result = api.tailscale.expose(_sctx(tmp_path), head_host="head1", ssh_kwargs={}, port=8000)
    assert result.url == "http://100.64.0.7:8000/v1" and result.target == "head1"
    assert "tailscale serve --bg --tcp=" in serve["primary"]
    # The MagicDNS convenience form is surfaced.
    assert any("http://head1:8000/v1" in w for w in result.warnings)


def test_api_expose_head_serve_failure_surfaces_error(tmp_path, monkeypatch):
    from sparkrun import api

    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        lambda hosts, script, **k: [_remote("head1", 0, "TS_STATE=Running\nTS_IP=100.64.0.7\n")],
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.sudo.run_with_sudo_fallback",
        lambda *a, **k: ({"head1": _remote("head1", 1, "flag provided but not defined: --tcp\nTS_ERROR=serve_failed\n")}, ["head1"]),
    )
    with pytest.raises(api.tailscale.TailscaleExposeError) as exc:
        api.tailscale.expose(_sctx(tmp_path), head_host="head1", ssh_kwargs={}, port=8000)
    assert "not defined: --tcp" in str(exc.value)


def test_api_join_bad_env_tag_translated(tmp_path, monkeypatch):
    from sparkrun import api

    monkeypatch.setenv("TS_API_CLIENT_ID", "cid")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "sec")
    monkeypatch.setenv("TS_TAG", 'tag:x" evil')
    # Even dry-run resolves settings first — the bad tag must surface as an
    # api-layer error, not a raw orchestration TailscaleError.
    with pytest.raises(api.tailscale.TailscaleSetupError):
        api.tailscale.join(_sctx(tmp_path), ["h1"], {}, dry_run=True)


def test_api_down_remove_fails_before_logout(tmp_path, monkeypatch):
    from sparkrun import api

    monkeypatch.delenv("TS_API_CLIENT_ID", raising=False)
    monkeypatch.delenv("TS_API_CLIENT_SECRET", raising=False)

    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        lambda hosts, script, **k: [_remote(h, 0, "TS_STATE=Running\nTS_IP=100.64.0.1\n") for h in hosts],
    )
    logout_calls = []
    monkeypatch.setattr(
        "sparkrun.orchestration.sudo.run_with_sudo_fallback",
        lambda *a, **k: logout_calls.append(a) or ({}, []),
    )

    with pytest.raises(api.tailscale.TailscaleNotConfigured):
        api.tailscale.down(_sctx(tmp_path), ["h1"], {}, remove=True)
    assert logout_calls == [], "hosts must not be logged out when --remove has no creds"


# ---------------------------------------------------------------------------
# CLI gating
# ---------------------------------------------------------------------------


def test_setup_tailscale_gate_helper_respects_env(monkeypatch):
    # The import-time help-visibility flag freezes at first import from the real
    # config, so assert the gate *helper* directly with an env override (which
    # takes precedence over config) — deterministic regardless of ~/.config state.
    from sparkrun.cli._setup._tailscale import _setup_tailscale_enabled_at_import

    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_TAILSCALE", "0")
    assert _setup_tailscale_enabled_at_import() is False
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_TAILSCALE", "1")
    assert _setup_tailscale_enabled_at_import() is True


def test_cli_join_gated_when_flag_off(monkeypatch):
    from click.testing import CliRunner

    # Force OFF via env override (beats any real/isolated config value).
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_TAILSCALE", "0")
    from sparkrun.cli import main

    r = CliRunner().invoke(main, ["setup", "tailscale", "join", "--dry-run", "-H", "h1"])
    assert r.exit_code != 0
    assert "features enable cli.setup.tailscale" in r.output


def test_cli_join_dry_run_when_flag_on(monkeypatch):
    from click.testing import CliRunner

    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_TAILSCALE", "1")
    monkeypatch.setenv("TS_API_CLIENT_ID", "cid")
    monkeypatch.setenv("TS_API_CLIENT_SECRET", "sec")
    from sparkrun.cli import main

    r = CliRunner().invoke(main, ["setup", "tailscale", "join", "--dry-run", "-H", "h1,h2"])
    assert r.exit_code == 0, r.output
    assert "Would join 2 host(s)" in r.output and "tag:dgx-spark" in r.output
