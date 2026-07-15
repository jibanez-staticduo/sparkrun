"""Tests for the generic transports seam.

The Thunder Compute transport (and its exhaustive tests) now lives in the
out-of-tree ``sparkrun_thunder`` plugin; these tests cover only the core seam
that ships in sparkrun: the SSH default, resolution, gating hooks, and
``ClusterDefinition`` transport serialization.
"""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition, ClusterManager
from sparkrun.transports import (
    DEFAULT_TRANSPORT,
    TransportError,
    list_transports,
    prepare_cluster_transport,
    resolve_transport,
)


# ---------------------------------------------------------------------------
# Registry / resolution
# ---------------------------------------------------------------------------


def test_registry_lists_builtins():
    # Only ssh ships in core; provider transports (thunder, …) arrive as plugins.
    assert list_transports() == ["ssh"]


def test_resolve_default_is_ssh():
    assert resolve_transport(None).transport_name == "ssh"
    assert resolve_transport("ssh").transport_name == "ssh"


def test_resolve_unknown_raises():
    with pytest.raises(TransportError):
        resolve_transport("nope")


def test_prepare_ssh_cluster_is_noop():
    # An ssh cluster (and a None cluster) short-circuit before any transport
    # machinery — no error, nothing to prepare.
    prepare_cluster_transport(ClusterDefinition(name="c", hosts=["h1"]))
    prepare_cluster_transport(None)
    assert DEFAULT_TRANSPORT == "ssh"


def test_prepare_unknown_transport_raises():
    # A cluster declaring a transport with no registered plugin must fail closed,
    # never silently run over plain SSH.
    c = ClusterDefinition(name="x", hosts=["h1"], transport="nosuch", provider_ref="p")
    with pytest.raises(TransportError):
        prepare_cluster_transport(c)


# ---------------------------------------------------------------------------
# ClusterDefinition serialization round-trip (transport is a plain string field,
# so this holds for any provider without the plugin being loaded)
# ---------------------------------------------------------------------------


def test_cluster_transport_fields_roundtrip(tmp_path):
    mgr = ClusterManager(tmp_path)
    mgr.create("prov-0", ["prov-abc"], transport="someprovider", provider_ref="abc")
    loaded = mgr.get("prov-0")
    assert loaded.transport == "someprovider"
    assert loaded.provider_ref == "abc"
    d = loaded.to_dict()
    assert d["transport"] == "someprovider" and d["provider_ref"] == "abc"


def test_ssh_cluster_omits_transport_key(tmp_path):
    mgr = ClusterManager(tmp_path)
    mgr.create("plain", ["h1"])
    loaded = mgr.get("plain")
    assert loaded.transport == "ssh"
    # Default ssh transport is not serialized (keeps existing YAML clean).
    assert "transport" not in loaded.to_dict()
    assert "provider_ref" not in loaded.to_dict()
