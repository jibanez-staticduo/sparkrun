"""Tests for native-cluster distributed-init network selection."""

from unittest import mock

from sparkrun.core.scheduler import RankAssignment, RankSlot
from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.runtimes import _cluster_ops, _init_network


def _make_ctx(hosts: list[str], *, dry_run: bool = False, placement=None) -> _cluster_ops.ClusterContext:
    return _cluster_ops.ClusterContext(
        hosts=list(hosts),
        head_host=hosts[0],
        worker_hosts=list(hosts[1:]),
        num_nodes=len(hosts),
        ssh_kwargs={},
        volumes={},
        all_env={},
        cluster_id="test-cluster",
        image="test:image",
        dry_run=dry_run,
        config=None,
        placement=placement,
    )


def _one_gpu_per_host_placement(hosts: list[str]) -> RankAssignment:
    return RankAssignment(
        by_rank=tuple(RankSlot(host=host, local_gpu=0) for host in hosts),
        hosts_used=tuple(hosts),
    )


class TestSelectInitNetwork:
    """Verify native init address selection keeps mgmt first, then falls back."""

    def test_management_path_stays_selected_when_reachable(self, monkeypatch):
        """Given reachable mgmt, when IB candidates exist, then mgmt remains selected."""
        ctx = _make_ctx(["node-1", "node-2"])
        candidates = _init_network.InitNetworkCandidates(
            management_head_ip="192.168.128.10",
            management_hosts=("192.168.128.10", "192.168.96.114"),
            ib_ip_map={"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
        )
        monkeypatch.setattr(_init_network, "workers_can_reach", lambda *_args, **_kwargs: True)

        selection = _init_network.select_init_network(ctx, candidates)

        assert selection.network == "management"
        assert selection.head_ip == "192.168.128.10"
        assert selection.hosts == ("192.168.128.10", "192.168.96.114")

    def test_reachable_ib_path_selected_when_management_fails(self, monkeypatch):
        """Given unreachable mgmt and reachable IB, then CX7 addresses are selected."""
        ctx = _make_ctx(["node-1", "node-2"])
        candidates = _init_network.InitNetworkCandidates(
            management_head_ip="192.168.128.10",
            management_hosts=("192.168.128.10", "192.168.96.114"),
            ib_ip_map={"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
        )
        reachable = {
            "192.168.128.10": False,
            "192.168.100.10": True,
        }
        monkeypatch.setattr(
            _init_network,
            "workers_can_reach",
            lambda _ctx, target_ip: reachable[target_ip],
        )

        selection = _init_network.select_init_network(ctx, candidates)

        assert selection.network == "ib"
        assert selection.head_ip == "192.168.100.10"
        assert selection.hosts == ("192.168.100.10", "192.168.100.11")

    def test_management_path_kept_when_ib_map_is_incomplete(self, monkeypatch):
        """Given unreachable mgmt but partial IB data, then unsafe substitution is skipped."""
        ctx = _make_ctx(["node-1", "node-2"])
        candidates = _init_network.InitNetworkCandidates(
            management_head_ip="192.168.128.10",
            management_hosts=("192.168.128.10", "192.168.96.114"),
            ib_ip_map={"node-1": "192.168.100.10"},
        )
        monkeypatch.setattr(_init_network, "workers_can_reach", lambda *_args, **_kwargs: False)

        selection = _init_network.select_init_network(ctx, candidates)

        assert selection.network == "management"
        assert selection.head_ip == "192.168.128.10"
        assert selection.hosts == ("192.168.128.10", "192.168.96.114")


def test_native_cluster_threads_reachable_ib_selection_into_node_commands(monkeypatch):
    """Given mgmt failure, when native cluster launches, then commands receive IB init hosts."""
    ctx = _make_ctx(["node-1", "node-2"])
    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _cluster_ops,
        "detect_ib_with_ips",
        lambda *_args, **_kwargs: (
            ClusterCommEnv.empty(),
            {"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
            {"node-1": "enp1s0f1np1", "node-2": "enp1s0f1np1"},
        ),
    )
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda _ctx: "192.168.128.10")
    monkeypatch.setattr(
        _cluster_ops,
        "resolve_hosts_for_init",
        lambda _ctx, _head_ip: ["192.168.128.10", "192.168.96.114"],
    )
    monkeypatch.setattr(_cluster_ops, "find_port", lambda _ctx, _host, port: port)
    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        _init_network,
        "workers_can_reach",
        lambda _ctx, target_ip: target_ip == "192.168.100.10",
    )

    runtime = mock.MagicMock()
    runtime._resolve_executor.return_value.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
    runtime.generate_node_command = mock.MagicMock(return_value="serve")
    runtime.get_extra_docker_opts = lambda: []
    runtime._print_cluster_banner = mock.MagicMock()

    rc = _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx)

    assert rc == 1
    call = runtime.generate_node_command.call_args
    assert call.kwargs["head_ip"] == "192.168.100.10"
    assert call.kwargs["hosts"] == ["192.168.100.10", "192.168.100.11"]


def test_native_cluster_pins_comm_env_to_fabric_on_ib_fallback(monkeypatch):
    """Given mgmt failure, the per-host comm env handed to containers is fabric-pinned."""
    ctx = _make_ctx(["node-1", "node-2"])

    def _mgmt_first(mgmt_ip):
        return {
            "NCCL_NET": "IB",
            "GLOO_SOCKET_IFNAME": "wlan0",
            "TP_SOCKET_IFNAME": "wlan0",
            "MN_IF_NAME": "wlan0",
            "OMPI_MCA_btl_tcp_if_include": "wlan0",
            "NCCL_SOCKET_IFNAME": "wlan0,cx0",
            "NODE_IP": mgmt_ip,
        }

    comm_env = ClusterCommEnv.from_per_host({"node-1": _mgmt_first("192.168.1.10"), "node-2": _mgmt_first("192.168.1.11")})

    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _cluster_ops,
        "detect_ib_with_ips",
        lambda *_a, **_k: (
            comm_env,
            {"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
            {"node-1": "cx0", "node-2": "cx0"},
        ),
    )
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda _ctx: "192.168.128.10")
    monkeypatch.setattr(_cluster_ops, "resolve_hosts_for_init", lambda _ctx, _head_ip: ["192.168.128.10", "192.168.96.114"])
    monkeypatch.setattr(_cluster_ops, "find_port", lambda _ctx, _host, port: port)
    # Reachable only over the IB head → forces the fabric verdict.
    monkeypatch.setattr(_init_network, "workers_can_reach", lambda _ctx, target_ip: target_ip == "192.168.100.10")

    captured = {}

    def _capture_launch(ctx_, containers, executor, ce, **_k):
        captured["comm_env"] = ce
        return 1

    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", _capture_launch)

    runtime = mock.MagicMock()
    runtime._resolve_executor.return_value.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
    runtime.generate_node_command = mock.MagicMock(return_value="serve")
    runtime.get_extra_docker_opts = lambda: []
    runtime._print_cluster_banner = mock.MagicMock()

    rc = _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx)

    assert rc == 1
    pinned = captured["comm_env"]
    for host, ib_ip in (("node-1", "192.168.100.10"), ("node-2", "192.168.100.11")):
        env = pinned.get_env(host)
        assert env["NODE_IP"] == ib_ip
        assert env["GLOO_SOCKET_IFNAME"] == "cx0"
        assert env["TP_SOCKET_IFNAME"] == "cx0"
        assert env["NCCL_SOCKET_IFNAME"] == "cx0"
        # VLLM_HOST_IP is mirrored from NODE_IP by the vLLM runtime hook at
        # container-launch time, not by the generic fabric pin.
        assert "VLLM_HOST_IP" not in env


def test_native_cluster_leaves_comm_env_untouched_when_mgmt_reachable(monkeypatch):
    """Given reachable mgmt, the per-host comm env is not re-pinned to the fabric."""
    ctx = _make_ctx(["node-1", "node-2"])
    comm_env = ClusterCommEnv.from_per_host(
        {
            "node-1": {"GLOO_SOCKET_IFNAME": "wlan0", "NODE_IP": "192.168.1.10"},
            "node-2": {"GLOO_SOCKET_IFNAME": "wlan0", "NODE_IP": "192.168.1.11"},
        }
    )

    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _cluster_ops,
        "detect_ib_with_ips",
        lambda *_a, **_k: (
            comm_env,
            {"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
            {"node-1": "cx0", "node-2": "cx0"},
        ),
    )
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda _ctx: "192.168.128.10")
    monkeypatch.setattr(_cluster_ops, "resolve_hosts_for_init", lambda _ctx, _head_ip: ["192.168.128.10", "192.168.96.114"])
    monkeypatch.setattr(_cluster_ops, "find_port", lambda _ctx, _host, port: port)
    # Management head is reachable → keep management, no re-pin.
    monkeypatch.setattr(_init_network, "workers_can_reach", lambda _ctx, _target_ip: True)

    captured = {}

    def _capture_launch(ctx_, containers, executor, ce, **_k):
        captured["comm_env"] = ce
        return 1

    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", _capture_launch)

    runtime = mock.MagicMock()
    runtime._resolve_executor.return_value.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
    runtime.generate_node_command = mock.MagicMock(return_value="serve")
    runtime.get_extra_docker_opts = lambda: []
    runtime._print_cluster_banner = mock.MagicMock()

    _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx)

    env = captured["comm_env"].get_env("node-1")
    assert env["GLOO_SOCKET_IFNAME"] == "wlan0"
    assert env["NODE_IP"] == "192.168.1.10"
    assert "VLLM_HOST_IP" not in env


def test_launch_containers_parallel_applies_runtime_finalize_hook():
    """launch_containers_parallel routes each host env through finalize_host_comm_env."""
    ctx = _make_ctx(["node-1"], dry_run=True)
    comm_env = ClusterCommEnv.from_per_host({"node-1": {"NODE_IP": "192.168.0.155"}})

    runtime = mock.MagicMock()
    runtime.finalize_host_comm_env.side_effect = lambda env: {**env, "VLLM_HOST_IP": env["NODE_IP"]}

    executor = mock.MagicMock()
    executor.workload_labels_for_cluster.return_value = {}

    captured = {}

    def _gen(**kwargs):
        captured["nccl_env"] = kwargs.get("nccl_env")
        return "script"

    executor.generate_launch_script.side_effect = _gen

    rc = _cluster_ops.launch_containers_parallel(ctx, [("node-1", "c0")], executor, comm_env, runtime=runtime)

    assert rc == 0
    runtime.finalize_host_comm_env.assert_called_once()
    assert captured["nccl_env"]["NODE_IP"] == "192.168.0.155"
    assert captured["nccl_env"]["VLLM_HOST_IP"] == "192.168.0.155"


class TestPlacementInitAddresses:
    """The scheduler's placement must speak the selected init network."""

    def test_address_map_zips_cluster_hosts_to_selection(self):
        """Given aligned lists, the map pairs each cluster host with its init address."""
        selection = _init_network.InitNetworkSelection(
            head_ip="10.113.145.138",
            hosts=("10.113.145.138", "10.113.145.65"),
            network="management",
        )

        assert _init_network.init_address_map(["127.0.0.1", "10.113.145.65"], selection) == {
            "127.0.0.1": "10.113.145.138",
            "10.113.145.65": "10.113.145.65",
        }

    def test_address_map_falls_back_to_identity_on_length_mismatch(self):
        """A selection not derived from these hosts must not produce a skewed map."""
        selection = _init_network.InitNetworkSelection(head_ip="10.0.0.1", hosts=("10.0.0.1",), network="management")

        assert _init_network.init_address_map(["127.0.0.1", "node-2"], selection) == {
            "127.0.0.1": "127.0.0.1",
            "node-2": "node-2",
        }

    def test_remap_rewrites_loopback_rank_host(self):
        """Given a loopback placement, the remapped view carries the routable head IP."""
        placement = _one_gpu_per_host_placement(["127.0.0.1", "10.113.145.65"])

        remapped = _init_network.remap_placement_addresses(
            placement,
            {"127.0.0.1": "10.113.145.138", "10.113.145.65": "10.113.145.65"},
        )

        assert remapped.host_for_rank(0) == "10.113.145.138"
        assert remapped.host_for_rank(1) == "10.113.145.65"
        assert remapped.hosts_used == ("10.113.145.138", "10.113.145.65")
        # Source placement is untouched — it stays the SSH-addressable truth.
        assert placement.host_for_rank(0) == "127.0.0.1"

    def test_remap_preserves_slot_resources_and_multi_gpu_ranks(self):
        """Per-rank GPU index and fractional claims survive the address rewrite."""
        placement = RankAssignment(
            by_rank=(
                RankSlot(host="127.0.0.1", local_gpu=0, util_fraction=0.5, memory_gb=40.0),
                RankSlot(host="127.0.0.1", local_gpu=1),
            ),
            hosts_used=("127.0.0.1",),
        )

        remapped = _init_network.remap_placement_addresses(placement, {"127.0.0.1": "10.0.0.1"})

        assert [(s.host, s.local_gpu, s.util_fraction, s.memory_gb) for s in remapped.by_rank] == [
            ("10.0.0.1", 0, 0.5, 40.0),
            ("10.0.0.1", 1, 1.0, None),
        ]

    def test_remap_is_identity_when_no_host_changes(self):
        """An all-routable placement is returned verbatim (no needless copy)."""
        placement = _one_gpu_per_host_placement(["node-1", "node-2"])

        assert _init_network.remap_placement_addresses(placement, {"node-1": "node-1", "node-2": "node-2"}) is placement

    def test_remap_of_none_is_none(self):
        """Callers that never scheduled keep their None placement."""
        assert _init_network.remap_placement_addresses(None, {"a": "b"}) is None


def test_native_cluster_remaps_loopback_placement_to_routable_head(monkeypatch):
    """Given a 127.0.0.1 head, node commands must not receive a loopback placement.

    Regression: ``_resolve_master_addr`` consults placement *before* the
    resolved host list, so a cluster whose head is listed as ``127.0.0.1``
    emitted ``--master-addr 127.0.0.1`` and every worker rendezvoused on
    its own loopback — even though head-IP detection and the reachability
    guard had both resolved the routable address.
    """
    hosts = ["127.0.0.1", "10.113.145.65"]
    ctx = _make_ctx(hosts, placement=_one_gpu_per_host_placement(hosts))
    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _cluster_ops,
        "detect_ib_with_ips",
        lambda *_a, **_k: (ClusterCommEnv.empty(), {}, {}),
    )
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda _ctx: "10.113.145.138")
    monkeypatch.setattr(_cluster_ops, "find_port", lambda _ctx, _host, port: port)
    # Management head is reachable from the worker — the real-world case.
    monkeypatch.setattr(_init_network, "workers_can_reach", lambda _ctx, _target: True)
    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", lambda *_a, **_k: 1)

    runtime = mock.MagicMock()
    runtime._resolve_executor.return_value.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
    runtime.generate_node_command = mock.MagicMock(return_value="serve")
    runtime.get_extra_docker_opts = lambda: []
    runtime._print_cluster_banner = mock.MagicMock()

    _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx)

    call = runtime.generate_node_command.call_args
    assert call.kwargs["head_ip"] == "10.113.145.138"
    assert call.kwargs["hosts"] == ["10.113.145.138", "10.113.145.65"]
    assert call.kwargs["placement"].host_for_rank(0) == "10.113.145.138"
    # SSH targeting still uses the cluster-config identifiers.
    assert ctx.hosts == ["127.0.0.1", "10.113.145.65"]


def test_native_cluster_remaps_placement_onto_fabric_on_ib_fallback(monkeypatch):
    """Given the IB fallback verdict, placement addresses follow the fabric too."""
    hosts = ["node-1", "node-2"]
    ctx = _make_ctx(hosts, placement=_one_gpu_per_host_placement(hosts))
    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _cluster_ops,
        "detect_ib_with_ips",
        lambda *_a, **_k: (
            ClusterCommEnv.empty(),
            {"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
            {"node-1": "cx0", "node-2": "cx0"},
        ),
    )
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda _ctx: "192.168.128.10")
    monkeypatch.setattr(
        _cluster_ops,
        "resolve_hosts_for_init",
        lambda _ctx, _head_ip: ["192.168.128.10", "192.168.96.114"],
    )
    monkeypatch.setattr(_cluster_ops, "find_port", lambda _ctx, _host, port: port)
    monkeypatch.setattr(_init_network, "workers_can_reach", lambda _ctx, target: target == "192.168.100.10")
    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", lambda *_a, **_k: 1)

    runtime = mock.MagicMock()
    runtime._resolve_executor.return_value.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
    runtime.generate_node_command = mock.MagicMock(return_value="serve")
    runtime.get_extra_docker_opts = lambda: []
    runtime._print_cluster_banner = mock.MagicMock()

    _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx)

    call = runtime.generate_node_command.call_args
    assert call.kwargs["placement"].host_for_rank(0) == "192.168.100.10"
    assert call.kwargs["placement"].host_for_rank(1) == "192.168.100.11"
