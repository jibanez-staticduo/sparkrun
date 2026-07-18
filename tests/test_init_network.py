"""Tests for native-cluster distributed-init network selection."""

from unittest import mock

from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.runtimes import _cluster_ops, _init_network


def _make_ctx(hosts: list[str], *, dry_run: bool = False) -> _cluster_ops.ClusterContext:
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
