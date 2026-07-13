from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.tokenary import TokenaryRuntime


def test_tokenary_node_command_rendezvous_at_head_with_hosts():
    recipe_data = {
        "name": "test-recipe",
        "model": "/mnt/quant/Minimax-M3-v0-NVFP4",
        "runtime": "tokenary",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = TokenaryRuntime()
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]

    for rank in range(4):
        cmd = runtime.generate_node_command(
            recipe,
            {},
            head_ip="10.0.0.1",
            num_nodes=4,
            node_rank=rank,
            hosts=hosts,
        )
        assert "--master-addr 10.0.0.1" in cmd, "rank %d -> %s" % (rank, cmd)
        assert "--world-size 4" in cmd
        assert "--rank %d" % rank in cmd


def test_tokenary_encode_only_and_ner_bool_flags_from_defaults():
    """encode_only / ner recipe defaults must emit the flag-only toggles."""
    runtime = TokenaryRuntime()

    encode = Recipe.from_dict(
        {
            "name": "embed-encode",
            "model": "Qwen/Qwen3.6-27B-FP8",
            "runtime": "tokenary",
            "defaults": {"port": 8005, "encode_only": True, "gpu_memory_utilization": 0.15},
        }
    )
    cmd = runtime.generate_command(encode, {}, is_cluster=False)
    assert "--encode-only" in cmd
    assert "--ner" not in cmd
    assert "--port 8005" in cmd

    ner = Recipe.from_dict(
        {
            "name": "ner-lane",
            "model": "org/ner-model",
            "runtime": "tokenary",
            "defaults": {"port": 8004, "ner": True},
        }
    )
    cmd = runtime.generate_command(ner, {}, is_cluster=False)
    assert "--ner" in cmd
    assert "--encode-only" not in cmd

    # Falsy / absent toggles must not leak the flag.
    plain = Recipe.from_dict(
        {
            "name": "chat",
            "model": "Qwen/Qwen3.6-27B-FP8",
            "runtime": "tokenary",
            "defaults": {"port": 8001, "encode_only": False},
        }
    )
    cmd = runtime.generate_command(plain, {}, is_cluster=False)
    assert "--encode-only" not in cmd
    assert "--ner" not in cmd
