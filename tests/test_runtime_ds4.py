"""Unit tests for the ds4 / DwarfStar runtime plugin.

The flag expectations here are transcribed from the argument parser in
upstream ``ds4_server.c``, not from the shared help text — ``ds4-server``
exits(2) on an unknown option, and several flags documented in the help are
parsed only by the ``ds4`` CLI binary.
"""

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.ds4 import _DS4_MODEL_ALIASES, Ds4Runtime


def _recipe(**overrides) -> Recipe:
    base = {
        "name": "test-ds4",
        "model": "antirez/deepseek-v4-gguf:IQ2XXS",
        "runtime": "ds4",
        "defaults": {"served_model_name": "deepseek-v4-flash"},
    }
    base.update(overrides)
    return Recipe.from_dict(base)


def _defaults(**extra) -> dict:
    return {"served_model_name": "deepseek-v4-flash", **extra}


# --- identity / container ---


def test_ds4_runtime_identity():
    runtime = Ds4Runtime()
    assert runtime.runtime_name == "ds4"
    assert runtime.get_family() == "ds4"
    assert runtime.cluster_strategy() == "native"


def test_ds4_default_container_follows_the_latest_convention():
    """Same ``<prefix>:latest`` default every other image-backed runtime uses."""
    runtime = Ds4Runtime()
    assert runtime.resolve_container(_recipe()) == "ghcr.io/spark-arena/dgx-ds4:latest"
    assert runtime.default_image_for() == "ghcr.io/spark-arena/dgx-ds4:latest"


def test_ds4_recipe_container_wins():
    runtime = Ds4Runtime()
    recipe = _recipe(container="ghcr.io/spark-arena/dgx-ds4:20260809-84cc882-cu131")
    assert runtime.resolve_container(recipe) == "ghcr.io/spark-arena/dgx-ds4:20260809-84cc882-cu131"


# --- placement ---


def test_ds4_world_size_is_always_one():
    """ds4-server has no distributed flags; solo placement is mandatory."""
    assert Ds4Runtime().world_size(None, recipe=None, cluster=None) == 1


# --- command generation ---


def test_ds4_structured_command_basics():
    cmd = Ds4Runtime().generate_command(_recipe(), {}, is_cluster=False)
    assert cmd.startswith("ds4-server -m antirez/deepseek-v4-gguf:IQ2XXS")
    # Injected defaults: ds4-server itself binds 127.0.0.1, which would be
    # unreachable from the control node.
    assert "--host 0.0.0.0" in cmd
    assert "--port 8000" in cmd
    assert "--backend cuda" in cmd


def test_ds4_recipe_defaults_override_injected_defaults():
    recipe = _recipe(defaults=_defaults(host="127.0.0.1", port=9999, backend="cpu"))
    cmd = Ds4Runtime().generate_command(recipe, {}, is_cluster=False)
    assert "--host 127.0.0.1" in cmd
    assert "--port 9999" in cmd
    assert "--backend cpu" in cmd
    assert "0.0.0.0" not in cmd


def test_ds4_cli_overrides_win():
    cmd = Ds4Runtime().generate_command(_recipe(), {"port": 9001}, is_cluster=False)
    assert "--port 9001" in cmd
    assert "--port 8000" not in cmd


def test_ds4_presynced_gguf_path_used_for_model():
    """The launcher injects the container-internal path; ds4 has no downloader."""
    cmd = Ds4Runtime().generate_command(
        _recipe(),
        {"_gguf_model_path": "/cache/huggingface/blobs/flash.gguf"},
        is_cluster=False,
    )
    assert cmd.startswith("ds4-server -m /cache/huggingface/blobs/flash.gguf")
    assert "antirez/deepseek-v4-gguf" not in cmd


def test_ds4_resolved_model_path_used_when_no_gguf_sync():
    """An absolute ``model:`` skips download and arrives as resolved_model_path."""
    recipe = _recipe(model="/mnt/models/flash.gguf")
    cmd = Ds4Runtime().generate_command(
        recipe,
        {"resolved_model_path": "/mnt/models/flash.gguf"},
        is_cluster=False,
    )
    assert cmd.startswith("ds4-server -m /mnt/models/flash.gguf")


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("ctx_size", 131072, "-c 131072"),
        ("max_tokens", 4096, "-n 4096"),
        ("threads", 8, "-t 8"),
        ("gpu_vram", "auto", "--gpu-vram auto"),
        ("gpu_devices", "0,1", "--gpu-devices 0,1"),
        ("batched_session", 16, "--batched-session 16"),
        ("kv_disk_dir", "/cache/ds4-kv", "--kv-disk-dir /cache/ds4-kv"),
        ("kv_disk_space_mb", 8192, "--kv-disk-space-mb 8192"),
        ("power", 60, "--power 60"),
        ("prefill_chunk", 2048, "--prefill-chunk 2048"),
        ("mtp", "/cache/dspark.gguf", "--mtp /cache/dspark.gguf"),
        ("dspark_confidence", 0.7, "--dspark-confidence 0.7"),
        ("ssd_streaming_cache_experts", "40GB", "--ssd-streaming-cache-experts 40GB"),
    ],
)
def test_ds4_valued_flags(key, value, expected):
    recipe = _recipe(defaults=_defaults(**{key: value}))
    assert expected in Ds4Runtime().generate_command(recipe, {}, is_cluster=False)


@pytest.mark.parametrize(
    "key,flag",
    [
        ("cuda_tensor_parallel", "--cuda-tensor-parallel"),
        ("quality", "--quality"),
        ("warm_weights", "--warm-weights"),
        ("ssd_streaming", "--ssd-streaming"),
        ("dspark", "--dspark"),
        ("dspark_strict", "--dspark-strict"),
        ("glm_mtp", "--glm-mtp"),
        ("cors", "--cors"),
    ],
)
def test_ds4_bool_flags(key, flag):
    runtime = Ds4Runtime()
    on = runtime.generate_command(_recipe(defaults=_defaults(**{key: True})), {}, is_cluster=False)
    off = runtime.generate_command(_recipe(defaults=_defaults(**{key: False})), {}, is_cluster=False)
    assert flag in on
    # A bare toggle takes no value — a falsy setting must omit it entirely.
    assert flag not in off


def test_ds4_bool_flag_emits_no_value():
    """``--cors`` is a bare toggle in ds4_server.c despite reading like a value."""
    cmd = Ds4Runtime().generate_command(_recipe(defaults=_defaults(cors=True)), {}, is_cluster=False)
    assert "--cors True" not in cmd
    assert cmd.rstrip().endswith("--cors") or " --cors " in cmd


def test_ds4_max_model_len_translates_to_ctx_size():
    """Cross-runtime portability with the vLLM/SGLang spelling."""
    recipe = _recipe(defaults=_defaults(max_model_len=65536))
    cmd = Ds4Runtime().generate_command(recipe, {}, is_cluster=False)
    assert "-c 65536" in cmd


def test_ds4_explicit_ctx_size_beats_max_model_len():
    recipe = _recipe(defaults=_defaults(max_model_len=65536, ctx_size=131072))
    cmd = Ds4Runtime().generate_command(recipe, {}, is_cluster=False)
    assert "-c 131072" in cmd
    assert "65536" not in cmd


def test_ds4_max_model_len_cli_override_translates():
    cmd = Ds4Runtime().generate_command(_recipe(), {"max_model_len": 32768}, is_cluster=False)
    assert "-c 32768" in cmd


def test_ds4_command_template_is_rendered_verbatim():
    recipe = _recipe(command="ds4-server -m {model} -c 100000 --host 0.0.0.0 --port {port}")
    cmd = Ds4Runtime().generate_command(recipe, {"port": 8080}, is_cluster=False)
    assert "--port 8080" in cmd
    assert "-c 100000" in cmd


def test_ds4_skip_keys_strip_flags_from_template():
    recipe = _recipe(command="ds4-server -m {model} --port 8000 --quality -c 4096")
    cmd = Ds4Runtime().generate_command(recipe, {}, is_cluster=False, skip_keys={"quality", "ctx_size"})
    assert "--quality" not in cmd
    assert "-c 4096" not in cmd
    assert "--port 8000" in cmd


def test_ds4_skip_keys_strip_flags_from_structured_command():
    recipe = _recipe(defaults=_defaults(ctx_size=4096, quality=True))
    cmd = Ds4Runtime().generate_command(recipe, {}, is_cluster=False, skip_keys={"quality", "ctx_size"})
    assert "--quality" not in cmd
    assert "-c 4096" not in cmd


def test_ds4_long_form_ctx_alias_is_stripped():
    """``--ctx`` is the long form of the canonical ``-c``."""
    recipe = _recipe(command="ds4-server -m {model} --ctx 4096 --port 8000")
    cmd = Ds4Runtime().generate_command(recipe, {}, is_cluster=False, skip_keys={"ctx_size"})
    assert "--ctx" not in cmd


# --- served-name allowlist ---


def test_ds4_no_api_key_support():
    """ds4-server has no auth; the base ``None`` must not be papered over."""
    assert Ds4Runtime().resolve_api_key(_recipe()) is None


@pytest.mark.parametrize(
    "model,expected",
    [
        ("DeepSeek-V4-Flash-IQ2XXS-chat-v2-imatrix-0731.gguf", "deepseek-v4-flash"),
        ("DeepSeek-V4-Pro-Q4K-Layers00-30.gguf", "deepseek-v4-pro"),
        ("GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS.gguf", "glm-5.2"),
        ("/models/deepseek-v4-flash-mxfp4.gguf", "deepseek-v4-flash"),
        ("some-other-model.gguf", None),
        (None, None),
    ],
)
def test_ds4_infer_model_alias(model, expected):
    assert Ds4Runtime.infer_model_alias(model) == expected


def test_ds4_known_aliases_match_upstream_allowlist():
    """Transcribed from ``server_model_alias_known()`` in ds4_server.c."""
    assert _DS4_MODEL_ALIASES == {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "glm-5.2",
        "glm-5.2-chat",
        "glm-5.2-no-think",
        "glm-5.2-nothink",
        "glm-5.2-reasoner",
        "zai/glm-5.2",
        "zai/glm-5.2-chat",
        "zai/glm-5.2-reasoner",
    }


# --- validation ---


def test_ds4_valid_recipe_has_no_issues():
    assert Ds4Runtime().validate_recipe(_recipe()) == []


def test_ds4_missing_served_model_name_is_flagged():
    recipe = _recipe(defaults={})
    issues = Ds4Runtime().validate_recipe(recipe)
    assert any("served_model_name" in i and "404" in i for i in issues)


def test_ds4_missing_served_model_name_suggests_the_right_alias():
    recipe = _recipe(model="antirez/deepseek-v4-gguf:DeepSeek-V4-Pro-IQ2XXS", defaults={})
    issues = Ds4Runtime().validate_recipe(recipe)
    assert any("deepseek-v4-pro" in i for i in issues)


def test_ds4_unknown_served_model_name_is_flagged():
    recipe = _recipe(defaults=_defaults(served_model_name="deepseek-v4-flash-q2"))
    issues = Ds4Runtime().validate_recipe(recipe)
    assert any("404" in i for i in issues)


def test_ds4_api_key_is_rejected_not_silently_ignored():
    recipe = _recipe(defaults=_defaults(api_key="sk-secret"))
    issues = Ds4Runtime().validate_recipe(recipe)
    assert any("no authentication" in i for i in issues)


def test_ds4_min_nodes_gt_one_is_rejected():
    issues = Ds4Runtime().validate_recipe(_recipe(min_nodes=2))
    assert any("single-node only" in i for i in issues)


@pytest.mark.parametrize("key", ["tensor_parallel", "pipeline_parallel", "data_parallel"])
def test_ds4_cross_host_parallelism_is_rejected(key):
    recipe = _recipe(defaults=_defaults(**{key: 2}))
    issues = Ds4Runtime().validate_recipe(recipe)
    assert any(key in i for i in issues)


@pytest.mark.parametrize("key", ["tensor_parallel", "pipeline_parallel", "data_parallel"])
def test_ds4_parallelism_of_one_is_fine(key):
    recipe = _recipe(defaults=_defaults(**{key: 1}))
    assert Ds4Runtime().validate_recipe(recipe) == []


def test_ds4_non_gguf_model_is_flagged():
    recipe = _recipe(model="Qwen/Qwen3-1.7B")
    issues = Ds4Runtime().validate_recipe(recipe)
    assert any("GGUF" in i for i in issues)


@pytest.mark.parametrize(
    "model",
    [
        "antirez/deepseek-v4-gguf:IQ2XXS",
        "/mnt/models/DeepSeek-V4-Flash-IQ2XXS.gguf",
        "DeepSeek-V4-Flash.gguf",
    ],
)
def test_ds4_gguf_models_accepted(model):
    recipe = _recipe(model=model)
    assert not any("GGUF" in i for i in Ds4Runtime().validate_recipe(recipe))
