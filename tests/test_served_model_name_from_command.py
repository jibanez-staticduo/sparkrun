"""Served-model-name resolution from a ``command:`` template (issue #257, part 3).

The supported spelling is ``defaults.served_model_name``, which every runtime
reconciles into the rendered command.  A recipe that instead hardcodes
``--served-model-name <name>`` in ``command:`` bypasses that, and the name
becomes invisible to the config chain — so the benchmark asked the endpoint for
the *model id* and every task 404'd.  The proxy (which routes on the name) and
the container labels had the same blind spot.
"""

from __future__ import annotations

import textwrap

import pytest

from sparkrun.core.recipe import (
    Recipe,
    extract_served_model_name_from_command,
    resolve_served_model_name,
)


# ---------------------------------------------------------------------------
# extract_served_model_name_from_command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        ("vllm serve org/m --served-model-name dsv4", "dsv4"),
        ("vllm serve org/m --served-model-name=dsv4", "dsv4"),
        ("sglang serve org/m --served-model-name  dsv4  --port 8000", "dsv4"),
        # atlas spelling
        ("atlas serve org/m --model-name dsv4", "dsv4"),
        # llama.cpp spelling
        ("llama-server -m x.gguf --alias dsv4", "dsv4"),
        # quoted value
        ('vllm serve org/m --served-model-name "dsv4"', "dsv4"),
        # vLLM accepts several names; the first is the canonical id
        ("vllm serve org/m --served-model-name dsv4 alt1 alt2", "dsv4"),
    ],
)
def test_extract_recognized_spellings(command, expected):
    assert extract_served_model_name_from_command(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        None,
        "",
        "vllm serve org/m --port 8000",
        # An unrendered placeholder means the value really is in the config
        # chain, which resolves it properly — returning "{served_model_name}"
        # would be worse than returning nothing.
        "vllm serve org/m --served-model-name {served_model_name}",
        # Bare short flag is deliberately not matched (collision-prone).
        "llama-server -m x.gguf -a dsv4",
    ],
)
def test_extract_returns_none(command):
    assert extract_served_model_name_from_command(command) is None


def test_extract_tolerates_non_string_command():
    """Callers reach this via ``getattr(recipe, "command", None)`` on objects
    that only duck-type as recipes, on the launch path's best-effort metadata
    write — a TypeError there would fail an otherwise-fine launch."""
    from unittest.mock import MagicMock

    assert extract_served_model_name_from_command(MagicMock()) is None
    assert extract_served_model_name_from_command(object()) is None
    assert extract_served_model_name_from_command(123) is None


# ---------------------------------------------------------------------------
# resolve_served_model_name — declared wins, command is last resort
# ---------------------------------------------------------------------------


def _recipe(defaults_block: str = "", command_tail: str = "") -> Recipe:
    text = textwrap.dedent(
        """
        recipe_version: "2"
        model: deepseek-ai/DeepSeek-V4-Flash-0731
        runtime: vllm
        container: example/img:latest
        defaults:
          port: 8000
        %s
        command: |
          vllm serve {model} --port {port}%s
        """
    ) % (defaults_block, command_tail)
    return Recipe.from_dict(__import__("yaml").safe_load(text))


def test_declared_value_beats_command():
    r = _recipe(command_tail=" --served-model-name from-command")
    assert resolve_served_model_name(r, "declared") == "declared"


def test_command_used_when_nothing_declared():
    r = _recipe(command_tail=" --served-model-name from-command")
    assert resolve_served_model_name(r, None) == "from-command"


def test_model_id_is_the_final_fallback():
    r = _recipe()
    assert resolve_served_model_name(r, None) == r.model


# ---------------------------------------------------------------------------
# Recipe.effective_served_model_name (container labels)
# ---------------------------------------------------------------------------


def test_effective_served_model_name_reads_command_template():
    r = _recipe(command_tail=" --served-model-name dsv4")
    assert r.effective_served_model_name == "dsv4"


def test_effective_served_model_name_defaults_still_win():
    r = _recipe(defaults_block="  served_model_name: from-defaults", command_tail=" --served-model-name from-command")
    assert r.effective_served_model_name == "from-defaults"


def test_effective_served_model_name_unchanged_without_the_flag():
    r = _recipe()
    assert r.effective_served_model_name == "deepseek-ai/DeepSeek-V4-Flash-0731"


# ---------------------------------------------------------------------------
# llama-benchy: the reported 404
# ---------------------------------------------------------------------------


def _prepare(recipe):
    from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework

    chain = recipe.build_config_chain()
    return LlamaBenchyFramework().prepare_benchmark_args(recipe, {k: chain.get(k) for k in chain.keys()}, {})


def test_benchmark_gets_served_name_from_command_template():
    """The reported failure: without this, llama-benchy requests ``--model``
    (the HF id) from a server that only answers to ``dsv4`` → HTTP 404."""
    r = _recipe(command_tail=" --served-model-name dsv4")
    assert _prepare(r) == {"served_model_name": "dsv4"}


def test_benchmark_still_prefers_the_config_chain():
    r = _recipe(defaults_block="  served_model_name: from-defaults", command_tail=" --served-model-name from-command")
    assert _prepare(r) == {"served_model_name": "from-defaults"}


def test_benchmark_emits_nothing_when_no_served_name_anywhere():
    """llama-benchy defaults --served-model-name to --model, so an absent key
    is correct here — not something to fill in with the model id."""
    assert _prepare(_recipe()) == {}


# ---------------------------------------------------------------------------
# Proxy discovery metadata
# ---------------------------------------------------------------------------


def test_job_metadata_records_served_name_from_command(tmp_path):
    """The proxy *routes* on this name, so the same blind spot mis-routed."""
    from sparkrun.orchestration.job_metadata import generate_cluster_id, generate_intent_id, save_job_metadata

    r = _recipe(command_tail=" --served-model-name dsv4")
    save_job_metadata(
        cluster_id=generate_cluster_id(generate_intent_id(r), "a" * 12),
        recipe=r,
        hosts=["h1"],
        cache_dir=str(tmp_path),
    )
    import yaml

    written = list((tmp_path / "jobs").glob("*.yaml"))
    assert written, "no job metadata file written"
    meta = yaml.safe_load(written[0].read_text())
    assert meta.get("served_model_name") == "dsv4"


# ---------------------------------------------------------------------------
# Intent id must NOT widen — see resolve_served_model_name's docstring
# ---------------------------------------------------------------------------


def test_intent_id_ignores_command_template_served_name():
    """Widening the intent id would orphan already-running workloads from
    ``stop`` / ``logs`` / ``--ensure``, which recompute it from the recipe."""
    from sparkrun.orchestration.job_metadata import generate_intent_id

    plain = _recipe()
    hardcoded = _recipe(command_tail=" --served-model-name dsv4")
    assert generate_intent_id(plain) == generate_intent_id(hardcoded)
