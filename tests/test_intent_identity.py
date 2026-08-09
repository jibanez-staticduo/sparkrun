"""What counts as "the same workload" — ``generate_intent_id``.

The intent_id is not only a discovery key.  ``api.run`` treats a *matching*
intent as this launch's own workload: placement subtracts it from the
occupancy snapshot (``exclude_intent_id``) and the launch then evicts it.  So
whatever the intent conflates, ``sparkrun run`` is willing to destroy.

Reported on a live 4-host cluster: two registry recipes serving
``DeepSeek-V4-Flash-0731`` at tp=2 — one on a stable image, one on the nightly
``…-b12x`` build — hashed to the *same* intent, because the container image
was not part of it.  Launching the second placed onto the two hosts the first
was serving from (their occupancy had been subtracted as "mine") and tore it
down.  These tests pin the distinctions that must survive.
"""

from __future__ import annotations

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.job_metadata import generate_intent_id

_BASE = {
    "sparkrun_version": "2",
    "runtime": "vllm-distributed",
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "container": "ghcr.io/spark-arena/dgx-vllm:stable",
    "defaults": {"port": 8000, "tensor_parallel": 2},
}


def _recipe(**changes):
    data = {**_BASE, **{k: v for k, v in changes.items() if k != "defaults"}}
    if "defaults" in changes:
        data["defaults"] = {**_BASE["defaults"], **changes["defaults"]}
    return Recipe(data)


def test_container_image_distinguishes_intents():
    """The reported bug: same model/runtime/port/tp, different image."""
    stable = generate_intent_id(_recipe())
    nightly = generate_intent_id(_recipe(container="ghcr.io/spark-arena/dgx-vllm-nightly-b12x:latest"))

    assert stable != nightly, "two images = two workloads; conflating them makes `run` evict the other"


def test_image_override_distinguishes_intents():
    """``--image`` writes through to ``recipe.container``, so it counts too."""
    base = _recipe()
    overridden = _recipe()
    overridden.container = "some/other:image"

    assert generate_intent_id(base) != generate_intent_id(overridden)


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "meta-llama/Llama-3.1-8B"},
        {"runtime": "sglang"},
        {"defaults": {"port": 8001}},
        {"defaults": {"tensor_parallel": 4}},
        {"defaults": {"served_model_name": "custom"}},
    ],
    ids=["model", "runtime", "port", "tp", "served_model_name"],
)
def test_identity_dimensions_are_distinct(changes):
    assert generate_intent_id(_recipe()) != generate_intent_id(_recipe(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"defaults": {"gpu_memory_utilization": 0.75}},
        {"defaults": {"max_model_len": 4096}},
        {"command": "vllm serve --some-new-flag"},
        {"env": {"VLLM_USE_V1": "1"}},
        {"description": "a different description"},
    ],
    ids=["gpu_mem", "max_model_len", "command", "env", "description"],
)
def test_serve_configuration_does_not_change_identity(changes):
    """The intent stays narrow on purpose.

    ``stop`` / ``logs`` / ``status`` / ``--ensure`` recompute the intent from
    the recipe *without* the flags the user typed at launch, so if a tweaked
    serve argument produced a new intent they would stop finding the running
    workload — and a relaunch would no longer evict the deployment it
    replaces, leaving it holding the GPUs.  Callers that must tell these apart
    hash ``derive_recipe_fingerprint`` alongside the intent.
    """
    assert generate_intent_id(_recipe()) == generate_intent_id(_recipe(**changes))


def test_recipe_without_a_container_still_hashes():
    """A recipe relying on the runtime's default image must not blow up."""
    data = {k: v for k, v in _BASE.items() if k != "container"}
    assert generate_intent_id(Recipe(data))


def test_intent_is_independent_of_placement():
    """Hosts are never hashed — that's what the placement token is for."""
    r = _recipe()
    assert generate_intent_id(r) == generate_intent_id(r)
