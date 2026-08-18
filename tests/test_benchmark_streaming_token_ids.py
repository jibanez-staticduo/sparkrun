"""Regression tests: llama-benchy's ``return_token_ids`` against SGLang.

llama-benchy sets ``stream`` and ``return_token_ids`` together on every
measurement request (``client.py:_build_generation_payload``).  SGLang's
``entrypoints/openai/serving_chat.py`` rejects that pair outright::

    HTTP 400: return_token_ids is not supported with streaming on
    /v1/chat/completions. Please set stream=false when using
    return_token_ids=true.

The warmup and coherence probes set neither field, so they returned 200 while
*every* measurement 400'd — a benchmark that looked healthy start to finish and
reported nothing.

``--extra-body`` is the only lever llama-benchy exposes here
(``payload.update(self.extra_body)`` runs last), so suppressing the field is
the whole workaround.  It costs fidelity — see the warning text — which is why
it must apply only where the server actually rejects it.
"""

from __future__ import annotations

from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework


def _recipe(runtime: str):
    from sparkrun.core.recipe import Recipe

    return Recipe.from_dict(
        {
            "recipe_version": "2",
            "name": "tok-ids",
            "model": "org/model",
            "runtime": runtime,
            "container": "img:tag",
        }
    )


def test_sglang_suppresses_return_token_ids(v):
    fw = LlamaBenchyFramework()
    assert fw.prepare_benchmark_args(_recipe("sglang"), {}, {}).get("extra_body") == "return_token_ids=false"


def test_other_runtimes_keep_token_ids(v):
    """The fallback costs measurement fidelity — it must not apply where it needn't."""
    fw = LlamaBenchyFramework()
    for runtime in ("vllm-distributed", "vllm-ray", "trtllm"):
        assert "extra_body" not in fw.prepare_benchmark_args(_recipe(runtime), {}, {}), runtime


def test_unresolvable_runtime_keeps_token_ids(v):
    """Failing to identify a runtime is not grounds for degrading its measurements."""

    class _Bogus:
        runtime = "not-a-real-runtime"
        command = None

    fw = LlamaBenchyFramework()
    assert "extra_body" not in fw.prepare_benchmark_args(_Bogus(), {}, {})


def test_suppression_reaches_the_command_line(v):
    """``--extra-body`` is llama-benchy's only lever here, so it has to be rendered."""
    fw = LlamaBenchyFramework()
    args = {"pp": [2048], "depth": [0], "prefix_caching": True}
    args.update(fw.prepare_benchmark_args(_recipe("sglang"), {}, {}))
    cmd = fw.build_benchmark_command("http://h:8000/v1", "org/model", args)

    assert "--extra-body" in cmd
    assert cmd[cmd.index("--extra-body") + 1] == "return_token_ids=false"


def test_user_supplied_extra_body_wins(v):
    """The caller merges with ``setdefault``; ``-b extra_body=…`` must survive that.

    This is the escape hatch for the day llama-benchy moves to
    ``/v1/completions`` or SGLang implements token ids for chat streaming.
    """
    fw = LlamaBenchyFramework()
    bench_args = {"extra_body": "return_token_ids=true"}
    for k, val in fw.prepare_benchmark_args(_recipe("sglang"), {}, {}).items():
        bench_args.setdefault(k, val)

    assert bench_args["extra_body"] == "return_token_ids=true"
