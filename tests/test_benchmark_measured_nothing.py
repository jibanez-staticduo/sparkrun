"""Regression tests: a benchmark that measured nothing must not report success.

llama-benchy prints each failed request and carries on, exiting **0** — and
still writes a result row per test, because the row echoes the *request*
parameters. Every metric in it is ``null``.

sparkrun took the zero exit code as the whole answer: it exported that file,
drew a results table of ``…`` and printed "Benchmark complete". A run in which
100% of requests failed was indistinguishable, by exit code and by output
shape, from a real measurement — and it propagated into the exported YAML/JSON/
CSV like any other result.

Observed live against SGLang (every measurement 400'd on ``return_token_ids``
while the warmup probes passed), which is where ``_ALL_NULL`` comes from.
"""

from __future__ import annotations

import json

from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework

#: Verbatim shape llama-benchy wrote when every request failed.
_ALL_NULL = {
    "benchmarks": [
        {
            "concurrency": 1,
            "context_size": 0,
            "prompt_size": 2048,
            "response_size": 32,
            "is_context_prefill_phase": False,
            "pp_throughput": None,
            "tg_throughput": None,
            "peak_throughput": None,
            "ttfr": None,
            "e2e_ttft": None,
        }
    ]
}

_MEASURED = {
    "benchmarks": [
        {
            "concurrency": 1,
            "prompt_size": 2048,
            "pp_throughput": {"mean": 1783.1, "std": 0.0, "values": [1783.1]},
            "tg_throughput": {"mean": 26.9, "std": 0.0, "values": [26.9]},
            "ttfr": {"mean": 1152.6, "std": 0.0, "values": [1152.6]},
        }
    ]
}


def test_all_null_rows_count_as_no_measurement():
    """A row's *existence* proves nothing — llama-benchy writes one either way."""
    assert LlamaBenchyFramework().measured_nothing({"json": _ALL_NULL}) is True


def test_real_results_are_not_flagged():
    assert LlamaBenchyFramework().measured_nothing({"json": _MEASURED}) is False


def test_a_single_measured_row_is_enough():
    """A partially-failed sweep still measured something; only a total loss fails."""
    mixed = {"benchmarks": [_ALL_NULL["benchmarks"][0], _MEASURED["benchmarks"][0]]}
    assert LlamaBenchyFramework().measured_nothing({"json": mixed}) is False


def test_missing_or_empty_payloads_count_as_no_measurement():
    fw = LlamaBenchyFramework()
    assert fw.measured_nothing({}) is True
    assert fw.measured_nothing({"json": {}}) is True
    assert fw.measured_nothing({"json": {"benchmarks": []}}) is True


def test_non_dict_rows_do_not_raise():
    assert LlamaBenchyFramework().measured_nothing({"json": {"benchmarks": ["junk", None]}}) is True


def test_base_default_has_no_opinion():
    """A framework that hasn't implemented this keeps the previous behaviour.

    The default must not be "empty means failed" — the base class cannot read an
    arbitrary framework's result shape, and guessing would fail working runs.
    """
    from sparkrun.benchmarking.base import BenchmarkingPlugin

    assert BenchmarkingPlugin.measured_nothing(object(), {"anything": True}) is False


def test_guard_matches_the_files_from_the_incident(tmp_path):
    """Round-trip through JSON, as ``parse_results`` does."""
    fw = LlamaBenchyFramework()
    for payload, expected in ((_ALL_NULL, True), (_MEASURED, False)):
        p = tmp_path / "r.json"
        p.write_text(json.dumps(payload))
        parsed = fw.parse_results("", "", result_file=str(p))
        assert fw.measured_nothing(parsed) is expected
