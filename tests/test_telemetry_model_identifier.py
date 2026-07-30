"""Tests for telemetry model-identifier disclosure rules.

The recipe's ``model`` value is the one telemetry field that can name a
user's work.  It is emitted verbatim **only** for a repo confirmed publicly
readable on the Hub; everything else collapses to a coarse placeholder that
still distinguishes *why* it was withheld.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sparkrun.models.vram import (
    MODEL_VISIBILITY_PRIVATE,
    MODEL_VISIBILITY_PUBLIC,
    MODEL_VISIBILITY_UNKNOWN,
    fetch_model_visibility,
)
from sparkrun.telemetry.util import (
    MODEL_LOCAL_PATH,
    MODEL_PRIVATE,
    MODEL_UNKNOWN,
    model_identifier,
)


# ---------------------------------------------------------------------------
# fetch_model_visibility
# ---------------------------------------------------------------------------


class _Info:
    def __init__(self, private=False, gated=False):
        self.private = private
        self.gated = gated


@pytest.fixture(autouse=True)
def _clear_visibility_memo():
    from sparkrun.models import vram

    vram._VISIBILITY_MEMO.clear()
    yield
    vram._VISIBILITY_MEMO.clear()


@pytest.mark.parametrize(
    "info,expected",
    [
        (_Info(), MODEL_VISIBILITY_PUBLIC),
        (_Info(private=True), MODEL_VISIBILITY_PRIVATE),
        (_Info(gated=True), MODEL_VISIBILITY_PRIVATE),
        # `gated` is False | "auto" | "manual"; any truthy value is not freely readable.
        (_Info(gated="manual"), MODEL_VISIBILITY_PRIVATE),
    ],
)
def test_visibility_reads_private_and_gated(info, expected):
    with patch("huggingface_hub.model_info", return_value=info):
        assert fetch_model_visibility("org/model") == expected


def test_visibility_is_unknown_when_the_lookup_fails():
    """Offline, rate-limited, or nonexistent must not read as 'public'."""
    with patch("huggingface_hub.model_info", side_effect=OSError("offline")):
        assert fetch_model_visibility("org/model") == MODEL_VISIBILITY_UNKNOWN


def test_visibility_is_memoized_per_process():
    with patch("huggingface_hub.model_info", return_value=_Info()) as mi:
        assert fetch_model_visibility("org/model") == MODEL_VISIBILITY_PUBLIC
        assert fetch_model_visibility("org/model") == MODEL_VISIBILITY_PUBLIC
    assert mi.call_count == 1


# ---------------------------------------------------------------------------
# model_identifier
# ---------------------------------------------------------------------------


def test_public_repo_is_sent_verbatim():
    with patch("sparkrun.models.vram.fetch_model_visibility", return_value=MODEL_VISIBILITY_PUBLIC):
        assert model_identifier("Qwen/Qwen3-1.7B") == "Qwen/Qwen3-1.7B"


def test_private_repo_becomes_its_own_placeholder():
    with patch("sparkrun.models.vram.fetch_model_visibility", return_value=MODEL_VISIBILITY_PRIVATE):
        assert model_identifier("acme-internal/secret-7b") == MODEL_PRIVATE


def test_unresolvable_visibility_fails_closed():
    """The disclosure-critical case: unknown must never fall through to the name."""
    with patch("sparkrun.models.vram.fetch_model_visibility", return_value=MODEL_VISIBILITY_UNKNOWN):
        assert model_identifier("acme-internal/secret-7b") == MODEL_UNKNOWN


@pytest.mark.parametrize(
    "path",
    [
        "/mnt/models/internal-v3",
        "~/models/internal-v3",
        "./weights",
        "../weights",
        "C:\\models\\internal",
        "/srv/nfs/team/model/checkpoint",  # >1 slash, absolute
    ],
)
def test_local_paths_never_reach_the_hub(path):
    """A filesystem path is classified without any network call at all."""
    with patch("sparkrun.models.vram.fetch_model_visibility", side_effect=AssertionError("probed")) as probe:
        assert model_identifier(path) == MODEL_LOCAL_PATH
    assert probe.call_count == 0


def test_probe_disabled_reports_unknown_without_lookup():
    with patch("sparkrun.models.vram.fetch_model_visibility", side_effect=AssertionError("probed")) as probe:
        assert model_identifier("org/model", probe=False) == MODEL_UNKNOWN
    assert probe.call_count == 0


def test_empty_model_is_omitted():
    assert model_identifier(None) is None
    assert model_identifier("   ") is None


def test_placeholders_do_not_leak_the_name():
    """Whatever the outcome, a withheld identifier must not embed the original."""
    secret = "acme-internal/project-orion"
    for verdict in (MODEL_VISIBILITY_PRIVATE, MODEL_VISIBILITY_UNKNOWN):
        with patch("sparkrun.models.vram.fetch_model_visibility", return_value=verdict):
            assert secret not in model_identifier(secret)
    assert secret not in model_identifier("/mnt/" + secret)


# ---------------------------------------------------------------------------
# Emission guard
# ---------------------------------------------------------------------------


def test_opted_out_run_never_probes_the_hub(tmp_path, monkeypatch):
    """An opted-out user must not pay for a lookup that only serves telemetry.

    `send_event` checks enablement, but the event is built *before* it is
    called — so the guard has to sit in the emitter, ahead of construction.
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.telemetry.emit import emit_run_telemetry

    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "1")
    config = SparkrunConfig(tmp_path / "config.yaml")

    with (
        patch("sparkrun.models.vram.fetch_model_visibility", side_effect=AssertionError("probed")) as probe,
        patch("sparkrun.telemetry.emit.build_run_event", side_effect=AssertionError("built")) as build,
    ):
        emit_run_telemetry(config, result=object(), recipe=object(), cluster=None, options=None)

    assert probe.call_count == 0
    assert build.call_count == 0
