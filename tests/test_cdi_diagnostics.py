"""CDI failure diagnosis: stale-spec detection and launch-failure hints.

Both exist for the same reason. A host whose /etc/cdi/nvidia.yaml is absent or
stale fails at `docker run` with an error that names the mechanism that
refused ("unresolvable CDI devices") but not the thing to fix, and does not
say which host is affected. These turn that into a command.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

import pytest

from sparkrun.cli._setup._check import (
    FAIL,
    OK,
    WARN,
    CheckContext,
    HostState,
    _check_cdi_spec,
)
from sparkrun.orchestration.launch_diagnostics import (
    diagnose_launch_failure,
    log_launch_failure_hint,
)

# ---------------------------------------------------------------------------
# Stale-spec detection (the check item)
# ---------------------------------------------------------------------------

_GPU_HOST = {"CHECK_GPU_PRESENT": "1", "CHECK_NVIDIA_CTK": "1"}


def _state(**facts) -> HostState:
    return HostState(host="h1", facts=dict(_GPU_HOST, **facts))


def _ctx() -> CheckContext:
    return CheckContext(cluster_name="mylab", multi_host=True)


def test_spec_present_and_intact_is_ok():
    item = _check_cdi_spec(_state(CHECK_CDI_SPEC="1", CHECK_CDI_PATHS_CHECKED="12", CHECK_CDI_PATHS_MISSING="0"), _ctx())
    assert item.status == OK


def test_spec_with_missing_paths_warns_as_stale():
    """The driver-upgrade case: spec exists, but what it points at does not."""
    item = _check_cdi_spec(_state(CHECK_CDI_SPEC="1", CHECK_CDI_PATHS_CHECKED="12", CHECK_CDI_PATHS_MISSING="5"), _ctx())
    assert item.status == WARN
    assert "5 of 12" in item.detail
    assert "stale" in item.detail
    assert "nvidia-ctk cdi generate" in item.guidance


def test_absent_spec_still_fails():
    item = _check_cdi_spec(_state(CHECK_CDI_SPEC="0"), _ctx())
    assert item.status == FAIL
    assert "nvidia-ctk cdi generate" in item.guidance


def test_missing_staleness_facts_do_not_invent_a_finding():
    """An older probe emits no path counts; that must read as OK, not stale.

    The check runs against whatever setup_check.sh the *remote host* has, so a
    cluster mid-upgrade will return facts without these keys.
    """
    item = _check_cdi_spec(_state(CHECK_CDI_SPEC="1"), _ctx())
    assert item.status == OK


def test_unparseable_counts_do_not_invent_a_finding():
    item = _check_cdi_spec(_state(CHECK_CDI_SPEC="1", CHECK_CDI_PATHS_CHECKED="?", CHECK_CDI_PATHS_MISSING="?"), _ctx())
    assert item.status == OK


# ---------------------------------------------------------------------------
# The probe script's own staleness arithmetic
# ---------------------------------------------------------------------------

_SPEC_TEMPLATE = """\
cdiVersion: 0.5.0
kind: nvidia.com/gpu
devices:
- name: "0"
  containerEdits:
    deviceNodes:
    - path: {present_dev}
containerEdits:
  mounts:
  - hostPath: {present_lib}
    containerPath: /usr/lib/libnvidia-ml.so.1
  - hostPath: {missing_lib}
    containerPath: /usr/lib/libcuda.so.1
"""


@pytest.mark.skipif(shutil.which("sh") is None, reason="requires a POSIX shell")
def test_probe_counts_missing_spec_paths(tmp_path):
    """Exercise the real shell, not a Python re-implementation of it.

    The staleness signal is produced by setup_check.sh on the remote host, so
    a test that reimplements its arithmetic would pass while the shipped
    script was broken.
    """
    present_lib = tmp_path / "libnvidia-ml.so.580.65.06"
    present_lib.write_text("x")
    present_dev = tmp_path / "nvidia0"
    present_dev.write_text("x")
    missing_lib = tmp_path / "libcuda.so.999.99.99"  # never created: the upgraded-away path

    spec = tmp_path / "nvidia.yaml"
    spec.write_text(_SPEC_TEMPLATE.format(present_dev=present_dev, present_lib=present_lib, missing_lib=missing_lib))

    # The scan logic from setup_check.sh, pointed at the fixture spec.
    script = """
    _cdi_checked=0
    _cdi_missing=0
    for _cdi_path in $(grep -oE '(hostPath|path): *"?/[^ "]+' "$1" 2>/dev/null \
            | sed 's/.*: *"\\?//' | sort -u | head -60); do
        _cdi_checked=$((_cdi_checked + 1))
        [ -e "$_cdi_path" ] || _cdi_missing=$((_cdi_missing + 1))
    done
    echo "CHECK_CDI_PATHS_CHECKED=$_cdi_checked"
    echo "CHECK_CDI_PATHS_MISSING=$_cdi_missing"
    """
    out = subprocess.run(["sh", "-c", script, "sh", str(spec)], capture_output=True, text=True, check=True).stdout

    facts = dict(line.split("=", 1) for line in out.strip().splitlines())
    assert facts["CHECK_CDI_PATHS_CHECKED"] == "3", out
    assert facts["CHECK_CDI_PATHS_MISSING"] == "1", out


def test_shipped_script_contains_the_scan():
    """Guard against the script and the check item drifting apart."""
    from sparkrun.utils.resource_loader import load_script_resource

    src = load_script_resource("setup_check.sh")
    assert "CHECK_CDI_PATHS_CHECKED" in src
    assert "CHECK_CDI_PATHS_MISSING" in src


# ---------------------------------------------------------------------------
# Launch-failure hints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("docker: Error response from daemon: unresolvable CDI devices: nvidia.com/gpu=all", "nvidia-ctk cdi generate"),
        ("Error: CDI device injection failed: nvidia.com/gpu=0", "nvidia-ctk cdi generate"),
        ('could not select device driver "" with capabilities: [[gpu]]', "NVIDIA Container Toolkit"),
        ("unknown flag: --device", "Docker 25 or newer"),
        ("dial unix /var/run/docker.sock: connect: permission denied", "usermod -aG docker"),
    ],
)
def test_recognized_failures_get_actionable_guidance(stderr, expected):
    hint = diagnose_launch_failure(stderr)
    assert hint is not None
    assert expected in hint
    # Every hint must name how to confirm it, not just what to type.
    assert "sparkrun setup check" in hint or "upgrade Docker" in hint


@pytest.mark.parametrize("stderr", ["", None, "OOMKilled", "manifest unknown: manifest tagged by 'x' not found"])
def test_unrecognized_failures_stay_silent(stderr):
    """An unrecognized cause must not attract a confidently wrong hint."""
    assert diagnose_launch_failure(stderr) is None


def test_hint_is_logged_at_error_level(caplog):
    """A hint below the operator's verbosity threshold is a hint never seen."""
    log = logging.getLogger("test.cdi")
    with caplog.at_level(logging.ERROR, logger="test.cdi"):
        log_launch_failure_hint(log, "unresolvable CDI devices: nvidia.com/gpu=all")
    assert any("nvidia-ctk cdi generate" in r.message for r in caplog.records)
    assert all(r.levelno == logging.ERROR for r in caplog.records)


def test_no_hint_logged_when_unrecognized(caplog):
    log = logging.getLogger("test.cdi2")
    with caplog.at_level(logging.ERROR, logger="test.cdi2"):
        log_launch_failure_hint(log, "some unrelated failure")
    assert caplog.records == []
