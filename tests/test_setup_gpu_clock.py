"""CLI tests for `sparkrun setup throttle-gpu-clock`.

SSH/sudo are mocked — the assertions are about which nvidia-smi invocation the
generated script would run, whether it touches the boot-time systemd unit, the
read-only status path a bare invocation takes, and the confirmation gates in
front of any change.
"""

from __future__ import annotations

import re
import subprocess

from unittest import mock

from click.testing import CliRunner

from sparkrun.cli._setup._gpu_clock import UNIT_NAME, setup_throttle_gpu_clock
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.scripts import read_script

# One GPU per host, as gpu_clock_status.sh emits it (UNIT line + CSV rows).
_STATUS_OUT = "UNIT: none\n0, 208, 3003, 2418, Enabled\n"


def _ok(host: str, stdout: str = "LOCKED: 0,2000 MHz\nCURRENT: 1200 MHz, Enabled \n") -> RemoteResult:
    return RemoteResult(host=host, returncode=0, stdout=stdout, stderr="")


def _norm(text: str) -> str:
    """Collapse the status table's column padding so rows can be matched."""
    return " ".join(text.split())


def _case_action(script: str) -> str:
    """Return the unit action the rendered script dispatches to.

    One script carries all three branches (install/remove/keep), so what
    matters is the literal the ``case`` selects on — not whether some other
    branch's text appears somewhere in the file.
    """
    match = re.search(r'case "([a-z]+)" in', script)
    assert match, "no rendered case selector in script"
    return match.group(1)


def _run(args, hosts=("h1", "h2"), sudo_result=None, status_results=None, input=None):
    """Invoke the command with host resolution, sudo, and SSH mocked.

    Returns ``(cli_result, captured)``; *captured* records what each mocked
    layer was handed, and stays empty for the layer that was never reached.
    """
    captured = {}

    def fake_sudo(host_list, script, fallback, ssh_kwargs, **kwargs):
        captured["script"] = script
        captured["fallback"] = fallback
        captured["sudo_hosts"] = list(host_list)
        if sudo_result is not None:
            return sudo_result
        return {h: _ok(h) for h in host_list}, []

    def fake_parallel(host_list, script, **kwargs):
        captured["status_script"] = script
        captured["status_hosts"] = list(host_list)
        if status_results is not None:
            return status_results
        return [_ok(h, _STATUS_OUT) for h in host_list]

    with (
        mock.patch(
            "sparkrun.cli._setup._gpu_clock._resolve_setup_context",
            return_value=(list(hosts), "me", {}),
        ),
        mock.patch("sparkrun.orchestration.sudo.run_with_sudo_fallback", fake_sudo),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", fake_parallel),
    ):
        return CliRunner().invoke(setup_throttle_gpu_clock, args, input=input), captured


# ---------------------------------------------------------------------------
# Bare invocation — report only
# ---------------------------------------------------------------------------


def test_bare_invocation_reports_and_changes_nothing():
    r, cap = _run(["--hosts", "h1,h2"])
    assert r.exit_code == 0, r.output
    # No sudo path was entered at all.
    assert "script" not in cap
    assert "--lock-gpu-clocks" not in cap["status_script"]
    assert "--reset-gpu-clocks" not in cap["status_script"]
    # Reporting is read-only: nothing in the probe escalates or writes a unit.
    assert "sudo -n" not in cap["status_script"]
    assert "sudo nvidia-smi" not in cap["status_script"]
    assert "systemctl enable" not in cap["status_script"]
    assert cap["status_hosts"] == ["h1", "h2"]


def test_status_renders_a_row_per_gpu():
    r, _ = _run(["--hosts", "h1,h2"])
    assert r.exit_code == 0, r.output
    assert "HOST" in r.output and "MAX SM" in r.output and "AT BOOT" in r.output
    assert "h1 0 208 MHz 3003 MHz 2418 MHz Enabled -" in _norm(r.output)
    assert "h2 0 208 MHz 3003 MHz 2418 MHz Enabled -" in _norm(r.output)
    # The lock itself is not readable back from the driver — say so.
    assert "does not report the --lock-gpu-clocks setting" in r.output


def test_status_footnotes_persistence_column():
    r, _ = _run(["--hosts", "h1"], hosts=("h1",))
    assert r.exit_code == 0, r.output
    assert "PERSISTENCE*" in r.output
    assert "* driver stays loaded while idle — holds the lock, but not across reboots" in r.output


def test_status_reports_installed_boot_unit():
    results = [_ok("h1", "UNIT: 0,2150 enabled\n0, 208, 3003, 2418, Enabled\n")]
    r, _ = _run([], hosts=("h1",), status_results=results)
    assert r.exit_code == 0, r.output
    assert "h1 0 208 MHz 3003 MHz 2418 MHz Enabled 0,2150" in _norm(r.output)


def test_status_flags_an_installed_but_disabled_unit():
    """A disabled unit would not reapply at boot — that distinction must show."""
    results = [_ok("h1", "UNIT: 0,2150 disabled\n0, 208, 3003, 2418, Enabled\n")]
    r, _ = _run([], hosts=("h1",), status_results=results)
    assert r.exit_code == 0, r.output
    assert "0,2150 (disabled)" in r.output


def test_status_reports_unavailable_fields_as_dash():
    results = [_ok("h1", "UNIT: none\n0, 208, 3003, [N/A], Enabled\n")]
    r, _ = _run([], hosts=("h1",), status_results=results)
    assert r.exit_code == 0, r.output
    assert "h1 0 208 MHz 3003 MHz - Enabled -" in _norm(r.output)


def test_status_never_prompts():
    r, _ = _run(["--hosts", "h1"], input="n\n")
    assert r.exit_code == 0, r.output
    assert "Proceed?" not in r.output


def test_status_failure_exits_nonzero():
    results = [RemoteResult(host="h1", returncode=1, stdout="", stderr="ERROR: nvidia-smi not found")]
    r, _ = _run([], hosts=("h1",), status_results=results)
    assert r.exit_code == 1
    assert "[FAIL] h1" in r.output
    assert "nvidia-smi not found" in r.output


# ---------------------------------------------------------------------------
# Locking / clearing
# ---------------------------------------------------------------------------


def test_max_clock_locks_after_confirmation():
    r, cap = _run(["2150", "--hosts", "h1,h2"], input="y\n")
    assert r.exit_code == 0, r.output
    assert "Locking GPU clocks to 0,2150 MHz on 2 host(s)" in r.output
    assert "sudo -n nvidia-smi --lock-gpu-clocks 0,2150" in cap["script"]
    assert "nvidia-smi --lock-gpu-clocks 0,2150" in cap["fallback"]
    # The fallback already runs as root — it must not re-invoke sudo.
    assert "sudo -n" not in cap["fallback"]
    # Without --persistent, nothing boot-related is touched.
    assert _case_action(cap["script"]) == "keep"


def test_declining_confirmation_changes_nothing():
    r, cap = _run(["2150", "--hosts", "h1"], input="n\n")
    assert r.exit_code == 0, r.output
    assert "Aborted." in r.output
    assert "script" not in cap


def test_yes_skips_confirmation():
    r, cap = _run(["2150", "--hosts", "h1", "--yes"], hosts=("h1",))
    assert r.exit_code == 0, r.output
    assert "Proceed?" not in r.output
    assert cap["sudo_hosts"] == ["h1"]


def test_dry_run_needs_no_confirmation():
    r, cap = _run(["2150", "--hosts", "h1", "--dry-run"], hosts=("h1",))
    assert r.exit_code == 0, r.output
    assert "Proceed?" not in r.output
    assert cap["sudo_hosts"] == ["h1"]


def test_clear_flag_resets_and_removes_the_unit():
    r, cap = _run(["--clear", "--hosts", "h1", "--yes"])
    assert r.exit_code == 0, r.output
    assert "Clearing GPU clock lock (and any boot-time unit)" in r.output
    assert "nvidia-smi --reset-gpu-clocks" in cap["script"]
    assert "--lock-gpu-clocks" not in cap["script"]
    assert _case_action(cap["script"]) == "remove"


def test_zero_max_clock_resets():
    r, cap = _run(["0", "--hosts", "h1", "--yes"])
    assert r.exit_code == 0, r.output
    assert "Clearing GPU clock lock" in r.output
    assert "nvidia-smi --reset-gpu-clocks" in cap["script"]


def test_clear_conflicts_with_explicit_max_clock():
    r, cap = _run(["2150", "--clear", "--hosts", "h1", "--yes"])
    assert r.exit_code != 0
    assert "--clear conflicts with MAX_CLOCK 2150" in r.output
    assert cap == {}


def test_clear_accepts_redundant_zero():
    r, cap = _run(["0", "--clear", "--hosts", "h1", "--yes"])
    assert r.exit_code == 0, r.output
    assert "nvidia-smi --reset-gpu-clocks" in cap["script"]


def test_out_of_range_max_clock_rejected():
    for bad in ("2", "99999"):
        r, cap = _run([bad, "--hosts", "h1", "--yes"])
        assert r.exit_code != 0, bad
        assert "MAX_CLOCK must be 0 (clear) or between 100 and 10000 MHz" in r.output
        assert cap == {}


def test_failed_host_exits_nonzero():
    failure = ({"h1": RemoteResult(host="h1", returncode=1, stdout="", stderr="ERROR: nvidia-smi not found")}, [])
    r, _ = _run(["2150", "--hosts", "h1", "--yes"], hosts=("h1",), sudo_result=failure)
    assert r.exit_code == 1
    assert "[FAIL] h1" in r.output
    assert "1 failed" in r.output


def test_existing_unit_is_reported_and_warned_about():
    """A lock that disagrees with what boots must not be silent."""
    stdout = "LOCKED: 0,2150 MHz\nUNIT: present and unchanged (lock-gpu-clocks 0,2000)\n"
    result = ({"h1": _ok("h1", stdout)}, [])
    r, _ = _run(["2150", "--hosts", "h1", "--yes"], hosts=("h1",), sudo_result=result)
    assert r.exit_code == 0, r.output
    assert "UNIT: present and unchanged" in r.output
    assert "Re-run with --persistent to update it" in r.output


# ---------------------------------------------------------------------------
# --persistent (boot-time unit)
# ---------------------------------------------------------------------------


def test_persistent_asks_a_second_time_and_installs():
    r, cap = _run(["2150", "--persistent", "--hosts", "h1"], hosts=("h1",), input="y\ny\n")
    assert r.exit_code == 0, r.output
    assert "Install the boot-time unit?" in r.output
    assert _case_action(cap["script"]) == "install"
    assert "UNIT_NAME=%s" % UNIT_NAME in cap["script"]
    assert "UNIT_FILE=/etc/systemd/system/$UNIT_NAME" in cap["script"]
    assert "ExecStart=$NVIDIA_SMI --lock-gpu-clocks 0,2150" in cap["script"]
    assert "systemctl enable" in cap["script"]


def test_persistent_second_prompt_lists_what_it_writes():
    r, _ = _run(["2150", "--persistent", "--hosts", "h1"], hosts=("h1",), input="y\nn\n")
    assert r.exit_code == 0, r.output
    assert UNIT_NAME in r.output
    assert "ExecStart=<nvidia-smi> --lock-gpu-clocks 0,2150" in r.output
    assert "nvidia-persistenced.service" in r.output


def test_declining_the_unit_still_applies_the_runtime_lock():
    """The first prompt already approved the lock; only the unit is declined."""
    r, cap = _run(["2150", "--persistent", "--hosts", "h1"], hosts=("h1",), input="y\nn\n")
    assert r.exit_code == 0, r.output
    assert "Skipping the boot-time unit; applying the runtime lock only." in r.output
    assert "nvidia-smi --lock-gpu-clocks 0,2150" in cap["script"]
    assert _case_action(cap["script"]) == "keep"


def test_declining_the_first_prompt_skips_the_unit_prompt():
    r, cap = _run(["2150", "--persistent", "--hosts", "h1"], hosts=("h1",), input="n\n")
    assert r.exit_code == 0, r.output
    assert "Install the boot-time unit?" not in r.output
    assert "script" not in cap


def test_yes_skips_both_prompts():
    r, cap = _run(["2150", "--hosts", "h1", "--persistent", "--yes"], hosts=("h1",))
    assert r.exit_code == 0, r.output
    assert "Install the boot-time unit?" not in r.output
    assert _case_action(cap["script"]) == "install"


def test_persistent_needs_a_max_clock():
    r, cap = _run(["--persistent", "--hosts", "h1", "--yes"])
    assert r.exit_code != 0
    assert "--persistent needs a MAX_CLOCK" in r.output
    assert cap == {}


def test_persistent_conflicts_with_clear():
    r, cap = _run(["--clear", "--persistent", "--hosts", "h1", "--yes"])
    assert r.exit_code != 0
    assert "--clear already removes the boot-time unit" in r.output
    assert cap == {}


# ---------------------------------------------------------------------------
# Rendered scripts must be valid bash in every mode
# ---------------------------------------------------------------------------


def test_rendered_scripts_are_syntactically_valid():
    cases = [
        ("--lock-gpu-clocks 0,2150", "LOCKED: 0,2150 MHz", "install"),
        ("--lock-gpu-clocks 0,2150", "LOCKED: 0,2150 MHz", "keep"),
        ("--reset-gpu-clocks", "CLEARED: gpu clock lock removed", "remove"),
    ]
    for name in ("gpu_clock_throttle.sh", "gpu_clock_throttle_fallback.sh"):
        for clock_args, result_label, unit_action in cases:
            rendered = read_script(name).format(
                clock_args=clock_args,
                result_label=result_label,
                unit_action=unit_action,
            )
            proc = subprocess.run(["bash", "-n"], input=rendered, text=True, capture_output=True)
            assert proc.returncode == 0, "%s/%s: %s" % (name, unit_action, proc.stderr)


def test_status_script_is_syntactically_valid():
    proc = subprocess.run(["bash", "-n"], input=read_script("gpu_clock_status.sh"), text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
