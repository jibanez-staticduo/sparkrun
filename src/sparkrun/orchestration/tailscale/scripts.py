"""Bash script builders for the Tailscale join / status / teardown flow.

The join scripts are stored as ``.sh`` files (primary + sudo fallback) and
filled via ``str.format``; the read-only status and logout scripts are short and
inlined here (mirrors the docker-group inline scripts in ``_setup/_phases.py``).
"""

from __future__ import annotations

from sparkrun.scripts import read_script

# Read-only per-host state probe. No sudo, no placeholders — used as a literal.
STATUS_SCRIPT = """\
#!/bin/bash
set -uo pipefail
if ! command -v tailscale >/dev/null 2>&1; then
    echo "TS_STATE=not_installed"
    exit 0
fi
BACKEND=$(tailscale status --json 2>/dev/null | grep -o '"BackendState": *"[^"]*"' | head -n1 | cut -d'"' -f4)
TS_IP=$(tailscale ip -4 2>/dev/null | head -n1)
HOST=$(tailscale status --json 2>/dev/null | grep -o '"HostName": *"[^"]*"' | head -n1 | cut -d'"' -f4)
echo "TS_STATE=$BACKEND"
echo "TS_IP=$TS_IP"
echo "TS_HOSTNAME=$HOST"
"""

# Logout — primary attempts non-interactive sudo, fallback runs bare under sudo.
LOGOUT_SCRIPT = """\
#!/bin/bash
set -uo pipefail
if ! command -v tailscale >/dev/null 2>&1; then
    echo "TS_DOWN=not_installed"
    exit 0
fi
if sudo -n tailscale logout >/dev/null 2>&1; then
    echo "TS_DOWN=logged_out"
else
    echo "TS_DOWN=needs_sudo"
    exit 1
fi
"""

LOGOUT_FALLBACK_SCRIPT = """\
#!/bin/bash
set -uo pipefail
if ! command -v tailscale >/dev/null 2>&1; then
    echo "TS_DOWN=not_installed"
    exit 0
fi
tailscale logout >/dev/null 2>&1
echo "TS_DOWN=logged_out"
"""


# TCP serve — forward the tailnet node:PORT to the local inference port. This is
# the inbound mechanism that works in userspace-networking mode (raw-port inbound
# does not). PORT is int-validated by the caller, so single-quoting is safe.
_SERVE_PRIMARY = """\
#!/bin/bash
set -uo pipefail
PORT='{port}'
if ! command -v tailscale >/dev/null 2>&1; then
    echo "TS_ERROR=not_installed"
    exit 1
fi
if ! sudo -n tailscale serve --bg --tcp="$PORT" "tcp://127.0.0.1:$PORT" 2>&1; then
    echo "TS_ERROR=serve_failed"
    exit 1
fi
TS_IP=$(tailscale ip -4 2>/dev/null | head -n1)
echo "TS_IP=$TS_IP"
echo "TS_SERVE_OK=1"
"""

_SERVE_FALLBACK = """\
#!/bin/bash
set -uo pipefail
PORT='{port}'
if ! command -v tailscale >/dev/null 2>&1; then
    echo "TS_ERROR=not_installed"
    exit 1
fi
if ! tailscale serve --bg --tcp="$PORT" "tcp://127.0.0.1:$PORT" 2>&1; then
    echo "TS_ERROR=serve_failed"
    exit 1
fi
TS_IP=$(tailscale ip -4 2>/dev/null | head -n1)
echo "TS_IP=$TS_IP"
echo "TS_SERVE_OK=1"
"""


def build_serve_scripts(port: int) -> tuple[str, str]:
    """Return ``(primary, fallback)`` scripts that TCP-serve *port* on the tailnet.

    *port* is coerced to an int before interpolation, so the value that reaches
    the shell is always a bare integer (no injection surface).
    """
    values = {"port": str(int(port))}
    return _SERVE_PRIMARY.format(**values), _SERVE_FALLBACK.format(**values)


def _hostname_arg(hostname: str | None) -> str:
    """Render the ``--hostname=<name>`` fragment, or empty when unset."""
    if not hostname:
        return ""
    # Tailscale hostnames are DNS labels; keep only safe chars.
    safe = "".join(c for c in hostname if c.isalnum() or c in "-.").strip("-.")
    return "--hostname=%s" % safe if safe else ""


def _extra_args(enable_ssh: bool) -> str:
    """Render extra ``tailscale up`` flags."""
    return "--ssh" if enable_ssh else ""


def build_join_scripts(
    authkey: str,
    tags: str,
    *,
    hostname: str | None = None,
    enable_ssh: bool = False,
) -> tuple[str, str]:
    """Return ``(primary, fallback)`` join scripts filled with the given values.

    *authkey* and *tags* are interpolated into **single-quoted** shell
    assignments (``VAR='…'``), which disable all bash interpolation ($, backtick,
    backslash). The only character that can break out of a single-quoted literal
    is ``'`` itself, so we reject it (plus newlines) — a malformed provider value
    can never reach the shell.
    """
    for label, value in (("authkey", authkey), ("tags", tags)):
        if "'" in value or "\n" in value or "\r" in value:
            raise ValueError("Refusing to build join script: %s contains shell-unsafe characters." % label)
    values = {
        "authkey": authkey,
        "tags": tags,
        "hostname_arg": _hostname_arg(hostname),
        "extra_args": _extra_args(enable_ssh),
    }
    primary = read_script("tailscale_join.sh").format(**values)
    fallback = read_script("tailscale_join_fallback.sh").format(**values)
    return primary, fallback


def parse_join_result(stdout: str) -> dict[str, str]:
    """Parse ``KEY=value`` markers emitted by the join / status scripts."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "=" in line and line.split("=", 1)[0].startswith("TS_"):
            key, val = line.split("=", 1)
            out[key] = val
    return out
