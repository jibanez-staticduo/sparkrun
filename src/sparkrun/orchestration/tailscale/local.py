"""Control-machine-local Tailscale probes (for ``expose --proxy``).

Reads the local ``tailscaled`` state via the ``tailscale`` CLI. Separate from
:mod:`sparkrun.orchestration.tailscale.api` (which talks to the REST control
plane) because this runs a local subprocess rather than an HTTP call.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def local_tailscale_ipv4() -> str | None:
    """Return this machine's tailnet IPv4, or ``None`` when unavailable.

    ``None`` when the ``tailscale`` CLI is missing, the daemon is not up, or the
    command fails — callers treat that as "not on a tailnet".
    """
    if shutil.which("tailscale") is None:
        return None
    try:
        proc = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover - defensive
        logger.debug("local tailscale ip failed: %s", e)
        return None
    if proc.returncode != 0:
        return None
    ip = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
    return ip or None


def local_tailscale_dnsname() -> str | None:
    """Return this machine's MagicDNS name (``host.tailnet.ts.net``) or ``None``."""
    if shutil.which("tailscale") is None:
        return None
    try:
        proc = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover - defensive
        logger.debug("local tailscale status failed: %s", e)
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except ValueError:  # pragma: no cover - defensive
        return None
    dns = (data.get("Self") or {}).get("DNSName") or ""
    return dns.rstrip(".") or None
