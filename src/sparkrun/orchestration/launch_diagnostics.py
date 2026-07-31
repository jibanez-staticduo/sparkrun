"""Turn a recognized container-launch failure into an actionable next step.

A failed ``docker run`` prints the daemon's own error, which names the
mechanism that refused but not the thing the operator has to fix.  The CDI
case is the clearest example: a host whose ``/etc/cdi/nvidia.yaml`` is absent
or stale fails with ``unresolvable CDI devices: nvidia.com/gpu=all``, which
says nothing about a driver upgrade having moved the paths the spec pins, nor
which of the cluster's hosts is affected.

This module maps those signatures onto the command that resolves them.  It is
deliberately advisory: a hint is appended to the error, never substituted for
it, so the daemon's own wording stays available for a cause we haven't seen.
"""

from __future__ import annotations

import re

#: (compiled signature, guidance). Ordered — the first match wins, so put the
#: specific causes ahead of the generic "no GPU driver" catch-all.
_SIGNATURES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"unresolvable CDI device|CDI device injection failed|cdi.*device.*not found", re.I),
        "The host's NVIDIA CDI spec is missing or stale — sparkrun requests GPUs as "
        "'nvidia.com/gpu', which Docker resolves through /etc/cdi/nvidia.yaml.\n"
        "  Diagnose:  sparkrun setup check\n"
        "  Fix:       sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   (on the affected host)\n"
        "  A driver upgrade invalidates an existing spec, so regenerate after one.",
    ),
    (
        # Pre-CDI failure shape: the daemon has no GPU driver wired up at all.
        re.compile(r"could not select device driver.*capabilities.*gpu", re.I),
        "Docker could not attach a GPU. The NVIDIA Container Toolkit is likely not installed "
        "or not registered with the daemon.\n"
        "  Diagnose:  sparkrun setup check\n"
        "  Fix:       install the NVIDIA Container Toolkit, then\n"
        "             sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml",
    ),
    (
        re.compile(r"unknown flag: --device|flag provided but not defined: -device", re.I),
        "This Docker daemon does not understand CDI device requests. sparkrun emits "
        "'--device nvidia.com/gpu=...', which needs Docker 25 or newer.\n"
        "  Fix:       upgrade Docker on the affected host.",
    ),
    (
        re.compile(r"permission denied.*docker\.sock|dial unix /var/run/docker\.sock", re.I),
        "The SSH user cannot reach the Docker socket.\n"
        "  Diagnose:  sparkrun setup check\n"
        "  Fix:       sudo usermod -aG docker $USER   (then reconnect the SSH session)",
    ),
)


def diagnose_launch_failure(stderr: str | None) -> str | None:
    """Return guidance for a recognized launch failure, or ``None``.

    ``None`` means "not a failure mode we recognize" — the caller should show
    the raw error unchanged rather than guessing.
    """
    if not stderr:
        return None
    for pattern, guidance in _SIGNATURES:
        if pattern.search(stderr):
            return guidance
    return None


def log_launch_failure_hint(logger, stderr: str | None) -> None:
    """Log the guidance for *stderr* at ERROR level, when one is recognized.

    Emitted at the same level as the failure it explains: a hint logged below
    the operator's verbosity threshold is a hint they never see, and the
    surrounding failure output is already ERROR.
    """
    hint = diagnose_launch_failure(stderr)
    if not hint:
        return
    for line in hint.splitlines():
        logger.error("  %s", line)


__all__ = ["diagnose_launch_failure", "log_launch_failure_hint"]
