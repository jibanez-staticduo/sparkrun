"""Preflight: does an image's ENTRYPOINT pass sparkrun's command through?

sparkrun composes every workload as ``docker run <opts> <image> bash -c
<b64 command>`` — the command is always appended as **CMD arguments**.  Docker
hands those arguments to the image's ENTRYPOINT, so what happens next depends
entirely on which of two opposite idioms the image uses:

- **Passthrough** — the ENTRYPOINT is a wrapper that does setup and ends in
  ``exec "$@"``.  ``/opt/nvidia/nvidia_entrypoint.sh``, which nearly every
  NGC-derived image inherits, is this.  sparkrun's command runs normally and
  clearing the ENTRYPOINT would *skip* the wrapper's setup.
- **Consuming** — the ENTRYPOINT names a binary plus subcommand, e.g.
  ``ENTRYPOINT ["vllm","serve"]``.  sparkrun's ``bash -c <cmd>`` is parsed as
  *that program's flags*, so the workload never starts.

``docker image inspect`` cannot tell them apart — both are simply a non-empty
ENTRYPOINT — so this module settles it empirically by running the real argv
shape and checking for a sentinel.  The failure it prevents is otherwise close
to undiagnosable: the consumed launcher surfaces as an error about whatever
flag the consuming program happened to mis-parse, and the container's own serve
log is destroyed with the container on exit.

Called through :meth:`~sparkrun.orchestration.executors._base.Executor.verify_command_passthrough`
so the question stays substrate-dispatched, and best-effort throughout — only a
*confirmed* consuming entrypoint is reported.  Kill switch:
``SPARKRUN_NO_IMAGE_PROBE=1``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from sparkrun.scripts import read_script
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)

#: Env kill switch, mirroring ``SPARKRUN_NO_SESSION_GUARD``.
NO_PROBE_ENV = "SPARKRUN_NO_IMAGE_PROBE"

#: Command run inside the probe container.  The sentinel is *computed* rather
#: than literal so that an entrypoint which echoes its rejected argv back
#: cannot accidentally satisfy the match — only a shell that really executed
#: the command can print the evaluated form.
PROBE_COMMAND = 'printf "sparkrun_probe_%s\\n" "$((21*2))"'

#: What :data:`PROBE_COMMAND` prints when it actually runs.
PROBE_TOKEN = "sparkrun_probe_42"

#: Verdicts.  Only :data:`VERDICT_CONSUMES` is actionable; everything else
#: (including "could not tell") lets the launch proceed.
VERDICT_ABSENT = "absent"
"""Image declares no ENTRYPOINT — the appended CMD is the argv."""
VERDICT_PASSTHROUGH = "pass"
"""ENTRYPOINT is present but ran sparkrun's command (``exec "$@"`` wrapper)."""
VERDICT_CONSUMES = "fail"
"""ENTRYPOINT swallowed sparkrun's command — this launch cannot work."""
VERDICT_UNKNOWN = "unknown"
"""Probe did not complete (unreachable host, timeout, no docker, disabled)."""


@dataclass(frozen=True)
class EntrypointProbe:
    """Outcome of one image probe on one host."""

    image: str
    host: str
    verdict: str
    entrypoint: str = ""

    @property
    def consumes_command(self) -> bool:
        """True only for a *confirmed* consuming ENTRYPOINT."""
        return self.verdict == VERDICT_CONSUMES


def probe_disabled() -> bool:
    """True when the ``SPARKRUN_NO_IMAGE_PROBE`` kill switch is set."""
    return os.environ.get(NO_PROBE_ENV, "").strip().lower() in ("1", "true", "yes")


def build_probe_script(image: str, accel_opts: list[str] | None = None) -> str:
    """Render the remote probe script for *image*.

    *accel_opts* are the executor's own accelerator flags (``--gpus all`` /
    CDI ``--device``), passed so the probe container starts under the same
    device conditions as the real launch.  Without them, an entrypoint that
    hard-fails on a missing GPU *before* reaching ``exec "$@"`` would be
    misreported as consuming.
    """
    return read_script("image_entrypoint_probe.sh").format(
        image=quote(image),
        accel_opts=" ".join(accel_opts or []),
        probe_command=PROBE_COMMAND,
        probe_token=PROBE_TOKEN,
    )


def parse_probe_output(output: str) -> tuple[str, str]:
    """Parse the script's ``key=value`` output into ``(verdict, entrypoint)``."""
    from sparkrun.utils import parse_kv_output

    parsed = parse_kv_output(output or "")
    verdict = parsed.get("SPARKRUN_PROBE", VERDICT_UNKNOWN)
    if verdict not in (VERDICT_ABSENT, VERDICT_PASSTHROUGH, VERDICT_CONSUMES):
        verdict = VERDICT_UNKNOWN
    entrypoint = parsed.get("SPARKRUN_ENTRYPOINT", "")
    if entrypoint in ("null", "[]"):
        entrypoint = ""
    return verdict, entrypoint


def probe_image_entrypoint(
    image: str,
    host: str,
    *,
    ssh_kwargs: dict | None = None,
    accel_opts: list[str] | None = None,
    timeout: int = 300,
) -> EntrypointProbe:
    """Run the ENTRYPOINT probe for *image* on *host*.

    Never raises: any failure to reach a verdict degrades to
    :data:`VERDICT_UNKNOWN`, which callers treat as "proceed".  Refusing a
    launch because we could not tell is strictly worse than the status quo.

    The generous *timeout* is deliberate — the passthrough case answers in
    well under a second (the container starts and immediately exits), but a
    consuming entrypoint pays for however much the program loads before it
    parses argv, which for a vLLM image means importing torch first.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    if probe_disabled():
        logger.debug("Image entrypoint probe disabled via %s", NO_PROBE_ENV)
        return EntrypointProbe(image=image, host=host, verdict=VERDICT_UNKNOWN)

    try:
        result = run_remote_script(
            host,
            build_probe_script(image, accel_opts),
            timeout=timeout,
            **(ssh_kwargs or {}),
        )
    except Exception:
        logger.debug("Image entrypoint probe errored on %s; skipping", host, exc_info=True)
        return EntrypointProbe(image=image, host=host, verdict=VERDICT_UNKNOWN)

    if not result.success:
        logger.debug("Image entrypoint probe failed on %s (rc=%s); skipping", host, result.returncode)
        return EntrypointProbe(image=image, host=host, verdict=VERDICT_UNKNOWN)

    verdict, entrypoint = parse_probe_output(result.stdout)
    logger.debug("Image entrypoint probe on %s: image=%s verdict=%s entrypoint=%s", host, image, verdict, entrypoint or "(none)")
    return EntrypointProbe(image=image, host=host, verdict=verdict, entrypoint=entrypoint)
