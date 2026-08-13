"""Shared internals for sparkrun tuning modules.

This is a private module — external code should import from
:mod:`sparkrun.tuning.sglang` or :mod:`sparkrun.tuning.vllm`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.core.config import DEFAULT_CACHE_DIR
from sparkrun.core.progress import PROGRESS
from sparkrun.utils import format_duration as _format_duration  # noqa: F401 — re-exported for local callers
from sparkrun.utils.shell import quote, safe_remote_path

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig

logger = logging.getLogger(__name__)

DEFAULT_TP_SIZES = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# Parameterized host-side helpers
# ---------------------------------------------------------------------------


def _get_tuning_dir(cache_subdir: str) -> Path:
    """Return the host-side directory for tuning configs under *cache_subdir*."""
    return DEFAULT_CACHE_DIR / cache_subdir


def _get_tuning_volumes(
    tuning_dir_fn: callable,
    container_path: str,
) -> dict[str, str] | None:
    """Return volume mapping for tuning configs if they exist.

    Args:
        tuning_dir_fn: Callable returning the host-side tuning :class:`Path`.
        container_path: Mount target inside the container.

    Returns:
        Dict mapping host dir to container dir, or ``None``.
    """
    tuning_dir = tuning_dir_fn()
    if tuning_dir.is_dir() and any(tuning_dir.rglob("*.json")):
        return {str(tuning_dir): container_path}
    return None


def _get_tuning_env(
    volumes_fn: callable,
    env_var: str,
    container_path: str,
) -> dict[str, str] | None:
    """Return env vars for tuning configs if they exist.

    Args:
        volumes_fn: Callable returning the volume dict (or ``None``).
        env_var: Environment variable name to set.
        container_path: Value to assign to the env var.

    Returns:
        Dict with *env_var* set, or ``None``.
    """
    if volumes_fn() is not None:
        return {env_var: container_path}
    return None


# ---------------------------------------------------------------------------
# Long-run plumbing shared by both tuners
# ---------------------------------------------------------------------------


def resolve_tuning_timeout(timeout_hours: float | None, config: SparkrunConfig | None) -> int | None:
    """Resolve the per-TP wall-clock cap in seconds: CLI → config → default.

    Returns ``None`` for "no cap", which a non-positive value selects.  That
    is a supported choice rather than a footgun: tuning sessions now run with
    SSH keepalives, so a dead link is reported in minutes instead of being
    indistinguishable from a slow sweep — which is what the cap was really
    guarding against.
    """
    from sparkrun.core.config import DEFAULT_TUNING_TIMEOUT_HOURS

    if timeout_hours is None:
        timeout_hours = config.tuning_timeout_hours if config is not None else DEFAULT_TUNING_TIMEOUT_HOURS
    try:
        hours = float(timeout_hours)
    except (TypeError, ValueError):
        logger.warning("Invalid tuning timeout %r; using %s hours", timeout_hours, DEFAULT_TUNING_TIMEOUT_HOURS)
        hours = DEFAULT_TUNING_TIMEOUT_HOURS
    if hours <= 0:
        return None
    return int(hours * 3600)


def describe_timeout(timeout_sec: int | None) -> str:
    """Render a resolved timeout for the run banner."""
    return "none" if timeout_sec is None else _format_duration(timeout_sec)


def resolve_tuning_parallel(parallel: int, n_jobs: int, force: bool = False) -> int:
    """Clamp ``--parallel`` to 1 unless *force*, explaining why.

    Tuning **measures kernel latency** to pick the fastest tile config, and
    every job in a run targets the same single host — so concurrent jobs
    contend for one GPU and the timings they compare are contaminated.  The
    result is not a slower correct answer, it is a wrong config written to the
    cache and auto-mounted by every later ``sparkrun run``.

    Downgrading (rather than erroring) keeps existing ``-j`` scripts working
    and merely makes them correct; ``force=True`` is the escape hatch for
    someone deliberately measuring the contention itself.
    """
    if parallel <= 1 or n_jobs <= 1:
        return max(1, parallel)
    if force:
        logger.warning(
            "Running %d tuning jobs concurrently on one GPU (--force-parallel): the measured "
            "kernel latencies will be contaminated by contention and the selected configs may "
            "be wrong.",
            parallel,
        )
        return parallel
    logger.warning(
        "Ignoring --parallel %d: tuning measures kernel latency on a single GPU, so concurrent "
        "jobs contend and produce wrong configs. Running sequentially. Pass --force-parallel to "
        "override.",
        parallel,
    )
    return 1


def _slugify_model(model: str) -> str:
    """Reduce a model id to a filename-safe token."""
    slug = "".join(ch if (ch.isalnum() or ch in "-._") else "-" for ch in model)
    return slug.strip("-")[:80] or "model"


def remote_tune_log_dir(remote_output_dir: str) -> str:
    """Return the host-side directory for tuning logs.

    A sibling of the flat config cache (``…/tuning/logs``), deliberately *not*
    inside it: that directory is bind-mounted into inference containers and
    rsynced back, and multi-hour tuning logs belong in neither.
    """
    import posixpath

    parent = posixpath.dirname(remote_output_dir.rstrip("/")) or remote_output_dir
    return posixpath.join(parent, "logs")


def remote_tune_log_path(remote_output_dir: str, backend: str, model: str, tp_size: int, stamp: str) -> str:
    """Return the host-side logfile path for one TP size's tuning run."""
    import posixpath

    name = "%s-%s-tp%d-%s.log" % (backend, _slugify_model(model), tp_size, stamp)
    return posixpath.join(remote_tune_log_dir(remote_output_dir), name)


def log_stamp() -> str:
    """A sortable timestamp for log filenames."""
    import time

    return time.strftime("%Y%m%d-%H%M%S")


def build_streamed_tune_script(command: str, log_path: str) -> str:
    """Wrap a tuning *command* so its output both streams and is recorded.

    Three things the bare command doesn't do on its own:

    * ``PYTHONUNBUFFERED`` — the tuner's progress is Python ``print``/tqdm
      output, and Python block-buffers when stdout is a pipe (which it is,
      over SSH).  Without this, "streaming" delivers 8 KB at a time and a
      quiet sweep looks hung.
    * ``tee`` to a host-side logfile — a dropped link, a Ctrl-C or an overrun
      timeout otherwise loses every trace of a run that may have been going
      for hours.  The log outlives the SSH session.
    * ``PIPESTATUS[0]`` — with ``tee`` in the pipeline the shell would
      otherwise report *tee's* exit status, turning any tuning failure into a
      success.

    The command runs inside a subshell because tuning commands are compound
    (``mkdir … && cd … && python3 …``).  A bare ``cmd 2>&1 | tee`` binds the
    redirection and the pipe to the *last* simple command only, so the earlier
    stages' output would bypass the log and ``PIPESTATUS[0]`` would report the
    wrong command's status.
    """
    import posixpath

    log = safe_remote_path(log_path)
    log_dir = safe_remote_path(posixpath.dirname(log_path))
    return (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        'mkdir -p "%s"\n'
        "export PYTHONUNBUFFERED=1\n"
        "(\n"
        "%s\n"
        ') 2>&1 | tee -a "%s"\n'
        "rc=${PIPESTATUS[0]}\n"
        'echo "[sparkrun] tuning command exited rc=$rc" >> "%s"\n'
        "exit $rc\n"
    ) % (log_dir, command, log, log)


def report_tune_failure(
    tp_size: int,
    result,
    elapsed: float,
    host: str,
    log_path: str,
    timeout_sec: int | None,
) -> None:
    """Explain a failed or timed-out TP run without dumping the firehose.

    Shared by both tuners: a timeout is reported as a timeout (naming the cap
    that fired and how to raise it), not as an opaque non-zero exit, and only
    the tail of the captured output is echoed — the full record is in the
    host-side log.
    """
    from sparkrun.orchestration.ssh import TIMEOUT_RETURNCODE

    if result.returncode == TIMEOUT_RETURNCODE:
        logger.error(
            "  Tuning for TP=%d hit its %s cap (%s elapsed) and was terminated.",
            tp_size,
            describe_timeout(timeout_sec),
            _format_duration(elapsed),
        )
        logger.error("  Raise it with --timeout HOURS (or tuning.timeout_hours in config.yaml); --timeout 0 removes the cap.")
    else:
        logger.error(
            "  Tuning for TP=%d failed (exit %d, %s)",
            tp_size,
            result.returncode,
            _format_duration(elapsed),
        )
    logger.error("  Full output: %s:%s", host, log_path)
    if result.stdout and result.stdout.strip():
        logger.error("  stdout (tail):\n%s", tail_text(result.stdout))
    if result.stderr and result.stderr.strip():
        logger.error("  stderr (tail):\n%s", tail_text(result.stderr))


def tail_text(text: str, max_lines: int = 60) -> str:
    """Return the last *max_lines* lines of *text*, noting what was dropped.

    Tuning emits tens of thousands of progress-bar lines; dumping all of them
    into an error message buries the actual failure (see issue #206).
    """
    lines = (text or "").rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    dropped = len(lines) - max_lines
    return "\n".join(["... (%d earlier lines omitted)" % dropped] + lines[-max_lines:])


# ---------------------------------------------------------------------------
# BaseTuner — shared orchestration skeleton
# ---------------------------------------------------------------------------


class BaseTuner:
    """Shared tuning orchestration logic.

    Subclasses must set the class attributes below and override
    :meth:`_run_tune_for_tp`.
    """

    # --- Class attributes set by subclasses ---
    runtime_label: str  # "SGLang" or "vLLM"
    container_name: str  # e.g. "sparkrun_tune"
    output_path: str  # container-side output mount point
    clone_script: str  # e.g. "sglang_clone_benchmarks.sh"

    def __init__(
        self,
        host: str,
        image: str,
        model: str,
        config: SparkrunConfig | None = None,
        cache_dir: str | None = None,
        output_dir: str | None = None,
        skip_clone: bool = False,
        dry_run: bool = False,
        timeout_hours: float | None = None,
        force_parallel: bool = False,
    ):
        self.host = host
        self.image = image
        self.model = model
        self.config = config
        self.cache_dir = cache_dir
        self._custom_output_dir = output_dir is not None
        self.output_dir = output_dir or str(self._default_output_dir())
        self.skip_clone = skip_clone
        self.dry_run = dry_run
        self.timeout_sec = resolve_tuning_timeout(timeout_hours, config)
        self.force_parallel = force_parallel
        self._log_stamp = log_stamp()

        from sparkrun.orchestration.primitives import build_ssh_kwargs

        self.ssh_kwargs = build_ssh_kwargs(config)

        # Compute remote output dir: on cross-OS or cross-user setups the
        # remote host path differs from the local one.
        self.remote_output_dir = self._resolve_remote_output_dir()

    def _default_output_dir(self) -> Path:
        """Return the default host-side output directory.

        Subclasses override to return their ``get_*_tuning_dir()`` result.
        """
        raise NotImplementedError

    def _resolve_remote_output_dir(self) -> str:
        """Derive the remote-host output directory.

        When the control machine is non-Linux (e.g. macOS) or the SSH user
        differs from the local user, the local ``DEFAULT_CACHE_DIR`` path
        won't exist on remote Linux hosts.  This method replaces the local
        cache prefix with a Linux-appropriate path derived from the SSH user.

        If a custom ``output_dir`` was provided explicitly, it is assumed
        to be valid on the remote host and returned as-is.
        """
        import os
        import sys

        # If the user gave an explicit output_dir, trust it for remote too
        if self.__dict__.get("_custom_output_dir"):
            return self.output_dir

        ssh_user = self.ssh_kwargs.get("ssh_user")
        local_user = os.environ.get("USER")

        if (ssh_user and ssh_user != local_user) or sys.platform != "linux":
            _user = ssh_user or local_user or "user"
            # Replace the local cache prefix with the remote user's cache dir.
            # output_dir is always under DEFAULT_CACHE_DIR/<subdir>.
            local_prefix = str(DEFAULT_CACHE_DIR)
            if self.output_dir.startswith(local_prefix):
                suffix = self.output_dir[len(local_prefix) :]
                return "/home/%s/.cache/sparkrun%s" % (_user, suffix)
            # Fallback: if output_dir doesn't start with DEFAULT_CACHE_DIR
            # (shouldn't happen in normal use), return as-is.
            return self.output_dir

        return self.output_dir

    # ----- public entry point -----

    def run_tuning(
        self,
        tp_sizes: tuple[int, ...] = DEFAULT_TP_SIZES,
        parallel: int = 1,
    ) -> int:
        """Run the full tuning flow.

        Args:
            tp_sizes: Tensor parallel sizes to tune for.
            parallel: Max concurrent tuning jobs (1 = sequential).

        Returns:
            Exit code (0 = success).
        """
        import time

        logger.log(PROGRESS, "=" * 60)
        logger.log(PROGRESS, "sparkrun %s Kernel Tuner", self.runtime_label)
        logger.log(PROGRESS, "=" * 60)
        parallel = resolve_tuning_parallel(parallel, len(tp_sizes), self.force_parallel)

        logger.log(PROGRESS, "Host:       %s", self.host)
        logger.log(PROGRESS, "Image:      %s", self.image)
        logger.log(PROGRESS, "Model:      %s", self.model)
        logger.log(PROGRESS, "TP sizes:   %s", ", ".join(str(t) for t in tp_sizes))
        logger.log(PROGRESS, "Parallel:   %d", parallel)
        logger.log(PROGRESS, "Output:     %s", self.output_dir)
        logger.log(PROGRESS, "Timeout:    %s (per TP size)", describe_timeout(self.timeout_sec))
        logger.log(PROGRESS, "Logs:       %s:%s", self.host, remote_tune_log_dir(self.remote_output_dir))
        logger.log(PROGRESS, "Mode:       %s", "DRY-RUN" if self.dry_run else "LIVE")
        logger.log(PROGRESS, "=" * 60)

        t_total = time.monotonic()
        tp_timings: list[tuple[int, float]] = []  # (tp_size, seconds)

        try:
            # Step 1: Launch container
            rc = self._launch_container()
            if rc != 0:
                return rc

            # Step 2: Clone benchmark scripts
            if not self.skip_clone:
                rc = self._clone_benchmarks()
                if rc != 0:
                    return rc
                self._apply_patches()
            else:
                logger.log(PROGRESS, "Step 2/5: Skipping clone (--skip-clone)")

            # Step 3: Detect Triton version
            triton_version = self._detect_triton_version()

            # Step 4: Run tuning for each TP size
            if parallel > 1 and len(tp_sizes) > 1:
                rc = self._run_tuning_parallel(
                    tp_sizes,
                    triton_version,
                    parallel,
                    tp_timings,
                )
            else:
                rc = self._run_tuning_sequential(
                    tp_sizes,
                    triton_version,
                    tp_timings,
                )

            # Step 5: Sync configs back to control node (remote hosts only).
            # Unconditional, and *before* the failure return: a run that dies
            # on TP=4 has usually already produced good configs for TP=1 and
            # TP=2, and returning early stranded hours of completed work on
            # the remote host.
            self._sync_back_configs()

            if rc != 0:
                return rc

            logger.log(PROGRESS, "Step 5/5: Tuning complete!")
            total_elapsed = time.monotonic() - t_total
            self._print_timing_summary(tp_timings, total_elapsed)
            return 0

        finally:
            self._cleanup_container()

    # ----- orchestration steps -----

    def _run_tuning_sequential(
        self,
        tp_sizes: tuple[int, ...],
        triton_version: str,
        tp_timings: list[tuple[int, float]],
    ) -> int:
        """Run tuning for each TP size sequentially."""
        import time

        for i, tp in enumerate(tp_sizes):
            if self._pre_check_tp(tp, triton_version):
                logger.log(
                    PROGRESS,
                    "Step 4/5: TP=%d configs already exist, skipping (%d/%d)",
                    tp,
                    i + 1,
                    len(tp_sizes),
                )
                continue
            logger.log(
                PROGRESS,
                "Step 4/5: Tuning TP=%d (%d/%d)...",
                tp,
                i + 1,
                len(tp_sizes),
            )
            t_tp = time.monotonic()
            rc = self._run_tune_for_tp(tp, triton_version)
            tp_timings.append((tp, time.monotonic() - t_tp))
            if rc != 0:
                logger.error("Tuning failed for TP=%d (exit %d)", tp, rc)
                return rc
        return 0

    def _run_tuning_parallel(
        self,
        tp_sizes: tuple[int, ...],
        triton_version: str,
        max_workers: int,
        tp_timings: list[tuple[int, float]],
    ) -> int:
        """Run tuning for TP sizes in parallel batches."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        effective_workers = min(max_workers, len(tp_sizes))
        logger.log(
            PROGRESS,
            "Step 4/5: Tuning %d TP sizes with %d parallel workers...",
            len(tp_sizes),
            effective_workers,
        )

        # Filter out TP sizes that already have configs
        needed_tp = []
        for tp in tp_sizes:
            if self._pre_check_tp(tp, triton_version):
                logger.log(PROGRESS, "  TP=%d configs already exist, skipping", tp)
            else:
                needed_tp.append(tp)

        if not needed_tp:
            logger.log(PROGRESS, "  All TP sizes already tuned, nothing to do")
            return 0

        failed: list[tuple[int, int]] = []  # (tp_size, exit_code)

        def _tune_one(tp: int) -> tuple[int, int, float]:
            t0 = time.monotonic()
            # quiet: N concurrent sweeps interleaved on one terminal are
            # unreadable.  Each still tees to its own host-side logfile.
            rc = self._run_tune_for_tp(tp, triton_version, quiet=True)
            return tp, rc, time.monotonic() - t0

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_tune_one, tp): tp for tp in needed_tp}
            for future in as_completed(futures):
                tp, rc, elapsed = future.result()
                tp_timings.append((tp, elapsed))
                if rc != 0:
                    logger.error("Tuning failed for TP=%d (exit %d)", tp, rc)
                    failed.append((tp, rc))
                else:
                    logger.log(PROGRESS, "  TP=%d done (%s)", tp, _format_duration(elapsed))

        # Sort timings by TP size for consistent display
        tp_timings.sort(key=lambda x: x[0])

        if failed:
            logger.error(
                "Tuning failed for TP size(s): %s",
                ", ".join(str(tp) for tp, _ in failed),
            )
            return failed[0][1]
        return 0

    def _launch_container(self) -> int:
        """Step 1: Launch a tuning container with sleep infinity."""
        import time
        from sparkrun.orchestration.primitives import build_volumes, run_script_on_host
        from sparkrun.orchestration.executor import get_executor

        # TODO: switch to being resolved executor instance?
        DockerExecutor = get_executor("docker")

        t0 = time.monotonic()
        logger.log(PROGRESS, "Step 1/5: Launching tuning container on %s...", self.host)

        # Ensure output directory exists on the remote host (as the SSH user, not root)
        mkdir_script = '#!/bin/bash\nset -uo pipefail\nmkdir -p "%s"\n' % safe_remote_path(self.remote_output_dir)
        mkdir_result = run_script_on_host(
            self.host,
            mkdir_script,
            ssh_kwargs=self.ssh_kwargs,
            timeout=30,
            dry_run=self.dry_run,
        )
        if not mkdir_result.success and not self.dry_run:
            logger.error(
                "Failed to create output directory %s: %s",
                self.remote_output_dir,
                mkdir_result.stderr,
            )
            return 1

        volumes = build_volumes(self.cache_dir)
        # Mount tuning output directory (use remote path for volume mount)
        volumes[self.remote_output_dir] = self.output_path

        launch_script = DockerExecutor().generate_launch_script(
            image=self.image,
            container_name=self.container_name,
            command="sleep infinity",
            volumes=volumes,
        )

        result = run_script_on_host(
            self.host,
            launch_script,
            ssh_kwargs=self.ssh_kwargs,
            timeout=120,
            dry_run=self.dry_run,
        )

        if not result.success and not self.dry_run:
            logger.error("Failed to launch tuning container: %s", result.stderr)
            return 1

        logger.log(PROGRESS, "Step 1/5: Container launched (%.1fs)", time.monotonic() - t0)
        return 0

    def _clone_benchmarks(self) -> int:
        """Step 2: Clone benchmark scripts inside the container."""
        import time
        from sparkrun.orchestration.primitives import run_script_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd
        from sparkrun.scripts import read_script

        t0 = time.monotonic()
        logger.log(PROGRESS, "Step 2/5: Cloning %s benchmark scripts...", self.runtime_label)

        clone_script = read_script(self.clone_script)
        exec_cmd = docker_exec_cmd(self.container_name, clone_script)

        # Wrap in a bash script for run_script_on_host
        script = "#!/bin/bash\nset -uo pipefail\n%s\n" % exec_cmd

        result = run_script_on_host(
            self.host,
            script,
            ssh_kwargs=self.ssh_kwargs,
            timeout=120,
            dry_run=self.dry_run,
        )

        if not result.success and not self.dry_run:
            logger.error("Failed to clone benchmark scripts: %s", result.stderr)
            return 1

        logger.log(PROGRESS, "Step 2/5: Clone done (%.1fs)", time.monotonic() - t0)
        return 0

    def _detect_triton_version(self) -> str:
        """Step 3: Detect Triton version inside the container."""
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd

        logger.log(PROGRESS, "Step 3/5: Detecting Triton version...")

        detect_cmd = docker_exec_cmd(
            self.container_name,
            'python3 -c "import triton; print(triton.__version__)"',
        )

        result = run_command_on_host(
            self.host,
            detect_cmd,
            ssh_kwargs=self.ssh_kwargs,
            timeout=30,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            logger.log(PROGRESS, "Step 3/5: [dry-run] Would detect Triton version")
            return "unknown"

        version = "unknown"
        if result.success and result.stdout.strip():
            version = result.stdout.strip().splitlines()[-1].strip()
            logger.log(PROGRESS, "Step 3/5: Triton version: %s", version)
        else:
            logger.warning(
                "Step 3/5: Could not detect Triton version, using 'unknown': %s",
                result.stderr[:200] if result.stderr else "(no output)",
            )

        return version

    def _apply_patches(self) -> None:
        """Apply post-clone patches to benchmark scripts.

        Called after cloning benchmark scripts into the container.
        Subclasses override to fix known upstream issues.
        """

    def _pre_check_output_dir(self, tp_size: int, triton_version: str) -> str:
        """Return the container-side output directory for pre-check.

        Subclasses override to apply versioning (e.g. SGLang uses
        ``triton_X_Y_Z`` subdirectories).  The default returns
        :attr:`output_path`.
        """
        return self.output_path

    def _pre_check_tp(self, tp_size: int, triton_version: str) -> bool:
        """Check if tuning configs already exist for this TP size.

        Runs a lightweight script inside the container that loads the model
        config to determine MoE shape params (E, N), then checks whether
        matching config files already exist in the output directory.

        Returns ``True`` if configs exist (skip tuning), ``False`` otherwise.
        On any error, returns ``False`` (safe default — tune anyway).
        """
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd

        if self.dry_run:
            return False

        output_dir = self._pre_check_output_dir(tp_size, triton_version)

        check_py = (
            "import sys, os, glob; "
            "from transformers import AutoConfig; "
            "c = AutoConfig.from_pretrained(%r, trust_remote_code=True); "
            "E = getattr(c, 'num_local_experts', getattr(c, 'num_experts', 0)); "
            "I = getattr(c, 'intermediate_size', getattr(c, 'moe_intermediate_size', 0)); "
            "N = (I * 2) // %d; "
            "pattern = os.path.join(%r, 'E=%%d,N=%%d,*' %% (E, N)); "
            "matches = glob.glob(pattern); "
            "sys.exit(0 if matches else 1)"
        ) % (self.model, tp_size, output_dir)

        exec_cmd = docker_exec_cmd(self.container_name, "python3 -c " + quote(check_py))
        try:
            result = run_command_on_host(
                self.host,
                exec_cmd,
                ssh_kwargs=self.ssh_kwargs,
                timeout=60,
                dry_run=False,
            )
            return result.success
        except Exception:
            logger.debug("Pre-check failed for TP=%d, will proceed with tuning", tp_size)
            return False

    def _build_tune_command(self, tp_size: int, triton_version: str) -> str:
        """Build the tuning command for a given TP size.

        Subclasses must override this — each runtime builds a different command.
        """
        raise NotImplementedError

    def _run_tune_for_tp(self, tp_size: int, triton_version: str, quiet: bool = False) -> int:
        """Step 4 (per-TP): Run the tuning script for a given TP size.

        Output streams to the terminal as it is produced and is teed to a
        host-side logfile that outlives the SSH session.  *quiet* captures
        instead of streaming — used when several TP sizes run concurrently
        and interleaved output would be unreadable.
        """
        import time
        from sparkrun.orchestration.primitives import run_script_on_host_streaming
        from sparkrun.orchestration.docker import docker_exec_cmd

        t0 = time.monotonic()
        tune_cmd = self._build_tune_command(tp_size, triton_version)
        exec_cmd = docker_exec_cmd(self.container_name, tune_cmd)
        log_path = self._tune_log_path(tp_size)
        script = build_streamed_tune_script(exec_cmd, log_path)

        logger.log(PROGRESS, "  TP=%d: log -> %s:%s", tp_size, self.host, log_path)
        result = run_script_on_host_streaming(
            self.host,
            script,
            ssh_kwargs=self.ssh_kwargs,
            timeout=self.timeout_sec,
            dry_run=self.dry_run,
            quiet=quiet,
            # A sweep holds the GPU for hours: if this process dies (Ctrl-C,
            # or the timeout above), the payload must die with its session
            # rather than keep tuning invisibly on the host.
            session_guard=True,
            keepalive=True,
        )

        if self.dry_run:
            logger.log(PROGRESS, "  [dry-run] Would run tuning for TP=%d", tp_size)
            return 0

        elapsed = time.monotonic() - t0
        if not result.success:
            self._report_tune_failure(tp_size, result, elapsed, log_path)
            return result.returncode

        logger.log(PROGRESS, "  TP=%d tuning complete (%s)", tp_size, _format_duration(elapsed))
        return 0

    def _tune_log_path(self, tp_size: int) -> str:
        """Host-side logfile for this TP size's tuning run."""
        return remote_tune_log_path(
            self.remote_output_dir,
            self.runtime_label.lower(),
            self.model,
            tp_size,
            self._log_stamp,
        )

    def _report_tune_failure(self, tp_size: int, result, elapsed: float, log_path: str) -> None:
        report_tune_failure(tp_size, result, elapsed, self.host, log_path, self.timeout_sec)

    def _print_timing_summary(
        self,
        tp_timings: list[tuple[int, float]],
        total_elapsed: float,
    ) -> None:
        """Print a timing summary table after tuning completes."""
        logger.log(PROGRESS, "")
        logger.log(PROGRESS, "=" * 60)
        logger.log(PROGRESS, "Tuning Summary")
        logger.log(PROGRESS, "=" * 60)
        logger.log(PROGRESS, "  %-8s  %s", "TP Size", "Duration")
        logger.log(PROGRESS, "  %-8s  %s", "-------", "--------")
        for tp, elapsed in tp_timings:
            logger.log(PROGRESS, "  %-8d  %s", tp, _format_duration(elapsed))
        logger.log(PROGRESS, "  %-8s  %s", "-------", "--------")
        logger.log(PROGRESS, "  %-8s  %s", "Total", _format_duration(total_elapsed))
        logger.log(PROGRESS, "")
        logger.log(PROGRESS, "Tuning configs saved to: %s", self.output_dir)
        logger.log(
            PROGRESS,
            "These will be auto-mounted in future 'sparkrun run' invocations for %s recipes.",
            self.runtime_label,
        )
        logger.log(PROGRESS, "=" * 60)

    def _sync_back_configs(self) -> None:
        """Sync tuning configs from remote host back to the control node.

        After tuning on a remote host, the configs exist only on that
        host's filesystem.  This step rsyncs them back to the local
        ``output_dir`` so they can be reviewed, exported, and
        distributed to other hosts in future ``sparkrun run`` invocations.

        No-op when the host is localhost (same filesystem).
        """
        from sparkrun.utils import is_local_host

        if is_local_host(self.host):
            return

        if self.dry_run:
            logger.log(PROGRESS, "  [dry-run] Would sync configs back from %s:%s", self.host, self.remote_output_dir)
            return

        from sparkrun.orchestration.ssh import run_rsync_from_remote

        logger.log(PROGRESS, "  Syncing tuning configs back from %s...", self.host)
        result = run_rsync_from_remote(
            host=self.host,
            source_path=self.remote_output_dir,
            dest_path=self.output_dir,
            ssh_user=self.ssh_kwargs.get("ssh_user"),
            ssh_key=self.ssh_kwargs.get("ssh_key"),
            ssh_options=self.ssh_kwargs.get("ssh_options"),
            rsync_options=["-az", "--mkpath", "--partial", "--links"],
            timeout=120,
        )
        if result.success:
            logger.log(PROGRESS, "  Tuning configs synced to local %s", self.output_dir)
        else:
            logger.warning(
                "  Failed to sync tuning configs back from %s: %s",
                self.host,
                result.stderr[:200],
            )

    def _cleanup_container(self) -> None:
        """Step 5: Remove the tuning container."""
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_stop_cmd

        logger.log(PROGRESS, "Cleaning up tuning container...")
        cmd = docker_stop_cmd(self.container_name)
        run_command_on_host(
            self.host,
            cmd,
            ssh_kwargs=self.ssh_kwargs,
            timeout=30,
            dry_run=self.dry_run,
        )
