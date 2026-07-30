"""Reading :class:`~sparkrun.core.log_source.LogSource` streams into lines.

The transport half of the log path.  A runtime says *what* to read
(:meth:`~sparkrun.runtimes.base.RuntimePlugin.log_sources`), an executor
says *how* to read it on its substrate
(:meth:`~sparkrun.orchestration.executors._base.Executor.read_logs_cmd`),
and this module actually runs those commands — locally or over SSH — and
turns their output into :class:`~sparkrun.core.log_source.LogLine` records.

Ordering contract (deliberate, and not the same in both modes):

* ``follow=True`` — one reader thread per source feeding a shared queue, so
  lines are yielded **as they arrive**.  That is true time order for live
  streaming, and each line is stamped with its arrival time.
* ``follow=False`` — sources are read **rank-grouped**: each source in full,
  head first, then workers by rank.  Cross-host time-interleaving is not
  possible here: the serve log written by ``scripts/exec_serve_detached.sh``
  carries no per-line capture timestamps, and a dump delivers every line at
  once, so arrival time carries no information either.  Inferring order from
  the runtime's own log prefixes would silently mis-order every unstamped
  line (tracebacks, progress bars, warnings), which is worse than honest
  grouping.  Getting true dump ordering requires stamping at capture time in
  the launch redirect — a deliberate, separate change.

``sparkrun.api.logs`` composes this into the public iterator; the CLI
renders it.  Nothing here imports ``api`` (layering).
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from typing import Iterator, Sequence

from sparkrun.core.log_source import LogLine, LogSource

logger = logging.getLogger(__name__)

_QUEUE_SENTINEL = object()


def build_read_command(
    executor,
    source: LogSource,
    *,
    follow: bool = False,
    tail: int | None = None,
    ssh_kwargs: dict | None = None,
) -> list[str]:
    """Build the argv that reads *source*, dispatching local vs SSH.

    Mirrors the dispatch every other orchestration primitive uses
    (:func:`~sparkrun.orchestration.ssh.should_run_locally`), so reading a
    workload on the control machine itself doesn't pointlessly round-trip
    through sshd — and doesn't break when SSH-to-self isn't configured.
    """
    from sparkrun.orchestration.ssh import build_ssh_cmd, should_run_locally

    kwargs = dict(ssh_kwargs or {})
    command = executor.read_logs_cmd(source, follow=follow, tail=tail)

    if should_run_locally(source.host, kwargs.get("ssh_user")):
        return ["bash", "-c", command]
    return build_ssh_cmd(source.host, **kwargs) + ["bash", "-c", command]


def _spawn(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


def _line(source: LogSource, text: str, *, timestamp: float | None = None) -> LogLine:
    return LogLine(
        host=source.host,
        container=source.container,
        text=text.rstrip("\n"),
        role=source.role,
        rank=source.rank,
        timestamp=timestamp,
    )


def _read_one(
    executor,
    source: LogSource,
    *,
    follow: bool,
    tail: int | None,
    ssh_kwargs: dict | None,
) -> Iterator[LogLine]:
    """Yield every line from a single source, then return."""
    cmd = build_read_command(executor, source, follow=follow, tail=tail, ssh_kwargs=ssh_kwargs)
    logger.debug("Reading logs from %s: %s", source.label, " ".join(cmd))
    proc = _spawn(cmd)
    try:
        assert proc.stdout is not None
        for text in proc.stdout:
            yield _line(source, text)
    finally:
        _terminate(proc)


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort teardown of a reader subprocess."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        logger.debug("Log reader terminate failed; killing", exc_info=True)
        try:
            proc.kill()
        except Exception:
            logger.debug("Log reader kill failed", exc_info=True)


def read_log_sources(
    executor,
    sources: Sequence[LogSource],
    *,
    follow: bool = False,
    tail: int | None = None,
    ssh_kwargs: dict | None = None,
) -> Iterator[LogLine]:
    """Yield :class:`LogLine` records from every source in *sources*.

    See the module docstring for the ordering contract: arrival-interleaved
    when *follow*, rank-grouped otherwise.

    Args:
        executor: Executor whose substrate the sources live on.
        sources: Sources to read, head-first (the non-follow emit order).
        follow: Keep streaming new lines instead of dumping and exiting.
        tail: Number of existing lines per source; ``None`` for all.
        ssh_kwargs: SSH connection parameters.
    """
    if not sources:
        return
    if not follow:
        for source in sources:
            yield from _read_one(executor, source, follow=False, tail=tail, ssh_kwargs=ssh_kwargs)
        return
    yield from _interleave(executor, sources, tail=tail, ssh_kwargs=ssh_kwargs)


def _interleave(
    executor,
    sources: Sequence[LogSource],
    *,
    tail: int | None,
    ssh_kwargs: dict | None,
) -> Iterator[LogLine]:
    """Follow every source concurrently, yielding lines as they arrive.

    One thread per source pushing into a shared queue.  Arrival order *is*
    time order for a live stream, and each line is stamped on arrival so
    downstream consumers can re-sort or display times.
    """
    lines: queue.Queue = queue.Queue(maxsize=1000)
    procs: list[subprocess.Popen] = []
    stop = threading.Event()

    def _pump(source: LogSource) -> None:
        try:
            cmd = build_read_command(executor, source, follow=True, tail=tail, ssh_kwargs=ssh_kwargs)
            logger.debug("Following logs from %s: %s", source.label, " ".join(cmd))
            proc = _spawn(cmd)
            procs.append(proc)
            assert proc.stdout is not None
            for text in proc.stdout:
                if stop.is_set():
                    break
                lines.put(_line(source, text, timestamp=time.time()))
        except Exception as e:  # pragma: no cover - defensive; one source must not kill the rest
            logger.debug("Log reader for %s failed: %s", source.label, e)
        finally:
            lines.put(_QUEUE_SENTINEL)

    threads = [threading.Thread(target=_pump, args=(source,), daemon=True, name="sparkrun-logs-%s" % source.label) for source in sources]
    for thread in threads:
        thread.start()

    remaining = len(threads)
    try:
        while remaining:
            item = lines.get()
            if item is _QUEUE_SENTINEL:
                remaining -= 1
                continue
            yield item
    finally:
        # Generator closed (Ctrl-C, break, consumer done) — stop the readers
        # and reap their subprocesses so we never leak an ssh child.
        stop.set()
        for proc in list(procs):
            _terminate(proc)


def print_log_sources(
    executor,
    sources: Sequence[LogSource],
    *,
    follow: bool = False,
    tail: int | None = None,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> None:
    """Print *sources* to stdout — the printing shim over :func:`read_log_sources`.

    Used by :meth:`~sparkrun.runtimes.base.RuntimePlugin.follow_logs` for the
    post-launch attach.  The ``sparkrun logs`` command renders
    :func:`sparkrun.api.logs` instead; both read through the same machinery.
    """
    if not sources:
        logger.warning("No log sources to read.")
        return

    if dry_run:
        for source in sources:
            cmd = build_read_command(executor, source, follow=follow, tail=tail, ssh_kwargs=ssh_kwargs)
            logger.info("[dry-run] Would stream logs from %s: %s", source.label, " ".join(cmd))
        return

    if follow:
        logger.info(
            "Following logs from %d source(s): %s (Ctrl-C to stop)...",
            len(sources),
            ", ".join(source.label for source in sources),
        )

    prefixed = len(sources) > 1
    try:
        for line in read_log_sources(executor, sources, follow=follow, tail=tail, ssh_kwargs=ssh_kwargs):
            print("[%s] %s" % (line.host, line.text) if prefixed else line.text)
    except KeyboardInterrupt:
        logger.info("\nLog following stopped.")


__all__ = ["build_read_command", "print_log_sources", "read_log_sources"]
