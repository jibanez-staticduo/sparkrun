"""Resumable benchmark run state.

Persists benchmark progress in ``~/.cache/sparkrun/benchmarks/<benchmark_id>/``
so that interrupted runs can be resumed with ``sparkrun benchmark resume <id>``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sparkrun.core.config import resolve_sparkrun_cache_dir

logger = logging.getLogger(__name__)

LOCK_FILE_NAME = "run.lock"

# A benchmark holds its state directory for the whole sweep, which for a large
# schedule is legitimately many hours — far longer than a launch's pending-op
# lock, hence a separate ceiling rather than reusing
# ``pending_ops.LOCK_MAX_AGE_SECONDS``.  Its only job is to eventually release
# a directory whose owner died in a way that skipped cleanup (SIGKILL, power
# loss) *and* whose PID we cannot evaluate because it belonged to another host.
LOCK_MAX_AGE_SECONDS = 72 * 60 * 60  # 72 hours


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def canonical_host_key(hosts: "list[str] | tuple[str, ...] | None") -> str:
    """Canonical, order-independent digest input for a host set.

    Sorted so that host ordering — which placement may vary without changing
    *what was measured* — never moves the digest.  Returns ``""`` for an empty
    or absent host set, which is what keeps the legacy no-hosts call shape
    hashing exactly as it did.
    """
    if not hosts:
        return ""
    return "\0".join(sorted(str(h) for h in hosts))


def derive_benchmark_id(
    cluster_id: str,
    framework: str,
    profile: str | None,
    base_args: dict[str, Any],
    schedule: list[dict[str, Any]] | None,
    recipe_fingerprint: str | None = None,
    hosts: "list[str] | tuple[str, ...] | None" = None,
) -> str:
    """Stable ID derived from canonical-JSON of inputs. Returns ``'bench_<12hex>'``.

    Benchmark identity follows the recipe *intent* (port, parallelism, runtime,
    model, served-model-name), **not** per-launch placement. The cluster_id is
    parsed via :func:`parse_cluster_id` and only its intent half is hashed, so a
    benchmark resumes successfully across relaunches that produce a fresh
    placement token but represent the same logical workload.

    ``hosts`` is the *measured node set*, and it is hashed separately from — and
    for the opposite reason to — the discarded placement token.  The two are
    easy to conflate but are not the same thing: ``api.run`` mints a **random**
    placement token per launch, so hashing that would break resume across
    relaunches; a host set is stable, and a run against different nodes is a
    different measurement.  Dropping both is what let two concurrent per-node
    runs share one state directory and silently serve one node's cached results
    as the other's (issue #267) — invisible in the output, and it read as "these
    nodes perform identically", which is a conclusion someone acts on.  Pass the
    *resolved* host list (what actually ran), not the candidate set.

    ``recipe_fingerprint`` extends the identity to the recipe's *content*
    (declared serve configuration + user overrides): two recipes that share an
    intent — same model, port, parallelism — but differ in a serve argument
    (e.g. ``--speculative-config``) are different workloads and must never
    resume into each other's results.  Obtain it from
    :func:`sparkrun.orchestration.job_metadata.derive_recipe_fingerprint`,
    which is the single definition of *what* gets hashed: declared
    configuration only, never resolved artifacts or placement, so it is stable
    across relaunches of the same logical workload.

    Malformed (legacy) cluster_ids that do not parse fall back to hashing the
    full string verbatim — they will not match a relaunch, but they also won't
    crash old callers.
    """
    from sparkrun.orchestration.job_metadata import parse_cluster_id

    try:
        intent_id, _placement_token = parse_cluster_id(cluster_id)
        identity = intent_id
    except (ValueError, TypeError):
        logger.debug(
            "derive_benchmark_id: cluster_id %r does not parse; hashing verbatim — benchmark will not be resumable across relaunches",
            cluster_id,
        )
        identity = cluster_id

    payload = {
        "intent_id": identity,
        "framework": framework,
        "profile": profile,
        "base_args": base_args,
        "schedule": schedule,
    }
    if recipe_fingerprint is not None:
        payload["recipe_fingerprint"] = recipe_fingerprint
    if host_key := canonical_host_key(hosts):
        payload["hosts"] = host_key
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return "bench_%s" % digest


class StateDirLocked(RuntimeError):
    """Raised when another live run already owns a benchmark state directory.

    Carries the holder's ``pid`` / ``host`` / ``started_at`` so the caller can
    say *who* holds it rather than only that something does.
    """

    def __init__(self, benchmark_id: str, info: dict[str, Any]) -> None:
        self.benchmark_id = benchmark_id
        self.info = info
        super().__init__(
            "benchmark state directory %s is held by pid %s on %s" % (benchmark_id, info.get("pid", "?"), info.get("host") or "?")
        )


def _state_dir_for(benchmark_id: str, cache_dir: str | None = None) -> Path:
    return resolve_sparkrun_cache_dir(cache_dir) / "benchmarks" / benchmark_id


def _lock_is_stale(info: dict[str, Any]) -> bool:
    """Return ``True`` when *info* describes a lock whose owner is gone.

    Mirrors :func:`sparkrun.core.pending_ops._is_stale`: PID liveness is only
    trusted for a lock written on *this* host, and the age ceiling is the sole
    signal for anything else (a PID from another host means nothing locally,
    and a recycled PID would otherwise read as alive forever).
    """
    from sparkrun.core.pending_ops import is_pid_alive, lock_hostname

    started_at = info.get("started_at")
    if isinstance(started_at, (int, float)) and (time.time() - started_at) > LOCK_MAX_AGE_SECONDS:
        return True

    lock_host = info.get("host")
    if lock_host in (None, "", lock_hostname()):
        return not is_pid_alive(info.get("pid", -1))

    return False


def clear_state_dir(benchmark_id: str, cache_dir: str | None = None) -> None:
    """Discard a benchmark's recorded state, keeping the directory's lock.

    The peer of ``shutil.rmtree(state_dir)`` for use *inside*
    :func:`hold_state_dir`.  Removing the directory outright would take our own
    ``run.lock`` with it and leave the benchmark unlocked for the rest of the
    run — precisely while it is about to write the fresh results a second run
    could then interleave with.  So the contents go and the lock stays.
    """
    import shutil

    sdir = _state_dir_for(benchmark_id, cache_dir)
    if not sdir.is_dir():
        return
    for child in sdir.iterdir():
        if child.name == LOCK_FILE_NAME:
            continue
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove benchmark state entry %s", child, exc_info=True)


@contextmanager
def hold_state_dir(benchmark_id: str, cache_dir: str | None = None):
    """Take exclusive ownership of a benchmark state directory for the block.

    Two runs sharing a state directory interleave into the same
    ``runs/<idx>.json`` files and the same ``state.yaml``: the per-task
    artefacts are keyed on task index alone, so the loser's measurements
    overwrite the winner's with nothing anywhere recording that it happened
    (issue #267).  Unlike a colliding *identity*, this cannot be fixed by
    keying the directory better — the same recipe on the same hosts is
    genuinely the same directory — so it is excluded rather than separated.

    Acquisition is an ``O_CREAT | O_EXCL`` create, which is atomic on POSIX
    and Windows alike; a lock left behind by a dead owner is reclaimed.

    Raises:
        StateDirLocked: another live run holds the directory.
    """
    from sparkrun.core.pending_ops import lock_hostname

    sdir = _state_dir_for(benchmark_id, cache_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    lock_path = sdir / LOCK_FILE_NAME

    payload = json.dumps(
        {
            "benchmark_id": benchmark_id,
            "pid": os.getpid(),
            "host": lock_hostname(),
            "started_at": time.time(),
        }
    )

    def _acquire() -> bool:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        except OSError:
            # An unwritable cache must not block a benchmark: the lock is a
            # concurrency guard, not a correctness gate for a single run.
            logger.debug("Could not create benchmark state lock at %s", lock_path, exc_info=True)
            return True
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
        except OSError:
            logger.debug("Could not write benchmark state lock at %s", lock_path, exc_info=True)
        return True

    acquired = _acquire()
    if not acquired:
        try:
            held = json.loads(lock_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Unreadable or truncated — a half-written lock from a run that
            # died mid-create. Treat as stale rather than blocking forever.
            held = {}
        if not _lock_is_stale(held):
            raise StateDirLocked(benchmark_id, held)
        logger.debug("Reclaiming stale benchmark state lock at %s (%s)", lock_path, held)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        if not _acquire():
            # Lost the reclaim race to another run that was also waiting.
            try:
                held = json.loads(lock_path.read_text())
            except (OSError, json.JSONDecodeError):
                held = {}
            raise StateDirLocked(benchmark_id, held)

    try:
        yield lock_path
    finally:
        # Release only our own lock: a run whose directory was rmtree'd and
        # recreated by someone else must not delete the new owner's lock.
        try:
            current = json.loads(lock_path.read_text())
        except (OSError, json.JSONDecodeError):
            current = None
        if current is None or (current.get("pid") == os.getpid() and current.get("host") == lock_hostname()):
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove benchmark state lock at %s", lock_path, exc_info=True)


@dataclass
class BenchmarkRunState:
    """Persistent progress state for a scheduled benchmark run.

    ``cluster_id`` carries the most recently observed concrete launch; it is
    refreshed on resume when the user relaunches inference for the same intent.
    ``intent_id`` is the stable identity used for resume matching — it does not
    change across relaunches of the same logical workload.

    ``host_list`` records the node set these measurements were taken on, so a
    reuse against a *different* set can be refused rather than silently mixed
    (see :meth:`matches_hosts`).
    """

    benchmark_id: str
    cluster_id: str
    recipe_qualified_name: str
    framework: str
    profile: str | None
    base_args: dict[str, Any]
    schedule: list[dict[str, Any]]  # raw schedule_entry dicts in order
    intent_id: str = ""  # derived from cluster_id; stable across relaunches
    host_list: list[str] = field(default_factory=list)
    completed_indices: list[int] = field(default_factory=list)
    failed_indices: list[int] = field(default_factory=list)
    crash_count: int = 0
    session_count: int = 0
    sessions: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)  # arena uses for submission_id, etc.
    created_at: str = ""  # ISO-8601 UTC
    updated_at: str = ""  # ISO-8601 UTC

    def __post_init__(self) -> None:
        """Derive ``intent_id`` from ``cluster_id`` if not already set."""
        if not self.intent_id and self.cluster_id:
            from sparkrun.orchestration.job_metadata import parse_cluster_id

            try:
                self.intent_id, _ = parse_cluster_id(self.cluster_id)
            except (ValueError, TypeError):
                logger.debug(
                    "BenchmarkRunState: cluster_id %r does not parse; intent_id left empty (unresumable across relaunches)",
                    self.cluster_id,
                )
                self.intent_id = ""

    def matches_hosts(self, hosts: "list[str] | tuple[str, ...] | None") -> bool:
        """Return ``True`` when *hosts* is the node set this state was measured on.

        With hosts folded into :func:`derive_benchmark_id` a mismatch is only
        reachable through state written before that change, where the ID alone
        could not tell two node sets apart — which is precisely the state most
        likely to hold another node's numbers.  State that recorded no host set
        (legacy) is treated as matching: refusing every pre-upgrade resume
        would be a worse trade than the mismatch it guards against, and the ID
        itself now separates the case going forward.
        """
        if not self.host_list:
            return True
        return canonical_host_key(self.host_list) == canonical_host_key(hosts)

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    def state_dir(self, cache_dir: str | None = None) -> Path:
        """Return ``~/.cache/sparkrun/benchmarks/<benchmark_id>/``."""
        return _state_dir_for(self.benchmark_id, cache_dir)

    def runs_dir(self, cache_dir: str | None = None) -> Path:
        """Return the per-run artefact directory (``state_dir / "runs"``)."""
        return self.state_dir(cache_dir) / "runs"

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, cache_dir: str | None = None) -> Path:
        """Atomically persist state to ``state_dir/state.yaml``.

        Sets ``created_at`` on first save; always updates ``updated_at``.
        Returns the path to the written file.
        """
        now = _now_iso()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now

        sdir = self.state_dir(cache_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        self.runs_dir(cache_dir).mkdir(parents=True, exist_ok=True)

        state_path = sdir / "state.yaml"
        tmp_path = sdir / "state.yaml.tmp"

        data = asdict(self)
        with open(tmp_path, "w") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)

        tmp_path.replace(state_path)
        logger.debug("Saved benchmark run state to %s", state_path)
        return state_path

    @classmethod
    def load(cls, benchmark_id: str, cache_dir: str | None = None) -> "BenchmarkRunState | None":
        """Load state from disk. Returns ``None`` if no state file exists."""
        state_path = _state_dir_for(benchmark_id, cache_dir) / "state.yaml"
        if not state_path.exists():
            return None
        try:
            with open(state_path) as fh:
                data = yaml.safe_load(fh)
            if not data:
                return None
            return cls(**data)
        except Exception:
            logger.debug("Failed to load benchmark run state for %s", benchmark_id, exc_info=True)
            return None

    # -------------------------------------------------------------------------
    # Progress tracking
    # -------------------------------------------------------------------------

    def mark_started(self, idx: int, pid: int | None = None) -> None:
        """Record that task *idx* has started (optionally with process *pid*)."""
        logger.debug("Benchmark %s: task %d started (pid=%s)", self.benchmark_id, idx, pid)

    def mark_completed(self, idx: int) -> None:
        """Record task *idx* as completed.

        Deduplicates — safe to call more than once.  Also removes *idx* from
        ``failed_indices`` if present.
        """
        if idx not in self.completed_indices:
            self.completed_indices.append(idx)
        if idx in self.failed_indices:
            self.failed_indices.remove(idx)

    def mark_failed(self, idx: int, error: str | None = None) -> None:
        """Record task *idx* as failed for this session."""
        if idx not in self.failed_indices:
            self.failed_indices.append(idx)
        logger.debug("Benchmark %s: task %d failed — %s", self.benchmark_id, idx, error or "no detail")

    def mark_session_started(self) -> None:
        """Increment ``session_count`` and append a new sessions entry."""
        self.session_count += 1
        self.sessions.append({"session": self.session_count, "started_at": _now_iso(), "status": "running"})

    def mark_session_ended(self, status: str) -> None:
        """Update the last sessions entry with *status* and end timestamp."""
        if self.sessions:
            self.sessions[-1]["ended_at"] = _now_iso()
            self.sessions[-1]["status"] = status

    def mark_crash(self) -> None:
        """Increment ``crash_count``."""
        self.crash_count += 1
        logger.debug("Benchmark %s: crash #%d recorded", self.benchmark_id, self.crash_count)

    # -------------------------------------------------------------------------
    # Scheduling helpers
    # -------------------------------------------------------------------------

    def next_pending(self, total_tasks: int) -> int | None:
        """Return the smallest index in ``[0, total_tasks)`` not yet completed.

        Failed indices from previous sessions are retried; failed indices in the
        *current* session are skipped to avoid tight crash loops.  Retry
        semantics can be refined in a later iteration — for now this returns
        the smallest idx not in ``completed_indices``.
        """
        for idx in range(total_tasks):
            if idx not in self.completed_indices:
                return idx
        return None

    def is_complete(self, total_tasks: int) -> bool:
        """Return ``True`` when every task in ``[0, total_tasks)`` is completed."""
        return all(idx in self.completed_indices for idx in range(total_tasks))
