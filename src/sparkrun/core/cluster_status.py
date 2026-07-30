"""Cluster status dataclasses — the shape of "what's running where?"

This module is **data only**.  Production of :class:`ClusterStatus`
lives behind the :class:`~sparkrun.orchestration.executors._base.Executor`
ABC's ``query_status`` method — there is no separate extension point
for status providers.  Each Executor knows how to inspect its own
backend (Docker via ``docker ps``, K8s via ``kubectl get pods``, …).

Consumers (the ``occupancy-sparse`` and ``occupancy-dense`` schedulers,
the ``sparkrun status`` CLI, the ``cluster monitor`` TUI) all flow
through ``sparkrun.api.status`` → ``executor.query_status``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ContainerDetail:
    """Per-container / per-process realization of a workload on one host.

    The concrete backend object behind a :class:`RunningWorkload` — a
    Docker container (``docker ps``) or a native process (LocalExecutor
    pidfile).  Carries the presentation-layer detail the ``sparkrun
    status`` CLI needs that the occupancy-shaped :class:`RunningWorkload`
    discards: the container ``name``, its ``role`` (``solo`` / ``head`` /
    ``worker`` / ``node_<rank>``) derived from the name, a human ``status``
    string (e.g. ``"Up 10 seconds"``), and the ``image``.
    """

    name: str
    role: str
    status: str
    image: str


@dataclass(frozen=True)
class RunningWorkload:
    """A single sparkrun-launched workload occupying slots on one host.

    :attr:`memory_used_gb` and :attr:`util_fraction` are optional
    per-workload resource accounting fields populated by status
    providers that can observe them (e.g. parsing ``nvidia-smi``).
    Fractional-capable schedulers read these to compute available
    capacity for occupancy-sparse / occupancy-dense placement.
    """

    cluster_id: str
    intent_id: str | None = None
    """Hex intent identifier (the deterministic prefix of cluster_id).

    Recovered from the ``sparkrun.intent_id`` container label when
    emitted; otherwise derived from the cluster_id's intent prefix or
    enriched from cached job metadata.  ``None`` indicates a workload
    whose container name does not parse as a canonical sparkrun
    identifier.
    """
    recipe_name: str | None = None
    runtime_name: str | None = None
    started_at: float | None = None
    ranks_on_host: int = 1
    container_ids: tuple[str, ...] = field(default_factory=tuple)
    memory_used_gb: float | None = None
    util_fraction: float | None = None
    containers: tuple[ContainerDetail, ...] = field(default_factory=tuple)
    """Per-container/process realizations of this workload on the host.

    Optional (default empty for back-compat).  Populated by executors
    that can observe per-container detail (docker ``Names``/``Status``/
    ``Image``, local pidfiles); consumed by the ``sparkrun status`` CLI.
    """


@dataclass(frozen=True)
class GpuOccupancy:
    """Per-accelerator occupancy detail on a host.

    Populated by status providers that can introspect per-GPU usage
    (e.g. via ``nvidia-smi --query-gpu=memory.used,utilization.gpu``).
    Consumed by fractional-capable schedulers to decide whether a new
    rank's :class:`~sparkrun.core.scheduler.ResourceRequest` fits on
    each accelerator.
    """

    gpu_index: int
    used_memory_gb: float = 0.0
    used_util_fraction: float = 0.0
    workloads: tuple[RunningWorkload, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HostOccupancy:
    """Per-host accelerator occupancy at the time of the snapshot.

    :attr:`gpus` carries per-accelerator detail when the producing
    executor can supply it; otherwise it's empty and consumers should
    fall back to host-level :attr:`used_slots` / :attr:`free_slots`.
    """

    host: str
    workloads: tuple[RunningWorkload, ...] = field(default_factory=tuple)
    used_slots: int = 0
    free_slots: int = 0
    free_memory_gb: float | None = None
    gpus: tuple[GpuOccupancy, ...] = field(default_factory=tuple)

    @property
    def total_slots(self) -> int:
        """Total accelerator slots = used + free."""
        return self.used_slots + self.free_slots


@dataclass(frozen=True)
class ClusterStatus:
    """Snapshot of cluster occupancy produced by an Executor.

    ``hosts`` carries one entry per host inspected (in input order).
    ``executor`` is the registered name of the executor that produced
    the snapshot (e.g. ``"docker"``) — primarily useful for diagnostics
    and asymmetry detection (e.g. asking the docker executor about a
    k8s-managed host).
    """

    hosts: tuple[HostOccupancy, ...] = field(default_factory=tuple)
    queried_at: float = 0.0
    executor: str = ""
    errors: dict[str, str] = field(default_factory=dict)
    """Hosts the executor could not reach, mapped to an error message.

    Populated by executors when a host's remote probe fails (non-zero
    return code / no result) — those hosts are absent from :attr:`hosts`,
    so this is how a caller distinguishes an *unreachable* host from a
    reachable-but-idle one.  Merged as a union across executors, minus any
    host that turns out reachable via at least one of them (see
    :meth:`merged_with`).
    """

    def for_host(self, host: str) -> HostOccupancy | None:
        """Return the :class:`HostOccupancy` for *host*, or ``None`` if absent."""
        for entry in self.hosts:
            if entry.host == host:
                return entry
        return None

    def free_slots(self, host: str) -> int:
        """Free accelerator slots on *host*; 0 when host is absent."""
        entry = self.for_host(host)
        return entry.free_slots if entry is not None else 0

    def running_cluster_ids(self) -> tuple[str, ...]:
        """Distinct cluster_ids running anywhere in the snapshot."""
        seen: list[str] = []
        for entry in self.hosts:
            for w in entry.workloads:
                if w.cluster_id not in seen:
                    seen.append(w.cluster_id)
        return tuple(seen)

    @classmethod
    def merge(cls, snapshots: "list[ClusterStatus]") -> "ClusterStatus":
        """Fold *snapshots* (in priority order) into one via :meth:`merged_with`.

        The fold is left-associative: ``snapshots[0]`` is authoritative and
        wins any per-``cluster_id`` collision (callers pass the cluster's
        default executor first).  Used slots sum across executors; capacity is
        the shared host hardware; containers union.  Returns an empty snapshot
        when *snapshots* is empty and the sole snapshot unchanged (same object)
        when there's exactly one, so a single-executor scope is zero-cost.
        """
        if not snapshots:
            return cls()
        result = snapshots[0]
        for snap in snapshots[1:]:
            result = result.merged_with(snap)
        return result

    def merged_with(self, other: "ClusterStatus") -> "ClusterStatus":
        """Merge *other* into this snapshot, preserving this snapshot's host order.

        Combines two snapshots of the (same) host set produced by different
        executors (e.g. ``docker`` + ``local``, which inspect *disjoint*
        backend state on the same hosts) into one honest view.  Per host
        present in both:

        - ``workloads`` are concatenated, deduped by ``cluster_id`` (this
          snapshot's entry wins on collision);
        - ``used_slots`` is this snapshot's used plus the peer's slots for
          only its *non-duplicate* workloads — a ``cluster_id`` present in
          both (e.g. the same recipe launched once via docker and once via
          local yields the same deterministic id) is counted once, matching
          the workload dedup above (``used_slots == sum(ranks_on_host)`` for
          both producing executors);
        - ``free_slots`` recomputes as ``max(capacity - used_slots, 0)`` where
          ``capacity`` is this snapshot's ``total_slots`` (both snapshots share
          host hardware, so capacity matches).

        A host present in only one snapshot is carried over unchanged.  This
        snapshot's ``queried_at`` and ``executor`` name are kept (the merged
        snapshot reports as the primary executor to keep single-name consumers
        working).
        """
        other_by_host = {entry.host: entry for entry in other.hosts}
        seen_hosts: set[str] = set()
        merged: list[HostOccupancy] = []
        for entry in self.hosts:
            seen_hosts.add(entry.host)
            peer = other_by_host.get(entry.host)
            if peer is None:
                merged.append(entry)
                continue
            workloads = list(entry.workloads)
            index_by_id = {w.cluster_id: i for i, w in enumerate(workloads)}
            # Add the peer's slots only for workloads not already counted in
            # this snapshot, so a shared cluster_id isn't double-counted.  For a
            # shared cluster_id the primary workload is kept but its
            # ``containers`` absorb the peer's disjoint realizations (docker +
            # local), deduped by name — a shared workload can have both a Docker
            # container and a native process.
            peer_added_slots = 0
            for w in peer.workloads:
                idx = index_by_id.get(w.cluster_id)
                if idx is None:
                    index_by_id[w.cluster_id] = len(workloads)
                    workloads.append(w)
                    peer_added_slots += w.ranks_on_host
                elif w.containers:
                    existing = workloads[idx]
                    merged_containers = _union_containers(existing.containers, w.containers)
                    if merged_containers != existing.containers:
                        workloads[idx] = replace(existing, containers=merged_containers)
            used = entry.used_slots + peer_added_slots
            capacity = entry.total_slots  # shared host hardware → == peer.total_slots
            merged.append(
                HostOccupancy(
                    host=entry.host,
                    workloads=tuple(workloads),
                    used_slots=used,
                    free_slots=max(capacity - used, 0),
                    free_memory_gb=entry.free_memory_gb,
                    gpus=entry.gpus,
                )
            )
        for entry in other.hosts:
            if entry.host not in seen_hosts:
                merged.append(entry)
        # Errors are the union of both snapshots (this snapshot wins on
        # collision), minus any host that is reachable via at least one
        # executor — a host present in the merged ``hosts`` is not an error.
        reachable = {entry.host for entry in merged}
        merged_errors: dict[str, str] = {}
        for host, msg in list(self.errors.items()) + list(other.errors.items()):
            if host in reachable or host in merged_errors:
                continue
            merged_errors[host] = msg
        return ClusterStatus(hosts=tuple(merged), queried_at=self.queried_at, executor=self.executor, errors=merged_errors)


def _union_containers(
    primary: tuple[ContainerDetail, ...],
    peer: tuple[ContainerDetail, ...],
) -> tuple[ContainerDetail, ...]:
    """Concatenate two container lists, deduped by ``name`` (primary wins)."""
    seen = {c.name for c in primary}
    extra = [c for c in peer if c.name not in seen]
    if not extra:
        return primary
    return primary + tuple(extra)


def empty_status(hosts: list[str], executor: str = "") -> ClusterStatus:
    """Build a zero-occupancy snapshot — every host fully free, no workloads.

    Used as a safe default by executors that don't implement
    introspection (and as a sentinel in tests).
    """
    return ClusterStatus(
        hosts=tuple(HostOccupancy(host=h) for h in hosts),
        queried_at=0.0,
        executor=executor,
    )
