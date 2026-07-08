"""NCCL rendezvous env for k8s JobSet workloads (TCP tier).

The first k8s multi-node cut uses pod-network TCP NCCL — portable, no
special cluster infra.  JobSet provisions a headless Service, so rank-0's
pod is reachable at a stable DNS name that every rank uses as
``MASTER_ADDR``.  Per-pod ``RANK`` is derived *inside* the pod from the
JobSet job-index (injected via the downward API) plus the podset's base
rank offset — see :func:`rank_prelude`.

RDMA/IB (the higher-perf tier) is step 6; this module is the TCP baseline.
"""

from __future__ import annotations

DEFAULT_MASTER_PORT = 29500
JOBSET_JOB_INDEX_ANNOTATION = "jobset.sigs.k8s.io/job-index"
JOB_INDEX_ENV = "SPARKRUN_JOB_INDEX"
BASE_RANK_ENV = "SPARKRUN_BASE_RANK"

# RDMA tier (opt-in; requires the NVIDIA Network Operator / RDMA shared
# device plugin — sparkrun consumes it, does not install it).
DEFAULT_RDMA_RESOURCE = "rdma/rdma_shared_device_a"
DEFAULT_RDMA_HCA = "mlx5_0"
RDMA_CAPABILITY = "IPC_LOCK"


def headless_pod_dns(jobset: str, replicated_job: str, *, job_index: int = 0, pod_index: int = 0) -> str:
    """Stable DNS name of a JobSet pod.

    JobSet names pods ``<jobset>-<replicatedJob>-<jobIndex>-<podIndex>`` and
    exposes them through a headless Service named after the JobSet, so the
    FQDN (within the namespace) is ``<pod>.<jobset>``.
    """
    return "%s-%s-%d-%d.%s" % (jobset, replicated_job, job_index, pod_index, jobset)


def master_addr(jobset: str, head_replicated_job: str) -> str:
    """Rendezvous address = pod 0 of job 0 of the head podset."""
    return headless_pod_dns(jobset, head_replicated_job, job_index=0, pod_index=0)


def base_tcp_nccl_env(
    world_size: int,
    master_address: str,
    *,
    master_port: int = DEFAULT_MASTER_PORT,
    socket_ifname: str = "eth0",
) -> dict[str, str]:
    """Cluster-wide TCP NCCL env shared by every rank.

    ``NCCL_IB_DISABLE=1`` forces the sockets transport — correct for the
    portable TCP tier (RDMA is opt-in later).
    """
    return {
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": master_address,
        "MASTER_PORT": str(master_port),
        "NCCL_SOCKET_IFNAME": socket_ifname,
        "NCCL_IB_DISABLE": "1",
    }


def base_rdma_nccl_env(
    world_size: int,
    master_address: str,
    *,
    master_port: int = DEFAULT_MASTER_PORT,
    socket_ifname: str = "eth0",
    hca: str = DEFAULT_RDMA_HCA,
) -> dict[str, str]:
    """Cluster-wide RDMA/IB NCCL env (opt-in tier).

    Leaves IB enabled (no ``NCCL_IB_DISABLE``) and pins ``NCCL_IB_HCA`` to
    the injected device; ``NCCL_SOCKET_IFNAME`` still seeds the bootstrap
    rendezvous.  Pods must also request the RDMA resource + ``IPC_LOCK``
    (added by the launch layer).
    """
    return {
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": master_address,
        "MASTER_PORT": str(master_port),
        "NCCL_SOCKET_IFNAME": socket_ifname,
        "NCCL_IB_HCA": hca,
    }


def rdma_available(node_labels: dict[str, str], allocatable: dict[str, str] | None = None) -> bool:
    """Heuristic: does this node expose RDMA (Mellanox NIC + rdma resource)?

    True when NFD reports the Mellanox/NVIDIA-Networking PCI vendor (15b3)
    *and* an ``rdma/*`` extended resource is allocatable (the shared device
    plugin is running).
    """
    mellanox = str(node_labels.get("feature.node.kubernetes.io/pci-15b3.present", "")).lower() == "true"
    has_rdma_resource = any(str(k).startswith("rdma/") for k in (allocatable or {}))
    return mellanox and has_rdma_resource


def field_ref_env() -> dict[str, str]:
    """Downward-API env mapping — exposes the JobSet job-index to the pod."""
    return {JOB_INDEX_ENV: "metadata.annotations['%s']" % JOBSET_JOB_INDEX_ANNOTATION}


def rank_prelude(base_rank: int) -> str:
    """Bash prelude computing this pod's global RANK before the serve command.

    ``RANK = base_rank + job_index`` — ``base_rank`` is the podset's offset
    into the global rank space (baked per podset) and ``job_index`` comes
    from the JobSet annotation via :func:`field_ref_env`.  Assumes one GPU
    per rank/pod (LOCAL_RANK=0), which matches single-GPU-node placement;
    multi-GPU-per-pod runtimes override LOCAL_RANK themselves.
    """
    return (
        'export %(job_idx)s="${%(job_idx)s:-0}"\n'
        "export RANK=$(( %(base)d + %(job_idx)s ))\n"
        'export NODE_RANK="$RANK"\n'
        'export LOCAL_RANK="${LOCAL_RANK:-0}"\n'
    ) % {"job_idx": JOB_INDEX_ENV, "base": base_rank}


__all__ = [
    "DEFAULT_MASTER_PORT",
    "JOBSET_JOB_INDEX_ANNOTATION",
    "JOB_INDEX_ENV",
    "DEFAULT_RDMA_RESOURCE",
    "DEFAULT_RDMA_HCA",
    "RDMA_CAPABILITY",
    "headless_pod_dns",
    "master_addr",
    "base_tcp_nccl_env",
    "base_rdma_nccl_env",
    "rdma_available",
    "field_ref_env",
    "rank_prelude",
]
