"""Privileged fallback hardware probe for clusters without GFD/NFD labels.

The label-based inventory (:mod:`.inventory`) is the happy path and needs
no privilege.  When a cluster has no NVIDIA GPU Operator / Node Feature
Discovery — so the ``nvidia.com/gpu.*`` labels are absent — this module
runs sparkrun's *existing* combined probe script (nvidia-smi + ibstat,
:func:`sparkrun.core.hardware_probe.generate_combined_probe_script`) inside
a **privileged** per-node Job and parses the pod's stdout with the *same*
parser the SSH path uses.  Same probe, same parser — only the transport
changes (pod logs instead of ``ssh``).

This is a distinct, higher privilege tier: the probe pod is ``privileged``
with ``hostPID`` and a ``/dev`` hostPath so it can reach the GPUs/IB
devices, and the namespace must permit privileged PodSecurity.  It is
opt-in — prefer installing the GPU Operator.
"""

from __future__ import annotations

import re

from sparkrun.core.fingerprint import build_host_hardware, parse_fingerprint_output
from sparkrun.core.hardware import HostHardware
from sparkrun.core.hardware_probe import generate_combined_probe_script, split_probe_output
from sparkrun.utils import parse_kv_output

from .client import KubectlClient
from .manifests import MANAGED_BY_LABEL, render_manifests

PROBE_NAME_PREFIX = "sparkrun-probe"
_TTL_SECONDS = 300
_WAIT_TIMEOUT = 120


def _safe_name(node: str) -> str:
    """RFC1123-ify a node name for use in a Job name."""
    slug = re.sub(r"[^a-z0-9-]+", "-", node.lower()).strip("-")
    return "%s-%s" % (PROBE_NAME_PREFIX, slug or "node")


def probe_job_manifest(node_name: str, *, namespace: str, image: str) -> dict:
    """A privileged Job pinned to *node_name* that prints the probe output."""
    labels = dict(MANAGED_BY_LABEL)
    labels["app.kubernetes.io/name"] = PROBE_NAME_PREFIX
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": _safe_name(node_name), "namespace": namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": _TTL_SECONDS,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "nodeName": node_name,
                    "restartPolicy": "Never",
                    "hostPID": True,
                    "containers": [
                        {
                            "name": "probe",
                            "image": image,
                            "command": ["bash", "-c", generate_combined_probe_script()],
                            "securityContext": {"privileged": True},
                            "volumeMounts": [{"name": "dev", "mountPath": "/dev"}],
                        }
                    ],
                    "volumes": [{"name": "dev", "hostPath": {"path": "/dev"}}],
                },
            },
        },
    }


def parse_probe_output(stdout: str) -> HostHardware:
    """Parse probe-pod stdout into :class:`HostHardware` (SSH-path parser)."""
    accel_section, ib_section = split_probe_output(stdout)
    if accel_section:
        hardware = build_host_hardware(parse_fingerprint_output(accel_section))
    else:
        hardware = HostHardware(notes="k8s probe: accelerator section missing")
    hardware.ib_info = parse_kv_output(ib_section) if ib_section else None
    return hardware


def probe_nodes_fallback(
    client: KubectlClient,
    node_names: list[str],
    *,
    image: str,
    namespace: str = "sparkrun",
    timeout: int = _WAIT_TIMEOUT,
) -> dict[str, HostHardware]:
    """Run the privileged probe on each node and collect the results.

    Best-effort: a node whose probe Job fails or produces no logs is
    omitted from the result (a warning-level gap, not a hard error).  Each
    Job is deleted after its logs are read.
    """
    results: dict[str, HostHardware] = {}
    for node in node_names:
        job = _safe_name(node)
        apply_res = client.apply(render_manifests([probe_job_manifest(node, namespace=namespace, image=image)]))
        if not apply_res.success:
            continue
        client.wait_for_job(job, condition="complete", timeout=timeout)
        logs = client.run(["logs", "job/%s" % job])
        if logs.success and logs.stdout.strip():
            results[node] = parse_probe_output(logs.stdout)
        client.run(["delete", "job", job, "--ignore-not-found"])
    return results


__all__ = [
    "PROBE_NAME_PREFIX",
    "probe_job_manifest",
    "parse_probe_output",
    "probe_nodes_fallback",
]
