"""Docker command string generation.

These functions are pure generators -- they produce command strings
that will be embedded into scripts and executed remotely via SSH.
They do not execute Docker commands directly.
"""

from __future__ import annotations

import logging

# The teardown reporting protocol is substrate-agnostic and lives in
# ``orchestration.teardown``, but it was born here and callers/tests import it
# from this module — re-exported so those imports keep resolving.
from sparkrun.orchestration.teardown import TEARDOWN_REMOVED_MARKER
from sparkrun.orchestration.teardown import parse_teardown_removed as parse_teardown_removed
from sparkrun.utils.shell import b64_wrap_bash, quote

logger = logging.getLogger(__name__)


def docker_exec_cmd(
    container_name: str,
    command: str,
    detach: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """Generate a ``docker exec`` command string.

    Args:
        container_name: Name of the running container.
        command: Command to execute inside the container.
        detach: Run in detached mode.
        env: Environment variables to set.

    Returns:
        Complete ``docker exec`` command string.
    """
    parts = ["docker", "exec"]
    if detach:
        parts.append("-d")
    if env:
        for key, value in sorted(env.items()):
            parts.extend(["-e", quote(f"{key}={value}")])

    parts.extend([quote(container_name), "bash", "-c", b64_wrap_bash(command)])
    return " ".join(parts)


def docker_stop_cmd(container_name: str, force: bool = True) -> str:
    """Generate a docker stop/rm command string.

    Args:
        container_name: Name of the container to stop.
        force: If True, use ``docker rm -f``; otherwise ``docker stop``.

    Returns:
        Command string that stops (and optionally removes) the container.
    """
    quoted = quote(container_name)
    if force:
        return "docker rm -f %s 2>/dev/null || true" % quoted
    return "docker stop %s 2>/dev/null || true" % quoted


def docker_teardown_script(container_names: list[str] | tuple[str, ...]) -> str:
    """Generate a teardown script that removes containers *and verifies it*.

    Docker's implementation of the substrate-agnostic teardown protocol (see
    :mod:`sparkrun.orchestration.teardown`), reached via
    :meth:`~sparkrun.orchestration.executors.docker.DockerExecutor.teardown_script`.

    :func:`docker_stop_cmd` ends in ``|| true`` so a candidate name that
    was never running doesn't fail the chain -- which also means a *real*
    docker failure (daemon down, permission denied, no docker binary)
    exits 0 and reads as a successful teardown.  Callers that need a
    truthful answer use this instead: a census, the removals, then a
    verification pass.  It

    - fails if docker could not be queried before or after,
    - fails, listing them, if any target container survived, and
    - prints ``sparkrun_removed=<n>`` counting the containers that were
      **present when we started**, not the removals attempted.

    That last distinction is the reason for the leading census.  ``docker rm
    -f`` exits **0 for a container that does not exist** (verified on Docker
    28 and on Thunder's custom daemon), so counting successful removals
    counted every candidate name -- and ``enumerate_containers`` hands this
    the solo / head / worker / node_N shapes for every host, of which at most
    one is real.  ``sparkrun stop`` reported "removed 4 containers" for a
    single-container job, and reported removals on hosts holding nothing.

    Args:
        container_names: Candidate container names to remove.  Names that
            aren't present are not failures -- teardown is idempotent.

    Returns:
        Bash script content; a no-op script when *container_names* is empty.
    """
    if not container_names:
        return 'echo "' + TEARDOWN_REMOVED_MARKER + '0"\n'

    # ``docker ps -a`` (not ``docker ps``) throughout: a container that has
    # exited but not been removed is still present, and ``rm -f`` must leave
    # nothing behind.
    patterns = " ".join("-e " + quote(name) for name in container_names)
    lines = [
        "if ! _sr_before=$(docker ps -a --format '{{.Names}}' 2>&1); then",
        '  echo "docker ps failed: $_sr_before" >&2',
        "  exit 1",
        "fi",
        "_sr_removed=$(printf '%s\\n' \"$_sr_before\" | grep -Fxc " + patterns + " || true)",
    ]
    for name in container_names:
        lines.append("docker rm -f " + quote(name) + " >/dev/null 2>&1 || true")

    lines.extend(
        [
            "if ! _sr_all=$(docker ps -a --format '{{.Names}}' 2>&1); then",
            '  echo "docker ps failed: $_sr_all" >&2',
            "  exit 1",
            "fi",
            "_sr_left=$(printf '%s\\n' \"$_sr_all\" | grep -Fx " + patterns + " || true)",
            'if [ -n "$_sr_left" ]; then',
            "  echo \"containers still present: $(printf '%s' \"$_sr_left\" | tr '\\n' ' ')\" >&2",
            "  exit 1",
            "fi",
            'echo "' + TEARDOWN_REMOVED_MARKER + '${_sr_removed:-0}"',
        ]
    )
    return "\n".join(lines) + "\n"


def docker_inspect_exists_cmd(image: str) -> str:
    """Generate a command to check if a docker image exists locally.

    Args:
        image: Image reference to check.

    Returns:
        Command string that exits 0 if the image exists locally.
    """
    return "docker image inspect %s >/dev/null 2>&1" % quote(image)


def docker_pull_cmd(image: str) -> str:
    """Generate a ``docker pull`` command.

    Args:
        image: Image reference to pull.

    Returns:
        Command string.
    """
    return "docker pull %s" % quote(image)


def docker_logs_cmd(
    container_name: str,
    follow: bool = False,
    tail: int | None = None,
) -> str:
    """Generate a ``docker logs`` command.

    Args:
        container_name: Name of the container.
        follow: If True, follow log output (``-f``).
        tail: Number of lines to show from the end.

    Returns:
        Command string.
    """
    parts = ["docker", "logs"]
    if follow:
        parts.append("-f")
    if tail is not None:
        parts.extend(["--tail", str(tail)])
    parts.append(quote(container_name))
    return " ".join(parts)


def generate_container_name(cluster_id: str, role: str = "head") -> str:
    """Generate a deterministic container name.

    Args:
        cluster_id: Cluster identifier (e.g. ``sparkrun0``).
        role: Container role -- ``"head"``, ``"worker"``, or ``"solo"``.

    Returns:
        Container name in the form ``{cluster_id}_{role}``.
    """
    return f"{cluster_id}_{role}"


def generate_node_container_name(cluster_id: str, rank: int) -> str:
    """Generate a container name for a ranked node: ``{cluster_id}_node_{rank}``.

    Used by native-cluster runtimes (SGLang, vllm-distributed) where
    each node gets a rank-indexed container name.

    Args:
        cluster_id: Cluster identifier (e.g. ``sparkrun0``).
        rank: Node rank (0 = head, 1+ = workers).

    Returns:
        Container name string.
    """
    return "%s_node_%d" % (cluster_id, rank)


def enumerate_cluster_containers(cluster_id: str, num_hosts: int) -> list[str]:
    """Return all possible container names for a cluster.

    Covers solo, Ray (head/worker), and native (node_N) patterns so
    callers can clean up containers regardless of the runtime that
    created them.

    Args:
        cluster_id: Cluster identifier (e.g. ``sparkrun0``).
        num_hosts: Number of hosts in the cluster (used to generate
            native ``node_N`` names).

    Returns:
        List of container name strings.
    """
    names = [
        generate_container_name(cluster_id, "solo"),
        generate_container_name(cluster_id, "head"),
        generate_container_name(cluster_id, "worker"),
    ]
    for rank in range(num_hosts):
        names.append(generate_node_container_name(cluster_id, rank))
    return names
