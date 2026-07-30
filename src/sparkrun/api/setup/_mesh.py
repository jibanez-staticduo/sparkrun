"""Bash-free host-to-host SSH key mesh.

``scripts/mesh_ssh_keys.sh`` needs a POSIX shell on the *control* machine,
which a Windows control machine does not have.  This module reproduces its
core — every host able to SSH to every other host as the same user — using
only the generic ``ssh <host> bash -s`` machinery, so all the shell work
happens on the (always-Linux) cluster hosts.

It is deliberately narrower than the bash script: it assumes control→host key
auth already works, which :mod:`sparkrun.api.setup._ssh_access` establishes
first.  Distributing ``known_hosts`` entries stays where it was, in
:func:`sparkrun.orchestration.networking.distribute_host_keys`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_HEREDOC_MARKER = "SPARKRUN_MESH_KEYS"

#: Emitted on each host so we can pick the key out of any shell noise.
_KEY_MARKER = "SPARKRUN_PUBKEY "


@dataclass(frozen=True)
class MeshResult:
    """Outcome of a native key mesh."""

    #: host -> its public key, for every host we could read.
    public_keys: dict[str, str] = field(default_factory=dict)
    #: Hosts whose key could not be read or generated.
    collect_failures: dict[str, str] = field(default_factory=dict)
    #: Hosts the collected keys could not be installed on.
    install_failures: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.collect_failures and not self.install_failures


def build_collect_key_script() -> str:
    """Return a script that ensures a host has an ed25519 key and prints it."""
    return (
        "set -eu\n"
        'mkdir -p "$HOME/.ssh"\n'
        'chmod 700 "$HOME/.ssh"\n'
        'if [ ! -f "$HOME/.ssh/id_ed25519" ]; then\n'
        '  ssh-keygen -t ed25519 -N "" -C "$(whoami)@$(hostname)" -f "$HOME/.ssh/id_ed25519" >/dev/null 2>&1\n'
        "fi\n"
        'printf \'%s%%s\\n\' "$(cat "$HOME/.ssh/id_ed25519.pub")"\n' % _KEY_MARKER
    )


def build_install_keys_script(public_keys: list[str]) -> str:
    """Return a script that idempotently authorizes every key in *public_keys*.

    Keys are carried in a quoted heredoc rather than argv, so no local shell
    quoting is involved and the payload is identical on every platform.

    Raises:
        ValueError: A key spans multiple lines or collides with the heredoc
            terminator.
    """
    cleaned: list[str] = []
    for key in public_keys:
        stripped = key.strip()
        if not stripped:
            continue
        if "\n" in stripped or "\r" in stripped:
            raise ValueError("public key must be a single line")
        if stripped == _HEREDOC_MARKER:
            raise ValueError("public key collides with the heredoc terminator")
        cleaned.append(stripped)

    if not cleaned:
        raise ValueError("no public keys to install")

    return (
        "set -eu\n"
        'mkdir -p "$HOME/.ssh"\n'
        'chmod 700 "$HOME/.ssh"\n'
        'touch "$HOME/.ssh/authorized_keys"\n'
        'chmod 600 "$HOME/.ssh/authorized_keys"\n'
        "installed=0\n"
        "while IFS= read -r k; do\n"
        '  [ -n "$k" ] || continue\n'
        '  if ! grep -qxF "$k" "$HOME/.ssh/authorized_keys"; then\n'
        '    printf \'%%s\\n\' "$k" >> "$HOME/.ssh/authorized_keys"\n'
        "    installed=$((installed + 1))\n"
        "  fi\n"
        "done <<'%s'\n"
        "%s\n"
        "%s\n"
        'echo "sparkrun: authorized $installed new key(s)"\n'
        % (
            _HEREDOC_MARKER,
            "\n".join(cleaned),
            _HEREDOC_MARKER,
        )
    )


def _extract_key(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith(_KEY_MARKER):
            key = line[len(_KEY_MARKER) :].strip()
            if key:
                return key
    return None


def mesh_ssh_keys_native(
    hosts: list[str],
    user: str | None,
    *,
    key: str | None = None,
    options: list[str] | None = None,
    timeout: int = 60,
    dry_run: bool = False,
) -> MeshResult:
    """Mesh SSH keys across *hosts* without needing a local shell.

    Every host gets an ed25519 key if it lacks one, then every host's key is
    authorized on every host (including itself, so ``ssh localhost`` works for
    runtimes that shell back into the node).

    Requires working control→host key auth; run
    :func:`sparkrun.api.setup.probe_ssh_access` first.

    Args:
        hosts: Cluster hosts to mesh.
        user: SSH username, the same on every host.
        key: Optional control-machine private key to authenticate with.
        options: Extra ``ssh`` options.
        timeout: Per-host timeout in seconds.
        dry_run: Log the plan without touching any host.

    Returns:
        A :class:`MeshResult` naming exactly which hosts failed and why.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if not hosts:
        return MeshResult()

    ssh_kwargs = {
        "ssh_user": user,
        "ssh_key": key,
        "ssh_options": list(options or []),
        "timeout": timeout,
    }

    if dry_run:
        logger.info("[dry-run] Would mesh SSH keys across %d host(s): %s", len(hosts), ", ".join(hosts))
        return MeshResult(public_keys={h: "[dry-run]" for h in hosts})

    logger.info("Collecting host keys from %d host(s)", len(hosts))
    collected = run_remote_scripts_parallel(list(hosts), build_collect_key_script(), quiet=True, **ssh_kwargs)

    public_keys: dict[str, str] = {}
    collect_failures: dict[str, str] = {}
    for r in collected:
        if not r.success:
            collect_failures[r.host] = (r.stderr or "").strip() or "ssh exited %d" % r.returncode
            continue
        extracted = _extract_key(r.stdout)
        if extracted is None:
            collect_failures[r.host] = "no public key in output"
        else:
            public_keys[r.host] = extracted

    if not public_keys:
        return MeshResult(collect_failures=collect_failures)

    # Deduplicate while keeping a stable order (hosts can share a key when the
    # home directory is on shared storage).
    ordered: list[str] = []
    seen: set[str] = set()
    for host in hosts:
        k = public_keys.get(host)
        if k and k not in seen:
            ordered.append(k)
            seen.add(k)

    targets = [h for h in hosts if h not in collect_failures]
    logger.info("Authorizing %d key(s) on %d host(s)", len(ordered), len(targets))
    installed = run_remote_scripts_parallel(targets, build_install_keys_script(ordered), quiet=True, **ssh_kwargs)

    install_failures = {r.host: ((r.stderr or "").strip() or "ssh exited %d" % r.returncode) for r in installed if not r.success}

    return MeshResult(
        public_keys=public_keys,
        collect_failures=collect_failures,
        install_failures=install_failures,
    )


__all__ = [
    "MeshResult",
    "build_collect_key_script",
    "build_install_keys_script",
    "mesh_ssh_keys_native",
]
