"""Console-free SSH access bootstrap for cluster setup.

Everything in this module works from a bare **Windows** control machine: no
local ``bash``, no third-party SSH library.  The only external binaries used
are ``ssh`` and ``ssh-keygen`` from OpenSSH, which ship with Windows 10+ and
with every Linux/macOS distribution sparkrun targets.

The bootstrap is a three-step loop, each step exposed separately so a CLI
wizard, the desktop app's sidecar, or a future setup-steps registry can drive
it with their own presentation:

1. :func:`probe_ssh_access` — can we already reach every host non-interactively?
2. :func:`ensure_local_key` — find (or generate) the key we would install.
3. :func:`install_public_key_interactive` — install it using password auth.

Step 3 is the one place the "console-free" rule bends: OpenSSH reads the
password from the terminal itself, so the child process inherits the console.
The api layer still writes nothing of its own, and the *result* of the install
is never inferred from the output — callers re-run :func:`probe_ssh_access` to
confirm, which is the only trustworthy signal anyway.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sparkrun.utils.shell import stdin_bytes

from ._errors import OpenSshUnavailable, SshKeyError

logger = logging.getLogger(__name__)

#: Name of the key sparkrun generates when the user has no usable SSH key.
#: Deliberately distinct from ``id_ed25519`` so setup never takes ownership of
#: (or silently reuses the passphrase policy of) a personal identity.
SPARKRUN_KEY_NAME = "sparkrun_ed25519"

#: Candidate existing identities, in the order OpenSSH itself would prefer.
_DEFAULT_KEY_NAMES = ("id_ed25519", "id_ecdsa", "id_rsa")

#: SSH options applied to every setup probe/install.  ``accept-new`` matters:
#: with ``BatchMode=yes`` and the stock ``StrictHostKeyChecking=ask``, a
#: never-before-seen host fails with "Host key verification failed" and would
#: otherwise be misreported as unreachable rather than simply unknown.
_ACCEPT_NEW = ("-o", "StrictHostKeyChecking=accept-new")

_AUTH_FAILURE_MARKERS = (
    "permission denied",
    "too many authentication failures",
    "no supported authentication methods",
    "authentication failed",
)

_HOST_KEY_FAILURE_MARKERS = (
    "host key verification failed",
    "remote host identification has changed",
)


@dataclass(frozen=True)
class SshProbe:
    """Outcome of a single non-interactive SSH reachability check."""

    host: str
    ok: bool
    #: The host answered but rejected our credentials — a bootstrap candidate.
    auth_failed: bool = False
    #: The host answered but its host key is unknown/changed — needs operator
    #: attention (we never silently overwrite a changed key).
    host_key_failed: bool = False
    error: str = ""

    @property
    def reachable(self) -> bool:
        """The host answered on the SSH port, whatever it thought of us."""
        return self.ok or self.auth_failed or self.host_key_failed


@dataclass(frozen=True)
class LocalSshKey:
    """A usable local SSH identity, guaranteed to have a ``.pub`` alongside."""

    path: Path
    public_path: Path
    public_key: str
    #: True when :func:`ensure_local_key` created this key just now.
    generated: bool = False


def _classify(host: str, returncode: int, stderr: str) -> SshProbe:
    """Turn an ``ssh`` exit into a diagnosis, not just a pass/fail."""
    if returncode == 0:
        return SshProbe(host=host, ok=True)

    text = (stderr or "").strip()
    lowered = text.lower()
    if any(marker in lowered for marker in _HOST_KEY_FAILURE_MARKERS):
        return SshProbe(host=host, ok=False, host_key_failed=True, error=text)
    if any(marker in lowered for marker in _AUTH_FAILURE_MARKERS):
        return SshProbe(host=host, ok=False, auth_failed=True, error=text)
    return SshProbe(host=host, ok=False, error=text or "ssh exited %d" % returncode)


def probe_ssh_access(
    hosts: list[str],
    user: str | None,
    *,
    key: str | None = None,
    options: list[str] | None = None,
    timeout: int = 10,
    max_workers: int | None = None,
) -> list[SshProbe]:
    """Check non-interactive (key-based) SSH access to every host.

    Runs ``true`` under ``BatchMode=yes`` so a host that would prompt for a
    password is reported as an auth failure instead of hanging.

    Args:
        hosts: Hostnames or IPs to probe.
        user: SSH username (``None`` uses the ssh client's own default).
        key: Optional path to a private key to offer.
        options: Extra ``ssh`` options from cluster/user config.
        timeout: Per-host timeout in seconds.
        max_workers: Cap on concurrent probes.

    Returns:
        One :class:`SshProbe` per host, in the order *hosts* was given.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if not hosts:
        return []

    results = run_remote_scripts_parallel(
        list(hosts),
        "true",
        ssh_user=user,
        ssh_key=key,
        ssh_options=[*_ACCEPT_NEW, *(options or [])],
        timeout=timeout,
        quiet=True,
        max_workers=max_workers,
    )
    by_host = {r.host: r for r in results}
    probes: list[SshProbe] = []
    for host in hosts:
        r = by_host.get(host)
        if r is None:
            probes.append(SshProbe(host=host, ok=False, error="no result"))
        else:
            probes.append(_classify(host, r.returncode, r.stderr))
    return probes


def _ssh_dir(explicit: Path | str | None = None) -> Path:
    return Path(explicit) if explicit else Path.home() / ".ssh"


def _sparkrun_key_dir(explicit: Path | str | None = None) -> Path:
    """Directory sparkrun keeps its *own* identity in.

    Defaults to ``<config root>/ssh`` rather than ``~/.ssh`` so the key travels
    with sparkrun's config: it can be bind-mounted into a containerized daemon,
    backed up with the rest of the config, and never pollutes the user's
    personal identities.  Callers holding SAF ``Variables`` should pass this
    explicitly — the default resolves the config root without them and so does
    not see a redirected stateful root.
    """
    if explicit:
        return Path(explicit)
    from sparkrun.core.config import get_config_root

    return get_config_root() / "ssh"


def _read_identity(private_path: Path) -> LocalSshKey | None:
    """Return the identity at *private_path* if both halves are present."""
    public_path = private_path.with_name(private_path.name + ".pub")
    if not private_path.is_file() or not public_path.is_file():
        return None
    try:
        public_key = public_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.debug("Could not read public key %s: %s", public_path, e)
        return None
    if not public_key:
        return None
    return LocalSshKey(path=private_path, public_path=public_path, public_key=public_key)


def ensure_local_key(
    *,
    preferred: str | Path | None = None,
    ssh_dir: Path | str | None = None,
    key_dir: Path | str | None = None,
    generate: bool = True,
) -> LocalSshKey:
    """Locate a usable local SSH identity, generating one if needed.

    Preference order: *preferred* (typically ``config.ssh_key``), then the
    stock OpenSSH identities, then a sparkrun-owned key (in *key_dir*, or the
    legacy ``~/.ssh`` location).  Only when none of those exist is a new
    ``sparkrun_ed25519`` created — setup never regenerates or overwrites an
    existing key.

    A generated key lands in *key_dir* (``<config root>/ssh``), not ``~/.ssh``,
    so sparkrun's identity travels with its config — mountable into a
    containerized daemon and separate from the user's personal keys.

    Args:
        preferred: Private-key path to try first.
        ssh_dir: Directory holding the user's stock identities (defaults to
            ``~/.ssh``).  Still searched, so an existing working setup keeps
            using the key it already installed on its hosts.
        key_dir: Directory for sparkrun's own key (defaults to
            ``<config root>/ssh``).
        generate: When False, raise instead of creating a key.

    Returns:
        The resolved :class:`LocalSshKey`.

    Raises:
        SshKeyError: No key was usable and none could be generated.
        OpenSshUnavailable: ``ssh-keygen`` is not on PATH.
    """
    directory = _ssh_dir(ssh_dir)
    own_dir = _sparkrun_key_dir(key_dir)

    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(os.path.expanduser(str(preferred))))
    candidates.extend(directory / name for name in _DEFAULT_KEY_NAMES)
    candidates.append(own_dir / SPARKRUN_KEY_NAME)
    # Keys generated before sparkrun owned its own directory.
    candidates.append(directory / SPARKRUN_KEY_NAME)

    for candidate in candidates:
        found = _read_identity(candidate)
        if found is not None:
            logger.debug("Using existing SSH identity %s", found.path)
            return found

    if not generate:
        raise SshKeyError("No usable SSH key found in %s or %s (looked for: %s)" % (directory, own_dir, ", ".join(_DEFAULT_KEY_NAMES)))

    keygen = shutil.which("ssh-keygen")
    if not keygen:
        raise OpenSshUnavailable(
            "ssh-keygen not found on PATH. Install the OpenSSH client (Windows: Settings > Apps > Optional Features > OpenSSH Client)."
        )

    target = own_dir / SPARKRUN_KEY_NAME
    try:
        own_dir.mkdir(parents=True, exist_ok=True)
        # No-op on Windows, where ACLs govern instead of mode bits.
        os.chmod(own_dir, 0o700)
    except OSError as e:
        logger.debug("Could not prepare %s: %s", own_dir, e)

    comment = "sparkrun@%s" % socket.gethostname()
    proc = subprocess.run(
        [keygen, "-t", "ed25519", "-N", "", "-C", comment, "-f", str(target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise SshKeyError("ssh-keygen failed (rc=%d): %s" % (proc.returncode, (proc.stderr or "").strip()))

    created = _read_identity(target)
    if created is None:
        raise SshKeyError("ssh-keygen reported success but %s.pub is missing" % target)
    logger.info("Generated SSH key %s", target)
    return LocalSshKey(
        path=created.path,
        public_path=created.public_path,
        public_key=created.public_key,
        generated=True,
    )


def build_authorized_key_script(public_key: str) -> str:
    """Return a bash script that idempotently authorizes *public_key*.

    The key travels inside the script (piped over stdin) rather than as an
    argv element, so nothing depends on the *local* platform's command-line
    quoting rules — the difference between working and mangling the key on
    Windows.

    Raises:
        ValueError: The key is not a single safe line.
    """
    key = public_key.strip()
    if not key:
        raise ValueError("public key is empty")
    if "\n" in key or "\r" in key:
        raise ValueError("public key must be a single line")
    if "'" in key:
        raise ValueError("public key contains an unexpected quote character")

    return (
        "set -eu\n"
        'mkdir -p "$HOME/.ssh"\n'
        'chmod 700 "$HOME/.ssh"\n'
        'touch "$HOME/.ssh/authorized_keys"\n'
        'chmod 600 "$HOME/.ssh/authorized_keys"\n'
        "KEY='%s'\n"
        'if grep -qxF "$KEY" "$HOME/.ssh/authorized_keys"; then\n'
        '  echo "sparkrun: key already authorized"\n'
        "else\n"
        '  printf \'%%s\\n\' "$KEY" >> "$HOME/.ssh/authorized_keys"\n'
        '  echo "sparkrun: key installed"\n'
        "fi\n" % key
    )


def install_public_key_interactive(
    host: str,
    user: str | None,
    public_key: str,
    *,
    options: list[str] | None = None,
    connect_timeout: int = 15,
    password_prompts: int = 3,
    dry_run: bool = False,
) -> bool:
    """Install *public_key* into ``~/.ssh/authorized_keys`` on *host*.

    Uses password (or keyboard-interactive) auth, so the caller's terminal is
    inherited and OpenSSH can prompt.  Public-key auth is disabled explicitly:
    otherwise a control machine with several identities can exhaust the
    server's ``MaxAuthTries`` before a password is ever offered.

    Because the console is shared, callers must invoke this **one host at a
    time**.

    Args:
        host: Target hostname or IP.
        user: SSH username.
        public_key: The single-line public key to authorize.
        options: Extra ``ssh`` options.
        connect_timeout: SSH connect timeout in seconds.
        password_prompts: Password retries allowed before ssh gives up.
        dry_run: Log the intent and report success without connecting.

    Returns:
        True when ``ssh`` exited 0.  This is a hint, not proof — confirm with
        :func:`probe_ssh_access`.

    Raises:
        OpenSshUnavailable: ``ssh`` is not on PATH.
        ValueError: *public_key* is not a single safe line.
    """
    script = build_authorized_key_script(public_key)

    if dry_run:
        logger.info("[dry-run] Would install public key on %s as %s", host, user)
        return True

    ssh = shutil.which("ssh")
    if not ssh:
        raise OpenSshUnavailable(
            "ssh not found on PATH. Install the OpenSSH client (Windows: Settings > Apps > Optional Features > OpenSSH Client)."
        )

    cmd = [
        ssh,
        *_ACCEPT_NEW,
        "-o",
        "ConnectTimeout=%d" % connect_timeout,
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "PreferredAuthentications=password,keyboard-interactive",
        "-o",
        "NumberOfPasswordPrompts=%d" % password_prompts,
        *(options or []),
        "%s@%s" % (user, host) if user else host,
        "bash",
        "-s",
    ]
    logger.debug("Installing public key on %s: %s", host, " ".join(cmd))

    # stdout/stderr are deliberately inherited: OpenSSH writes its password
    # prompt and its failure reasons straight to the terminal.
    try:
        proc = subprocess.run(cmd, input=stdin_bytes(script), text=False)
    except OSError as e:
        logger.debug("Public key install on %s failed to launch: %s", host, e)
        return False
    return proc.returncode == 0


__all__ = [
    "SPARKRUN_KEY_NAME",
    "LocalSshKey",
    "SshProbe",
    "build_authorized_key_script",
    "ensure_local_key",
    "install_public_key_interactive",
    "probe_ssh_access",
]
