"""Materialize Thunder instances as plain SSH hosts.

Writes a **sparkrun-managed** ssh_config file (``<config-root>/ssh/thunder.conf``)
containing one ``Host tnr-<uuid>`` alias per instance, and ensures a single
``Include`` line points ``~/.ssh/config`` at it.  This keeps all per-instance
port/key variance inside an ssh alias so the rest of sparkrun treats each host
as an opaque name — no changes to ``orchestration.ssh`` needed.

Private keys are cached under Thunder's own key dir (``~/.thunder/keys/<uuid>``)
so we share tnr's cache rather than duplicating it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from sparkrun.transports.thunder import api as thunder_api
from sparkrun.transports.thunder.api import ThunderInstance

logger = logging.getLogger(__name__)

ALIAS_PREFIX = "tnr-"

# Thunder instances always expose the ``ubuntu`` login user. Kept as a constant
# so the ssh alias block and the imported cluster's ``user`` field can't drift.
THUNDER_SSH_USER = "ubuntu"

# Executor overrides for Thunder-imported clusters. Thunder runs a custom
# proot/fastvfs docker with two constraints sparkrun's rootless defaults trip on:
#   * ``user: root`` — proot cannot run containers as a non-root user (the
#     ``--user $(id -u):$(id -g)`` from auto_user dies with
#     "proot ... can't sanitize binding ... Permission denied"). Forcing root
#     also drops the /etc/passwd + /etc/group bind mounts (gated on $SHELL_USER).
#   * ``ulimit: [stack=...]`` — zero-capability containers can't raise
#     RLIMIT_MEMLOCK, so the rootless ``memlock=-1:-1`` fails at container start
#     ("error setting rlimit type 8: operation not permitted"); drop it, keep stack.
# Both use replace semantics and outrank the rootless adjustments in the chain.
THUNDER_EXECUTOR_CONFIG = {"user": "root", "ulimit": ["stack=67108864"]}


def alias_for(inst: ThunderInstance) -> str:
    """Stable ssh alias for *inst* — ``tnr-<uuid>``.

    Uses the stable uuid (not the positional ``id``) so the alias — and thus the
    cluster's stored host — survives instance-id reassignment.  Distinct from
    tnr's own ``tnr-<id>`` blocks, so our managed file never collides with them.
    """
    return ALIAS_PREFIX + inst.uuid


def _thunder_dir() -> Path:
    explicit = os.environ.get("TNR_HOME")
    return Path(explicit) if explicit else Path.home() / ".thunder"


def key_path(uuid: str) -> Path:
    """Path to the cached private key for *uuid* (shared with tnr)."""
    return _thunder_dir() / "keys" / uuid


def _managed_conf_path() -> Path:
    from sparkrun.core.config import get_config_root

    return Path(get_config_root()) / "ssh" / "thunder.conf"


def ensure_key(token: str, base: str, inst: ThunderInstance, *, force: bool = False) -> Path:
    """Ensure a private key exists for *inst*; provision via ``add_key`` if not.

    Returns the key path.  When *force* is True (rotation / connection failure),
    re-provisions unconditionally.  Written 0600.
    """
    path = key_path(inst.uuid)
    if path.exists() and not force:
        return path
    pem = thunder_api.add_key(token, base, inst.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write restrictively from the start (avoid a world-readable window).
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, pem.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    logger.debug("Provisioned Thunder key for %s -> %s", inst.uuid, path)
    return path


def _render_block(inst: ThunderInstance, key: Path) -> str:
    """Render one ``Host tnr-<uuid>`` ssh_config block."""
    return "\n".join(
        [
            "Host %s" % alias_for(inst),
            "    HostName %s" % (inst.ip or ""),
            "    User %s" % THUNDER_SSH_USER,
            '    IdentityFile "%s"' % key,
            "    IdentitiesOnly yes",
            "    IdentityAgent none",
            "    StrictHostKeyChecking no",
            "    Port %d" % inst.port,
        ]
    )


_HEADER = "# Managed by sparkrun — do not edit.\n# Thunder Compute instance ssh aliases (see `sparkrun cluster import thunder`).\n"


def _parse_blocks(text: str) -> dict[str, str]:
    """Parse managed thunder.conf into ``{alias: block-text}`` keyed by Host name.

    Comment/blank lines before the first ``Host`` are dropped (regenerated as the
    header).  Preserves each ``Host`` block verbatim so unrelated aliases survive.
    """
    blocks: dict[str, str] = {}
    current_alias: str | None = None
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Host "):
            if current_alias is not None:
                blocks[current_alias] = "\n".join(current).rstrip()
            current_alias = stripped[len("Host ") :].strip()
            current = [line]
        elif current_alias is not None:
            current.append(line)
    if current_alias is not None:
        blocks[current_alias] = "\n".join(current).rstrip()
    return blocks


def write_aliases(entries: list[tuple[ThunderInstance, Path]]) -> dict[str, str]:
    """Upsert *entries* into the managed ``thunder.conf`` and ensure the Include.

    Existing aliases for **other** instances are preserved (upsert by alias), so a
    per-host refresh at run time never clobbers other Thunder clusters.  Stale
    aliases for deleted instances are cleaned up separately (cluster delete /
    ``import thunder --prune``).  Returns ``{uuid: alias}`` for *entries*.
    """
    conf = _managed_conf_path()
    conf.parent.mkdir(parents=True, exist_ok=True)

    blocks: dict[str, str] = {}
    if conf.exists():
        blocks = _parse_blocks(conf.read_text())
    for inst, key in entries:
        blocks[alias_for(inst)] = _render_block(inst, key)

    ordered = [blocks[a] for a in sorted(blocks)]
    content = _HEADER + "\n" + "\n\n".join(ordered) + ("\n" if ordered else "")
    fd = os.open(str(conf), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(conf, 0o600)

    ensure_include(conf)
    return {inst.uuid: alias_for(inst) for inst, _ in entries}


def remove_alias(uuid: str) -> None:
    """Drop the ``tnr-<uuid>`` block from the managed thunder.conf, if present."""
    conf = _managed_conf_path()
    if not conf.exists():
        return
    blocks = _parse_blocks(conf.read_text())
    if blocks.pop(ALIAS_PREFIX + uuid, None) is None:
        return
    ordered = [blocks[a] for a in sorted(blocks)]
    content = _HEADER + "\n" + "\n\n".join(ordered) + ("\n" if ordered else "")
    fd = os.open(str(conf), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(conf, 0o600)


def ensure_include(conf: Path | None = None) -> None:
    """Ensure ``~/.ssh/config`` includes the managed thunder.conf (once, at top).

    Uses an absolute path (ssh does not reliably expand ``~`` in ``Include``).
    Idempotent — does nothing if the line is already present.
    """
    if conf is None:
        conf = _managed_conf_path()
    include_line = "Include %s" % conf
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    ssh_config = ssh_dir / "config"

    existing = ""
    if ssh_config.exists():
        existing = ssh_config.read_text()
        if include_line in existing:
            return

    # Prepend so host-specific settings win (ssh takes the first value).
    marker = "# sparkrun-managed transport includes\n"
    new_content = marker + include_line + "\n"
    if existing:
        new_content += "\n" + existing
    fd = os.open(str(ssh_config), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, new_content.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(ssh_config, 0o600)
    logger.debug("Added Include %s to %s", conf, ssh_config)
