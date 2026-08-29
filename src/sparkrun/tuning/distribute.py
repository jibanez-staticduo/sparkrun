"""Distribute tuning configs from local cache to remote hosts."""

from __future__ import annotations

import logging
import os

from sparkrun.utils import is_local_host
from sparkrun.tuning.sync import _get_local_tuning_dir, _get_remote_tuning_dir

logger = logging.getLogger(__name__)


def distribute_tuning_to_hosts(
    runtime: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
    transfer_mode: str = "local",
    preserve_perms: bool = True,
    skip_fan_out: bool = False,
) -> list[str]:
    """Distribute local tuning configs to remote hosts via rsync.

    Pushes the local tuning config directory (populated by
    :func:`sparkrun.tuning.sync.sync_registry_tuning` or local tuning
    runs) to all remote hosts so that worker nodes have the same
    configs mounted into their containers.

    For ``push`` and ``delegated`` modes, rsyncs to the head node first,
    then runs a distribution rsync from head to workers.  Tuning configs
    are small, so the two-hop overhead is negligible.

    Args:
        runtime: Runtime name (e.g. ``"sglang"``, ``"vllm-ray"``).
        hosts: Target hostnames or IPs.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        dry_run: If True, log what would be done without executing.
        transfer_mode: Distribution strategy (``"local"``, ``"push"``,
            or ``"delegated"``).
        preserve_perms: When ``False``, rsync drops owner/group/perm/time
            preservation (``-r --links`` instead of ``-a``) — the harder
            relaxation for a destination where even the NFS-safe default
            set cannot apply attributes.  Mirrors the model path.
        skip_fan_out: When ``True``, the per-host rsync is skipped entirely
            because the tuning cache is already visible on every node (a
            shared ``$HOME``), so copying it there is redundant.

    Returns:
        List of hostnames where distribution failed (empty = success).
    """
    tuning_dir = _get_local_tuning_dir(runtime)

    # No-op if local tuning directory doesn't exist or has no JSON files
    if not tuning_dir.is_dir() or not any(tuning_dir.rglob("*.json")):
        logger.debug("No local tuning configs for %s, skipping distribution", runtime)
        return []

    # Shared-cache fast path: every host already sees this directory, so the
    # fan-out would copy it onto itself.  Mirrors the model path's skip.
    if skip_fan_out:
        logger.debug("Shared tuning cache: skipping per-host tuning distribution")
        return []

    # Filter out localhost — no need to rsync to self
    remote_hosts = [h for h in hosts if not is_local_host(h)]
    if not remote_hosts:
        logger.debug("No remote hosts for tuning distribution")
        return []

    from sparkrun.orchestration.ssh import NFS_SAFE_ATTR_OPTS, run_rsync_parallel, build_ssh_opts_string, run_remote_script
    from sparkrun.orchestration.transfer import map_transfer_failures

    source = str(tuning_dir)
    remote_dest = _get_remote_tuning_dir(runtime, ssh_user=ssh_user)

    # Tuning configs land in the SSH user's own cache dir, which on a shared
    # /home is routinely owned by a different uid than the one we connect as.
    # Without the NFS-safe relaxation rsync transfers every config and then
    # exits 23 setting times on the destination root.  --mkpath because the
    # per-runtime subdirectory may not exist on a host that has never tuned.
    if preserve_perms:
        tuning_rsync_options = ["-az", "--mkpath", "--partial", *NFS_SAFE_ATTR_OPTS]
    else:
        tuning_rsync_options = ["-rz", "--links", "--mkpath", "--partial"]

    # --delete prunes tuning configs that were removed upstream, but only when
    # we can be sure the destination is a *different* directory.  When the
    # remote path equals the local one (_get_remote_tuning_dir returns exactly
    # that for a matching SSH user on Linux) a shared $HOME makes source and
    # destination the same physical directory, and --delete against your own
    # source is how a sync becomes a deletion.  Pruning is a convenience;
    # not destroying the cache is not.
    if os.path.normpath(source) != os.path.normpath(remote_dest):
        tuning_rsync_options.append("--delete")
    else:
        logger.debug(
            "Tuning source and destination paths are identical (%s); omitting --delete in case $HOME is shared",
            source,
        )

    if transfer_mode in ("push", "delegated") and len(remote_hosts) > 1:
        # Two-hop: rsync to head, then head distributes to workers
        head = remote_hosts[0]
        workers = remote_hosts[1:]

        logger.info(
            "Distributing tuning configs (%s) via %s mode: head=%s, %d worker(s)",
            runtime,
            transfer_mode,
            head,
            len(workers),
        )

        # Step 1: rsync to head
        head_results = run_rsync_parallel(
            source,
            [head],
            remote_dest,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
            rsync_options=tuning_rsync_options,
            dry_run=dry_run,
        )
        head_failed = map_transfer_failures(head_results, [head], [head])
        if head_failed:
            logger.warning("Tuning config push to head failed: %s", head)
            return list(remote_hosts)

        # Step 2: distribute from head to workers
        ssh_opts = build_ssh_opts_string(
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
        )
        user_prefix = "%s@" % ssh_user if ssh_user else ""
        targets_str = " ".join(workers)
        dist_script = (
            "set -euo pipefail\n"
            'SOURCE="{source}"\n'
            "for TARGET in {targets}; do\n"
            '  rsync {attr_flags} -e "ssh {ssh_opts}" '
            '"$SOURCE/" {user_prefix}$TARGET:"$SOURCE/"\n'
            "done\n"
        ).format(
            source=remote_dest,
            targets=targets_str,
            # Never --delete on this hop: it uses $SOURCE as *both* sides, so on
            # a cluster with a shared $HOME the source and destination are one
            # directory and --delete would prune the cache against itself.  The
            # control→head hop above already did the pruning.
            attr_flags=" ".join(o for o in tuning_rsync_options if not o.startswith("--delete")),
            ssh_opts=ssh_opts,
            user_prefix=user_prefix,
        )

        dist_result = run_remote_script(
            head,
            dist_script,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
            timeout=120,
            dry_run=dry_run,
        )
        if not dist_result.success:
            logger.warning("Tuning config distribution from head failed (rc=%d)", dist_result.returncode)
            return list(workers)

        logger.info("Tuning configs distributed via %s mode to all %d host(s)", transfer_mode, len(remote_hosts))
        return []

    # Default (local mode) or single remote host: direct rsync to all
    logger.info(
        "Distributing tuning configs (%s) to %d host(s)",
        runtime,
        len(remote_hosts),
    )

    results = run_rsync_parallel(
        source,
        remote_hosts,
        remote_dest,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        rsync_options=tuning_rsync_options,
        dry_run=dry_run,
    )

    failed = map_transfer_failures(results, remote_hosts, remote_hosts)
    if failed:
        logger.warning("Tuning config distribution failed on hosts: %s", failed)
    else:
        logger.info("Tuning configs distributed to all %d host(s)", len(remote_hosts))

    return failed
