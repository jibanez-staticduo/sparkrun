"""Host-substrate implementation of the runtime cache lifecycle.

Generates and runs the one script that creates the cache directories, stamps
the last-used marker, and prunes aged sibling trees.  Shared by the docker and
local executors' :meth:`~sparkrun.orchestration.executors._base.Executor.ensure_runtime_cache`,
exactly as :func:`sparkrun.orchestration.ssh.verify_host_paths` is shared by
their :meth:`~sparkrun.orchestration.executors._base.Executor.verify_mount_sources`.

All three steps ride one SSH round-trip.  Folding the ``mkdir`` into the
generated launch script instead would save that round-trip but binds to the
docker path, leaves nowhere for the touch/prune step, and gives provider
executors no seam — see ``.slop/runtime-cache-design.md`` §7.
"""

from __future__ import annotations

import logging

from sparkrun.core.runtime_cache import LAST_USED_MARKER, RuntimeCacheMounts
from sparkrun.orchestration.ssh import run_remote_scripts_parallel
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)

#: Minimum path segments required of a directory before the prune sweep will
#: consider ``rm -rf``-ing its children.  ``/`` and other shallow paths are
#: refused outright regardless of how the caller built them.
_MIN_PRUNE_DEPTH = 3


def _is_prunable_root(path: str) -> bool:
    """Guard the one place this module deletes.

    The prune root is computed, not user-typed, but it feeds ``rm -rf`` — so it
    is validated independently of how it was built.
    """
    if not path.startswith("/") or path.rstrip("/") in ("", "/"):
        return False
    segments = [seg for seg in path.strip("/").split("/") if seg not in ("", ".", "..")]
    if len(segments) != len([seg for seg in path.strip("/").split("/") if seg]):
        return False  # contains . or .. — refuse rather than normalize
    return len(segments) >= _MIN_PRUNE_DEPTH


def generate_runtime_cache_script(mounts: RuntimeCacheMounts) -> str:
    """Build the mkdir + touch + prune script for *mounts*.

    Emits the prune stanza only when there is something to prune: with no key
    components the leaf *is* the family root and has no siblings.
    """
    lines = ["set -u"]

    if mounts.dirs:
        lines.append("mkdir -p %s" % " ".join(quote(d) for d in mounts.dirs))

    marker = "%s/%s" % (mounts.leaf.rstrip("/"), LAST_USED_MARKER)
    lines.append("touch %s 2>/dev/null || true" % quote(marker))

    lines.append(_prune_stanza(mounts))
    return "\n".join(line for line in lines if line) + "\n"


def _prune_stanza(mounts: RuntimeCacheMounts) -> str:
    """The sibling-tree sweep, or ``""`` when pruning cannot apply."""
    if not mounts.prune_enabled:
        return ""

    family_root = mounts.family_root.rstrip("/")
    leaf = mounts.leaf.rstrip("/")
    if leaf == family_root:
        return ""  # unkeyed: the leaf has no siblings
    if not _is_prunable_root(family_root) or not leaf.startswith(family_root + "/"):
        logger.debug("runtime_cache: refusing to prune under %r", family_root)
        return ""

    depth = len([seg for seg in leaf[len(family_root) :].strip("/").split("/") if seg])
    if depth < 1:
        return ""

    # Age by the marker, never by directory mtime: reading a cache does not
    # touch the directory, so an mtime-aged sweep would delete exactly the
    # warm trees it should keep.  A tree with no marker was not created by
    # sparkrun and is left alone.
    return "\n".join(
        (
            "if [ -d %(root)s ]; then" % {"root": quote(family_root)},
            "  find %(root)s -mindepth %(d)d -maxdepth %(d)d -type d -print 2>/dev/null | while IFS= read -r _tree; do"
            % {"root": quote(family_root), "d": depth},
            '    [ "$_tree" = %s ] && continue' % quote(leaf),
            '    _marker="$_tree/%s"' % LAST_USED_MARKER,
            '    [ -f "$_marker" ] || continue',
            '    if [ -n "$(find "$_marker" -maxdepth 0 -mtime +%d -print 2>/dev/null)" ]; then' % int(mounts.prune_max_age_days),
            # No %-format is applied to this element, so the printf directive is
            # written literally (a doubled %% would reach the shell verbatim).
            '      rm -rf -- "$_tree" 2>/dev/null && printf \'runtime-cache: pruned %s\\n\' "$_tree"',
            "    fi",
            "  done",
            "fi",
        )
    )


def generate_runtime_cache_sweep_script(
    root: str,
    *,
    max_age_days: int,
    dry_run: bool = False,
    purge_all: bool = False,
) -> str:
    """Build the manual-sweep script for ``sparkrun setup prune-runtime-cache``.

    Unlike the launch-time sweep (:func:`generate_runtime_cache_script`) this
    has no active leaf to protect and no family scope — it walks every runtime
    family under *root*.  It always reports per-tree sizes, which is the half
    of the answer the launch-time sweep has no reason to compute.

    ``purge_all`` ignores the age cutoff.  Nothing here is protected, which is
    why the command's help says to check ``sparkrun status`` first.
    """
    root = root.rstrip("/")
    if not _is_prunable_root(root):
        raise ValueError("refusing to sweep %r: not a plausible runtime-cache root" % root)

    age_test = "true" if purge_all else '[ -n "$(find "$_marker" -maxdepth 0 -mtime +%d -print 2>/dev/null)" ]' % int(max_age_days)
    action = (
        '      printf \'would-remove\\t%s\\t%s\\n\' "$_size" "$_tree"'
        if dry_run
        else '      rm -rf -- "$_tree" 2>/dev/null && printf \'removed\\t%s\\t%s\\n\' "$_size" "$_tree"'
    )

    return "\n".join(
        (
            "set -u",
            "if [ ! -d %s ]; then exit 0; fi" % quote(root),
            # Depth 2 and 3 cover <family>/<key> and <family>/<image>/<model>;
            # -mindepth 2 keeps the family directory itself out of reach.
            "find %s -mindepth 2 -maxdepth 3 -type d -print 2>/dev/null | while IFS= read -r _tree; do" % quote(root),
            '  _marker="$_tree/%s"' % LAST_USED_MARKER,
            '  [ -f "$_marker" ] || continue',
            '  _size="$(du -sk "$_tree" 2>/dev/null | cut -f1)"',
            "  if %s; then" % age_test,
            action,
            "  fi",
            "done",
        )
    )


def ensure_runtime_cache_on_hosts(
    mounts: RuntimeCacheMounts,
    hosts: list[str],
    ssh_kwargs: dict | None = None,
) -> None:
    """Create, stamp and sweep the runtime cache on every host in *hosts*.

    Best-effort throughout, like :func:`sparkrun.orchestration.ssh.verify_host_paths`
    and :meth:`Executor.query_status`: a host we could not prepare costs a
    recompile, never a launch.  Failures are logged at debug and swallowed.
    """
    if not hosts or mounts is None:
        return

    script = generate_runtime_cache_script(mounts)
    ssh_kwargs = ssh_kwargs or {}

    try:
        results = run_remote_scripts_parallel(
            hosts,
            script,
            ssh_user=ssh_kwargs.get("ssh_user"),
            ssh_key=ssh_kwargs.get("ssh_key"),
            ssh_options=ssh_kwargs.get("ssh_options"),
            timeout=ssh_kwargs.get("timeout", 30),
            quiet=True,
            allow_local=True,
        )
    except Exception:
        logger.debug("runtime_cache: preparation sweep failed; continuing", exc_info=True)
        return

    for r in results:
        if r.returncode != 0:
            logger.debug("runtime_cache: could not prepare %s on %s (rc=%s)", mounts.leaf, r.host, r.returncode)
            continue
        for line in r.stdout.splitlines():
            if line.startswith("runtime-cache: pruned "):
                logger.info("%s (%s)", line, r.host)
