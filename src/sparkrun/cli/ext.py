"""Plugin-contributed CLI commands.

External plugins add Click commands/groups to the sparkrun CLI by calling
:func:`register_cli_command` at import time (or in their ``register(v)`` hook),
e.g.::

    from sparkrun.cli.ext import register_cli_command
    register_cli_command(my_command)                          # top-level: `sparkrun my-command`
    register_cli_command(import_foo, parent=("cluster", "import"))  # `sparkrun cluster import foo`

The registry itself lives in :mod:`sparkrun.core.cli_registry`, deliberately
free of Click: a plugin that also loads on the console-free ``sparkrun.api``
path should import *that* and register a **loader** rather than a built
command, so registering costs no Click import.  This module re-exports the
registry API and owns the attach half, which needs Click.

**Timing.** The CLI command tree is built at import of :mod:`sparkrun.cli`, but
external plugins load later, during :func:`sparkrun.core.bootstrap.init_sparkrun`.
:class:`PluggableGroup` bridges the gap: the top-level ``main`` group is a
``PluggableGroup``, so the first time Click resolves *any* subcommand it runs
:func:`ensure_cli_extensions` — which loads plugins (they call
``register_cli_command``) and attaches every registered command to its target
group (walking into nested groups like ``cluster import``) before Click
dispatches.  A plugin command therefore shows up in ``--help`` and dispatches
like a built-in.

Command selection is deliberately *not* tied to the transport/executor
abstractions — plugins map to commands however makes sense for them.  Per-command
gating (feature flags, ``hidden=``) is the command's own concern.
"""

from __future__ import annotations

import logging

import click

# Re-exported so ``sparkrun.cli.ext`` remains the documented entry point.
from sparkrun.core.cli_registry import (  # noqa: F401 - re-export
    CliCommandSpec,
    register_cli_command,
    registered_cli_commands,
)

logger = logging.getLogger(__name__)


def _resolve_group(root: click.Group, path: tuple[str, ...]) -> "click.Group | None":
    """Walk *path* from *root* using the raw ``.commands`` dicts.

    Uses ``.commands`` rather than ``get_command`` so attaching never re-enters
    :meth:`PluggableGroup.get_command` (which would recurse into the ensure
    hook).  Returns ``None`` if any segment is missing or not a group.
    """
    group: click.Command = root
    for segment in path:
        if not isinstance(group, click.Group):
            return None
        group = group.commands.get(segment)
        if group is None:
            return None
    return group if isinstance(group, click.Group) else None


def attach_cli_extensions(root: click.Group) -> None:
    """Attach every registered command to its target group under *root*.

    A command whose parent group is missing is logged and skipped; a command
    whose name already exists on the target group is left as-is (built-in wins,
    and re-attach passes are no-ops).
    """
    for spec in registered_cli_commands():
        parent = _resolve_group(root, spec.parent)
        if parent is None:
            logger.warning(
                "Cannot attach plugin CLI command %r: parent group %r not found",
                spec.name,
                "/".join(spec.parent) or "<root>",
            )
            continue
        if spec.name in parent.commands:
            continue
        try:
            command = spec.resolve()
        except Exception:  # noqa: BLE001 - one bad loader must not break the CLI
            logger.exception("Could not build plugin CLI command %r", spec.name)
            continue
        parent.add_command(command)


def ensure_cli_extensions(root: click.Group) -> None:
    """Load external plugins, then attach their registered CLI commands.

    ``init_sparkrun`` imports external plugin modules, which call
    ``register_cli_command`` at import time; the subsequent attach drains the
    registry onto *root*.
    """
    from sparkrun.core.bootstrap import init_sparkrun

    init_sparkrun()
    attach_cli_extensions(root)


class PluggableGroup(click.Group):
    """A Click group that lazily merges plugin-contributed commands.

    On the first command resolution (``get_command`` / ``list_commands``) it
    runs :func:`ensure_cli_extensions` once, so plugin commands are present
    before Click dispatches — including into nested groups reached through this
    one.  The load is guarded per-instance and fully defensive: a broken plugin
    is logged and never breaks the CLI.
    """

    def _ensure_cli_extensions_loaded(self) -> None:
        if getattr(self, "_cli_ext_loaded", False):
            return
        # Set before running so the attach pass (which touches .commands, not
        # get_command) can't re-enter this hook.
        self._cli_ext_loaded = True
        try:
            ensure_cli_extensions(self)
        except Exception:  # noqa: BLE001 - plugin CLI loading must never break the CLI
            logger.exception("Failed to load plugin CLI commands")

    def list_commands(self, ctx):
        self._ensure_cli_extensions_loaded()
        return super().list_commands(ctx)

    def get_command(self, ctx, name):
        self._ensure_cli_extensions_loaded()
        return super().get_command(ctx, name)
