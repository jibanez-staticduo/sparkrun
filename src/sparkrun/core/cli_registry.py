"""Click-free registry of plugin-contributed CLI commands.

The registry lives in ``core`` rather than ``cli`` for one reason: plugins load
during :func:`sparkrun.core.bootstrap.init_sparkrun`, which the **console-free**
``sparkrun.api`` layer also runs. If a plugin had to import ``sparkrun.cli.ext``
to register a command, that import would pull in ``sparkrun.cli`` — and with it
Click — for every ``api`` caller, including the desktop sidecar. Keeping the
registry here means registering a command costs no Click import at all.

Which is also why a command may be registered as a **loader**: building a
``click.Command`` needs Click, so a plugin that constructs one eagerly would
re-introduce exactly the import it is avoiding. The loader is called only when
the CLI actually attaches commands.

:mod:`sparkrun.cli.ext` re-exports everything here and owns the attach half,
which legitimately needs Click.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import click


@dataclass(frozen=True)
class CliCommandSpec:
    """A plugin command plus the group path it attaches under.

    :attr:`command` is either a ``click.Command`` or a zero-arg callable
    returning one; :meth:`resolve` normalizes that away.
    """

    name: str
    command: Any
    parent: tuple[str, ...] = ()

    def resolve(self) -> "click.Command":
        """Return the command, invoking the loader if that is what was stored.

        A ``click.Command`` carries ``.name``; a loader function does not —
        which is a discriminator that needs no Click import.
        """
        return self.command if hasattr(self.command, "name") else self.command()


_CLI_EXTENSIONS: list[CliCommandSpec] = []


def register_cli_command(
    command: "click.Command | Callable[[], click.Command]",
    *,
    parent: "tuple[str, ...] | list[str]" = (),
    name: str | None = None,
) -> None:
    """Register *command* to attach under the group path *parent*.

    ``parent=()`` attaches to the top-level ``sparkrun`` group;
    ``parent=("cluster", "import")`` attaches under ``sparkrun cluster import``.

    Args:
        command: A ``click.Command``, or a zero-arg callable returning one.
            Prefer the callable when the plugin is loaded on the ``api`` path
            too — it keeps Click out of that import graph.
        parent: Group path to attach under.
        name: Command name. Required with a callable (idempotence must not
            have to build the command); inferred from a ``click.Command``.

    Idempotent by ``(parent, name)`` so repeated plugin loads — which happen
    routinely in tests — don't double-register.
    """
    resolved_name = name or getattr(command, "name", None)
    if not resolved_name:
        raise ValueError("register_cli_command needs an explicit name= when given a loader")

    spec = CliCommandSpec(name=resolved_name, command=command, parent=tuple(parent))
    for existing in _CLI_EXTENSIONS:
        if existing.parent == spec.parent and existing.name == spec.name:
            return
    _CLI_EXTENSIONS.append(spec)


def registered_cli_commands() -> list[CliCommandSpec]:
    """Return a copy of the registered command specs (introspection/tests)."""
    return list(_CLI_EXTENSIONS)


__all__ = ["CliCommandSpec", "register_cli_command", "registered_cli_commands"]
