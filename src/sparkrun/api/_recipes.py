"""``sparkrun.api.search_recipes`` — the recipe catalog.

The console-free counterpart to ``sparkrun list`` / ``sparkrun recipe
search``: enumerate and filter the recipes reachable from the configured
registries plus the working directory, without resolving any of them.

Scoping a query to one registry is part of the query itself —
``@community`` and ``@community/qwen`` — so a GUI search box gets the same
shorthand the CLI offers, and the filter is validated identically (see
:func:`sparkrun.core.registry.resolve_registry_filter`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import InvalidRegistryFilter
from sparkrun.api._models import RecipeSummary

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def resolve_recipe_filter(
    query: str | None = None,
    *,
    registry: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> tuple[str | None, str | None]:
    """Resolve a query's ``@registry`` scope into an explicit filter.

    :func:`search_recipes` applies this itself; call it directly when you
    need to *report* the resolved filter — e.g. to say which registry came
    back empty, since a bare ``@community`` scope leaves no query text to
    quote back at the user.

    Args:
        query: Free-text query, optionally carrying an ``@registry`` scope.
        registry: Explicit registry filter (may be None).
        sctx: Optional shared :class:`SparkrunContext`.

    Returns:
        ``(registry, query)`` with the scope stripped off the query.

    Raises:
        InvalidRegistryFilter: The filter is unknown, disabled, or
            self-contradictory.
    """
    from sparkrun.api._context import resolve_sctx
    from sparkrun.core.registry import RegistryFilterError, resolve_registry_filter

    sctx = resolve_sctx(sctx)
    try:
        return resolve_registry_filter(query, registry, sctx.config.get_registry_manager())
    except RegistryFilterError as e:
        raise InvalidRegistryFilter(str(e), registry=e.registry, reason=e.reason, available=e.available) from e


def search_recipes(
    query: str | None = None,
    *,
    registry: str | None = None,
    runtime: str | None = None,
    include_hidden: bool = False,
    include_local: bool = True,
    unique_names: bool = False,
    ensure_initialized: bool = True,
    sctx: "SparkrunContext | None" = None,
) -> list[RecipeSummary]:
    """Search the recipe catalog across registries and the working directory.

    Args:
        query: Free-text query matched (case-insensitively, substring) against
            recipe name, file stem, model, and description.  ``None`` or empty
            matches everything.  A leading ``@registry`` scopes the search:
            ``@community`` and ``@community/`` mean "that registry", and
            anything after the ``/`` is the remaining query, so
            ``@community/qwen`` is the community registry filtered by ``qwen``.
        registry: Explicit registry filter.  Equivalent to the ``@registry``
            scope, and validated the same way; supplying both with different
            names raises rather than silently picking one.
        runtime: Keep only recipes with this runtime (case-insensitive).
        include_hidden: Include registries flagged ``visible: false``.  A
            registry filter implies this — naming a registry is a stronger
            signal than its default visibility — so it only matters for an
            unscoped search.
        include_local: Include recipes discovered in the current working
            directory.  They carry no registry and are listed first, so a
            local override is easy to spot.  Ignored when a registry filter
            is in effect, since a local recipe belongs to no registry.
        unique_names: Collapse to one entry per unqualified name (first match
            wins, working-directory recipes first) — the view in which typing
            a bare recipe name is unambiguous, as ``sparkrun list`` shows it.
            The default keeps every copy, so a catalog browser can show
            ``@atlas/foo`` and ``@eugr/foo`` — or two same-stem variants
            within one registry — side by side.
        ensure_initialized: Clone not-yet-cached registries on first use.
            Best-effort: a failure is logged and the search proceeds against
            whatever is already cached.  Pass ``False`` to guarantee no
            network access.
        sctx: Optional shared :class:`SparkrunContext`.

    Returns:
        :class:`RecipeSummary` entries.  Working-directory recipes first,
        then registry recipes in configured-registry order.

    Raises:
        InvalidRegistryFilter: The registry filter (explicit or from the
            query scope) is unknown, disabled, or self-contradictory.
    """
    from sparkrun.api._context import resolve_sctx
    from sparkrun.core.recipe import discover_cwd_recipes, filter_recipes, recipe_matches_query, recipe_summary

    sctx = resolve_sctx(sctx)
    registry_mgr = sctx.config.get_registry_manager()

    if ensure_initialized:
        # Registries listed in registries.yaml aren't searchable until they
        # are cloned into the cache, so on a fresh machine the catalog would
        # come back empty.  Idempotent and fast once cached; the first call
        # may take a while.
        try:
            registry_mgr.ensure_initialized()
        except Exception:
            logger.warning("registry initialization failed; searching what is already cached", exc_info=True)

    registry, query = resolve_recipe_filter(query, registry=registry, sctx=sctx)

    # An explicit registry is a stronger signal than that registry's
    # visibility default, which exists to keep unqualified CLI names and
    # tab-completion sane rather than to hide recipes from someone asking
    # for them by name.
    entries = registry_mgr.search_recipes(query or "", include_hidden=include_hidden or registry is not None)

    # Drop entries that are literally the same file reached twice (shared
    # clones are symlinked per registry).  Deliberately keyed on path, not on
    # the qualified name: a registry's recipe dir is scanned with rglob, so
    # sibling directories can hold same-stem variants of a recipe
    # (``3x-spark-cluster/foo.yaml`` and ``4x-spark-cluster/foo.yaml``) that
    # share a qualified name while being different recipes.  Collapsing those
    # is `unique_names`' job, not this one.
    entries = _dedupe(entries, key=lambda e: str(e.get("path", "")))

    local: list[dict] = []
    if include_local and registry is None:
        for path in discover_cwd_recipes():
            entry = recipe_summary(path)
            if entry is not None and recipe_matches_query(entry, query):
                local.append(entry)

    entries = local + entries
    if unique_names:
        entries = _dedupe(entries, key=lambda e: str(e.get("file", "")))

    entries = filter_recipes(entries, runtime=runtime, registry=registry)
    return [RecipeSummary.from_summary(e) for e in entries]


def _dedupe(entries, key):
    """Keep the first entry per *key*, preserving order."""
    seen: set[str] = set()
    result = []
    for entry in entries:
        k = key(entry)
        if k in seen:
            continue
        seen.add(k)
        result.append(entry)
    return result


__all__ = ["search_recipes", "resolve_recipe_filter"]
