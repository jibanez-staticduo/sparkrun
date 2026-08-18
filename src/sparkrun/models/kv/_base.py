"""Base types for architecture-specific KV cache sizing.

A KV cache strategy answers three questions about one model:

======================================  ==========================================
"how big is the cache at length L?"     :meth:`KVCacheStrategy.size`
"how big is it per token?"              :attr:`KVSizing.per_token_bytes`
"how long a context fits in N bytes?"   :meth:`KVCacheStrategy.tokens_for_budget`
======================================  ==========================================

Sizing is expressed as a **total for a requested length**, not as a per-token
figure multiplied by that length.  Per-token is the special case: it holds for
dense attention and for MLA, but not for a sliding-window layer (which caches
``min(max_model_len, window)`` tokens) nor for a Mamba/SSM layer (whose state is
per *sequence*).  Making the total the primitive keeps those architectures
addable without reworking the seam.

Alongside sizing, a strategy **declares** the architecture parameters it reads
(:class:`ArchField`).  That declaration is the single source of truth for four
things that would otherwise be four hand-maintained copies of the same list:
HuggingFace config extraction, what a recipe may pin in ``metadata``, how those
values validate, and what gets written back to ``metadata`` after an estimate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

__all__ = [
    "ArchField",
    "ArchInfo",
    "KVCacheStrategy",
    "KVDetection",
    "KVSizing",
]


@dataclass(frozen=True)
class ArchField:
    """One architecture parameter a KV strategy consumes.

    ``name`` is used verbatim as the recipe ``metadata`` key, the
    :attr:`ArchInfo.extra` key and the key
    :func:`sparkrun.models.vram.extract_model_info` returns, so a field is
    spelled once and reaches every layer under that spelling.
    """

    name: str

    hf_keys: tuple[str, ...] = ()
    """HuggingFace ``config.json`` keys supplying this field, first match wins.

    Empty means the field is derived rather than read directly, and the owning
    strategy overrides :meth:`KVCacheStrategy.extract` to produce it.
    """

    kind: type = int
    """``int``, ``str`` or ``list`` — drives both validation and coercion."""

    positive: bool = True
    """For ``int`` fields: reject zero and negatives. Ignored for other kinds."""

    doc: str = ""
    """One-line description, for ``RECIPES.md`` and error messages."""

    def coerce(self, value: Any) -> Any:
        """Convert an accepted value to its canonical stored form."""
        if self.kind is int:
            return int(value)
        if self.kind is str:
            return str(value)
        if self.kind is list:
            return [int(v) for v in value]
        return value

    def validate(self, value: Any) -> str | None:
        """Return a human-readable issue with *value*, or ``None`` if it is fine.

        ``bool`` is rejected for ``int`` fields explicitly: it is an ``int``
        subclass, so ``metadata.kv_lora_rank: true`` would otherwise coerce to
        ``1`` and silently size the latent as one element wide.
        """
        if value is None:
            return None
        if self.kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                return "metadata.%s %r must be a positive integer" % (self.name, value)
            if self.positive and value <= 0:
                return "metadata.%s %r must be a positive integer" % (self.name, value)
            return None
        if self.kind is str:
            if not isinstance(value, str):
                return "metadata.%s %r must be a string" % (self.name, value)
            return None
        if self.kind is list:
            if not isinstance(value, list) or not value:
                return "metadata.%s %r must be a non-empty list of integers" % (self.name, value)
            if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
                return "metadata.%s must contain only integers" % self.name
            return None
        return None


@dataclass(frozen=True)
class ArchInfo:
    """The resolved architecture of one model, as a KV strategy sees it.

    Universal fields are explicit; anything a single architecture owns lives in
    :attr:`extra` under its :attr:`ArchField.name`, so adding an architecture
    never widens this dataclass.
    """

    num_layers: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    model_type: str | None = None

    kv_dtype: str | None = None
    """The *effective* KV dtype — the caller's ``bfloat16`` default already applied."""

    extra: Mapping[str, Any] = field(default_factory=dict)

    #: Names carried as real attributes rather than in :attr:`extra`.
    UNIVERSAL: ClassVar[frozenset[str]] = frozenset({"num_layers", "num_kv_heads", "head_dim", "model_type", "kv_dtype"})

    def get(self, name: str, default: Any = None) -> Any:
        """Read a field by name, whether it is universal or strategy-owned.

        A present-but-``None`` entry reads as absent, so a caller can build
        :attr:`extra` from a plain ``{field: metadata.get(field)}`` sweep
        without filtering the misses out first.
        """
        value = self.extra.get(name)
        if value is not None:
            return value
        if name in self.UNIVERSAL:
            value = getattr(self, name)
            return default if value is None else value
        return default


@dataclass(frozen=True)
class KVDetection:
    """Why a strategy claimed a model.

    Warnings raised here are about *identification* — "I am sizing this as MLA,
    and here is why you may not want me to" — as distinct from
    :attr:`KVSizing.warnings`, which qualify the resulting number.
    """

    source: str
    """``"architecture"``, ``"kv_layout"``, ``"model_type"`` or ``"fallback"``."""

    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class KVSizing:
    """How large one rank's KV cache is, and how it shards."""

    total_bytes: float | None
    """Bytes for one rank's full cache at the requested length.

    ``None`` when the architecture is understood but the numbers to size it are
    not — see :attr:`unsizable_reason`.  Also ``None`` when no length was
    requested; :attr:`per_token_bytes` may still be set in that case.
    """

    per_token_bytes: float | None = None
    """Marginal bytes per token, for display and for the default budget math.

    ``None`` when the cache is not linear in context length (a windowed or
    per-sequence cache), in which case only :attr:`total_bytes` is meaningful.
    """

    replicated_across_tp: bool = False
    """``True`` when every tensor-parallel rank holds a full copy of the cache.

    MLA's compressed latent has no head dimension to shard, so raising ``--tp``
    does not shrink it; pipeline parallelism still splits it by layer.
    """

    is_floor: bool = False
    """``True`` when auxiliary caches exist that this estimate does not count."""

    warnings: tuple[str, ...] = ()

    unsizable_reason: str | None = None
    """Why :attr:`total_bytes` is ``None``, phrased to name the actual gap.

    Set only for a genuine failure to size, never merely because no
    ``max_model_len`` was supplied.
    """


class KVCacheStrategy(ABC):
    """Sizes the KV cache for one family of attention architectures.

    Strategies are resolved in :attr:`priority` order by
    :func:`sparkrun.models.kv.resolve_kv_strategy`, most specific first, and the
    first whose :meth:`detect` claims the model wins.  Selection is
    order-sensitive, which is why this registry is in-process rather than SAF
    (see the ``platforms/`` precedent).
    """

    name: ClassVar[str]
    """Stable identifier, reported as ``VRAMEstimate.kv_arch``."""

    label: ClassVar[str | None] = None
    """Human-readable architecture line for the CLI, or ``None`` to show the
    generic layers/heads/head_dim summary."""

    priority: ClassVar[int] = 100
    """Lower is more specific. The catch-all dense strategy sits at 1000."""

    replicates_kv: ClassVar[bool] = False
    """Whether this architecture's cache is replicated across TP ranks.

    Duplicated on the class rather than read from :class:`KVSizing` alone
    because the sharding rule must survive a user-supplied
    ``kv_vram_per_token`` override, which replaces the *sizing* but not the
    architecture.
    """

    # -- declaration ---------------------------------------------------

    def arch_fields(self) -> tuple[ArchField, ...]:
        """Architecture parameters this strategy consumes."""
        return ()

    def kv_layouts(self) -> frozenset[str]:
        """Packed KV dtype names this strategy claims (e.g. ``fp8_ds_mla``).

        These name a whole padded slot rather than an element width, so they
        have no :func:`~sparkrun.models.dtypes.kv_bytes_per_element` entry and
        would otherwise fail recipe validation as unknown dtypes.
        """
        return frozenset()

    def extract(self, cfg: Mapping[str, Any], core: Mapping[str, Any]) -> dict[str, Any]:
        """Pull this strategy's fields out of one HuggingFace config dict.

        The default walks :meth:`arch_fields` and takes the first present
        :attr:`ArchField.hf_keys` entry.  Override to add a derivation that
        depends on other fields; *core* carries the already-extracted universal
        keys (``head_dim``, ``num_layers``, …) for exactly that.
        """
        found: dict[str, Any] = {}
        for f in self.arch_fields():
            for key in f.hf_keys:
                if key in cfg:
                    found[f.name] = cfg[key]
                    break
        return found

    # -- behaviour -----------------------------------------------------

    @abstractmethod
    def detect(self, arch: ArchInfo) -> KVDetection | None:
        """Claim *arch*, or return ``None`` to pass it to the next strategy."""

    @abstractmethod
    def size(self, arch: ArchInfo, *, max_model_len: int | None) -> KVSizing:
        """Size one rank's KV cache for *arch* at *max_model_len*."""

    def tokens_for_budget(self, arch: ArchInfo, sizing: KVSizing, budget_bytes: float) -> int | None:
        """Longest context fitting in *budget_bytes*.

        The default inverts :attr:`KVSizing.per_token_bytes`, which is exact for
        any cache linear in context length.  A strategy whose cache saturates
        (sliding window) or has a constant component (SSM state) must override
        this rather than let a linear extrapolation report a context length that
        cannot be served.
        """
        per_token = sizing.per_token_bytes
        if not per_token or per_token <= 0:
            return None
        return int(budget_bytes / per_token)
