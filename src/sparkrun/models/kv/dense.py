"""Dense (MHA / GQA / MQA) KV cache sizing — the fallback strategy.

Every layer caches a K and a V entry per KV head per token, so the cache is
linear in context length and shards cleanly across tensor-parallel ranks along
the head dimension.
"""

from __future__ import annotations

from typing import ClassVar

from sparkrun.models.dtypes import kv_bytes_per_element
from sparkrun.models.kv._base import ArchInfo, KVCacheStrategy, KVDetection, KVSizing

__all__ = ["DenseKVStrategy"]


class DenseKVStrategy(KVCacheStrategy):
    """Ordinary attention: ``2 * layers * kv_heads * head_dim * bytes`` per token.

    Claims every model, at the lowest priority in the registry, so resolution
    always terminates.  A more specific strategy that recognises the model but
    cannot size it does **not** fall through to here: sizing a compressed-latent
    cache with this formula overestimates by one to two orders of magnitude, and
    an estimate that wrong is worse than no estimate.
    """

    name: ClassVar[str] = "dense"
    label: ClassVar[str | None] = None  # the generic layers/heads/head_dim line says it better
    priority: ClassVar[int] = 1000
    replicates_kv: ClassVar[bool] = False

    def detect(self, arch: ArchInfo) -> KVDetection | None:
        return KVDetection(source="fallback")

    def size(self, arch: ArchInfo, *, max_model_len: int | None) -> KVSizing:
        if not (arch.num_layers and arch.num_kv_heads and arch.head_dim):
            missing = [
                name
                for name, value in (
                    ("num_layers", arch.num_layers),
                    ("num_kv_heads", arch.num_kv_heads),
                    ("head_dim", arch.head_dim),
                )
                if not value
            ]
            return KVSizing(
                total_bytes=None,
                unsizable_reason="Missing architecture info (%s); KV cache estimate unavailable" % ", ".join(missing),
            )

        bpe = kv_bytes_per_element(arch.kv_dtype or "")
        if bpe is None:
            return KVSizing(total_bytes=None, unsizable_reason="Unknown KV cache dtype %r" % arch.kv_dtype)

        per_token = 2.0 * arch.num_layers * arch.num_kv_heads * arch.head_dim * bpe
        return KVSizing(
            total_bytes=per_token * max_model_len if max_model_len else None,
            per_token_bytes=per_token,
        )
