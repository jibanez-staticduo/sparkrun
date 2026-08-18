"""ds4 / DwarfStar runtime for sparkrun via ``ds4-server``.

ds4 (https://github.com/antirez/ds4) is a self-contained native C/CUDA
inference engine — no PyTorch, no Python, and it does not link GGML.  It is
deliberately *not* a general GGUF loader: only the DeepSeek V4 Flash/PRO and
GLM 5.2 GGUFs published at ``antirez/deepseek-v4-gguf`` have the tensor
layout, quantization mix and metadata the engine expects.

Prebuilt CUDA images are published from https://github.com/spark-arena/dgx-ds4
to ``ghcr.io/spark-arena/dgx-ds4``, so this runtime uses the ordinary Docker
executor like every other image-backed runtime.  ``executor: local`` still
works for anyone who builds ``ds4-server`` on the host themselves.

Four properties of ``ds4-server`` shape this runtime, all verified against
the argument parser in ``ds4_server.c`` rather than the shared help text:

**Single node.** ``ds4-server`` does not parse ``--role`` / ``--layers`` /
``--listen`` / ``--coordinator`` / ``--tensor-parallel`` at all — those belong
to the ``ds4`` *CLI* binary, which has no HTTP server.  ds4's multi-machine
pipeline and two-machine tensor parallelism are therefore unreachable through
the OpenAI-compatible endpoint today, and an unknown option is a hard
``exit(2)``.  :meth:`world_size` pins the runtime to one node accordingly.
``--cuda-tensor-parallel`` is a *local* multi-GPU path (paired devices on one
host), which does not apply to a DGX Spark's single GB10.

**No authentication.** There is no ``--api-key``, no ``Authorization``
handling, nothing.  :meth:`resolve_api_key` inherits the base ``None`` and
:meth:`validate_recipe` rejects an ``api_key`` in the recipe rather than
letting a user believe their endpoint is protected.

**No served-name flag.** ``/v1/models`` answers to a fixed alias allowlist
compiled into the server (see :data:`_DS4_MODEL_ALIASES`).  The name cannot be
configured, so a recipe must *declare* the matching alias for benchmark
targeting and proxy routing to work — the failure mode otherwise is an HTTP
404 on every request (issue #257).

**No health endpoint.** ``GET /v1/models`` is the readiness probe.  Note that
loading a 60-160 GB GGUF takes minutes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.parallelism import ParallelismConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

# Recipe ``defaults`` key -> ``ds4-server`` CLI flag.  Every entry is a valued
# flag consuming exactly one argument (``need_arg`` in ds4_server.c).
_DS4_FLAG_MAP = {
    # Sparkrun standard keys.
    "port": "--port",
    "host": "--host",
    "ctx_size": "-c",
    "max_tokens": "-n",
    # Engine / placement.
    "backend": "--backend",
    "threads": "-t",
    "gpu_vram": "--gpu-vram",
    "gpu_devices": "--gpu-devices",
    "power": "--power",
    "prefill_chunk": "--prefill-chunk",
    "batched_session": "--batched-session",
    "mixed_prefill_quantum": "--mixed-prefill-quantum",
    # Speculative decoding (DSpark / MTP).  ``--dspark-confidence`` implies
    # ``--dspark``, so a recipe may set the threshold alone.
    "mtp": "--mtp",
    "mtp_draft": "--mtp-draft",
    "mtp_margin": "--mtp-margin",
    "dspark_confidence": "--dspark-confidence",
    # Persistent KV cache.
    "kv_disk_dir": "--kv-disk-dir",
    "kv_disk_space_mb": "--kv-disk-space-mb",
    "kv_cache_min_tokens": "--kv-cache-min-tokens",
    "kv_cache_cold_max_tokens": "--kv-cache-cold-max-tokens",
    "kv_cache_continued_interval_tokens": "--kv-cache-continued-interval-tokens",
    "kv_cache_boundary_trim_tokens": "--kv-cache-boundary-trim-tokens",
    "kv_cache_boundary_align_tokens": "--kv-cache-boundary-align-tokens",
    # SSD streaming (run a model larger than memory).
    "ssd_streaming_cache_experts": "--ssd-streaming-cache-experts",
    "ssd_streaming_full_layers": "--ssd-streaming-full-layers",
    "ssd_streaming_preload_experts": "--ssd-streaming-preload-experts",
    # Directional steering.
    "dir_steering_file": "--dir-steering-file",
    "dir_steering_ffn": "--dir-steering-ffn",
    "dir_steering_attn": "--dir-steering-attn",
    # Misc.
    "tool_memory_max_ids": "--tool-memory-max-ids",
    "trace": "--trace",
}

# Bare toggles — emitted when truthy, omitted otherwise.
_DS4_BOOL_FLAGS = {
    "cuda_tensor_parallel": "--cuda-tensor-parallel",
    "quality": "--quality",
    "warm_weights": "--warm-weights",
    "ssd_streaming": "--ssd-streaming",
    "ssd_streaming_cold": "--ssd-streaming-cold",
    "dspark": "--dspark",
    "dspark_strict": "--dspark-strict",
    "glm_mtp": "--glm-mtp",
    "glm_mtp_timing": "--glm-mtp-timing",
    "cors": "--cors",
    "kv_cache_reject_different_quant": "--kv-cache-reject-different-quant",
    "disable_exact_dsml_tool_replay": "--disable-exact-dsml-tool-replay",
}

# Short-form aliases recognised when stripping flags from a rendered
# ``command:`` template.  ``-c``/``-n``/``-t`` are the canonical spellings in
# the flag map, so the long forms are the aliases here.
_DS4_FLAG_ALIASES: dict[str, list[str]] = {
    "ctx_size": ["--ctx"],
    "max_tokens": ["--tokens"],
    "threads": ["--threads"],
}

# Injected when a recipe does not set them.
#
# ``host`` is the important one: ds4-server defaults to ``127.0.0.1``, which
# would leave the endpoint unreachable from the control node and invisible to
# proxy discovery.  ``ctx_size`` is deliberately absent — ds4's own 32768
# default is a large KV allocation decision that belongs in a recipe.
_DS4_CONFIG_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 8000,
    "backend": "cuda",
}

# The model ids ``ds4-server`` answers to, compiled into
# ``server_model_alias_known()``.  There is no flag to add or change one.
_DS4_MODEL_ALIASES = frozenset(
    {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "glm-5.2",
        "glm-5.2-chat",
        "glm-5.2-no-think",
        "glm-5.2-nothink",
        "glm-5.2-reasoner",
        "zai/glm-5.2",
        "zai/glm-5.2-chat",
        "zai/glm-5.2-reasoner",
    }
)


class Ds4Runtime(RuntimePlugin):
    """Single-node ds4 runtime serving DeepSeek V4 / GLM 5.2 GGUFs.

    Launches ``ds4-server`` through the solo path (container with
    ``sleep infinity`` + ``docker exec``) from the prebuilt
    ``ghcr.io/spark-arena/dgx-ds4:latest`` image.  Upstream ships no releases
    and moves fast, so pin an immutable ``<date>-<sha7>-cu131`` tag in
    ``container:`` for anything that has to be reproducible.

    Weights are ordinary GGUF files: point ``model`` at
    ``antirez/deepseek-v4-gguf:<quant-marker>`` and sparkrun's existing GGUF
    download and distribution path pre-syncs the file, injects the
    container-internal path, and this runtime passes it as ``-m``.
    """

    runtime_name = "ds4"
    default_image_prefix = "ghcr.io/spark-arena/dgx-ds4"

    def cluster_strategy(self) -> str:
        """ds4 has no multi-node serving path; nothing Ray-shaped applies."""
        return "native"

    def get_family(self) -> str:
        return "ds4"

    # --- placement: single node, always ---

    # noinspection PyUnusedLocal
    def world_size(
        self,
        parallelism: "ParallelismConfig",
        *,
        recipe: "Recipe",
        cluster: "ClusterDefinition",
    ) -> int:
        """ds4-server is one process on one node — world size is always ``1``.

        The engine's multi-machine pipeline and tensor-parallel modes live in
        the ``ds4`` CLI binary and are not reachable through ``ds4-server``,
        whose argument parser rejects those flags outright.  Returning ``1``
        forces solo placement and guarantees the base ``_run_cluster``
        (``NotImplementedError``) path is never reached.
        """
        return 1

    # --- served name ---

    @staticmethod
    def infer_model_alias(model: str | None) -> str | None:
        """Best guess at the ds4 alias a *model* will be served under.

        Used for validation messages only — nothing is auto-injected, because
        ``served_model_name`` feeds ``generate_intent_id`` and silently
        supplying one would change the identity of a workload between the
        recipe the user wrote and the job sparkrun records.

        Returns ``None`` when the model name carries no recognisable marker.
        """
        if not model:
            return None
        stem = os.path.basename(str(model)).lower()
        if "glm" in stem:
            return "glm-5.2"
        if "v4-pro" in stem or "v4pro" in stem:
            return "deepseek-v4-pro"
        if "v4-flash" in stem or "v4flash" in stem:
            return "deepseek-v4-flash"
        return None

    # --- command generation ---

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the ``ds4-server`` command (always single-node).

        ``is_cluster`` / ``num_nodes`` / ``head_ip`` are accepted for interface
        compatibility but unused: :meth:`world_size` pins this runtime to one
        node, so it is only ever invoked in solo mode.

        ``max_model_len`` is translated to ``ctx_size`` so a recipe stays
        portable with the vLLM/SGLang spelling of the same knob.
        """
        overrides = self._translate_ctx_size(recipe, overrides)
        config = recipe.build_config_chain(overrides)

        rendered = recipe.render_command(config)
        if rendered:
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    {**_DS4_FLAG_MAP, **_DS4_BOOL_FLAGS},
                    set(_DS4_BOOL_FLAGS),
                    flag_aliases=_DS4_FLAG_ALIASES,
                )
            return rendered

        return self._build_command(recipe, config, skip_keys=skip_keys)

    @staticmethod
    def _translate_ctx_size(recipe: "Recipe", overrides: dict[str, Any]) -> dict[str, Any]:
        """Fold ``max_model_len`` into ``ctx_size`` when the latter is unset.

        Mirrors the llama.cpp runtime: ``max_model_len`` is the cross-runtime
        spelling, ``-c`` is what ds4 accepts.  An explicit ``ctx_size`` on
        either layer always wins.
        """
        if "ctx_size" in overrides:
            return overrides
        if "max_model_len" in overrides:
            return {**overrides, "ctx_size": overrides["max_model_len"]}
        config = recipe.build_config_chain(overrides)
        if config.get("ctx_size") is None:
            max_model_len = config.get("max_model_len")
            if max_model_len is not None:
                return {**overrides, "ctx_size": max_model_len}
        return overrides

    def _build_command(
        self,
        recipe: "Recipe",
        config,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build ``ds4-server -m <gguf> [flags...]`` from structured config."""
        from scitrera_app_framework.api import EnvPlacement, Variables

        # Export first: Variables.get() returns None for a missing key rather
        # than raising, which would stop a nested chain from consulting the
        # defaults layer at all.
        config_dict = config.export_all_variables() if isinstance(config, Variables) else config
        config = Variables(sources=(config_dict, _DS4_CONFIG_DEFAULTS), env_placement=EnvPlacement.IGNORED)

        parts = ["ds4-server", "-m", str(self._resolve_model_arg(recipe, config))]
        parts.extend(
            self.build_flags_from_map(
                config,
                {**_DS4_FLAG_MAP, **_DS4_BOOL_FLAGS},
                bool_keys=set(_DS4_BOOL_FLAGS),
                skip_keys=skip_keys,
            )
        )
        return " ".join(parts)

    @staticmethod
    def _resolve_model_arg(recipe: "Recipe", config) -> str:
        """The value for ``-m``: a GGUF file path, never a HuggingFace repo id.

        ds4 has no downloader — it mmaps a local file.  The launcher pre-syncs
        GGUF models and injects the container-internal path as
        ``_gguf_model_path``; that is the normal case.  Everything else falls
        through to whatever the recipe declared, which :meth:`validate_recipe`
        has already flagged if it cannot be a file.
        """
        gguf_path = config.get("_gguf_model_path")
        if gguf_path:
            return str(gguf_path)
        resolved = config.get("resolved_model_path")
        if resolved:
            return str(resolved)
        return recipe.model or ""

    # --- validation ---

    def validate_recipe(self, recipe: "Recipe") -> list[str]:
        """Reject configurations ``ds4-server`` cannot honour.

        Each check exists because the failure it prevents is silent: a wrong
        served name is an HTTP 404 on every request, an ``api_key`` is an
        endpoint the user believes is protected and is not, and a non-GGUF
        model is a load failure minutes into a launch.
        """
        issues = super().validate_recipe(recipe)
        defaults = recipe.defaults or {}

        issues.extend(self._validate_single_node(recipe, defaults))
        issues.extend(self._validate_served_name(recipe, defaults))
        issues.extend(self._validate_model(recipe))

        if defaults.get("api_key"):
            issues.append(
                "[ds4] api_key is set but ds4-server has no authentication — it accepts "
                "no --api-key flag and never inspects the Authorization header. The "
                "endpoint would be unauthenticated despite the setting; remove it and "
                "restrict access at the network layer instead."
            )
        return issues

    @staticmethod
    def _validate_single_node(recipe: "Recipe", defaults: dict) -> list[str]:
        """Multi-node and cross-host parallelism intent."""
        issues: list[str] = []
        if recipe.min_nodes and recipe.min_nodes > 1:
            issues.append(
                "[ds4] is single-node only: min_nodes=%d is not supported. ds4-server "
                "does not parse the distributed flags (--role/--layers/--coordinator/"
                "--tensor-parallel) — those belong to the `ds4` CLI binary, which has "
                "no HTTP server." % recipe.min_nodes
            )
        for key in ("tensor_parallel", "pipeline_parallel", "data_parallel"):
            value = defaults.get(key)
            try:
                value = int(value) if value is not None else 1
            except (TypeError, ValueError):
                continue
            if value > 1:
                issues.append(
                    "[ds4] %s=%d is not supported: ds4-server serves from one process on "
                    "one node. Multi-GPU within a single host uses --gpu-devices plus "
                    "cuda_tensor_parallel, which does not apply to a DGX Spark's single "
                    "GB10." % (key, value)
                )
        return issues

    @classmethod
    def _validate_served_name(cls, recipe: "Recipe", defaults: dict) -> list[str]:
        """The served name must match ds4's compiled-in alias allowlist."""
        declared = defaults.get("served_model_name")
        guess = cls.infer_model_alias(recipe.model)
        hint = " Expected %r for this model." % guess if guess else ""

        if not declared:
            return [
                "[ds4] defaults.served_model_name is not set. ds4-server has no alias "
                "flag and answers only to a fixed set of model ids, so benchmarks and "
                "proxy routing will request a name the server does not recognise and "
                "get HTTP 404.%s Known ids: %s" % (hint, ", ".join(sorted(_DS4_MODEL_ALIASES)))
            ]
        if str(declared) not in _DS4_MODEL_ALIASES:
            return [
                "[ds4] served_model_name=%r is not one of the ids ds4-server answers to, "
                "so every request will return HTTP 404.%s Known ids: %s" % (declared, hint, ", ".join(sorted(_DS4_MODEL_ALIASES)))
            ]
        return []

    @staticmethod
    def _validate_model(recipe: "Recipe") -> list[str]:
        """ds4 loads a local GGUF file and nothing else."""
        model = recipe.model
        if not model:
            return []
        from sparkrun.models.download import is_gguf_model

        if is_gguf_model(model) or str(model).lower().endswith(".gguf") or str(model).startswith("/"):
            return []
        return [
            "[ds4] model=%r does not look like a GGUF file or GGUF repo. ds4 has no "
            "downloader and is not a general GGUF loader — it mmaps a local file and "
            "only accepts the DeepSeek V4 Flash/PRO and GLM 5.2 layouts published at "
            "antirez/deepseek-v4-gguf." % model
        ]
