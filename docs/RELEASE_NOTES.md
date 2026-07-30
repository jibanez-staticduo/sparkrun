# sparkrun 0.3.0 — Release Notes

Multiplatform foundations, unified executor resolution, and the B-workstream
security tightening land in this release. The shipped DGX Spark path is
unchanged byte-for-byte (NCCL output, container launch, post-launch lifecycle);
the surface area beneath it has been factored so AMD, Intel Gaudi, Local, and
Kubernetes hooks plug in without re-plumbing runtimes.

Repository: <https://github.com/spark-arena/sparkrun>

## Highlights

- **Multiplatform seams**: `HostHardware` / `AcceleratorSpec` /
  `CollectiveBackend` / `HardwarePlatformPlugin` are the abstractions every
  future vendor implementation builds against. NVIDIA stays the only fully
  wired vendor; AMD (RCCL) and Intel Gaudi (HCCL) ship as scaffolds.
- **Unified executor resolution**: a single `resolve_executor()` chain
  (CLI → recipe → runtime → SAF defaults → fallback) replaces ad-hoc executor
  construction. Two experimental executors land: `LocalExecutor` (native
  subprocess, no container) and `K8sExecutor` (`kubectl run`-driven draft).
- **Security hardening**: URL recipes never auto-trusted + hardened recipe
  fetch, explicit/persisted proxy bind host, arena OAuth state binding, trust
  gating on every recipe hook surface (`pre_exec` / `post_exec` /
  `post_commands`), git URL allowlist, validated sudo usernames and cache
  paths, strict trtllm host-key checking, delegated-copy host/path validation.
- **Scales to larger clusters**: bounded SSH/rsync fan-out, parallel
  head→worker distribution, execution timeouts on the launch path, and
  parallel per-host cleanup keep 8/16/32-host launches from saturating the
  control node or stalling on one slow host.
- **Console-free library API**: the `sparkrun.api` package no longer imports
  the CLI; `api.run` / `api.benchmark` / `api.resume_benchmark` share a single
  placement authority and raise typed errors instead of writing to stdout.
  The CLI is now a renderer over it, so anything the CLI does is reachable
  from Python.
- **Pluggable placement**: `occupancy-sparse` (new-cluster default),
  `occupancy-dense`, and `greedy` (the 0.2.x behavior). Occupancy schedulers
  place against *live* cluster state, so concurrent runs stop colliding.
  Clusters with no `scheduler` key keep `greedy`, so upgrading never silently
  moves existing workloads.
- **Unified, cross-executor status**: `api.status` sweeps every enabled
  executor on a cluster's substrate and merges. A native `local` workload and
  a docker container are invisible to each other's introspection; both now
  appear in `status`, `cluster monitor`, and `stop --all`.
- **Feature flags**: channel-aware gating (`stable` / `beta` / `alpha`) for
  experimental capabilities, managed with `sparkrun setup features`. Every
  experimental surface in this release is off by default on every channel.
- **Out-of-tree plugins**: runtimes, executors, transports, and CLI commands
  can ship outside the repo, loaded from `plugins.paths`. The Thunder Compute
  transport moved out of core to become the first consumer.
- **Anonymous, opt-out telemetry**: coarse usage signal to guide what gets
  built next (`sparkrun setup telemetry`, `SPARKRUN_NO_TELEMETRY`). Model
  identifiers are sent only for confirmed-public HuggingFace repos.

## New

### Multiplatform foundations

- `core/hardware.py` introduces `AcceleratorSpec` and `HostHardware`, the data
  model every multiplatform path consumes. Hosts without explicit metadata
  default to DGX Spark via `default_dgx_spark_hardware()`.
- `core/hardware_probe.py:probe_host()` / `probe_hosts()` collapse the
  accelerator fingerprint and InfiniBand detection into a single SSH
  round-trip. Output is split by sentinel markers and routed to existing
  parsers.
- `core/fingerprint.py:fingerprint_host` is now a thin shim retained for
  callers that don't pay for the IB section.
- `core/backend_select.py:select_backends(host_hardware)` returns a
  `BackendBundle(accelerator_vendor, collective)`. `launcher.py` calls this
  per host and threads the result through `runtime.run(..., backends=...)`.
- `core/placement.py:compute_placement()` maps `ParallelismConfig` onto hosts
  honoring an optional `RecipeLayout`. Auto-packs single-vendor clusters,
  raises `LayoutRequiredError` for multi-vendor clusters.

### `CollectiveBackend` abstraction

- `orchestration/collectives/base.py` defines the ABC.
- `orchestration/collectives/nccl.py` wraps `infiniband.generate_nccl_env` /
  `generate_ring_nccl_overrides`. Byte-identical to legacy DGX output.
- `orchestration/collectives/rccl.py` and `hccl.py` are scaffolds that raise
  `NotImplementedError` from `env_for_host` — surfaces the missing
  implementation rather than silently emitting NCCL.
- `orchestration/collectives/__init__.py:get_backend(vendor)` is the lookup;
  `UnsupportedCollectiveError` covers vendors with no scaffold (Apple, CPU,
  ...).

### `HardwarePlatformPlugin` + `validate_host` hook

- `platforms/base.py` defines the ABC. Built-ins in `platforms/dgx_spark.py`
  (`DgxSparkPlatform`) and `platforms/nvidia_generic.py`
  (`GenericNvidiaPlatform`).
- `DgxSparkPlatform.validate_host()` warns when the GB10 accelerator is
  missing the `rdma:roce-v2` capability — multi-node collective health
  concern.
- `GenericNvidiaPlatform.validate_host()` warns when `matches()` was called on
  a host without any NVIDIA accelerator.
- `launcher.py` runs `validate_host` per placed host after backend resolution.
  Warnings are logged; not raised.

### Executor subsystem

- `orchestration/executor.py` is the public facade. Exports `Executor`,
  `ExecutorConfig`, `EXT_EXECUTOR`, `get_executor()`, `list_executors()`,
  `resolve_executor()`.
- `orchestration/executors/_base.py` holds the ABC and `ExecutorConfig`
  dataclass; `from_chain()` parses CLI/recipe/runtime layers.
- `orchestration/executors/docker.py` owns `DOCKER_DEFAULTS` and the
  `rootless` / `auto_user` adjustment lever (`apply_runtime_adjustments`).
- `orchestration/executors/local.py` (`LocalExecutor`): native subprocess via
  `setsid`, pid/log-file lifecycle, no images. See `docs/EXECUTORS.md` for
  limitations.
- `orchestration/executors/k8s.py` (`K8sExecutor`): `kubectl run`-driven
  draft. Drops Docker-only options (`--privileged`, `--shm-size`, ...).
- `core/bootstrap.py` discovers executors via
  `find_types_in_modules("sparkrun.orchestration.executors", Executor)`.

### Recipe / config surface

- New recipe field `executor: docker | local | k8s` (default `docker`).
- New recipe block `executor_config:` for per-executor knobs (Local:
  `working_dir`, `log_dir`, `pid_dir`, `env_file`, `command_prefix`; K8s:
  `k8s_namespace`, `k8s_context`, `k8s_node_selector`, `kubeconfig`, ...).
- `SparkrunConfig` accepts `default_executor` + `executor_config` to set the
  fleet-wide default below recipe overrides.

### `core/launcher.py`

- `resolve_per_host_backends(host_list, cluster=...)` builds the per-host
  `BackendBundle` map.
- Centralized compatibility check (`runtimes/compatibility.py:
  check_runtime_host_compatibility`) walks every host before any side effects
  (container pull, model sync, ...). Raises `IncompatibleHardwareError`,
  surfaced cleanly by the CLI.
- `resolve_recipe_trust(recipe, trust_cli)` returns one trust verdict per
  recipe, shared by `pre_exec` and `post_exec`/`post_commands`.

### Scalability & robustness

- `SparkrunConfig.max_parallel_ssh` (config key `ssh.max_parallel_ssh`, default
  20) caps every parallel SSH/rsync fan-out; small clusters are unchanged.
- Head→worker image/model distribution (`image_distribute.sh` /
  `model_distribute.sh`) runs transfers with bounded concurrency instead of
  serially.
- Launch-path parallel SSH calls carry explicit execution timeouts so a host
  that connects then hangs no longer blocks the whole batch.
- Container cleanup runs in parallel and reports per-host success/failure,
  naming hosts that may still hold VRAM after a failed launch.
- `pending_ops` locks record host + token + timestamp, prune by max age (guards
  PID reuse / NFS-shared caches), and verify ownership before removal.

### Library API

- `sparkrun.api` is a console-free facade over `core/`: it no longer imports
  from `sparkrun.cli`. Shared resolution moved to `core/resolve.py` and
  `api/_hosts.py:resolve_effective_hosts` (the single placement authority used
  by `run`, `benchmark`, and the CLI).
- New `api.resume_benchmark(...)` entry point; benchmark resume is no longer
  CLI-only.
- Sub-namespaces group the larger surfaces — `api.proxy`, `api.setup`,
  `api.tailscale`, `api.k8s` — each with `_ops.py` for the console-free logic
  and `_errors.py` for its typed exceptions.
- Every entry point takes an optional `SparkrunContext` (`sctx`) so a chain of
  calls shares config, registry manager, and cluster manager rather than
  rebuilding them per call.

### Scheduling and placement

- `schedulers/` is a SAF extension point (`EXT_SCHEDULER`). Three ship:
  `greedy`, `occupancy-sparse`, `occupancy-dense`.
- Selection chain: `--scheduler` → recipe `scheduler:` → cluster `scheduler`.
  Nothing named → `greedy`, with a hint recommending the upgrade — that is
  what preserves 0.2.x placement for existing clusters.
- Occupancy schedulers spread *between* workloads, not within one: a single
  job's ranks still pack onto as few hosts as possible so tensor-parallel
  traffic rides the fastest link.
- `core/limits.py` resolves a per-accelerator usable-memory cap for scheduling
  and fit (`--max-gpu-mem-util`; GB10 defaults to 0.85 because unified memory
  contends with the host). It does **not** affect the serving
  `--gpu-memory-utilization`.
- `InsufficientCapacity` carries the status snapshot, host list, and requested
  slot count, so callers can render capacity diagnostics without a second SSH
  round-trip.

### Status, monitoring, and logs

- `api.status` (lean occupancy) and `api.status_report` (display tier) are the
  two entry points; `query_status_for_cluster` merges every enabled executor
  sharing the cluster's `status_scope`. A single failing executor is skipped
  rather than breaking the report.
- `TelemetryProvider` (`EXT_TELEMETRY`) is the telemetry peer of
  `Executor.query_status`, selected by scope. `api.live_monitor` composes
  telemetry with the occupancy poll into `MonitorFrame` snapshots, which drive
  the `cluster monitor` TUI — so its Jobs column shows every executor's
  workloads.
- Log locations are data (`core/log_source.py`). Runtimes that serve via
  `docker exec` inside a sleep-infinity container are structurally invisible
  to `docker logs`; `sparkrun logs` resolves the right source per runtime and
  gains `-f/--follow`, `-a/--all-sources`, and `-n/--lines`.

### Feature flags, plugins, transports

- `core/features.py`: `FeatureFlag` registry with per-channel defaults.
  Resolution is env (`SPARKRUN_FEATURE_<NAME>`) → config → channel → baseline
  → fail-closed for unknown flags. Managed with `sparkrun setup features`.
- Plugins self-gate by declaring `required_feature_flag`; a gated plugin stays
  registered but is absent from resolution and tab-completion. An explicitly
  requested but unavailable executor raises `ExecutorUnavailableError` rather
  than silently downgrading to docker.
- `transports/` is the connectivity seam — how hosts are reached and prepared,
  orthogonal to the executor. `ssh` is the default and a no-op, so existing
  clusters are byte-identical to before transports existed.
- `core/external_plugins.py` loads out-of-tree plugins from `plugins.paths`
  (gated behind `core.external_plugins`, off by default), and `cli/ext.py`
  lets them attach Click commands to the built-in tree.

### Setup and platform work

- `sparkrun setup check` — a non-destructive readiness probe over the things
  the wizard configures, with `--json`. The seed of a per-platform step
  registry with paired check/apply stages.
- SSH access bootstrap (`api/setup/`) runs first in the wizard, since every
  later phase assumes passwordless control→host SSH. Written to work on a bare
  Windows control machine: only `ssh` and `ssh-keygen`, no local `bash`, no
  paramiko, and key material travels over stdin rather than argv.
- NVIDIA CDI spec generation; GPUs are requested via CDI rather than `--gpus`.
- `sparkrun setup tailscale` (gated) publishes an inference endpoint on a
  tailnet using JIT-minted, tagged OAuth keys — control-plane only, never a
  data path (NCCL stays on InfiniBand/CX7).
- `sparkrun setup k8s` (gated) — kubectl acquisition, least-privilege service
  account, Kueue install, JobSet launch.

### Benchmarking

- Per-category CLI: `sparkrun benchmark performance` (alias `perf`) and
  `sparkrun benchmark tools`; bare `benchmark <recipe>` falls through to
  `performance`. New categories appear automatically when a plugin registers
  them.
- Non-interactive `--resume` / `--fresh`, and `--arena` to run the Spark Arena
  flow from either entry point.
- Run identity is tied to recipe *content*, and the container image is pinned
  by content-addressable digest on first launch, so a re-pushed tag or a
  rebuilt local image cannot change the bits mid-sweep.

### Proxy

- The generated config file is the single source of truth for the model list.
  LiteLLM's runtime mutation endpoints need a DB-backed model store, so
  applying a change means rewriting the config and restarting — skipped
  entirely when the desired model set already matches disk.
- A gateway seam (`proxy/gateway.py`): `gateway.litellm` gates availability,
  `proxy.gateway` pins selection, and exactly one gateway is used at a time.
  Bringing a gateway *up* is gated; stop/status/sync are not, so a running
  proxy stays manageable after the flag is turned off.

## Breaking changes

- **`RuntimePlugin.run()` signature** gains a keyword-only `backends:
  dict[str, BackendBundle] | None = None`. Existing runtimes that subclass
  `RuntimePlugin.run()` must accept and forward the new kwarg. Solo-mode and
  every in-tree native multi-node runtime are migrated.
- **`RuntimePlugin.executor` property removed.** Replaced by
  `_resolve_executor()` which delegates to
  `orchestration.executor:resolve_executor()`. Subclasses that previously read
  `self.executor` directly should call `self._resolve_executor()`.
- **`_KNOWN_EXECUTORS` set removed.** Executor selector validation now queries
  the SAF plugin registry via `get_extensions(EXT_EXECUTOR)`; the static set
  remains as an in-process fallback for test harnesses that bypass
  `init_sparkrun()`.
- **Native node-command generation** now goes through
  `RuntimePlugin._make_node_command_args` (template). Subclasses that emitted
  rank-specific argv lists by hand should switch to the template; the
  signatures and overall shape are documented in `runtimes/base.py`.
- **vLLM `_build_command`** moved out of `runtimes/vllm_distributed.py` and
  `runtimes/vllm_ray.py` into the new `VllmMixin._build_command()`
  (`runtimes/_vllm_mixin.py`). External subclasses inheriting the previous
  per-runtime method need to inherit from the mixin instead.

## Security fixes

- **URL-sourced recipes are never auto-trusted.** Recipes fetched from a URL
  (e.g. `@spark-arena/<uuid>` links) previously carried no `source_registry`
  and were treated as local — so their `pre_exec`/`post_exec`/`post_commands`
  hooks ran with **no confirmation**. `Recipe.is_url_sourced` now forces these
  through `--trust` / interactive confirmation, closing a "run this link" code-
  execution path.
- **Recipe fetch hardening.** `fetch_and_cache_recipe` enforces `https://`
  only (no plaintext-HTTP tampering), an allowed-host list (`spark-arena.com`;
  off-allowlist hosts require confirmation), redirect re-validation, and a
  response size cap.
- **Proxy bind host is now explicit and persisted.** A new `proxy.host` setting
  is saved and reused (recommended `127.0.0.1`). For backward compatibility an
  unconfigured proxy still binds `0.0.0.0`, but now emits a loud warning each
  start — escalated when no master key is set (unauthenticated gateway exposed
  to the network). Proxy `state.yaml` / `autodiscover.yaml` are written `0600`.
- **Arena OAuth callback** binds to a per-flow random `state` nonce and rejects
  mismatched/missing state before storing the token (token-fixation / CSRF).
- **Sudoers `cache_dir` validation.** `validate_sudoers_path()` (absolute path,
  conservative charset) gates `cache_dir` before it is interpolated into the
  passwordless `chown` sudoers rule.
- **Recipe env no longer expanded.** Control-machine environment variables are
  no longer substituted into recipe `env` values, preventing a third-party
  recipe from exfiltrating host secrets (`$AWS_SECRET_ACCESS_KEY`, ...) into a
  container.
- **Secret masking in logs.** `token`/`key`/`password`/`secret` env values are
  masked in the docker executor DEBUG output (per-var dump and full command).
- Trust gating on `pre_exec` / `post_exec` / `post_commands` (single decision
  via `resolve_recipe_trust`). Local recipes and recipes from a registry
  marked `trusted` are auto-trusted; anything else requires `--trust` or
  prompts.
- **Trust became a per-registry local decision**, stored as a `trusted:` field
  on each entry in the user's own `registries.yaml`. A repository manifest
  cannot grant itself trust. Every registry sparkrun ships as a built-in
  default is first-party and ships trusted; a registry added with
  `sparkrun registry add <url>` is untrusted until the user opts in with
  `--trust` or `sparkrun registry trust <name>`.
- **Container-escape surfaces are trust-gated**
  (`core/launcher.py:_enforce_recipe_mount_trust`). An untrusted recipe may
  not supply host bind-mounts, may not set the `executor_config` privilege
  keys (`privileged`, `cap_add`, `security_opt`, `devices`, `user`,
  `volumes`), and may not select an executor other than `docker` — the
  rootless, namespaced container is the sandbox that justifies running a
  registry recipe's serve command at all. Innocuous resource knobs
  (`shm_size`, `ipc`, `network`, `memory_limit`, `ulimit`, …) are
  deliberately not gated.
- **A mount denylist applies regardless of trust**
  (`utils/shell.py:assert_safe_mount_source`): the host root, the docker
  control socket, SSH keys, and kernel pseudo-filesystems are refused even for
  a trusted recipe. It validates the literal path shape rather than trusting
  control-machine `realpath`, because the mount happens on a remote host whose
  symlink layout differs.
- `orchestration/transfer.py:_run_delegated_copy` validates `source_host`
  against the validated host list and rejects `dest` paths that escape the
  cache root.
- `cli/_setup/_sudo.py`, `_phases.py`, `_uninstall.py`, `_commands.py` all
  call `utils.shell.validate_unix_username()` before interpolating user
  identifiers into sudoers fragments.
- `runtimes/trtllm.py` no longer relaxes SSH host-key checking in the
  MPI rsh wrapper.
- `core/registry.py:_validate_git_url` allowlists `https://`, `git@`,
  `ssh://`, `file://` schemes before any `git clone` invocation.
- OAuth callback CORS allowlist restricted to `AUTH_PROXY_BASE` (no wildcard).
- Token-prefix logging removed from debug paths.
- `utils/shell.py:quote()` wraps `shlex.quote()`; every shell-command
  construction in-tree routes through it.

## Deprecations

- The legacy `resolve_ib_env(ctx, comm_env)` wrapper has been **removed**.
  Per-host communication env now flows exclusively through
  `runtimes/_cluster_ops.py:resolve_comm_env(ctx, comm_env, backends)` with a
  `backends: dict[host, BackendBundle]` map; NVIDIA-only call sites continue to
  emit identical output.
- Recipe topology fields `mode`, `solo_only`, `cluster_only` remain
  deprecated in favor of `min_nodes` / `max_nodes` (carry-over from earlier
  releases; documented in `RECIPES.md`).
