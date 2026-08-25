"""``SYS_PTRACE`` is carried by default for vLLM / SGLang / TRT-LLM.

The capability is what makes a hung engine inspectable (``py-spy dump``,
``gdb -p``) from a ``docker exec -u root``.  It is contributed through
:meth:`RuntimePlugin.default_executor_config` — chain layer 7 — so every layer
above it can override or drop it.

The non-obvious half is that this only works because the docker executor's
rootless ``apply_runtime_adjustments`` (layer 6) no longer sets ``cap_add: []``.
That entry could not harden anything (Docker grants no capabilities unless
asked, so ``[]`` and ``None`` emit identical flags); it only suppressed layers
7-9 while layers 1-4 set ``cap_add`` freely.  These tests pin the resulting
behaviour so the suppressor can't come back unnoticed.
"""

import pytest

from sparkrun.orchestration.executor import resolve_executor
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.trtllm import TrtllmRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.runtimes.vllm_ray import VllmRayRuntime

_RUNTIMES = [VllmRayRuntime, VllmDistributedRuntime, SglangRuntime, TrtllmRuntime]


class _FakeRecipe:
    """Minimal stand-in for the recipe layer of the resolution chain."""

    def __init__(self, executor_config=None):
        self.executor = "docker"
        self.executor_config = executor_config or {}
        self.builder = None


@pytest.mark.parametrize("cls", _RUNTIMES, ids=lambda c: c.runtime_name)
def test_runtime_declares_the_cap(cls):
    assert cls().default_executor_config()["cap_add"] == ["SYS_PTRACE"]


def test_eugr_vllm_inherits_the_cap():
    """eugr-vllm extends VllmRayRuntime, so it must inherit the mixin's default."""
    from sparkrun.runtimes.eugr_vllm_ray import EugrVllmRayRuntime

    assert EugrVllmRayRuntime().default_executor_config()["cap_add"] == ["SYS_PTRACE"]


def test_trtllm_keeps_its_ulimits():
    """The cap moved to executor_config — TRT-LLM's raw ulimit opts are untouched."""
    opts = TrtllmRuntime().get_extra_docker_opts()
    assert opts == ["--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]
    assert "SYS_PTRACE" not in " ".join(opts)


@pytest.mark.parametrize("cls", _RUNTIMES, ids=lambda c: c.runtime_name)
def test_cap_survives_rootless_hardening(cls):
    """Layer 6 must not clear ``cap_add`` out from under the runtime layer."""
    executor = resolve_executor(runtime=cls(), rootless=True, auto_user=True)
    assert executor.config.cap_add == ["SYS_PTRACE"]
    assert "--cap-add SYS_PTRACE" in executor.run_cmd(
        "example/image:latest",
        command="true",
        container_name="ptrace-test",
    )


def test_rootless_hardening_otherwise_intact():
    """Dropping the ``cap_add`` entry must not weaken the rest of rootless mode."""
    executor = resolve_executor(runtime=SglangRuntime(), rootless=True, auto_user=True)
    cfg = executor.config
    assert cfg.privileged is False
    assert cfg.security_opt == ["no-new-privileges"]
    assert "memlock=-1:-1" in (cfg.ulimit or [])
    assert cfg.user == "$SHELL_USER"


def test_recipe_can_override_the_cap():
    """A recipe's ``cap_add`` replaces the runtime default (list, not merge)."""
    executor = resolve_executor(
        runtime=SglangRuntime(),
        recipe=_FakeRecipe({"cap_add": ["IPC_LOCK"]}),
        rootless=True,
    )
    assert executor.config.cap_add == ["IPC_LOCK"]


def test_cli_can_drop_the_cap():
    """``-o cap_add=`` is the opt-out — the raw-flag route had none."""
    executor = resolve_executor(
        runtime=SglangRuntime(),
        cli_overrides={"cap_add": []},
        rootless=True,
    )
    assert executor.config.cap_add is None
    assert "--cap-add" not in executor.run_cmd("example/image:latest", command="true")


def test_runtimes_without_the_default_get_no_caps():
    """Runtimes that don't ask keep Docker's default capability set."""
    from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

    assert LlamaCppRuntime().default_executor_config() == {}
    executor = resolve_executor(runtime=LlamaCppRuntime(), rootless=True)
    assert executor.config.cap_add is None
