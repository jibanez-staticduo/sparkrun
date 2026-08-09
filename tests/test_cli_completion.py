"""Tests for Click ``shell_complete`` methods on sparkrun CLI parameter types.

Covers the kubectl-style completion added for ``logs``/``stop`` (and the
other ``TARGET``-taking commands): ``TargetType.shell_complete`` returns
running-workload cluster_ids from the local job metadata cache, falling
back to recipe-name completion when no jobs are cached.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparkrun.cli._common import (
    TARGET,
    _complete_cluster_ids,
    _describe_job,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_job_meta(jobs_dir: Path, digest: str, **fields) -> Path:
    """Write a YAML job metadata file with sane defaults (mirrors test_api_schedule_status_jobs)."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "cluster_id": f"sparkrun_{digest}",
        "recipe": "test-recipe",
        "runtime": "vllm",
        "hosts": ["h1"],
        **fields,
    }
    path = jobs_dir / (f"{digest}.yaml")
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def jobs_cache(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the sparkrun cache (and thus ``~/.cache/sparkrun/jobs/``) to tmp_path."""
    import sparkrun.core.config as _config_module

    monkeypatch.setattr(_config_module, "DEFAULT_CACHE_DIR", tmp_path, raising=False)
    return tmp_path / "jobs"


# ---------------------------------------------------------------------------
# _describe_job
# ---------------------------------------------------------------------------


def test_describe_job_full():
    """recipe + runtime + hosts all present → all three in the description."""
    from sparkrun.api._models import JobInfo

    job = JobInfo(cluster_id="sparkrun_abc", recipe="my-recipe", runtime="vllm", hosts=("h1", "h2"))
    desc = _describe_job(job)
    assert "my-recipe" in desc
    assert "vllm" in desc
    assert "h1,h2" in desc


def test_describe_job_empty():
    """No recipe / runtime / hosts → empty string (no crash)."""
    from sparkrun.api._models import JobInfo

    job = JobInfo(cluster_id="sparkrun_abc")
    assert _describe_job(job) == ""


def test_describe_job_partial():
    """Only recipe set → just the recipe."""
    from sparkrun.api._models import JobInfo

    job = JobInfo(cluster_id="sparkrun_abc", recipe="my-recipe")
    assert _describe_job(job) == "my-recipe"


# ---------------------------------------------------------------------------
# _complete_cluster_ids
# ---------------------------------------------------------------------------


def test_complete_cluster_ids_empty_when_no_cache(jobs_cache: Path):
    """No jobs directory → empty list, never raises."""
    assert _complete_cluster_ids("") == []


def test_complete_cluster_ids_returns_all_on_empty_prefix(jobs_cache: Path):
    """Empty incomplete → all cached cluster_ids."""
    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="beta")

    items = _complete_cluster_ids("")
    assert len(items) == 2
    values = {i.value for i in items}
    assert values == {"sparkrun_aaaaaaaaaaaa", "sparkrun_bbbbbbbbbbbb"}


def test_complete_cluster_ids_matches_full_form(jobs_cache: Path):
    """``sparkrun_abc…`` prefix matches the canonical cluster_id."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbb1234567", recipe="beta")

    items = _complete_cluster_ids("sparkrun_abc")
    assert len(items) == 1
    assert items[0].value == "sparkrun_abcdef123456"


def test_complete_cluster_ids_matches_bare_digest(jobs_cache: Path):
    """Bare hex digest prefix also matches (short-form CLI input)."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbb1234567", recipe="beta")

    items = _complete_cluster_ids("abcd")
    assert len(items) == 1
    # The returned value is always the canonical form.
    assert items[0].value == "sparkrun_abcdef123456"


def test_complete_cluster_ids_descriptions(jobs_cache: Path):
    """Completion items carry recipe + runtime + hosts in the description."""
    _write_job_meta(
        jobs_cache,
        "abcdef123456",
        recipe="@eugr/inkling-small-nvfp4",
        runtime="vllm",
        hosts=["127.0.0.1", "192.168.70.8"],
    )

    items = _complete_cluster_ids("")
    assert len(items) == 1
    desc = items[0].help or ""
    assert "@eugr/inkling-small-nvfp4" in desc
    assert "vllm" in desc
    assert "127.0.0.1,192.168.70.8" in desc


def test_complete_cluster_ids_no_match(jobs_cache: Path):
    """Prefix that matches nothing → empty list."""
    _write_job_meta(jobs_cache, "abcdef123456")
    assert _complete_cluster_ids("zzz") == []


def test_complete_cluster_ids_handles_exception(jobs_cache: Path, monkeypatch):
    """If api.list_jobs() raises, completion degrades to empty list (never crashes)."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")

    def _boom(*a, **kw):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("sparkrun.api.list_jobs", _boom)
    assert _complete_cluster_ids("") == []


# ---------------------------------------------------------------------------
# TargetType.shell_complete
# ---------------------------------------------------------------------------


def test_target_shell_complete_returns_cluster_ids(jobs_cache: Path):
    """When jobs are cached, TargetType completes cluster_ids (not recipes)."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe")

    items = TARGET.shell_complete(ctx=None, param=None, incomplete="")
    values = {i.value for i in items}
    assert "sparkrun_abcdef123456" in values


def test_target_shell_complete_falls_back_to_recipes(jobs_cache: Path):
    """No jobs cached → recipe-name completion (inherited from RecipeNameType)."""
    # jobs_cache exists but is empty (no .yaml files)
    items = TARGET.shell_complete(ctx=None, param=None, incomplete="")
    # No jobs and no registries in the test sandbox → empty list, but must
    # not raise.
    assert items == []


def test_target_shell_complete_defers_for_paths(jobs_cache: Path):
    """Path-like incomplete (./) defers to recipe/file completion, not cluster_ids."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe")

    # A path prefix should never match a cluster_id — it goes to the parent
    # RecipeNameType.shell_complete (file / recipe path completion).
    items = TARGET.shell_complete(ctx=None, param=None, incomplete="./")
    # File completion returns [] for a non-existent directory in the test sandbox.
    cluster_id_values = {i.value for i in items} & {"sparkrun_abcdef123456"}
    assert cluster_id_values == set()


@pytest.mark.parametrize("prefix", ["~/", "/abs/"])
def test_target_shell_complete_defers_for_other_paths(jobs_cache: Path, prefix: str):
    """``~`` and ``/`` path prefixes also defer to recipe/file completion."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe")

    items = TARGET.shell_complete(ctx=None, param=None, incomplete=prefix)
    cluster_id_values = {i.value for i in items} & {"sparkrun_abcdef123456"}
    assert cluster_id_values == set()


def test_target_shell_complete_defers_for_at_registry(jobs_cache: Path):
    """``@registry`` incomplete defers to recipe completion, not cluster_ids."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe")

    items = TARGET.shell_complete(ctx=None, param=None, incomplete="@")
    # @ prefix triggers registry-name completion; cluster_ids should not appear.
    cluster_id_values = {i.value for i in items} & {"sparkrun_abcdef123456"}
    assert cluster_id_values == set()


def test_target_shell_complete_bare_digest(jobs_cache: Path):
    """Typing a bare hex prefix completes to the full cluster_id."""
    _write_job_meta(jobs_cache, "628f56e0461d", recipe="inkling")

    items = TARGET.shell_complete(ctx=None, param=None, incomplete="628f")
    assert len(items) == 1
    assert items[0].value == "sparkrun_628f56e0461d"
