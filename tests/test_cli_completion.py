"""Tests for Click ``shell_complete`` methods on sparkrun CLI parameter types.

Covers the kubectl-style completion added for ``logs``/``stop`` (and the
other ``TARGET``-taking commands): ``TargetType.shell_complete`` returns
running-workload cluster_ids from the local job metadata cache, falling
back to recipe-name completion when no jobs are cached.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unittest import mock
import yaml

from sparkrun.cli._common import (
    TARGET,
    _complete_targets,
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
# _complete_targets
# ---------------------------------------------------------------------------


def test_complete_targets_empty_when_no_cache(jobs_cache: Path):
    """No jobs directory → empty list, never raises."""
    assert _complete_targets("") == []


def test_complete_targets_returns_all_on_empty_prefix(jobs_cache: Path):
    """Empty incomplete → all cached cluster_ids."""
    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="beta")

    items = _complete_targets("")
    assert len(items) == 2
    values = {i.value for i in items}
    assert values == {"sparkrun_aaaaaaaaaaaa", "sparkrun_bbbbbbbbbbbb"}


def test_complete_targets_matches_full_form(jobs_cache: Path):
    """``sparkrun_abc…`` prefix matches the canonical cluster_id."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbb1234567", recipe="beta")

    items = _complete_targets("sparkrun_abc")
    assert len(items) == 1
    assert items[0].value == "sparkrun_abcdef123456"


def test_complete_targets_matches_bare_digest(jobs_cache: Path):
    """Bare hex digest prefix also matches (short-form CLI input)."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbb1234567", recipe="beta")

    items = _complete_targets("abcd")
    assert len(items) == 1
    # The returned value is always the canonical form.
    assert items[0].value == "sparkrun_abcdef123456"


def test_complete_targets_descriptions(jobs_cache: Path):
    """Completion items carry recipe + runtime + hosts in the description."""
    _write_job_meta(
        jobs_cache,
        "abcdef123456",
        recipe="@eugr/inkling-small-nvfp4",
        runtime="vllm",
        hosts=["127.0.0.1", "192.168.70.8"],
    )

    items = _complete_targets("")
    assert len(items) == 1
    desc = items[0].help or ""
    assert "@eugr/inkling-small-nvfp4" in desc
    assert "vllm" in desc
    assert "127.0.0.1,192.168.70.8" in desc


def test_complete_targets_no_match(jobs_cache: Path):
    """Prefix that matches nothing → empty list."""
    _write_job_meta(jobs_cache, "abcdef123456")
    assert _complete_targets("zzz") == []


def test_complete_targets_handles_exception(jobs_cache: Path, monkeypatch):
    """If api.list_jobs() raises, completion degrades to empty list (never crashes)."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")

    def _boom(*a, **kw):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("sparkrun.api.list_jobs", _boom)
    assert _complete_targets("") == []


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


# ---------------------------------------------------------------------------
# Recipe-name completion, default-cluster scoping, and the running filter
# ---------------------------------------------------------------------------


@pytest.fixture
def default_cluster(monkeypatch):
    """Make ``resolve_cluster()`` (no args) resolve to a known host set.

    That is what a bare ``logs <recipe>`` resolves against, so it decides
    which jobs may be offered by recipe name.
    """
    from sparkrun.core.cluster_manager import ClusterDefinition

    def _fake(*args, **kwargs):
        return ClusterDefinition(name="default", hosts=["h1", "h2"])

    monkeypatch.setattr("sparkrun.api._resolve.resolve_cluster", _fake)
    return {"h1", "h2"}


def test_recipe_name_offered_for_default_cluster_jobs(jobs_cache: Path, default_cluster):
    """On bash the value is all the user sees, so it must be the meaningful one.

    ``BashComplete.format_completion`` emits ``type,value`` and drops the help
    text, so a hex cluster_id completes to something unreadable.  A recipe name
    resolves through intent discovery and is what a human actually types.
    """
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe", hosts=["h1"])

    values = {i.value for i in _complete_targets("")}
    assert values == {"my-recipe"}


def test_offcluster_job_is_offered_by_id_not_recipe(jobs_cache: Path, default_cluster):
    """A recipe name only resolves against the default cluster.

    Offering one whose deployment lives elsewhere would complete to something
    that then fails to resolve — worse than not offering it.  The cluster_id
    form carries its own hosts, so it stays available.
    """
    _write_job_meta(jobs_cache, "abcdef123456", recipe="elsewhere", hosts=["tnr-dead"])

    values = {i.value for i in _complete_targets("")}
    assert values == {"sparkrun_abcdef123456"}


def test_recipe_names_are_deduped(jobs_cache: Path, default_cluster):
    """Relaunches share a recipe; the name is worth offering exactly once."""
    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="same", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="same", hosts=["h1"])

    values = [i.value for i in _complete_targets("")]
    assert values.count("same") == 1
    # The second deployment stays addressable by id.
    assert "sparkrun_bbbbbbbbbbbb" in values or "sparkrun_aaaaaaaaaaaa" in values


def test_running_snapshot_filters_dead_jobs(jobs_cache: Path, default_cluster):
    """A fresh snapshot hides what it saw *not* running."""
    from sparkrun.orchestration.job_metadata import save_running_snapshot

    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="dead", hosts=["h1"])
    save_running_snapshot({"sparkrun_aaaaaaaaaaaa"}, ["h1", "h2"], cache_dir=str(jobs_cache.parent))

    values = {i.value for i in _complete_targets("")}
    assert values == {"alive"}


def test_stale_snapshot_hides_nothing(jobs_cache: Path, default_cluster):
    """Completion must not hide a workload on information it can't vouch for."""
    from sparkrun.orchestration.job_metadata import save_running_snapshot

    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="dead", hosts=["h1"])
    save_running_snapshot({"sparkrun_aaaaaaaaaaaa"}, ["h1"], cache_dir=str(jobs_cache.parent))

    with mock.patch("sparkrun.orchestration.job_metadata.RUNNING_SNAPSHOT_MAX_AGE_S", -1):
        values = {i.value for i in _complete_targets("")}
    assert values == {"alive", "dead"}


def test_uncovered_hosts_are_unknown_not_dead(jobs_cache: Path, default_cluster):
    """A partial sweep must not read as "everything else is dead".

    Placement queries a candidate subset, not the whole cluster, so a snapshot
    routinely covers only some hosts.  A job outside it was never looked at.
    """
    from sparkrun.orchestration.job_metadata import save_running_snapshot

    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="onh1", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="onh2", hosts=["h2"])
    # Only h1 was swept, and nothing was running there.
    save_running_snapshot(set(), ["h1"], cache_dir=str(jobs_cache.parent))

    values = {i.value for i in _complete_targets("")}
    assert values == {"onh2"}, "the unswept host's job must survive; the swept host's must not"


def test_completion_parses_at_most_the_limit(jobs_cache: Path, default_cluster, monkeypatch):
    """Cost is dominated by YAML parsing, so the cap must bound the parse."""
    from sparkrun.cli import _common

    for i in range(40):
        _write_job_meta(jobs_cache, "%012x" % i, recipe="r%d" % i, hosts=["h1"])

    monkeypatch.setattr(_common, "COMPLETION_JOB_LIMIT", 5)
    parsed: list = []

    import sparkrun.api._jobs as jobs_mod

    real_from_file = jobs_mod._job_info_from_file

    def _counting(path):
        parsed.append(path)
        return real_from_file(path)

    monkeypatch.setattr(jobs_mod, "_job_info_from_file", _counting)
    _complete_targets("")
    assert len(parsed) == 5


def test_cluster_option_on_the_line_scopes_recipe_names(jobs_cache: Path, monkeypatch):
    """``logs --cluster lab <TAB>`` scopes to lab, not to the default cluster.

    Completion runs after Click has parsed the options it has seen, so the
    target is knowable.  Without this a user who always passes ``--cluster``
    (and has no ``default_hosts``) is never offered a recipe name at all.
    """
    from sparkrun.core.cluster_manager import ClusterDefinition

    def _fake(cluster_input=None, hosts_input=None, **kwargs):
        if cluster_input == "lab":
            return ClusterDefinition(name="lab", hosts=["h9"])
        raise RuntimeError("no default cluster configured")

    monkeypatch.setattr("sparkrun.api._resolve.resolve_cluster", _fake)
    _write_job_meta(jobs_cache, "abcdef123456", recipe="lab-recipe", hosts=["h9"])

    class _Ctx:
        params = {"cluster_name": "lab", "hosts": None}

    assert {i.value for i in _complete_targets("", _Ctx())} == {"lab-recipe"}
    # With no cluster on the line the default resolution raises, so the job is
    # still offered — by id rather than by a name that wouldn't resolve.
    assert {i.value for i in _complete_targets("")} == {"sparkrun_abcdef123456"}


def test_url_sourced_recipe_completes_by_id(jobs_cache: Path, default_cluster):
    """A URL "name" is valid input but useless as a completion value."""
    _write_job_meta(
        jobs_cache,
        "abcdef123456",
        recipe="https://spark-arena.com/api/recipes/cd00e976/raw",
        hosts=["h1"],
    )
    assert {i.value for i in _complete_targets("")} == {"sparkrun_abcdef123456"}
