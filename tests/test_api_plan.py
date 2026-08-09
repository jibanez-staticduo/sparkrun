"""``sparkrun.api.plan`` — the decide half of the launch path.

``run`` used to do everything, which forced any caller that needed to *show*
the target hosts before launching to schedule separately and hand the winners
back as ``options.hosts``.  That silently made the display pass authoritative
over placement.  ``plan`` exists so the decision can be made once and reused:
``run(options, plan=plan)`` launches exactly what the plan describes.

These tests pin the contract that makes that safe — ``run`` must not
re-resolve, re-prepare, or re-place when a plan is supplied.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml

import sparkrun.api as api
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

_HOSTS = ["h1", "h2", "h3", "h4"]

_RECIPE_DATA = {
    "sparkrun_version": "2",
    "name": "api.plan test recipe",
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "sglang",
    "mode": "cluster",
    "container": "scitrera/dgx-spark-sglang:latest",
    "defaults": {"port": 30000, "tensor_parallel": 2},
}


@pytest.fixture
def recipe(tmp_path):
    from sparkrun.core.recipe import Recipe

    path = tmp_path / "plan-test.yaml"
    path.write_text(yaml.safe_dump(_RECIPE_DATA))
    return Recipe.load(str(path), resolve=False)


@pytest.fixture
def idle_cluster(monkeypatch):
    """A 4-host cluster where h1/h2 are busy and h3/h4 are idle."""
    busy = set(_HOSTS[:2])

    def _stub_status(hosts, **kwargs):
        return ClusterStatus(
            hosts=tuple(
                HostOccupancy(
                    host=h,
                    workloads=(RunningWorkload(cluster_id="sparkrun_aaaa_bbbb"),) if h in busy else (),
                    used_slots=1 if h in busy else 0,
                    free_slots=0 if h in busy else 1,
                )
                for h in hosts
            ),
            executor="docker",
        )

    monkeypatch.setattr("sparkrun.api._status.status", _stub_status)
    monkeypatch.setattr("sparkrun.api.status", _stub_status)


def _options(recipe, **kw):
    return api.RunOptions(
        recipe=recipe,
        cluster=ClusterDefinition(name="c", hosts=list(_HOSTS), scheduler="occupancy-sparse"),
        hosts=tuple(_HOSTS),
        dry_run=True,
        **kw,
    )


def test_plan_separates_candidates_from_targets(recipe, idle_cluster, v):
    """The plan keeps both host sets: what it could pick and what it picked."""
    plan = api.plan(_options(recipe))

    assert list(plan.candidate_hosts) == _HOSTS
    assert list(plan.host_list) == ["h3", "h4"]
    assert plan.scheduler == "occupancy-sparse"
    assert plan.scheduler_defaulted is False
    assert plan.cluster_id.startswith("sparkrun_")
    assert plan.notes  # "N nodes required, using N of M hosts"


def test_plan_changes_no_cluster_state(recipe, idle_cluster, v):
    """Planning is safe to do and throw away — it must not launch anything."""
    with mock.patch("sparkrun.core.launcher.launch_inference") as launch:
        with mock.patch("sparkrun.api._run._evict_superseded_deployments") as evict:
            api.plan(_options(recipe))

    launch.assert_not_called()
    evict.assert_not_called()


def test_run_with_plan_does_not_replace(recipe, idle_cluster, v):
    """``run(plan=…)`` reuses the decision instead of scheduling again.

    This is the whole point of the split: a second placement pass over the
    first pass's survivors is what turned a scheduler disagreement into a
    launch failure on a cluster with free capacity.
    """
    plan = api.plan(_options(recipe))

    with mock.patch("sparkrun.api._hosts.resolve_effective_hosts") as replace:
        with mock.patch("sparkrun.core.launcher.launch_inference") as launch:
            launch.return_value = mock.MagicMock(
                cluster_id=plan.cluster_id,
                host_list=list(plan.host_list),
                rc=0,
                is_solo=False,
                runtime_info={},
                recipe_ref=None,
                serve_command="",
                container_image="",
                serve_port=0,
                effective_cache_dir="",
            )
            api.run(_options(recipe), plan=plan)

    replace.assert_not_called()
    assert list(launch.call_args.kwargs["host_list"]) == ["h3", "h4"]


def test_run_without_plan_still_places(recipe, idle_cluster, v):
    """``run(options)`` alone is unchanged — it plans internally.

    Library callers that render nothing should keep calling it this way.
    """
    with mock.patch("sparkrun.core.launcher.launch_inference") as launch:
        launch.return_value = mock.MagicMock(
            cluster_id="sparkrun_aaaa_bbbb",
            host_list=["h3", "h4"],
            rc=0,
            is_solo=False,
            runtime_info={},
            recipe_ref=None,
            serve_command="",
            container_image="",
            serve_port=0,
            effective_cache_dir="",
        )
        result = api.run(_options(recipe))

    assert list(launch.call_args.kwargs["host_list"]) == ["h3", "h4"]
    assert result.scheduler == "occupancy-sparse"


def test_run_ensure_skips_launch_when_intent_is_up(recipe, idle_cluster, v):
    """``RunOptions.ensure`` is honoured — it used to be documented but dead."""
    from sparkrun.api._intent import IntentMatch

    match = IntentMatch(
        intent_id="0123456789abcdef",
        cluster_id="sparkrun_0123456789abcdef_aabbccddeeff",
        hosts=("h3", "h4"),
        recipe="test-recipe",
        runtime="sglang",
    )

    with mock.patch("sparkrun.api._intent.find_running_intent", return_value=match):
        with mock.patch("sparkrun.core.launcher.launch_inference") as launch:
            result = api.run(_options(recipe, ensure=True))

    launch.assert_not_called()
    assert result.already_running is True
    assert result.cluster_id == match.cluster_id
    assert result.host_list == ("h3", "h4")
    assert result.launch_result is None
    assert result.rc == 0


def test_run_ensure_launches_when_intent_is_not_up(recipe, idle_cluster, v):
    """No match → ``ensure`` is a no-op and the launch proceeds normally."""
    with mock.patch("sparkrun.api._intent.find_running_intent", return_value=None):
        with mock.patch("sparkrun.core.launcher.launch_inference") as launch:
            launch.return_value = mock.MagicMock(
                cluster_id="sparkrun_aaaa_bbbb",
                host_list=["h3", "h4"],
                rc=0,
                is_solo=False,
                runtime_info={},
                recipe_ref=None,
                serve_command="",
                container_image="",
                serve_port=0,
                effective_cache_dir="",
            )
            result = api.run(_options(recipe, ensure=True))

    launch.assert_called_once()
    assert result.already_running is False


def test_plan_prepares_transport_once(recipe, idle_cluster, v):
    """Transport preparation belongs to planning, not to every pass.

    A provider transport refreshes ephemeral connection details on prepare,
    so preparing again after the plan was built could hand the launch a
    different endpoint than the one the plan was computed against.
    """
    with mock.patch("sparkrun.api._resolve.prepare_transport") as prepare:
        plan = api.plan(_options(recipe))
        assert prepare.call_count == 1

        with mock.patch("sparkrun.core.launcher.launch_inference") as launch:
            launch.return_value = mock.MagicMock(
                cluster_id=plan.cluster_id,
                host_list=list(plan.host_list),
                rc=0,
                is_solo=False,
                runtime_info={},
                recipe_ref=None,
                serve_command="",
                container_image="",
                serve_port=0,
                effective_cache_dir="",
            )
            api.run(_options(recipe), plan=plan)

    assert prepare.call_count == 1, "run() re-prepared a transport the plan already prepared"
