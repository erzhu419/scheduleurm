from __future__ import annotations

from skill import scheduler_staging_state as staging_state


def test_ttl_cache_hit_preserves_fresh_and_pops_stale():
    cache = {
        "fresh": 95.0,
        "stale": 80.0,
    }

    assert staging_state.ttl_cache_hit(cache, "fresh", ttl_s=10, now=lambda: 100.0) is True
    assert "fresh" in cache
    assert staging_state.ttl_cache_hit(cache, "stale", ttl_s=10, now=lambda: 100.0) is False
    assert "stale" not in cache
    assert staging_state.ttl_cache_hit(cache, "missing", ttl_s=10, now=lambda: 100.0) is False


def test_recent_failure_and_record_staging_failure_are_ttl_and_lru_bounded():
    failed = {("old0", "n"): 1.0, ("old1", "n"): 2.0, ("old2", "n"): 3.0, ("old3", "n"): 4.0}

    staging_state.record_staging_failure(
        failed,
        "new",
        "n",
        max_entries=4,
        now=lambda: 20.0,
    )

    assert ("new", "n") in failed
    assert len(failed) < 5
    assert staging_state.recently_failed(
        failed,
        "new",
        "n",
        cooldown_s=10,
        now=lambda: 25.0,
    ) is True
    assert staging_state.recently_failed(
        failed,
        "new",
        "n",
        cooldown_s=10,
        now=lambda: 31.0,
    ) is False
    assert ("new", "n") not in failed


def test_conda_sync_marker_success_failure_and_expiry():
    markers = {}

    staging_state.record_conda_sync_ok(markers, "node", "/env", now=lambda: 10.0)
    assert staging_state.conda_sync_ok(markers, "node", "/env", ttl_s=10, now=lambda: 19.0) is True

    staging_state.record_conda_sync_failed(markers, "node", "/env")
    assert staging_state.conda_sync_ok(markers, "node", "/env", ttl_s=10, now=lambda: 19.0) is False

    markers[("node", "/old")] = 1.0
    assert staging_state.conda_sync_ok(markers, "node", "/old", ttl_s=10, now=lambda: 20.0) is False
    assert ("node", "/old") not in markers


def test_stage_cwd_check_covers_ready_cap_failed_and_needs_stage():
    node_configs = {
        "local": {"host": None},
        "skip": {"host": "remote", "skip_launch_staging": True},
        "remote": {"host": "remote"},
    }
    cache = set()
    caps = {}
    fails = {}

    assert staging_state.stage_cwd_check(
        "local",
        "/work",
        node_configs=node_configs,
        staging_cache_hit=lambda key: key in cache,
        staging_cap_exceeded=caps,
        staging_fails=fails,
        staging_ttl_s=10,
        staging_fail_cooldown_s=20,
        now=lambda: 100.0,
    ) == "ready"
    assert staging_state.stage_cwd_check(
        "skip",
        "/work",
        node_configs=node_configs,
        staging_cache_hit=lambda key: key in cache,
        staging_cap_exceeded=caps,
        staging_fails=fails,
        staging_ttl_s=10,
        staging_fail_cooldown_s=20,
        now=lambda: 100.0,
    ) == "ready"

    key = ("local", "remote", "/work")
    cache.add(key)
    assert staging_state.stage_cwd_check(
        "remote",
        "/work",
        node_configs=node_configs,
        staging_cache_hit=lambda item: item in cache,
        staging_cap_exceeded=caps,
        staging_fails=fails,
        staging_ttl_s=10,
        staging_fail_cooldown_s=20,
        now=lambda: 100.0,
    ) == "ready"
    cache.clear()

    caps[key] = 95.0
    assert staging_state.stage_cwd_check(
        "remote",
        "/work",
        node_configs=node_configs,
        staging_cache_hit=lambda item: False,
        staging_cap_exceeded=caps,
        staging_fails=fails,
        staging_ttl_s=10,
        staging_fail_cooldown_s=20,
        now=lambda: 100.0,
    ) == "cap_exceeded"
    caps[key] = 80.0
    assert staging_state.stage_cwd_check(
        "remote",
        "/work",
        node_configs=node_configs,
        staging_cache_hit=lambda item: False,
        staging_cap_exceeded=caps,
        staging_fails=fails,
        staging_ttl_s=10,
        staging_fail_cooldown_s=20,
        now=lambda: 100.0,
    ) == "needs_stage"
    assert key not in caps

    fails[key] = (90.0, "rsync rc=12")
    assert staging_state.stage_cwd_check(
        "remote",
        "/work",
        node_configs=node_configs,
        staging_cache_hit=lambda item: False,
        staging_cap_exceeded=caps,
        staging_fails=fails,
        staging_ttl_s=10,
        staging_fail_cooldown_s=20,
        now=lambda: 100.0,
    ) == "stage_failed"
    assert staging_state.stage_failure_reason(fails, "remote", "/work") == "rsync rc=12"
    fails[key] = (70.0, "old")
    assert staging_state.stage_cwd_check(
        "remote",
        "/work",
        node_configs=node_configs,
        staging_cache_hit=lambda item: False,
        staging_cap_exceeded=caps,
        staging_fails=fails,
        staging_ttl_s=10,
        staging_fail_cooldown_s=20,
        now=lambda: 100.0,
    ) == "needs_stage"
    assert key not in fails


def test_stage_markers_and_can_migrate_to_use_same_ttl_semantics():
    cache = {}
    caps = {("local", "n", "/work"): 1.0}
    fails = {("local", "n", "/work"): (1.0, "old")}
    key = ("local", "n", "/work")

    staging_state.mark_launch_stage_success(cache, caps, fails, key, now=lambda: 10.0)
    assert cache[key] == 10.0
    assert key not in caps
    assert key not in fails

    staging_state.mark_launch_stage_cap_exceeded(caps, key, now=lambda: 11.0)
    assert caps[key] == 11.0
    staging_state.mark_staging_cache_success(cache, ("src", "dst", "/ckpt"), now=lambda: 12.0)
    assert cache[("src", "dst", "/ckpt")] == 12.0

    staged_tasks = {("t1", "n"): 95.0, ("t2", "n"): 80.0}
    assert staging_state.can_migrate_to(
        {"id": "t1"}, "n", staged_tasks=staged_tasks, staging_ttl_s=10, now=lambda: 100.0
    ) is True
    assert staging_state.can_migrate_to(
        {"id": "t2"}, "n", staged_tasks=staged_tasks, staging_ttl_s=10, now=lambda: 100.0
    ) is False
    assert ("t2", "n") not in staged_tasks
