from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skill"
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skill.scheduler_staging.runtime import build_staging_runtime_exports


class _Clock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def time(self) -> float:
        return self.value


def _namespace(clock: _Clock, state_dir: Path):
    return {
        "time": clock,
        "STATE_DIR": state_dir,
        "STAGING_TTL_S": 10,
        "STAGING_FAIL_COOLDOWN_S": 20,
        "NODES": {
            "local": {"host": None},
            "remote": {"host": "user@host"},
            "shared1": {"host": "shared-host", "shared_workspace_group": "shared"},
            "shared2": {"host": "shared-host", "shared_workspace_group": "shared"},
        },
    }


def test_staging_runtime_exports_live_cache_objects_and_ttl_helpers(tmp_path):
    clock = _Clock()
    exports = build_staging_runtime_exports(_namespace(clock, tmp_path))

    cache = exports["_STAGING_CACHE"]
    fresh = ("src", "dst", "/fresh")
    stale = ("src", "dst", "/stale")
    cache[fresh] = 95.0
    cache[stale] = 80.0

    assert exports["_staging_cache_hit"](fresh) is True
    assert fresh in cache
    assert exports["_staging_cache_hit"](stale) is False
    assert stale not in cache

    exports["_mark_staging_cache_success"](("src", "dst", "/new"))
    assert cache[("src", "dst", "/new")] == 100.0

    restarted = build_staging_runtime_exports(_namespace(clock, tmp_path))
    assert restarted["_staging_cache_hit"](("src", "dst", "/new")) is True


def test_code_tar_marker_is_invalidated_when_include_paths_change(tmp_path):
    clock = _Clock()
    namespace = _namespace(clock, tmp_path)
    namespace["NODES"]["code-tar"] = {
        "host": "user@host",
        "launch_staging_method": "code_tar",
        "launch_stage_include_paths": ["src", "configs"],
    }
    key = ("local", "code-tar", "/work")
    exports = build_staging_runtime_exports(namespace)
    exports["_mark_launch_stage_success"](key)

    restarted = build_staging_runtime_exports(namespace)
    assert restarted["_staging_cache_hit"](key) is True

    changed = _namespace(clock, tmp_path)
    changed["NODES"]["code-tar"] = {
        "host": "user@host",
        "launch_staging_method": "code_tar",
        "launch_stage_include_paths": ["src", "configs", "analysis"],
    }
    changed_runtime = build_staging_runtime_exports(changed)
    assert changed_runtime["_staging_cache_hit"](key) is False


def test_code_tar_marker_is_invalidated_when_included_source_changes(tmp_path):
    clock = _Clock()
    source = tmp_path / "src"
    source.mkdir()
    main = source / "main.py"
    main.write_text("print(1)\n", encoding="utf-8")
    namespace = _namespace(clock, tmp_path / "state")
    namespace["NODES"]["code-tar"] = {
        "host": "user@host",
        "launch_staging_method": "code_tar",
        "launch_stage_include_paths": ["src"],
    }
    key = ("local", "code-tar", str(tmp_path))
    exports = build_staging_runtime_exports(namespace)
    exports["_mark_launch_stage_success"](key)

    restarted = build_staging_runtime_exports(namespace)
    assert restarted["_staging_cache_hit"](key) is True

    main.write_text("print('changed')\n", encoding="utf-8")
    changed = build_staging_runtime_exports(namespace)
    assert changed["_staging_cache_hit"](key) is False


def test_rsync_marker_is_invalidated_when_configured_source_changes(tmp_path):
    clock = _Clock()
    source = tmp_path / "src"
    source.mkdir()
    main = source / "main.py"
    main.write_text("print(1)\n", encoding="utf-8")
    namespace = _namespace(clock, tmp_path / "state")
    namespace["NODES"]["rsync"] = {
        "host": "user@host",
        "launch_stage_include_paths": ["src"],
    }
    key = ("local", "rsync", str(tmp_path))
    exports = build_staging_runtime_exports(namespace)
    exports["_mark_launch_stage_success"](key)

    restarted = build_staging_runtime_exports(namespace)
    assert restarted["_staging_cache_hit"](key) is True

    main.write_text("print('changed')\n", encoding="utf-8")
    changed = build_staging_runtime_exports(namespace)
    assert changed["_staging_cache_hit"](key) is False


def test_staging_fingerprint_ignores_excluded_result_sync_outputs(tmp_path):
    clock = _Clock()
    source = tmp_path / "SC-OLH-KG"
    source.mkdir()
    main = source / "main.py"
    main.write_text("print(1)\n", encoding="utf-8")
    profiles = source / "profiles" / "run-1"
    profiles.mkdir(parents=True)
    namespace = _namespace(clock, tmp_path / "state")
    namespace["NODES"]["remote"] = {
        "host": "user@host",
        "launch_stage_include_paths": ["SC-OLH-KG"],
        "launch_stage_fingerprint_exclude_paths": ["SC-OLH-KG/profiles"],
    }
    key = ("local", "remote", str(tmp_path))
    exports = build_staging_runtime_exports(namespace)
    exports["_mark_launch_stage_success"](key)

    (profiles / "result.json").write_text('{"done": true}\n', encoding="utf-8")
    restarted = build_staging_runtime_exports(namespace)
    assert restarted["_staging_cache_hit"](key) is True

    main.write_text("print('changed')\n", encoding="utf-8")
    changed = build_staging_runtime_exports(namespace)
    assert changed["_staging_cache_hit"](key) is False


def test_staging_marker_is_invalidated_when_workspace_topology_changes(tmp_path):
    clock = _Clock()
    source = tmp_path / "src"
    source.mkdir()
    (source / "main.py").write_text("print(1)\n", encoding="utf-8")
    key = ("local", "remote", str(tmp_path))
    shared = _namespace(clock, tmp_path / "state")
    shared["NODES"]["remote"] = {
        "host": "user@host",
        "launch_stage_include_paths": ["src"],
        "shared_workspace_group": "shared-home",
    }
    exports = build_staging_runtime_exports(shared)
    exports["_mark_launch_stage_success"](key)

    private = _namespace(clock, tmp_path / "state")
    private["NODES"]["remote"] = {
        "host": "user@host",
        "launch_stage_include_paths": ["src"],
    }
    assert build_staging_runtime_exports(private)["_staging_cache_hit"](key) is False


def test_staging_runtime_failure_conda_and_launch_state_share_exported_dicts(tmp_path):
    clock = _Clock()
    exports = build_staging_runtime_exports(_namespace(clock, tmp_path))

    exports["_record_staging_failure"]("t1", "remote")
    assert exports["_staging_recently_failed"]("t1", "remote") is True
    clock.value = 130.0
    assert exports["_staging_recently_failed"]("t1", "remote") is False
    assert ("t1", "remote") not in exports["_STAGING_FAILED"]

    clock.value = 200.0
    exports["_record_conda_sync_ok"]("remote", "/env")
    assert exports["_conda_sync_ok"]("remote", "/env") is True
    exports["_record_conda_sync_failed"]("remote", "/env")
    assert exports["_conda_sync_ok"]("remote", "/env") is False

    cwd_key = ("local", "remote", "/work")
    assert exports["_stage_cwd_check"]("remote", "/work") == "needs_stage"
    exports["_mark_launch_stage_cap_exceeded"](cwd_key)
    assert exports["_stage_cwd_check"]("remote", "/work") == "cap_exceeded"
    exports["_mark_launch_stage_success"](cwd_key)
    assert exports["_stage_cwd_check"]("remote", "/work") == "ready"
    assert cwd_key not in exports["_STAGING_CAP_EXCEEDED"]

    clock.value = 211.0
    exports["_STAGING_FAILS"][cwd_key] = (clock.time(), "rsync failed")
    exports["_STAGING_CACHE"].pop(cwd_key, None)
    assert exports["_stage_cwd_check"]("remote", "/work") == "stage_failed"
    assert exports["_stage_failure_reason"]("remote", "/work") == "rsync failed"


def test_staging_runtime_can_migrate_to_is_cache_only_with_ttl(tmp_path):
    clock = _Clock()
    exports = build_staging_runtime_exports(_namespace(clock, tmp_path))
    staged = exports["_STAGED_TASKS"]

    staged[("fresh", "remote")] = 95.0
    staged[("stale", "remote")] = 80.0

    assert exports["_can_migrate_to"]({"id": "fresh"}, "remote") is True
    assert exports["_can_migrate_to"]({"id": "stale"}, "remote") is False
    assert ("stale", "remote") not in staged
    assert exports["_can_migrate_to"]({"id": "missing"}, "remote") is False


def test_staging_runtime_shares_cwd_success_and_guard_across_workspace_group(tmp_path):
    clock = _Clock()
    exports = build_staging_runtime_exports(_namespace(clock, tmp_path))
    key1 = ("local", "shared1", "/work")
    key2 = ("local", "shared2", "/work")

    exports["_mark_launch_stage_success"](key1)

    assert exports["_staging_cache_hit"](key2) is True
    assert key1 in exports["_STAGING_CACHE"]
    assert key2 in exports["_STAGING_CACHE"]
    with exports["_staging_key_guard"](key1) as first:
        with exports["_staging_key_guard"](key2) as second:
            assert first is True
            assert second is False


def test_staging_runtime_shares_checkpoint_success_and_guard_across_workspace_group(tmp_path):
    clock = _Clock()
    exports = build_staging_runtime_exports(_namespace(clock, tmp_path))
    key1 = ("resume_ckpt", "local", "shared1", "/work/checkpoints")
    key2 = ("resume_ckpt", "local", "shared2", "/work/checkpoints")

    exports["_mark_staging_cache_success"](key1)

    assert exports["_staging_cache_hit"](key2) is True
    assert key1 in exports["_STAGING_CACHE"]
    assert key2 in exports["_STAGING_CACHE"]
    with exports["_staging_key_guard"](key1) as first:
        with exports["_staging_key_guard"](key2) as second:
            assert first is True
            assert second is False
