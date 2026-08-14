from __future__ import annotations

from pathlib import Path

from skill.scheduler_state_runtime import build_state_runtime_exports


def _namespace(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return {
        "STATE_DIR": state_dir,
        "LOG_DIR": tmp_path / "logs",
        "QUEUE_FILE": state_dir / "queue.json",
        "VRAM_FILE": state_dir / "vram_history.json",
        "RUNTIME_FILE": state_dir / "runtime_history.json",
        "NODE_PROBE_CACHE_FILE": state_dir / "node_probe_cache.json",
        "DISPATCH_INTENT_FILE": state_dir / ".dispatch_intent.json",
        "LOCK_FILE": state_dir / ".scheduler.lock",
        "WATCHER_LOCK_FILE": state_dir / ".watcher.lock",
        "LAUNCH_EXEC_LOCK_FILE": state_dir / ".launch_exec.lock",
        "LAUNCH_EXEC_LOCK_TIMEOUT_S": 1.0,
        "LAUNCH_EXEC_MAX_PER_NODE": 2,
        "DISPATCH_INTENT_TTL_S": 60.0,
        "NODE_PROBE_CACHE_TTL_S": 300,
        "ALLOW_EMPTY_QUEUE_RESET_ENV": "SCHEDULEURM_TEST_ALLOW_EMPTY",
        "QUEUE_MISSING_GUARD_MAX_AGE_S": 60,
        "EMPTY_QUEUE_RESET_GUARD_MIN_TASKS": 999,
        "EMPTY_QUEUE_RESET_GUARD_MIN_ACTIVE": 999,
        "STATE_LOCK_WARN_WAIT_S": 999,
        "STATE_LOCK_WARN_HOLD_S": 999,
        "_canonicalize_state_node_names": lambda state: state.setdefault("canonicalized", True),
        "_compact_state_for_persistence": lambda state: state.setdefault("compacted", True),
    }


def test_state_runtime_exports_bind_state_and_history_paths(tmp_path):
    namespace = _namespace(tmp_path)
    exports = build_state_runtime_exports(namespace)

    state = {"tasks": [{"id": "t1", "status": "queued", "node": "node001"}], "next_id": 2}
    exports["save_state"](state)
    loaded = exports["load_state"]()

    assert loaded["tasks"][0]["id"] == "t1"
    assert loaded["canonicalized"] is True
    assert loaded["compacted"] is True

    exports["save_history"]({"sig": {"vram_mb": 123}})
    exports["save_runtime_history"]({"sig": {"duration_s": 456}})

    assert exports["load_history"]() == {"sig": {"vram_mb": 123}}
    assert exports["load_runtime_history"]() == {"sig": {"duration_s": 456}}


def test_state_runtime_exports_dispatch_intent_and_probe_cache(tmp_path):
    namespace = _namespace(tmp_path)
    exports = build_state_runtime_exports(namespace)

    intent = exports["_write_dispatch_intent"]("bulk-test", ttl_s=30)
    read_back = exports["_read_dispatch_intent"]()

    assert read_back["label"] == "bulk-test"
    assert read_back["pid"] == intent["pid"]
    assert "bulk dispatch active" in exports["_dispatch_intent_message"](read_back)

    exports["_clear_dispatch_intent"]()
    assert exports["_read_dispatch_intent"]() is None

    exports["_save_node_probe_cache"]({"node001": {"alive": True}})
    assert exports["_load_node_probe_cache"]()["node001"]["alive"] is True


def test_launch_exec_slot_prefix_sanitizes_node_name(tmp_path):
    namespace = _namespace(tmp_path)
    exports = build_state_runtime_exports(namespace)

    prefix = exports["_launch_exec_slot_prefix"]("node/007 weird")

    assert prefix.name == ".launch_exec.lock.node_007_weird"
    assert prefix.parent == namespace["LAUNCH_EXEC_LOCK_FILE"].parent
