from __future__ import annotations

from argparse import Namespace


def _bulk_spec(i: int) -> dict:
    return {
        "description": f"bulk task {i}",
        "cmd": "echo ok",
        "cwd": "/tmp",
        "signature": f"bulk/sig-{i}",
        "vram": 0,
        "ram_mb": 128,
        "cpu": 1,
        "allowed_nodes": ["node001", "node002"],
    }


def test_submit_jsonl_uses_batched_ids_and_cached_history(monkeypatch, sch, capsys):
    specs = [_bulk_spec(i) for i in range(25)]
    calls = {
        "load_history": 0,
        "load_runtime_history": 0,
        "allocate_task_ids": 0,
        "allocate_task_id": 0,
        "history_get": 0,
    }

    monkeypatch.setattr(sch, "_load_submit_jsonl_specs", lambda args: specs)

    def fake_load_history():
        calls["load_history"] += 1
        return {}

    def fake_load_runtime_history():
        calls["load_runtime_history"] += 1
        return {}

    real_allocate_task_ids = sch._allocate_task_ids

    def tracked_allocate_task_ids(state, count):
        calls["allocate_task_ids"] += 1
        return real_allocate_task_ids(state, count)

    def forbidden_allocate_task_id(state):
        calls["allocate_task_id"] += 1
        raise AssertionError("_allocate_task_id should not be called per bulk item")

    def forbidden_history_get(signature):
        calls["history_get"] += 1
        raise AssertionError("history_get should not be called per bulk item")

    monkeypatch.setattr(sch, "load_history", fake_load_history)
    monkeypatch.setattr(sch, "load_runtime_history", fake_load_runtime_history)
    monkeypatch.setattr(sch, "_allocate_task_ids", tracked_allocate_task_ids)
    monkeypatch.setattr(sch, "_allocate_task_id", forbidden_allocate_task_id)
    monkeypatch.setattr(sch, "history_get", forbidden_history_get)
    sch.save_state({"tasks": [], "next_id": 20846})

    args = Namespace(
        trusted=True,
        stdin=True,
        file=None,
        json=True,
        intent_label="test-bulk",
        intent_ttl=60,
        lock_timeout=10,
    )

    sch.cmd_submit_jsonl(args)
    out = capsys.readouterr().out
    state = sch.load_state()

    assert '"count": 25' in out
    assert [t["id"] for t in state["tasks"][:3]] == ["t20846", "t20847", "t20848"]
    assert state["next_id"] == 20871
    assert calls == {
        "load_history": 1,
        "load_runtime_history": 1,
        "allocate_task_ids": 1,
        "allocate_task_id": 0,
        "history_get": 0,
    }
    assert not sch.DISPATCH_INTENT_FILE.exists()
