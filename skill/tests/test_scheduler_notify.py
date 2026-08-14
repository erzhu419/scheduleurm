from __future__ import annotations

import json

from skill.scheduler_notification.notify import (
    NotifyDeps,
    compact_task_event_payload,
    format_feishu,
    load_feishu_cfg,
    notify,
    send_feishu,
)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b"ok"


def _deps(tmp_path, *, urlopen=None, request_factory=None):
    calls = {"rotate": [], "requests": []}

    def rotate(path, max_mb, generations):
        calls["rotate"].append((path, max_mb, generations))

    def request(url, **kwargs):
        calls["requests"].append((url, kwargs))
        return {"url": url, **kwargs}

    deps = NotifyDeps(
        watcher_log=tmp_path / "watcher.log",
        watcher_log_max_mb=50,
        watcher_log_generations=3,
        feishu_config=tmp_path / "feishu.json",
        maybe_rotate_log=rotate,
        format_task_location=lambda task: f"{task.get('node', '?')}:GPU{task.get('gpu_idx', '?')}",
        format_mem_gb=lambda mb: f"{mb / 1024:.2f}GB",
        max_launch_retry=3,
        now=lambda: 1000.0,
        request_factory=request_factory or request,
        urlopen=urlopen or (lambda req, timeout=5: _Response()),
    )
    return deps, calls


def test_load_feishu_cfg_requires_push_mode_and_webhook(tmp_path):
    cfg = tmp_path / "feishu.json"

    assert load_feishu_cfg(cfg) is None
    cfg.write_text("{bad json", encoding="utf-8")
    assert load_feishu_cfg(cfg) is None
    cfg.write_text(json.dumps({"mode": "pull", "webhook_url": "x"}), encoding="utf-8")
    assert load_feishu_cfg(cfg) is None
    cfg.write_text(json.dumps({"mode": "push"}), encoding="utf-8")
    assert load_feishu_cfg(cfg) is None
    cfg.write_text(json.dumps({"mode": "push", "webhook_url": "https://example"}), encoding="utf-8")
    assert load_feishu_cfg(cfg) == {"mode": "push", "webhook_url": "https://example"}


def test_notify_writes_jsonl_and_skips_feishu_when_disabled(tmp_path):
    deps, calls = _deps(tmp_path)

    notify("heartbeat", {"running": 1, "launching": 0, "queued": 2, "nodes": ["n0:0GB"]},
           feishu_enabled=False, deps=deps)

    row = json.loads((tmp_path / "watcher.log").read_text().strip())
    assert row == {
        "ts": 1000.0,
        "type": "heartbeat",
        "payload": {"running": 1, "launching": 0, "queued": 2, "nodes": ["n0:0GB"]},
    }
    assert calls["rotate"] == [(tmp_path / "watcher.log", 50, 3)]
    assert calls["requests"] == []


def test_notify_posts_feishu_when_configured(tmp_path):
    deps, calls = _deps(tmp_path)
    deps.feishu_config.write_text(
        json.dumps({"mode": "push", "webhook_url": "https://example/webhook"}),
        encoding="utf-8",
    )

    notify(
        "task_launched",
        {
            "id": "t1",
            "project": "Proj",
            "node": "node001",
            "gpu_idx": 2,
            "remote_pids": [123],
            "description": "hello",
        },
        deps=deps,
    )

    assert calls["requests"][0][0] == "https://example/webhook"
    body = json.loads(calls["requests"][0][1]["data"].decode())
    assert body["msg_type"] == "text"
    assert "t1 Proj launched on node001:GPU2 pid=123" in body["content"]["text"]


def test_send_feishu_failure_is_logged_without_raising(tmp_path):
    def bad_urlopen(_req, timeout=5):
        raise RuntimeError("network down")

    deps, _calls = _deps(tmp_path, urlopen=bad_urlopen)

    send_feishu("https://example/webhook", "hello", deps=deps)

    row = json.loads((tmp_path / "watcher.log").read_text().strip())
    assert row["type"] == "feishu_send_failed"
    assert "network down" in row["error"]


def test_format_feishu_and_compact_task_event_payload_preserve_existing_fields(tmp_path):
    deps, _calls = _deps(tmp_path)
    retry = format_feishu("task_launch_retry", {"task_id": "t1", "attempt": 2, "error": "ssh failed"}, deps=deps)
    killed = format_feishu("task_killed", {
        "task_id": "t2",
        "actor": {"label": "tester"},
        "action": "force",
        "node": "node001",
        "reason": "requested",
    }, deps=deps)

    assert "(2/3)" in retry and "ssh failed" in retry
    assert "kill t2 by tester" in killed

    compact = compact_task_event_payload({
        "id": "t1",
        "status": "done",
        "project": "Proj",
        "large_unused_blob": "x" * 1000,
        "_diagnosis": {
            "is_crash": False,
            "reason": "ok",
            "tail": "large tail omitted",
            "success_marker": "DONE",
        },
    })

    assert compact["id"] == "t1"
    assert compact["project"] == "Proj"
    assert "large_unused_blob" not in compact
    assert compact["_diagnosis"] == {
        "is_crash": False,
        "reason": "ok",
        "success_marker": "DONE",
    }
