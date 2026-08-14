from __future__ import annotations

import json

from skill import scheduler_dispatch_intent as dispatch_intent


def test_dispatch_intent_write_read_message_and_clear(tmp_path):
    path = tmp_path / ".dispatch_intent.json"
    payload = dispatch_intent.write_dispatch_intent(
        path,
        label="bulk-a",
        ttl_s=10,
        default_ttl_s=900,
        pid=123,
        now_fn=lambda: 100.0,
    )

    assert payload["label"] == "bulk-a"
    assert payload["pid"] == 123
    assert payload["expires_at"] == 110.0
    assert dispatch_intent.read_dispatch_intent(
        path, pid_alive=lambda pid: int(pid) == 123, now=101.0
    )["label"] == "bulk-a"
    assert dispatch_intent.dispatch_intent_message(payload, now=105.0) == (
        "bulk dispatch active pid=123 label=bulk-a age=5s ttl=5s"
    )

    dispatch_intent.clear_dispatch_intent(path, pid=123)
    assert not path.exists()


def test_dispatch_intent_read_drops_expired_dead_and_corrupt_files(tmp_path):
    expired = tmp_path / "expired.json"
    expired.write_text(json.dumps({"pid": 123, "started_at": 1, "expires_at": 2}), encoding="utf-8")
    assert dispatch_intent.read_dispatch_intent(expired, pid_alive=lambda pid: True, now=3.0) is None
    assert not expired.exists()

    dead = tmp_path / "dead.json"
    dead.write_text(json.dumps({"pid": 123, "started_at": 1, "expires_at": 100}), encoding="utf-8")
    assert dispatch_intent.read_dispatch_intent(dead, pid_alive=lambda pid: False, now=3.0) is None
    assert not dead.exists()

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    assert dispatch_intent.read_dispatch_intent(corrupt, pid_alive=lambda pid: True, now=3.0) is None
    assert not corrupt.exists()


def test_dispatch_intent_preserves_non_dict_and_other_pid(tmp_path):
    non_dict = tmp_path / "non-dict.json"
    non_dict.write_text("[1]", encoding="utf-8")
    assert dispatch_intent.read_dispatch_intent(non_dict, pid_alive=lambda pid: True, now=3.0) is None
    assert non_dict.exists()

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"pid": 999, "started_at": 1, "expires_at": 100}), encoding="utf-8")
    dispatch_intent.clear_dispatch_intent(other, pid=123)
    assert other.exists()
    dispatch_intent.clear_dispatch_intent(other, pid=999)
    assert not other.exists()


def test_scheduler_dispatch_intent_wrappers_use_configured_state_file(monkeypatch, sch):
    monkeypatch.setattr(sch, "_pid_alive_local", lambda pid: True)

    payload = sch._write_dispatch_intent(label="wrapper", ttl_s=2)
    read_back = sch._read_dispatch_intent()

    assert sch.DISPATCH_INTENT_FILE.exists()
    assert read_back["label"] == "wrapper"
    assert read_back["pid"] == payload["pid"]
    assert "label=wrapper" in sch._dispatch_intent_message(read_back)

    sch._clear_dispatch_intent()
    assert not sch.DISPATCH_INTENT_FILE.exists()
