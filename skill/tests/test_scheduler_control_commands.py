from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from skill.scheduler_control_commands import (
    ControlCommandDeps,
    cmd_claims,
    cmd_compact,
    cmd_priority,
    cmd_show,
    cmd_task_log,
)


@contextmanager
def _lock(*args, **kwargs):
    yield


def _deps(tmp_path, **overrides):
    queue_file = tmp_path / "queue.json"
    queue_file.write_text("{}", encoding="utf-8")
    values = dict(
        state_lock=_lock,
        load_state=lambda: {"tasks": []},
        save_state=lambda state: None,
        lock_timeout_error=TimeoutError,
        queue_file=queue_file,
        archive_file=tmp_path / "archive.jsonl",
        archive_max_hot_terminal=5,
        compact_age_days_from_args=lambda args: 1.0,
        terminal_archive_stats=lambda *args, **kwargs: {
            "count": 0,
            "age_count": 0,
            "extra_count": 0,
            "excess_count": 0,
            "approx_bytes": 0,
            "by_status": {},
        },
        archive_terminal_tasks=lambda *args, **kwargs: 0,
        node_configs={"local": {"host": None}, "node001": {"host": "node001"}},
        claim_enabled_for=lambda node: False,
        claim_snapshot=lambda node: {"ok": True, "claims": [], "intents": []},
        format_claim_record=lambda claim: f"{claim.get('task_id')}@gpu{claim.get('gpu_idx')}",
        find_task_record=lambda task_id, include_archive=True: (None, ""),
        tail_task_log=lambda task, lines=80: (False, "", "missing"),
        node_is_windows=lambda node: False,
        ssh_base_args=lambda node: ["ssh", node],
        print_result_artifacts=lambda task, include_log=True: None,
    )
    values.update(overrides)
    return ControlCommandDeps(**values)


def test_task_log_json_uses_hot_or_archive_lookup(tmp_path, capsys):
    task = {"id": "t1", "status": "running", "node": "node001", "log_path": "/tmp/t1.log"}

    cmd_task_log(
        SimpleNamespace(id="t1", no_archive=False, lines=3, json=True, no_header=False),
        deps=_deps(
            tmp_path,
            find_task_record=lambda task_id, include_archive=True: (task, "archive"),
            tail_task_log=lambda task, lines=80: (True, "/tmp/t1.log", "line1\nline2\n"),
        ),
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["source"] == "archive"
    assert payload["tail"] == "line1\nline2\n"


def test_priority_only_updates_queued_tasks(tmp_path):
    state = {
        "tasks": [
            {"id": "tQ", "status": "queued", "priority": "normal", "description": "queued"},
            {"id": "tR", "status": "running", "priority": "normal"},
        ]
    }
    saved = []
    deps = _deps(tmp_path, load_state=lambda: state, save_state=lambda value: saved.append(value))

    cmd_priority(SimpleNamespace(id="tQ", level="high"), deps=deps)

    assert state["tasks"][0]["priority"] == "high"
    assert saved == [state]
    with pytest.raises(SystemExit) as exc:
        cmd_priority(SimpleNamespace(id="tR", level="low"), deps=deps)
    assert "not queued" in str(exc.value)


def test_compact_dry_run_reports_stats_without_archiving(tmp_path, capsys):
    archive_calls = []

    cmd_compact(
        SimpleNamespace(
            keep_terminal=2,
            all_terminal=False,
            lock_timeout=None,
            dry_run=True,
        ),
        deps=_deps(
            tmp_path,
            load_state=lambda: {"tasks": [{"id": "done", "status": "done"}]},
            terminal_archive_stats=lambda *args, **kwargs: {
                "count": 1,
                "age_count": 1,
                "extra_count": 0,
                "excess_count": 0,
                "approx_bytes": 123,
                "by_status": {"done": 1},
            },
            archive_terminal_tasks=lambda *args, **kwargs: archive_calls.append(args) or 1,
        ),
    )

    out = capsys.readouterr().out
    assert "would archive 1 terminal tasks" in out
    assert "by_status={'done': 1}" in out
    assert archive_calls == []


def test_claims_text_sorts_claims_and_intents(tmp_path, capsys):
    snapshots = {
        "node001": {
            "ok": True,
            "claims": [
                {"task_id": "t2", "gpu_idx": 1, "pid": 22},
                {"task_id": "t1", "gpu_idx": 0},
            ],
            "intents": [
                {"task_id": "late", "gpu_idx": 0, "intent_at": 2, "scheduler_id": "b"},
                {"task_id": "early", "gpu_idx": 0, "intent_at": 1, "scheduler_id": "a"},
            ],
        }
    }

    cmd_claims(
        SimpleNamespace(node=None, json=False),
        deps=_deps(
            tmp_path,
            node_configs={"node001": {}},
            claim_enabled_for=lambda node: True,
            claim_snapshot=lambda node: snapshots[node],
        ),
    )

    out = capsys.readouterr().out
    assert out.index("t1@gpu0 pending") < out.index("t2@gpu1 pid=22")
    assert out.index("01. early@gpu0") < out.index("02. late@gpu0")


def test_show_prints_local_tail_hint_and_result_artifacts(tmp_path, capsys):
    artifact_calls = []
    task = {"id": "t1", "status": "done", "node": "local", "log_path": "/tmp/t1.log"}

    cmd_show(
        SimpleNamespace(id="t1"),
        deps=_deps(
            tmp_path,
            find_task_record=lambda task_id, include_archive=True: (task, "hot"),
            print_result_artifacts=lambda task, include_log=True: artifact_calls.append((task, include_log)),
        ),
    )

    out = capsys.readouterr().out
    assert '"id": "t1"' in out
    assert "# tail log: tail -f /tmp/t1.log" in out
    assert artifact_calls == [(task, True)]
