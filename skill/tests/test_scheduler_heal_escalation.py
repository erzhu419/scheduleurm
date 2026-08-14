from __future__ import annotations

import json
from pathlib import Path

from skill.scheduler_diagnostics.heal_escalation import (
    EscalationDeps,
    HealSessionDeps,
    escalation_record,
    fire_heal_session,
    latest_escalations,
    recover_resolved_staging_failures,
    resolve_staged_cwd_escalations,
    write_escalation,
)


def _heal_deps(tmp_path: Path, calls: list, *, now=1000.0):
    def popen(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    return HealSessionDeps(
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        heal_fire_lock=tmp_path / "state" / ".heal_fire.lock",
        heal_debounce_s=90,
        claude_bin="/opt/node/bin/claude",
        node_bin="/opt/node/bin/node",
        claude_cli_js="/opt/node/lib/cli.js",
        heal_fire_log=tmp_path / "logs" / "heal_fires.log",
        home=lambda: tmp_path / "home",
        env_get=lambda key, default: "tester" if key == "USER" else default,
        now=lambda: now,
        strftime=lambda fmt: "2026-07-09 12:34:56",
        popen=popen,
        devnull=object(),
    )


def test_fire_heal_session_spawns_clean_headless_process(tmp_path):
    calls = []

    assert fire_heal_session(deps=_heal_deps(tmp_path, calls)) is True

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == [
        "/opt/node/bin/node",
        "/opt/node/lib/cli.js",
        "-p",
        "/scheduler-heal",
        "--dangerously-skip-permissions",
    ]
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == str(tmp_path / "home")
    assert kwargs["env"] == {
        "HOME": str(tmp_path / "home"),
        "USER": "tester",
        "PATH": "/opt/node/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "CLAUDE_HEAL_FIRE": "1",
    }
    assert "fire @ 2026-07-09 12:34:56" in (tmp_path / "logs" / "heal_fires.log").read_text()


def test_fire_heal_session_debounces_recent_fire(tmp_path):
    calls = []
    deps = _heal_deps(tmp_path, calls, now=1000.0)

    assert fire_heal_session(deps=deps) is True
    assert fire_heal_session(deps=_heal_deps(tmp_path, calls, now=1030.0)) is False

    assert len(calls) == 1


def test_escalation_record_truncates_large_fields():
    record = escalation_record(
        {
            "id": "t1",
            "signature": "sig",
            "project": "proj",
            "node": "node001",
            "gpu_idx": 2,
            "retry_count": 3,
            "cmd": "x" * 250,
            "cwd": "/work",
        },
        "APP_BUG",
        {"reason": "boom", "tail": "y" * 700, "log_path": "/tmp/log"},
        now=lambda: 123.0,
    )

    assert record["ts"] == 123.0
    assert record["task_id"] == "t1"
    assert record["category"] == "APP_BUG"
    assert record["cmd"] == "x" * 200
    assert record["tail"] == "y" * 500
    assert record["status"] == "pending"


def test_write_escalation_appends_jsonl_and_fires_heal(tmp_path):
    fires = []
    path = tmp_path / "state" / "escalations.jsonl"

    record = write_escalation(
        {"id": "t2", "cmd": "python train.py"},
        "OOM",
        {"reason": "out of memory", "tail": "CUDA out of memory"},
        deps=EscalationDeps(
            escalations_file=path,
            fire_heal_session=lambda: fires.append("fire") or True,
            now=lambda: 456.0,
        ),
    )

    assert fires == ["fire"]
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved == record
    assert saved["category"] == "OOM"


def test_staging_resolves_only_missing_project_scripts(tmp_path):
    path = tmp_path / "state" / "escalations.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {
            "ts": 1.0,
            "task_id": "script",
            "status": "pending",
            "category": "ENV_MISSING",
            "node": "node001",
            "cwd": "/local/project",
            "tail": "python: can't open file '/remote/project/run.py': [Errno 2] No such file",
        },
        {
            "ts": 1.0,
            "task_id": "conda",
            "status": "pending",
            "category": "ENV_MISSING",
            "node": "node001",
            "cwd": "/local/project",
            "tail": "/env/bin/python: No such file or directory",
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    assert resolve_staged_cwd_escalations(
        "/local/project",
        ["node001", "node002"],
        path=path,
        now=lambda: 10.0,
    ) == ["script"]

    latest = latest_escalations(path)
    assert latest["script"]["status"] == "resolved"
    assert latest["script"]["resolution"] == "staging_success"
    assert latest["conda"]["status"] == "pending"


def test_recover_resolved_staging_failures_requeues_once():
    failed = {"id": "t1", "status": "failed"}
    state = {"tasks": [failed]}
    calls = []
    records = {
        "t1": {
            "task_id": "t1",
            "status": "resolved",
            "resolution": "staging_success",
            "ts": 10.0,
            "node": "node001",
        }
    }

    def requeue(task, state_arg):
        calls.append((task["id"], state_arg))
        state_arg["tasks"].append({"id": "t2", "status": "queued"})
        return "t2"

    assert recover_resolved_staging_failures(
        state,
        records,
        requeue_after_crash=requeue,
    ) == 1
    assert failed["requeued_as"] == "t2"
    assert failed["resolved_environment_retry"]["resolution"] == "staging_success"
    assert recover_resolved_staging_failures(
        state,
        records,
        requeue_after_crash=requeue,
    ) == 0
    assert len(calls) == 1
