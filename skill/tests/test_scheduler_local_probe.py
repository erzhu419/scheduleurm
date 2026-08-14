from __future__ import annotations

import subprocess

from skill.scheduler_local_probe import (
    MAX_LOCAL_PROBE_SHELL_BYTES,
    LocalBatchProbeDeps,
    local_backend_batch_probe,
)


def _task(task_id, *, node="n1", pids=None, status="running"):
    return {
        "id": task_id,
        "status": status,
        "node": node,
        "remote_pids": list(pids or []),
    }


def _deps(*, outputs=None, calls=None, descendants=None, raise_timeout=False):
    outputs = outputs or {}
    calls = calls if calls is not None else []
    descendants = descendants or {}

    def run_on(node, cmd, **kwargs):
        calls.append((node, cmd, kwargs))
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))
        result = outputs.get(node)
        if result is None:
            return 1, "", "ssh failed"
        return result

    def task_pids(task):
        return list(task.get("remote_pids") or [])

    def descendants_of(pids, ppid_of):
        key = tuple(sorted(pids))
        if key in descendants:
            return set(descendants[key])
        out = set()
        frontier = list(pids)
        while frontier:
            parent = frontier.pop()
            children = [pid for pid, ppid in ppid_of.items() if ppid == parent and pid not in out]
            out.update(children)
            frontier.extend(children)
        return out

    return LocalBatchProbeDeps(
        run_on=run_on,
        task_pids=task_pids,
        descendants_of=descendants_of,
    )


def test_no_running_tasks_returns_empty_without_ssh():
    calls = []
    state = {"tasks": [_task("queued", status="queued"), _task("nopid", pids=[])]}

    assert local_backend_batch_probe(state, deps=_deps(calls=calls)) == {}
    assert calls == []


def test_ssh_failure_marks_node_tasks_unknown():
    state = {"tasks": [_task("a", pids=[100]), _task("b", pids=[200])]}

    out = local_backend_batch_probe(state, deps=_deps(outputs={}))

    assert out["a"]["state"] == "unknown"
    assert out["b"]["state"] == "unknown"
    assert out["a"]["error"].startswith("local backend ssh/proc probe failed on n1")


def test_timeout_marks_unknown_with_timeout_message():
    state = {"tasks": [_task("a", pids=[100])]}

    deps = _deps(raise_timeout=True)
    out = local_backend_batch_probe(state, deps=deps)

    assert out["a"]["state"] == "unknown"
    assert "timed out" in out["a"]["error"]
    assert calls_timeout(deps) == 30


def test_alive_roots_and_descendants_are_aggregated_for_resources():
    probe = "\n".join(
        [
            "A100",
            "===VRAM===",
            "100, 512",
            "101, 256",
            "===PSALL===",
            "100 1 2048 10.5 S",
            "101 100 4096 20.0 R",
            "102 101 1024 1.5 S",
        ]
    )
    state = {"tasks": [_task("a", pids=[100])]}

    out = local_backend_batch_probe(state, deps=_deps(outputs={"n1": (0, probe, "")}))

    assert out["a"]["state"] == "alive"
    assert out["a"]["alive_pids"] == [100, 101, 102]
    assert out["a"]["vram_mb"] == 768
    assert out["a"]["ram_mb"] == 7
    assert out["a"]["pcpu"] == 32.0


def test_zombie_and_dead_ps_rows_do_not_revive_task():
    probe = "\n".join(
        [
            "===VRAM===",
            "101, 256",
            "===PSALL===",
            "100 1 2048 10.0 Z",
            "101 100 4096 20.0 X",
        ]
    )
    state = {"tasks": [_task("a", pids=[100])]}

    out = local_backend_batch_probe(
        state,
        deps=_deps(outputs={"n1": (0, probe, "")}, descendants={(100,): {101}}),
    )

    assert out["a"] == {
        "state": "dead",
        "alive_pids": [],
        "vram_mb": 0,
        "ram_mb": 0,
        "pcpu": 0.0,
        "backend_state": "LOCAL_PID_DEAD",
        "terminal_reason": "local backend pid probe found no live tracked process",
        "terminal_diagnosis_deferred": True,
    }


def test_matching_exit_status_sentinel_is_authoritative_over_live_pid():
    task = _task("a", pids=[100])
    task.update(
        {
            "exit_status_path": "/tmp/a.status",
            "exit_status_token": "launch-a",
            "remote_pid_start_ticks": {"100": 777},
        }
    )
    probe = "\n".join(
        [
            "T\ta\t0\t123.5\tlaunch-a",
            "I\ta\t100",
            "===VRAM===",
            "===PSALL===",
            "100 1 2048 10.0 S",
        ]
    )

    out = local_backend_batch_probe(
        {"tasks": [task]},
        deps=_deps(outputs={"n1": (0, probe, "")}),
    )

    assert out["a"]["state"] == "dead"
    assert out["a"]["exit_code"] == 0
    assert out["a"]["backend_finished_at"] == 123.5
    assert out["a"]["backend_state"] == "LOCAL_EXIT_STATUS"
    assert out["a"]["terminal_diagnosis_deferred"] is True


def test_nonzero_exit_status_forces_terminal_failure():
    task = _task("a", pids=[100])
    task.update(
        {
            "exit_status_path": "/tmp/a.status",
            "exit_status_token": "launch-a",
        }
    )
    probe = "\n".join(
        [
            "T\ta\t7\t123.5\tlaunch-a",
            "A100",
            "===VRAM===",
            "===PSALL===",
            "100 1 2048 10.0 S",
        ]
    )

    out = local_backend_batch_probe(
        {"tasks": [task]},
        deps=_deps(outputs={"n1": (0, probe, "")}),
    )

    assert out["a"]["state"] == "dead"
    assert out["a"]["exit_code"] == 7
    assert out["a"]["terminal_ok"] is False


def test_stale_exit_status_token_is_ignored():
    task = _task("a", pids=[100])
    task.update(
        {
            "exit_status_path": "/tmp/a.status",
            "exit_status_token": "current-launch",
        }
    )
    probe = "\n".join(
        [
            "T\ta\t0\t123.5\tstale-launch",
            "A100",
            "===VRAM===",
            "===PSALL===",
            "100 1 2048 10.0 S",
        ]
    )

    out = local_backend_batch_probe(
        {"tasks": [task]},
        deps=_deps(outputs={"n1": (0, probe, "")}),
    )

    assert out["a"]["state"] == "alive"
    assert "exit_code" not in out["a"]


def test_matching_pid_start_ticks_preserve_live_identity():
    task = _task("a", pids=[100])
    task["remote_pid_start_ticks"] = {"100": 777}
    probe = "\n".join(
        [
            "I\ta\t100",
            "===VRAM===",
            "===PSALL===",
            "100 1 2048 10.0 S",
        ]
    )

    out = local_backend_batch_probe(
        {"tasks": [task]},
        deps=_deps(outputs={"n1": (0, probe, "")}),
    )

    assert out["a"]["state"] == "alive"
    assert out["a"]["alive_pids"] == [100]


def test_pid_start_ticks_mismatch_does_not_revive_reused_pid():
    task = _task("a", pids=[100])
    task["remote_pid_start_ticks"] = {"100": 777}
    probe = "\n".join(
        [
            "M\ta\t100",
            "===VRAM===",
            "===PSALL===",
            "100 1 2048 10.0 S",
        ]
    )

    out = local_backend_batch_probe(
        {"tasks": [task]},
        deps=_deps(outputs={"n1": (0, probe, "")}),
    )

    assert out["a"]["state"] == "dead"
    assert out["a"]["backend_state"] == "LOCAL_PID_IDENTITY_MISMATCH"
    assert "identity" in out["a"]["terminal_reason"]


def test_probe_command_contains_proc_state_guard_and_ps_stat_column():
    calls = []
    state = {"tasks": [_task("a", pids=[12345])]}

    local_backend_batch_probe(
        state,
        deps=_deps(
            outputs={"n1": (0, "===VRAM===\n===PSALL===\n", "")},
            calls=calls,
        ),
    )

    cmd = calls[0][1]
    assert "kill -0 12345" in cmd
    assert "/proc/12345/status" in cmd
    assert 's!="Z" && s!="X"' in cmd
    assert "ps -eo pid= -o ppid= -o rss= -o pcpu= -o stat=" in cmd


def test_probe_command_reads_exit_status_and_checks_pid_start_ticks():
    calls = []
    task = _task("a", pids=[12345])
    task.update(
        {
            "exit_status_path": "/tmp/a status",
            "exit_status_token": "launch-a",
            "remote_pid_start_ticks": {"12345": 9876},
        }
    )

    local_backend_batch_probe(
        {"tasks": [task]},
        deps=_deps(
            outputs={"n1": (0, "===VRAM===\n===PSALL===\n", "")},
            calls=calls,
        ),
    )

    cmd = calls[0][1]
    assert "'/tmp/a status'" in cmd
    assert "/proc/12345/stat" in cmd
    assert "9876" in cmd
    assert "printf 'T\\t%s" in cmd
    assert "printf 'I\\t%s" in cmd
    assert "printf 'M\\t%s" in cmd
    assert cmd.index("/proc/12345/stat") < cmd.index("'/tmp/a status'")


def test_batch_probe_chunks_high_fanout_node_before_ssh_argument_limit():
    calls = []
    tasks = []
    output = []
    for index in range(90):
        task_id = f"task-{index}"
        pid = 10000 + index
        task = _task(task_id, pids=[pid])
        task.update(
            {
                "exit_status_path": f"/tmp/scheduleurm_exit_{task_id}_{'x' * 80}.status",
                "exit_status_token": f"launch-token-{index}",
                "remote_pid_start_ticks": {str(pid): 20000 + index},
            }
        )
        tasks.append(task)
        output.extend((f"I\t{task_id}\t{pid}",))
    output.extend(("===VRAM===", "===PSALL==="))
    output.extend(f"{10000 + index} 1 1024 1.0 S" for index in range(90))

    result = local_backend_batch_probe(
        {"tasks": tasks},
        deps=_deps(outputs={"n1": (0, "\n".join(output), "")}, calls=calls),
    )

    assert len(calls) > 1
    assert all(len(cmd.encode("utf-8")) < MAX_LOCAL_PROBE_SHELL_BYTES + 2048 for _, cmd, _ in calls)
    assert set(result) == {task["id"] for task in tasks}
    assert all(item["state"] == "alive" for item in result.values())


def test_batch_probe_limits_concurrent_node_probes():
    calls = []
    active = {"count": 0, "max": 0}
    release = {}

    def run_on(node, cmd, **kwargs):
        calls.append(node)
        active["count"] += 1
        active["max"] = max(active["max"], active["count"])
        if node == "n1":
            release["n1_started"].set()
            release["n2_started"].wait(timeout=2)
        elif node == "n2":
            release["n2_started"].set()
        active["count"] -= 1
        return 0, "===VRAM===\n===PSALL===\n1 0 1 0.0 S\n2 0 1 0.0 S\n3 0 1 0.0 S\n", ""

    import threading

    release["n1_started"] = threading.Event()
    release["n2_started"] = threading.Event()
    state = {"tasks": [_task("a", node="n1", pids=[1]), _task("b", node="n2", pids=[2]), _task("c", node="n3", pids=[3])]}
    deps = LocalBatchProbeDeps(
        run_on=run_on,
        task_pids=lambda task: list(task.get("remote_pids") or []),
        descendants_of=lambda pids, ppid_of: set(),
        max_workers=2,
    )

    local_backend_batch_probe(state, deps=deps)

    assert active["max"] <= 2
    assert set(calls) == {"n1", "n2", "n3"}


def calls_timeout(deps):
    calls = []

    def run_on(node, cmd, **kwargs):
        calls.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    local_backend_batch_probe(
        {"tasks": [_task("timeout", pids=[100])]},
        deps=LocalBatchProbeDeps(
            run_on=run_on,
            task_pids=lambda task: list(task.get("remote_pids") or []),
            descendants_of=lambda pids, ppid_of: set(),
            timeout_s=deps.timeout_s,
        ),
    )
    return calls[0]
