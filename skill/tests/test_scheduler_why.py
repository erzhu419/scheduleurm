from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

from skill.scheduler_commands import why


@contextmanager
def _noop_lock(**kwargs):
    yield


def _deps(*, state=None, history=None, nodes=None, node_configs=None, gpu_fits=None):
    state = state if state is not None else {"tasks": []}
    history = history if history is not None else {}
    nodes = nodes if nodes is not None else []
    node_configs = node_configs if node_configs is not None else {}
    return why.WhyCommandDeps(
        node_configs=node_configs,
        vram_margin_mb=500,
        hpc_cpu_pool_soft_require_nodes=lambda task: [],
        blocked_nodes_for_task=lambda task: set(),
        launch_failed_nodes_for_task=lambda task: set(),
        node_resources_ok=lambda task, node_state, node_info: (True, "ok"),
        node_is_windows=lambda node: False,
        node_cpu_fallback_block_reason=lambda task, node, node_info: None,
        gpu_fits=gpu_fits or (lambda task, gpu, node_info: True),
        algorithm_gpu_fit_block_reason=lambda task, gpu, node_info: None,
        hard_rule_bypassed=lambda *args, **kwargs: False,
        gpu_freeze_line_mb=lambda total: total // 3 + 512,
        task_ignores_one_third_pack_rule=lambda task, node_info: False,
        node_gpu_util_limit=lambda node_info: node_info.get("gpu_util_limit"),
        state_lock=lambda **kwargs: _noop_lock(**kwargs),
        lock_timeout_error=TimeoutError,
        load_state=lambda: state,
        load_history=lambda: history,
        probe_all=lambda: list(nodes),
        apply_cpu_slot_accounting_to_nodes=lambda state_arg, nodes_arg: None,
        claim_enabled_for=lambda node: False,
        format_claim_intent_hint_for_task=lambda task, node, snap: "",
    )


def test_explain_node_fit_marks_monitor_only_login_node_disabled():
    deps = _deps(node_configs={"login": {"monitor_only": True}})

    got = why.explain_node_fit(
        {"id": "t1", "est_vram_mb": 0},
        {"name": "login", "alive": True, "gpus": []},
        deps=deps,
    )

    assert "login-node-disabled" in got


def test_explain_node_fit_reports_vram_margin_rejection():
    deps = _deps(
        node_configs={"node001": {"max_vram_per_task": None}},
        gpu_fits=lambda task, gpu, node_info: False,
    )
    task = {"id": "t1", "est_vram_mb": 2000}
    node = {
        "name": "node001",
        "alive": True,
        "free_cpu": 8,
        "free_ram_mb": 64000,
        "gpus": [{"idx": 0, "used_mb": 0, "free_mb": 2200, "total_mb": 12000, "util_pct": 0}],
    }

    got = why.explain_node_fit(task, node, deps=deps)

    assert "no-GPU-fit" in got
    assert "free=2200MB < est+margin (2000+500)" in got


def test_cmd_why_prints_history_and_per_node_fit(capsys):
    task = {
        "id": "t1",
        "status": "queued",
        "description": "unit queued",
        "signature": "proj/run/a",
        "priority": "normal",
        "est_vram_mb": 0,
        "ram_mb": 1024,
        "cpu_cores": 1,
        "last_block_reason": "waiting for resources",
    }
    deps = _deps(
        state={"tasks": [task]},
        history={
            "proj/run/a": {"vram_mb": 100, "ram_mb": 200, "cpu_cores": 1, "dur_s_runs": 2},
            "proj/run/b": {"vram_mb": 50},
        },
        nodes=[{
            "name": "node001",
            "alive": True,
            "free_cpu": 8,
            "free_ram_mb": 64000,
            "gpus": [],
        }],
        node_configs={"node001": {"max_vram_per_task": None}},
    )

    why.cmd_why(SimpleNamespace(id="t1", lock_timeout=None), deps=deps)

    out = capsys.readouterr().out
    assert "=== why t1 ===" in out
    assert "history['proj/run/a']" in out
    assert "sibling signatures under 'proj/run'/" in out
    assert "node001    : FITS (CPU-only)" in out


def test_cmd_why_applies_startup_cpu_reservations_before_fit_analysis(capsys):
    task = {
        "id": "t1",
        "status": "queued",
        "signature": "proj/run/a",
        "est_vram_mb": 0,
        "ram_mb": 1024,
        "cpu_cores": 12,
    }
    nodes = [{
        "name": "node001",
        "alive": True,
        "free_cpu": 48,
        "free_ram_mb": 64000,
        "gpus": [],
    }]
    calls = []

    def apply_accounting(state_arg, nodes_arg):
        calls.append((state_arg, nodes_arg))
        nodes_arg[0]["cpu_hard_free"] = 0

    deps = replace(
        _deps(
            state={"tasks": [task]},
            nodes=nodes,
            node_configs={"node001": {"max_vram_per_task": None}},
        ),
        apply_cpu_slot_accounting_to_nodes=apply_accounting,
        node_resources_ok=lambda task_arg, node, _info: (
            node.get("cpu_hard_free", node.get("free_cpu", 0)) >= task_arg["cpu_cores"],
            f"cpu: need {task_arg['cpu_cores']}, free {node.get('cpu_hard_free', 0)}",
        ),
    )

    why.cmd_why(SimpleNamespace(id="t1", lock_timeout=None), deps=deps)

    out = capsys.readouterr().out
    assert calls and calls[0][0]["tasks"][0]["id"] == "t1"
    assert "node-reject: cpu: need 12, free 0" in out
