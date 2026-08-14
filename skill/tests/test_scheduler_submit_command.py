from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from skill.scheduler_submit_command import (
    SubmitCommandDeps,
    SubmitJsonlCommandDeps,
    cmd_submit,
    cmd_submit_jsonl,
)


class _Refusal(BaseException):
    def __init__(self, lines, code=2, stream="stderr"):
        self.lines = lines
        self.code = code
        self.stream = stream


def _state_lock(calls):
    @contextmanager
    def lock(**kwargs):
        calls.append(("lock_enter", kwargs))
        try:
            yield
        finally:
            calls.append(("lock_exit", kwargs))

    return lock


def test_cmd_submit_records_test_peaks_and_saves_task():
    calls = []
    state = {"tasks": []}
    printed = []

    deps = SubmitCommandDeps(
        split_bapr_seed_batch_submit_args=lambda args: [],
        submit_one=lambda args: calls.append(("submit_one", args)),
        run_submit_preflight=lambda args, **kwargs: calls.append(("preflight", kwargs)) or SimpleNamespace(notes=["note"]),
        submit_preflight_deps=lambda: "preflight-deps",
        submit_preflight_refusal_type=_Refusal,
        history_record=lambda signature, **kwargs: calls.append(("history", signature, kwargs)),
        state_lock=_state_lock(calls),
        load_state=lambda: calls.append(("load_state", None)) or state,
        build_submit_task_for_state=lambda state_arg, args, preflight, **kwargs: (
            calls.append(("build", state_arg, preflight, kwargs))
            or SimpleNamespace(
                task={"id": "t1", "cpu_parallel_items": 0},
                cpu_cores=2,
                ram_mb=3000,
                est_vram=4000,
                source_label="declared",
            )
        ),
        submit_task_deps=lambda: "submit-task-deps",
        save_state=lambda saved: calls.append(("save_state", list(saved["tasks"]))),
        submit_task_refusal_type=_Refusal,
        default_vram_mb=3500,
        default_ram_mb=2000,
        default_cpu_cores=1,
        print_fn=lambda *args, **kwargs: printed.append((args, kwargs)),
        exit_fn=lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )
    args = SimpleNamespace(
        signature="sig",
        priority="normal",
        description="desc",
        test_peak_vram_mb=10,
        test_peak_ram_mb=20,
        test_cpu=3,
    )

    cmd_submit(args, deps=deps)

    assert state["tasks"] == [{"id": "t1", "cpu_parallel_items": 0}]
    assert ("history", "sig", {"peak_vram_mb": 10, "peak_ram_mb": 20, "cpu_cores": 3}) in calls
    assert ("lock_enter", {}) in calls
    assert any("submitted t1" in item[0][0] for item in printed if item[0])


def test_cmd_submit_preflight_refusal_exits_with_code():
    printed = []
    deps = SubmitCommandDeps(
        split_bapr_seed_batch_submit_args=lambda args: [],
        submit_one=lambda args: None,
        run_submit_preflight=lambda args, **kwargs: (_ for _ in ()).throw(_Refusal(["no"], code=7)),
        submit_preflight_deps=lambda: None,
        submit_preflight_refusal_type=_Refusal,
        history_record=lambda *args, **kwargs: None,
        state_lock=_state_lock([]),
        load_state=lambda: {"tasks": []},
        build_submit_task_for_state=lambda *args, **kwargs: None,
        submit_task_deps=lambda: None,
        save_state=lambda state: None,
        submit_task_refusal_type=_Refusal,
        default_vram_mb=1,
        default_ram_mb=1,
        default_cpu_cores=1,
        print_fn=lambda *args, **kwargs: printed.append((args, kwargs)),
        exit_fn=lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    with pytest.raises(SystemExit) as exc:
        cmd_submit(SimpleNamespace(), deps=deps)

    assert exc.value.code == 7
    assert printed[0][0] == ("no",)


def test_cmd_submit_jsonl_clears_dispatch_intent_after_build_failure():
    calls = []
    deps = SubmitJsonlCommandDeps(
        load_submit_jsonl_specs=lambda args: [{"id": "spec"}],
        write_dispatch_intent=lambda **kwargs: calls.append(("write_intent", kwargs)),
        clear_dispatch_intent=lambda: calls.append(("clear_intent", None)),
        load_history=lambda: calls.append(("load_history", None)) or {"h": True},
        load_runtime_history=lambda: calls.append(("load_runtime", None)) or {"r": True},
        runtime_history_closest_index=lambda history: calls.append(("closest", history)) or ["idx"],
        state_lock=_state_lock(calls),
        load_state=lambda: calls.append(("load_state", None)) or {"tasks": []},
        allocate_task_ids=lambda state, count: calls.append(("alloc", count)) or ["t1"],
        build_bulk_submit_tasks=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("build failed")),
        save_state=lambda state: calls.append(("save_state", state)),
        dispatch_intent_ttl_s=12.0,
        print_fn=lambda *args, **kwargs: calls.append(("print", args, kwargs)),
        exit_fn=lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    with pytest.raises(RuntimeError, match="build failed"):
        cmd_submit_jsonl(
            SimpleNamespace(trusted=True, json=True, intent_label="", intent_ttl=None, lock_timeout=5),
            deps=deps,
        )

    assert ("write_intent", {"label": "bulk-submit-1", "ttl_s": 12.0}) in calls
    assert ("clear_intent", None) in calls


def test_cmd_submit_jsonl_empty_specs_prints_zero_without_intent():
    calls = []
    deps = SubmitJsonlCommandDeps(
        load_submit_jsonl_specs=lambda args: [],
        write_dispatch_intent=lambda **kwargs: calls.append(("write_intent", kwargs)),
        clear_dispatch_intent=lambda: calls.append(("clear_intent", None)),
        load_history=lambda: {},
        load_runtime_history=lambda: {},
        runtime_history_closest_index=lambda history: [],
        state_lock=_state_lock(calls),
        load_state=lambda: {"tasks": []},
        allocate_task_ids=lambda state, count: [],
        build_bulk_submit_tasks=lambda *args, **kwargs: [],
        save_state=lambda state: None,
        dispatch_intent_ttl_s=12.0,
        print_fn=lambda *args, **kwargs: calls.append(("print", args, kwargs)),
        exit_fn=lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    cmd_submit_jsonl(SimpleNamespace(trusted=True, json=False), deps=deps)

    assert calls == [("print", ("submitted 0 tasks",), {})]
