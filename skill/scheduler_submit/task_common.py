"""Shared types and small helpers for submit task construction."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SubmitTaskDeps:
    known_nodes: Any
    parse_env: Callable[[Any], dict]
    task_run_identity: Callable[[dict], Any]
    task_has_recorded_launch_artifacts: Callable[[dict], bool]
    recorded_launch_safety_state: Callable[[dict], tuple[str, str]]
    same_declared_path: Callable[[Any, Any], bool]
    active_or_unsynced_result_status: Callable[[dict], bool]
    history_get: Callable[[str], dict]
    load_history: Callable[[], dict]
    effective_est_vram: Callable[[dict, dict, dict], int]
    effective_est_ram: Callable[[dict, dict, dict], int]
    project_from_path: Callable[[str], str]
    cpu_worker_plan_for_items: Callable[[int, int], dict]
    node_physical_cores: Callable[[str], int]
    cpu_labor_node_names: Callable[[], list[str]]
    allocate_task_id: Callable[[dict], str]
    local_user: Callable[[], str]
    local_host_short: Callable[[], str]
    scheduler_id: Callable[[], str]
    apply_test_log_runtime_profile: Callable[[dict, str], bool]
    seed_pending_eta_from_history: Callable[..., Any]
    now: Callable[[], float] = _time.time


@dataclass(frozen=True)
class SubmitTaskResult:
    task: dict
    hist: dict
    est_vram: int
    ram_mb: int
    cpu_cores: int
    source_label: str


class SubmitTaskRefusal(Exception):
    def __init__(self, lines: list[str], *, code: int = 2, stream: str = "stderr"):
        super().__init__("\n".join(lines))
        self.lines = lines
        self.code = code
        self.stream = stream


def _refuse(*lines: str, code: int = 2, stream: str = "stderr") -> None:
    raise SubmitTaskRefusal(list(lines), code=code, stream=stream)


def _int_arg(args: Any, name: str, default: int = 0) -> int:
    return int(getattr(args, name, default) or default)


def _submit_source_label(args: Any, hist: dict, cpu_parallel_items: int) -> str:
    sources: list[str] = []
    if hist.get("vram_mb") and args.vram is None:
        sources.append("vram=hist")
    elif args.vram is not None:
        sources.append("vram=explicit")
    if hist.get("ram_mb") and args.ram_mb is None:
        sources.append("ram=hist")
    elif args.ram_mb is not None:
        sources.append("ram=explicit")
    if cpu_parallel_items > 0 and args.cpu is None:
        sources.append("cpu=auto-workers")
    elif hist.get("cpu_cores") and args.cpu is None:
        sources.append("cpu=hist")
    elif args.cpu is not None:
        sources.append("cpu=explicit")
    return ",".join(sources) or "all-defaults"


def _submit_parallel_fields(args: Any) -> dict:
    items = _int_arg(args, "cpu_parallel_items")
    total_items = _int_arg(args, "cpu_parallel_total_items") or items
    logical_items = _int_arg(args, "cpu_parallel_logical_items")
    if items and not logical_items:
        logical_items = total_items
    return {
        "items": items,
        "total_items": total_items,
        "logical_items": logical_items,
        "item_multiplier": _int_arg(args, "cpu_parallel_item_multiplier", 1),
        "start": _int_arg(args, "cpu_parallel_start"),
        "end": _int_arg(args, "cpu_parallel_end") or items,
        "shard_index": _int_arg(args, "cpu_parallel_shard_index"),
        "num_shards": _int_arg(args, "cpu_parallel_num_shards", 1),
    }


def _stage_excludes(args: Any) -> list[str]:
    return [
        str(x).strip()
        for x in (getattr(args, "stage_excludes", None) or [])
        if str(x).strip()
    ]
