"""Experiment-only hard-rule override hooks.

The production scheduler owns safety policy.  This module only exposes a
scoped switch for A/B experiments that need to measure clean placement policy
behavior without mixing in legacy guard rails.
"""
from __future__ import annotations

import os
from typing import Any, Mapping


DEFAULT_CLEAN_RULES = frozenset(
    {
        "one_third_pack",
        "gpu_util_saturation",
        "max_tasks_per_gpu",
        "node_concurrency",
        "thread_pressure",
        "vram_margin",
        "startup_vram_reserve",
        "post_dispatch_gpu_freeze",
    }
)


def _csv_set(raw: str | None) -> set[str]:
    out: set[str] = set()
    for part in str(raw or "").replace(";", ",").split(","):
        item = part.strip().lower().replace("-", "_")
        if item:
            out.add(item)
    return out


def _truthy(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() not in ("", "0", "false", "no", "off", "none")
    return bool(raw)


def hard_rule_mode() -> str:
    raw = (
        os.environ.get("SCHEDULEURM_AB_HARD_RULE_MODE")
        or os.environ.get("SCHEDULEURM_EXPERIMENT_HARD_RULE_MODE")
        or ""
    )
    return str(raw).strip().lower().replace("-", "_")


def active_rules() -> set[str]:
    mode = hard_rule_mode()
    if mode in ("", "none", "default", "safety", "safe"):
        return set()
    raw_rules = _csv_set(os.environ.get("SCHEDULEURM_AB_DISABLED_HARD_RULES"))
    if raw_rules:
        return set(DEFAULT_CLEAN_RULES) if "all" in raw_rules else raw_rules
    if mode in ("clean", "clean_bench", "no_esp", "bench_clean", "ab_clean"):
        return set(DEFAULT_CLEAN_RULES)
    return set()


def _scope_prefixes() -> list[str]:
    raw = os.environ.get("SCHEDULEURM_AB_SCOPE_PREFIXES")
    if raw:
        return [p for p in (x.strip() for x in raw.replace(";", ",").split(",")) if p]
    return ["ScheduleurmBench"]


def task_in_scope(task: Mapping[str, Any] | None) -> bool:
    if _truthy(os.environ.get("SCHEDULEURM_AB_ALLOW_ALL_TASKS")):
        return True
    task = task or {}
    fields = [
        str(task.get("project") or ""),
        str(task.get("signature") or ""),
        str(task.get("description") or ""),
    ]
    prefixes = _scope_prefixes()
    return any(value.startswith(prefix) for value in fields for prefix in prefixes)


def should_bypass_hard_rule(
    rule: str,
    *,
    task: Mapping[str, Any] | None = None,
    node_info: Mapping[str, Any] | None = None,
    node_state: Mapping[str, Any] | None = None,
    gpu: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> bool:
    del node_info, node_state, gpu, context
    normalized = str(rule or "").strip().lower().replace("-", "_")
    if normalized not in active_rules():
        return False
    return task_in_scope(task)


def snapshot(task: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rules = sorted(active_rules())
    return {
        "mode": hard_rule_mode() or "none",
        "active_rules": rules,
        "scope_prefixes": _scope_prefixes(),
        "task_in_scope": task_in_scope(task) if task is not None else None,
    }
