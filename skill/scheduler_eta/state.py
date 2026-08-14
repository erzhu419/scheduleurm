"""ETA display/state helpers shared by scheduler commands and runtime updates."""

from __future__ import annotations


LIVE_REMAINING_ETA_SOURCES = frozenset({
    "tqdm",
    "inline_eta",
    "iter_time_window",
    "progress_window",
    "freqduet_shard",
    "seconds_per_unit",
    "progress_rate",
    "saas_native_eta",
    "growing_iter_cost",
    "kg_inner_progress",
    "scolhkg_progress",
    "bapr_fork_protocol",
    "bapr_independent_specialist",
    "bapr_seed_batch",
    "bapr_run_seed",
    "runtime_history_fallback",
    "duration_ewma_fallback",
    "runtime_history_overrun",
    "duration_ewma_overrun",
})

LIVE_RUNTIME_PROJECTION_SOURCES = frozenset({
    "tqdm",
    "inline_eta",
    "iter_time_window",
    "progress_window",
    "freqduet_shard",
    "seconds_per_unit",
    "progress_rate",
    "saas_native_eta",
    "growing_iter_cost",
    "kg_inner_progress",
    "scolhkg_progress",
    "bapr_fork_protocol",
    "bapr_independent_specialist",
    "bapr_seed_batch",
    "bapr_run_seed",
})

ETA_AUDIT_FIELDS = (
    "eta_seconds", "eta_source", "eta_confidence", "eta_updated_at",
    "eta_detail", "eta_log_bytes", "eta_probe_error", "last_progress_line",
)

RUNTIME_PROJECTION_FIELDS = (
    "runtime_total_s_est", "runtime_eta_s_est", "runtime_est_source",
    "runtime_progress_at", "runtime_current_unit", "runtime_total_units",
    "runtime_unit_s_est",
)


def eta_source_base(source: object) -> str:
    return str(source or "").split(":", 1)[0]


def is_live_remaining_eta_source(source: object) -> bool:
    return eta_source_base(source) in LIVE_REMAINING_ETA_SOURCES


def is_live_runtime_projection_source(source: object) -> bool:
    return eta_source_base(source) in LIVE_RUNTIME_PROJECTION_SOURCES


def clear_live_eta_fields(task: dict, clear_runtime_projection: bool = False) -> bool:
    """Remove running-run ETA/progress fields before a task re-enters queue."""
    changed = False
    for key in ETA_AUDIT_FIELDS:
        if key in task:
            task.pop(key, None)
            changed = True
    if clear_runtime_projection or is_live_runtime_projection_source(task.get("runtime_est_source")):
        for key in RUNTIME_PROJECTION_FIELDS:
            if key in task:
                task.pop(key, None)
                changed = True
    return changed


def queued_has_stale_live_eta(task: dict) -> bool:
    if task.get("status") not in ("queued", "launching"):
        return False
    if is_live_remaining_eta_source(task.get("eta_source")):
        return True
    if is_live_runtime_projection_source(task.get("runtime_est_source")):
        return True
    if task.get("last_progress_line") and not task.get("started_at"):
        return True
    return False


def eta_confidence_for_source(source: str) -> str:
    base = eta_source_base(source)
    if base in ("tqdm", "inline_eta", "local_test_tqdm"):
        return "high"
    if base in (
        "iter_time_window",
        "progress_window",
        "freqduet_shard",
        "seconds_per_unit",
        "progress_rate",
        "saas_native_eta",
        "growing_iter_cost",
        "kg_inner_progress",
        "scolhkg_progress",
        "local_test_progress",
        "runtime_history",
        "peer_progress",
        "bapr_fork_protocol",
        "bapr_independent_specialist",
        "bapr_seed_batch",
    ):
        return "medium"
    return "low"


def fmt_eta_seconds(seconds: int) -> str:
    seconds = int(seconds or 0)
    if seconds <= 0:
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def eta_source_tag(source: object) -> str:
    base = eta_source_base(source)
    if base in LIVE_RUNTIME_PROJECTION_SOURCES:
        return "live"
    if base in (
        "runtime_history",
        "runtime_history_fallback",
        "runtime_history_overrun",
        "duration_ewma",
        "duration_ewma_fallback",
        "duration_ewma_overrun",
        "peer_progress",
    ) or base.startswith("closest"):
        return "hist"
    if base.startswith("local_test"):
        return "test"
    return "est"


def format_task_eta(task: dict) -> str:
    eta = int(task.get("eta_seconds") or 0)
    if eta <= 0:
        return ""
    return f"eta~{fmt_eta_seconds(eta)}/{eta_source_tag(task.get('eta_source'))}"


def history_fallback_eta_seconds(
    total_s: int,
    elapsed_s: float,
    *,
    overrun_floor_s: int,
    overrun_fraction: float,
    overrun_max_s: int,
) -> tuple[int, bool]:
    """Remaining ETA from a historical full-run estimate.

    When elapsed time already exceeds history, keep a bounded positive overrun
    floor so overloaded nodes remain visible while the source tag shows that
    this is not progress-derived ETA.
    """
    try:
        total = max(0, int(total_s or 0))
        elapsed = max(0.0, float(elapsed_s or 0.0))
    except Exception:
        return 0, False
    if total <= 0:
        return 0, False
    remaining = int(total - elapsed)
    if remaining > 0:
        return remaining, False
    floor = int(overrun_floor_s)
    if floor <= 0:
        return 0, True
    proportional = int(total * float(overrun_fraction))
    estimate = max(floor, proportional)
    max_s = int(overrun_max_s)
    if max_s > 0:
        estimate = min(estimate, max_s)
    return int(max(1, estimate)), True
