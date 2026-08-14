from __future__ import annotations

from skill.scheduler_eta.state import (
    clear_live_eta_fields,
    eta_confidence_for_source,
    eta_source_base,
    eta_source_tag,
    fmt_eta_seconds,
    format_task_eta,
    history_fallback_eta_seconds,
    queued_has_stale_live_eta,
)


def test_eta_source_helpers_group_internal_sources_for_display():
    assert eta_source_base("inline_eta:tail") == "inline_eta"
    assert eta_source_tag("inline_eta") == "live"
    assert eta_source_tag("progress_window") == "live"
    assert eta_source_tag("freqduet_shard") == "live"
    assert eta_source_tag("bapr_fork_protocol") == "live"
    assert eta_source_tag("kg_inner_progress") == "live"
    assert eta_source_tag("runtime_history_overrun") == "hist"
    assert eta_source_tag("peer_progress") == "hist"
    assert eta_source_tag("closest_runtime") == "hist"
    assert eta_confidence_for_source("tqdm") == "high"
    assert eta_confidence_for_source("progress_rate") == "medium"
    assert eta_confidence_for_source("bapr_fork_protocol") == "medium"
    assert eta_confidence_for_source("kg_inner_progress") == "medium"
    assert eta_confidence_for_source("unknown") == "low"


def test_eta_formatting_boundaries():
    assert fmt_eta_seconds(0) == "?"
    assert fmt_eta_seconds(59) == "59s"
    assert fmt_eta_seconds(60) == "1.0m"
    assert fmt_eta_seconds(3600) == "1.0h"
    assert fmt_eta_seconds(86400) == "1.0d"
    assert format_task_eta({"eta_seconds": 90, "eta_source": "inline_eta"}) == "eta~1.5m/live"
    assert format_task_eta({"eta_seconds": 0, "eta_source": "inline_eta"}) == ""


def test_clear_live_eta_fields_removes_running_eta_and_projection():
    task = {
        "status": "queued",
        "eta_seconds": 10,
        "eta_source": "progress_rate",
        "eta_confidence": "medium",
        "last_progress_line": "Iter 1",
        "runtime_est_source": "progress_rate",
        "runtime_total_s_est": 100,
        "runtime_eta_s_est": 90,
        "keep": "value",
    }

    assert queued_has_stale_live_eta(task) is True
    assert clear_live_eta_fields(task) is True

    assert task == {"status": "queued", "keep": "value"}


def test_queued_has_stale_live_eta_ignores_history_and_running_rows():
    assert queued_has_stale_live_eta({"status": "queued", "eta_source": "runtime_history"}) is False
    assert queued_has_stale_live_eta({"status": "running", "eta_source": "progress_rate"}) is False
    assert queued_has_stale_live_eta({"status": "queued", "last_progress_line": "x"}) is True
    assert queued_has_stale_live_eta({"status": "queued", "last_progress_line": "x", "started_at": 1}) is False


def test_history_fallback_eta_positive_overrun_floor_and_cap():
    assert history_fallback_eta_seconds(
        1000,
        200,
        overrun_floor_s=300,
        overrun_fraction=0.25,
        overrun_max_s=7200,
    ) == (800, False)
    assert history_fallback_eta_seconds(
        1000,
        1200,
        overrun_floor_s=300,
        overrun_fraction=0.25,
        overrun_max_s=7200,
    ) == (300, True)
    assert history_fallback_eta_seconds(
        10000,
        12000,
        overrun_floor_s=300,
        overrun_fraction=0.25,
        overrun_max_s=2000,
    ) == (2000, True)
    assert history_fallback_eta_seconds(
        100,
        200,
        overrun_floor_s=0,
        overrun_fraction=0.25,
        overrun_max_s=7200,
    ) == (0, True)
