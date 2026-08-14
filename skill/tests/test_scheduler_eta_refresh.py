from __future__ import annotations

from skill.scheduler_eta.refresh import EtaRefreshDeps, refresh_eta_from_logs


class _EtaTracker:
    @staticmethod
    def compute_eta_seconds(text, *, elapsed_s, fallback_ewma_s, cmd=None):
        if "INLINE" in text:
            return 111
        if "PROGRESS" in text:
            return 200
        if fallback_ewma_s:
            return max(0, int(fallback_ewma_s - elapsed_s))
        return 0

    @staticmethod
    def runtime_projection(text, *, elapsed_s, cmd=None):
        if "INLINE" in text:
            return {
                "source": "inline_eta",
                "eta_s": 111,
                "total_s": int(elapsed_s + 111),
                "current": 4,
                "total_units": 10,
            }
        if "PROGRESS" in text:
            return {
                "source": "progress_rate",
                "eta_s": 200,
                "total_s": int(elapsed_s + 200),
                "current": 8,
                "total_units": 10,
            }
        return None


class _KgEtaTracker(_EtaTracker):
    @staticmethod
    def compute_eta_seconds(text, *, elapsed_s, fallback_ewma_s, cmd=None):
        return 300 if "KG_INNER" in text else _EtaTracker.compute_eta_seconds(
            text, elapsed_s=elapsed_s, fallback_ewma_s=fallback_ewma_s, cmd=cmd)

    @staticmethod
    def runtime_projection(text, *, elapsed_s, cmd=None):
        if "KG_INNER" in text:
            return {
                "source": "kg_inner_progress",
                "eta_s": 300,
                "total_s": int(elapsed_s + 300),
                "current": 1400,
                "total_units": 2000,
                "stage": 13,
                "total_stages": 20,
                "base_step_s": 69.0,
                "future_eval_count": 2,
            }
        return _EtaTracker.runtime_projection(text, elapsed_s=elapsed_s, cmd=cmd)


def _task(task_id, **overrides):
    task = {
        "id": task_id,
        "status": "running",
        "node": "n1",
        "signature": f"sig/{task_id}",
        "started_at": 100.0,
        "cmd": "python train.py",
    }
    task.update(overrides)
    return task


def _deps(
    *,
    by_node=None,
    pure_ewma=None,
    history=None,
    runtime_total_by_id=None,
    tail_outputs=None,
    bapr_projection=None,
    now=1000.0,
    calls=None,
    eta_tracker=None,
):
    by_node = by_node or {}
    pure_ewma = pure_ewma or []
    history = history or {}
    runtime_total_by_id = runtime_total_by_id or {}
    calls = calls if calls is not None else []
    eta_tracker = eta_tracker if eta_tracker is not None else _EtaTracker()

    def apply_projection(task, projection):
        calls.append(("apply_projection", task["id"], projection))
        if not projection:
            return
        task["runtime_total_s_est"] = projection.get("total_s")
        task["runtime_eta_s_est"] = projection.get("eta_s")
        task["runtime_current_unit"] = projection.get("current")
        task["runtime_total_units"] = projection.get("total_units")
        task["runtime_est_source"] = projection.get("source")

    return EtaRefreshDeps(
        load_eta_tracker_module=lambda: eta_tracker,
        load_runtime_history=lambda: {"loaded": True},
        runtime_history_closest_index=lambda history_obj: {"index": True},
        eta_tail_targets=lambda state: (by_node, pure_ewma),
        history_get=lambda sig: history.get(sig),
        runtime_total_history_s=lambda task, **kwargs: runtime_total_by_id.get(task["id"], 0),
        effective_elapsed_s=lambda task: float(task.get("elapsed", 0)),
        history_fallback_eta_seconds=lambda ewma, elapsed: (
            (1800, True) if ewma and elapsed > ewma else (max(0, int(ewma - elapsed)), False)
        ),
        eta_tail_outputs_by_node=lambda targets: tail_outputs if tail_outputs is not None else {},
        last_progress_line=lambda text: next(
            (line for line in reversed(text.splitlines()) if "PROGRESS" in line or "INLINE" in line),
            None,
        ),
        bapr_batch_projection=lambda task, text, elapsed: bapr_projection.get(task["id"]) if bapr_projection else None,
        apply_runtime_projection=apply_projection,
        eta_confidence_for_source=lambda source: "high" if source == "inline_eta" else "medium",
        now=lambda: now,
    )


def test_no_log_task_uses_runtime_history_fallback_and_clears_probe_error():
    task = _task("nolog", log_path=None, elapsed=100, eta_probe_error="old")
    state = {"tasks": [task]}

    refresh_eta_from_logs(
        state,
        deps=_deps(
            pure_ewma=[task],
            runtime_total_by_id={"nolog": 1000},
            now=1234.0,
        ),
    )

    assert task["eta_seconds"] == 900
    assert task["eta_source"] == "runtime_history_fallback"
    assert task["eta_confidence"] == "low"
    assert task["eta_updated_at"] == 1234
    assert task["eta_log_bytes"] == 0
    assert "eta_probe_error" not in task


def test_no_history_fallback_clears_stale_source_and_says_history_is_missing():
    task = _task(
        "nohistory",
        log_path=None,
        elapsed=100,
        eta_source="duration_ewma_fallback",
        eta_confidence="low",
    )

    refresh_eta_from_logs(
        {"tasks": [task]},
        deps=_deps(pure_ewma=[task]),
    )

    assert task["eta_seconds"] == 0
    assert "eta_source" not in task
    assert "eta_confidence" not in task
    assert task["eta_detail"] == (
        "no scheduler log_path and no runtime history fallback"
    )


def test_marker_tail_updates_eta_projection_progress_line_and_detail():
    task = _task("t-prog", log_path="/tmp/t.log", elapsed=800)
    state = {"tasks": [task]}
    output = "===ETA_LOG_t-prog===\nnoise\nPROGRESS 8/10\n"
    calls = []

    refresh_eta_from_logs(
        state,
        tail_outputs={"n1": output},
        deps=_deps(by_node={"n1": [(task, "/tmp/t.log")]}, calls=calls),
    )

    assert task["eta_seconds"] == 200
    assert task["eta_source"] == "progress_rate"
    assert task["eta_confidence"] == "medium"
    assert task["last_progress_line"] == "PROGRESS 8/10"
    assert task["runtime_total_s_est"] == 1000
    assert task["runtime_est_source"] == "progress_rate"
    assert task["eta_detail"] == "parsed progress_rate from scheduler log tail"
    assert calls[0][0] == "apply_projection"


def test_inline_projection_source_gets_high_confidence():
    task = _task("t-inline", log_path="/tmp/t.log", elapsed=50)
    output = "===ETA_LOG_t-inline===\nINLINE eta line\n"

    refresh_eta_from_logs(
        {"tasks": [task]},
        tail_outputs={"n1": output},
        deps=_deps(by_node={"n1": [(task, "/tmp/t.log")]}),
    )

    assert task["eta_seconds"] == 111
    assert task["eta_source"] == "inline_eta"
    assert task["eta_confidence"] == "high"
    assert task["eta_detail"] == "parsed inline_eta from scheduler log tail"


def test_bapr_projection_overrides_tracker_projection_and_detail():
    task = _task("bapr", log_path="/tmp/b.log", elapsed=100)
    projection = {
        "source": "bapr_seed_batch",
        "eta_s": 321,
        "total_s": 421,
        "completed_seeds": 3,
        "seed_count": 8,
    }

    refresh_eta_from_logs(
        {"tasks": [task]},
        tail_outputs={"n1": "===ETA_LOG_bapr===\nPROGRESS but BAPR wins\n"},
        deps=_deps(
            by_node={"n1": [(task, "/tmp/b.log")]},
            bapr_projection={"bapr": projection},
        ),
    )

    assert task["eta_seconds"] == 321
    assert task["eta_source"] == "bapr_seed_batch"
    assert task["eta_detail"].endswith("completed_seeds=3/8")


def test_kg_projection_prefers_mature_live_model_over_stale_history():
    task = _task("kg", log_path="/tmp/kg.log", elapsed=800)

    refresh_eta_from_logs(
        {"tasks": [task]},
        tail_outputs={"n1": "===ETA_LOG_kg===\nKG_INNER\n"},
        deps=_deps(
            by_node={"n1": [(task, "/tmp/kg.log")]},
            runtime_total_by_id={"kg": 1500},
            eta_tracker=_KgEtaTracker(),
        ),
    )

    assert task["eta_seconds"] == 300
    assert task["eta_source"] == "kg_inner_progress"
    assert task["runtime_eta_s_est"] == 300
    assert task["runtime_total_s_est"] == 1100
    assert "history_floor" not in task["eta_detail"]


def test_probe_failure_sets_error_and_uses_fallback_only_when_eta_missing():
    missing = _task("missing", log_path="/tmp/m.log", elapsed=100)
    existing = _task("existing", log_path="/tmp/e.log", elapsed=100, eta_seconds=77)

    refresh_eta_from_logs(
        {"tasks": [missing, existing]},
        tail_outputs={"n1": None},
        deps=_deps(
            by_node={"n1": [(missing, "/tmp/m.log"), (existing, "/tmp/e.log")]},
            history={"sig/missing": {"dur_s_ewma": 600}, "sig/existing": {"dur_s_ewma": 600}},
        ),
    )

    assert missing["eta_seconds"] == 500
    assert missing["eta_source"] == "duration_ewma_fallback"
    assert missing["eta_probe_error"].startswith("log tail probe failed")
    assert existing["eta_seconds"] == 77
    assert existing["eta_probe_error"].startswith("log tail probe failed")


def test_probed_task_ids_filter_skips_unprobed_tasks():
    probed = _task("probed", log_path="/tmp/p.log", elapsed=10)
    skipped = _task("skipped", log_path="/tmp/s.log", elapsed=10)

    refresh_eta_from_logs(
        {"tasks": [probed, skipped]},
        tail_outputs={
            "n1": (
                "===ETA_LOG_probed===\nPROGRESS yes\n"
                "===ETA_LOG_skipped===\nPROGRESS no\n"
            )
        },
        probed_task_ids={"probed"},
        deps=_deps(by_node={"n1": [(probed, "/tmp/p.log"), (skipped, "/tmp/s.log")]}),
    )

    assert probed["eta_seconds"] == 200
    assert "eta_seconds" not in skipped
