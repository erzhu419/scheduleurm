"""Runtime-bound resource history wrappers for scheduler.py."""

from __future__ import annotations

import os
from typing import Any, Mapping

from scheduler_runtime.history_batch import HistoryBatchCoordinator, HistoryBatchDeps


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_resource_history_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {
        "HISTORY_MAX_ENTRIES": int(os.environ.get("SCHEDULEURM_HISTORY_MAX_ENTRIES", "5000")),
        "HISTORY_SAMPLES_PER_SIG": 10,
        "HISTORY_PERCENTILE": 80,
        "RUNTIME_HISTORY_MAX_ENTRIES": int(os.environ.get("SCHEDULEURM_RUNTIME_HISTORY_MAX_ENTRIES", "5000")),
        "RUNTIME_HISTORY_SAMPLES_PER_KEY": 10,
        "RUNTIME_HISTORY_PERCENTILE": 80,
        "RUNTIME_WALLTIME_MULT": 1.20,
        "RUNTIME_MIN_WALLTIME_S": 10 * 60,
        "RUNTIME_CLOSEST_MIN_SCORE": 0.58,
    }

    def history_get(sig):
        return _ns(namespace, "_history_get_impl")(
            sig,
            load_history=_ns(namespace, "load_history"),
        )

    def _task_duration_s(task):
        return _ns(namespace, "_task_duration_s_impl")(task)

    def _task_has_progress_marker(task):
        return _ns(namespace, "_task_has_progress_marker_impl")(task)

    def _is_oom_like_task(task):
        return _ns(namespace, "_is_oom_like_task_impl")(task)

    def _untrusted_startup_oom_sample(task, duration_s=None):
        return _ns(namespace, "_untrusted_startup_oom_sample_impl")(
            task,
            duration_s=duration_s,
        )

    def _effective_est_vram(task, state, history):
        return _ns(namespace, "_effective_est_vram_impl")(
            task,
            state,
            history,
            deps=_ns(namespace, "_EffectiveResourceEstimateDeps")(
                default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
                default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
                untrusted_startup_oom_sample=_ns(namespace, "_untrusted_startup_oom_sample"),
            ),
        )

    def _effective_est_ram(task, state, history):
        return _ns(namespace, "_effective_est_ram_impl")(
            task,
            state,
            history,
            deps=_ns(namespace, "_EffectiveResourceEstimateDeps")(
                default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
                default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
                untrusted_startup_oom_sample=_ns(namespace, "_untrusted_startup_oom_sample"),
            ),
        )

    def _live_sibling_ram_floor(task, state):
        return _ns(namespace, "_live_sibling_ram_floor_impl")(task, state)

    def _live_sibling_resource_estimate(task, state, kind):
        return _ns(namespace, "_live_sibling_resource_estimate_impl")(task, state, kind)

    def _live_sibling_resource_estimate_index(state):
        return _ns(namespace, "_live_sibling_resource_estimate_index_impl")(state)

    def _description_sibling_resource_estimate(task, state, kind):
        return _ns(namespace, "_description_sibling_resource_estimate_impl")(task, state, kind)

    def _description_sibling_resource_estimate_index(state):
        return _ns(namespace, "_description_sibling_resource_estimate_index_impl")(state)

    def _with_resource_slack(observed_mb: int, *, min_mb: int) -> int:
        return _ns(namespace, "_with_resource_slack_impl")(observed_mb, min_mb=min_mb)

    def _maybe_lower_explicit_resource_estimate(
        task: dict,
        key: str,
        observed_mb: int,
        *,
        min_mb: int,
        kind: str,
    ) -> bool:
        return _ns(namespace, "_maybe_lower_explicit_resource_estimate_impl")(
            task,
            key,
            observed_mb,
            min_mb=min_mb,
            kind=kind,
            now=_ns(namespace, "time").time,
        )

    def _percentile(samples, p):
        return _ns(namespace, "_percentile_impl")(samples, p)

    def _resource_history_deps():
        return _ns(namespace, "_build_resource_history_deps")(namespace)

    history_batch = HistoryBatchCoordinator(
        HistoryBatchDeps(
            history_lock=lambda **kwargs: _ns(namespace, "history_lock")(**kwargs),
            load_history=lambda: _ns(namespace, "load_history")(),
            save_history=lambda history: _ns(namespace, "save_history")(history),
            resource_record=lambda *args, **kwargs: _ns(namespace, "_history_record_impl")(
                *args, **kwargs
            ),
            resource_deps_factory=lambda: _ns(namespace, "_resource_history_deps")(),
            load_runtime_history=lambda: _ns(namespace, "load_runtime_history")(),
            save_runtime_history=lambda history: _ns(namespace, "save_runtime_history")(history),
            runtime_record=lambda *args, **kwargs: _ns(namespace, "_runtime_history_record_impl")(
                *args, **kwargs
            ),
            runtime_deps_factory=lambda: _ns(namespace, "_runtime_history_record_deps")(),
        )
    )

    def history_record(
        sig, peak_vram_mb=0, peak_ram_mb=0, cpu_cores=0, duration_s=0,
        task=None,
    ):
        sample = {
            "sig": sig,
            "peak_vram_mb": peak_vram_mb,
            "peak_ram_mb": peak_ram_mb,
            "cpu_cores": cpu_cores,
            "duration_s": duration_s,
        }
        if isinstance(task, dict):
            sample["metadata"] = _ns(namespace, "_resource_history_metadata_impl")(task)
        if history_batch.queue_resource(sample):
            return None
        with _ns(namespace, "history_lock")(purpose="history:resource-record"):
            return _ns(namespace, "_history_record_impl")(
                **sample,
                deps=_ns(namespace, "_resource_history_deps")(),
            )

    exports.update({
        "history_get": history_get,
        "_task_duration_s": _task_duration_s,
        "_task_has_progress_marker": _task_has_progress_marker,
        "_is_oom_like_task": _is_oom_like_task,
        "_untrusted_startup_oom_sample": _untrusted_startup_oom_sample,
        "_effective_est_vram": _effective_est_vram,
        "_effective_est_ram": _effective_est_ram,
        "_live_sibling_ram_floor": _live_sibling_ram_floor,
        "_live_sibling_resource_estimate": _live_sibling_resource_estimate,
        "_live_sibling_resource_estimate_index": _live_sibling_resource_estimate_index,
        "_description_sibling_resource_estimate": _description_sibling_resource_estimate,
        "_description_sibling_resource_estimate_index": _description_sibling_resource_estimate_index,
        "_with_resource_slack": _with_resource_slack,
        "_maybe_lower_explicit_resource_estimate": _maybe_lower_explicit_resource_estimate,
        "_percentile": _percentile,
        "_resource_history_deps": _resource_history_deps,
        "history_record": history_record,
        "_history_batch_coordinator": history_batch,
        "_begin_history_batch": history_batch.begin,
        "_end_history_batch": history_batch.end,
        "_history_batch_active": history_batch.active,
        "_flush_pending_history_updates": history_batch.flush,
    })
    return exports
