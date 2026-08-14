from __future__ import annotations

import re
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_resource_forensics_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _node_ram_headroom_mb(node_state: dict, node_info: dict) -> int:
        return _ns(namespace, "_node_ram_headroom_mb_impl")(
            node_state,
            node_info,
            default_frac=_ns(namespace, "RAM_HEADROOM_FRAC"),
        )

    def _gpu_freeze_line_mb(total_mb: int) -> int:
        return _ns(namespace, "_gpu_freeze_line_mb_impl")(
            total_mb,
            one_third_pack_grace_mb=_ns(namespace, "ONE_THIRD_PACK_GRACE_MB"),
        )

    def _gpu_threshold_snapshot(gpu: Optional[dict]) -> Optional[dict]:
        return _ns(namespace, "_gpu_threshold_snapshot_impl")(
            gpu,
            gpu_freeze_line_mb=_ns(namespace, "_gpu_freeze_line_mb"),
        )

    def _node_ram_snapshot(node_state: Optional[dict]) -> Optional[dict]:
        return _ns(namespace, "_node_ram_snapshot_impl")(
            node_state,
            node_configs=_ns(namespace, "NODES"),
            node_ram_headroom_mb=_ns(namespace, "_node_ram_headroom_mb"),
            ram_headroom_eviction_grace_mb=_ns(
                namespace, "RAM_HEADROOM_EVICTION_GRACE_MB"
            ),
        )

    def _resource_forensics_deps():
        return _ns(namespace, "_build_resource_forensics_deps")(namespace)

    def _task_progress_ratio(task: dict) -> Optional[float]:
        return _ns(namespace, "_task_progress_ratio_impl")(task)

    def _task_has_progress_evidence(task: dict) -> bool:
        return _ns(namespace, "_task_has_progress_evidence_impl")(task)

    def _task_ram_pressure_mb(task: dict) -> int:
        return _ns(namespace, "_task_ram_pressure_mb_impl")(task)

    def _task_has_resume_evidence(task: dict) -> bool:
        return _ns(namespace, "_task_has_resume_evidence_impl")(task)

    def _ckpt_task_evict_protected(task: dict) -> bool:
        return _ns(namespace, "_ckpt_task_evict_protected_impl")(
            task,
            deps=_ns(namespace, "_resource_forensics_deps")(),
        )

    def _task_has_incremental_result_resume(task: dict) -> bool:
        return _ns(namespace, "_task_has_incremental_result_resume_impl")(
            task,
            deps=_ns(namespace, "_resource_forensics_deps")(),
        )

    def _task_evict_loss_protected(task: dict) -> bool:
        return _ns(namespace, "_task_evict_loss_protected_impl")(
            task,
            deps=_ns(namespace, "_resource_forensics_deps")(),
        )

    def _summarize_task_for_resource_log(task: dict) -> dict:
        return _ns(namespace, "_summarize_task_for_resource_log_impl")(
            task,
            deps=_ns(namespace, "_resource_forensics_deps")(),
        )

    def _assign_windows_pin_plan(task: dict, state: dict, node_state: Optional[dict]):
        return _ns(namespace, "_assign_windows_pin_plan_impl")(
            task,
            state,
            node_state,
            node_is_windows=_ns(namespace, "_node_is_windows"),
            node_physical_cores=_ns(namespace, "_node_physical_cores"),
            node_configs=_ns(namespace, "NODES"),
            default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        )

    def _resource_snapshot_for_placement(
        state: dict,
        nodes: list,
        task: dict,
        node_name: Optional[str],
        gpu_idx,
        stage: str,
    ) -> dict:
        return _ns(namespace, "_resource_snapshot_for_placement_impl")(
            state,
            nodes,
            task,
            node_name,
            gpu_idx,
            stage,
            deps=_ns(namespace, "_resource_forensics_deps")(),
        )

    def _remember_running_resource_snapshots(state: dict, nodes: list):
        return _ns(namespace, "_remember_running_resource_snapshots_impl")(
            state,
            nodes,
            deps=_ns(namespace, "_resource_forensics_deps")(),
        )

    def _reconcile_aggregate_only_vram(state: dict, nodes: list) -> int:
        return _ns(namespace, "_reconcile_aggregate_only_vram_impl")(
            state,
            nodes,
            deps=_ns(namespace, "_AggregateVramReconcileDeps")(
                is_slurm_managed=_ns(namespace, "_is_legacy_external_managed")
            ),
        )

    def _last_progress_line(text: str) -> str:
        if not text:
            return ""
        progress_re = re.compile(
            r"(\d+\s*/\s*\d+\s*\[|\[\s*\d+\s*/\s*\d+\s*\]|"
            r"(?:^|[^\w])(?:Iter|Iteration|Epoch|Step)\s*(?:[:=]\s*)?\d+|"
            r"Starting training|JAX devices|Training complete|DONE)",
            re.IGNORECASE,
        )
        last = ""
        for line in text.splitlines():
            clean = line.replace("\r", "\n").splitlines()[-1] if "\r" in line else line
            if progress_re.search(clean):
                last = clean.strip()
        return last[-500:]

    def _log_progress_forensics(
        task: dict,
        tail_text: str,
        elapsed_s: Optional[float] = None,
    ) -> dict:
        return _ns(namespace, "_log_progress_forensics_impl")(
            task,
            tail_text,
            elapsed_s=elapsed_s,
            deps=_ns(namespace, "_resource_forensics_deps")(),
        )

    def _build_crash_forensics_payload(
        task: dict,
        state: dict,
        nodes: Optional[list] = None,
    ) -> dict:
        return _ns(namespace, "_build_crash_forensics_payload_impl")(
            task,
            state,
            nodes,
            deps=_ns(namespace, "_resource_forensics_deps")(),
        )

    def _is_oom_like_forensics(payload: dict) -> bool:
        return _ns(namespace, "_is_oom_like_forensics_impl")(payload)

    def _build_oom_forensics_payload(crash_payload: dict, state: dict) -> dict:
        return _ns(namespace, "_build_oom_forensics_payload_impl")(crash_payload, state)

    return {
        "_node_ram_headroom_mb": _node_ram_headroom_mb,
        "_gpu_freeze_line_mb": _gpu_freeze_line_mb,
        "_gpu_threshold_snapshot": _gpu_threshold_snapshot,
        "_node_ram_snapshot": _node_ram_snapshot,
        "_resource_forensics_deps": _resource_forensics_deps,
        "_task_progress_ratio": _task_progress_ratio,
        "_task_has_progress_evidence": _task_has_progress_evidence,
        "_task_ram_pressure_mb": _task_ram_pressure_mb,
        "_task_has_resume_evidence": _task_has_resume_evidence,
        "_ckpt_task_evict_protected": _ckpt_task_evict_protected,
        "_task_has_incremental_result_resume": _task_has_incremental_result_resume,
        "_task_evict_loss_protected": _task_evict_loss_protected,
        "_summarize_task_for_resource_log": _summarize_task_for_resource_log,
        "_assign_windows_pin_plan": _assign_windows_pin_plan,
        "_resource_snapshot_for_placement": _resource_snapshot_for_placement,
        "_remember_running_resource_snapshots": _remember_running_resource_snapshots,
        "_reconcile_aggregate_only_vram": _reconcile_aggregate_only_vram,
        "_last_progress_line": _last_progress_line,
        "_log_progress_forensics": _log_progress_forensics,
        "_build_crash_forensics_payload": _build_crash_forensics_payload,
        "_is_oom_like_forensics": _is_oom_like_forensics,
        "_build_oom_forensics_payload": _build_oom_forensics_payload,
    }
