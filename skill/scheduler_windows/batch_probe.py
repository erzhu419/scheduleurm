from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class WindowsBatchProbeDeps:
    collect_windows_probe_targets: Callable[..., dict]
    node_is_windows: Callable[[str], bool]
    task_pids: Callable[[dict], list]
    windows_path_for_task: Callable[[dict, str], str]
    success_patterns: list
    windows_log_exit_probe_specs: Callable[..., list]
    windows_log_exit_probe_script: Callable[[str], str]
    windows_log_exit_results_from_rows: Callable[[list], dict]
    windows_process_probe_specs: Callable[..., list]
    windows_process_probe_script: Callable[[str, str], str]
    normalize_windows_json_rows: Callable[..., list]
    windows_queue_accounting_fallback_result: Callable[[dict, list, str], dict]
    windows_process_results_from_rows: Callable[..., dict]
    local_launch_transport_alive: Callable[[dict], bool]
    ps_quote: Callable[[str], str]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    environ: dict


def windows_backend_batch_probe(state: dict, *, deps: WindowsBatchProbeDeps) -> dict:
    by_node = deps.collect_windows_probe_targets(
        state,
        node_is_windows=deps.node_is_windows,
        task_pids=deps.task_pids,
    )
    results = {}
    if not by_node:
        return results

    def probe_log_exits(node: str):
        specs = deps.windows_log_exit_probe_specs(
            by_node[node],
            windows_path_for_task=deps.windows_path_for_task,
            success_patterns=deps.success_patterns,
        )
        if not specs:
            return {}
        ps = deps.windows_log_exit_probe_script(deps.ps_quote(json.dumps(specs)))
        try:
            fallback_timeout = max(20, min(60, 8 + len(specs) // 2))
            rc, out, _ = deps.run_windows_ps(node, ps, timeout=fallback_timeout, check=False)
            if rc != 0 or not (out or "").strip():
                return {}
            data = deps.normalize_windows_json_rows(out, require_nonempty=False)
            return deps.windows_log_exit_results_from_rows(data)
        except Exception:
            return {}

    def probe_processes(node: str):
        specs = deps.windows_process_probe_specs(
            by_node[node],
            windows_path_for_task=deps.windows_path_for_task,
        )
        roots = sorted({int(spec["root"]) for spec in specs})
        root_list = ",".join(str(pid) for pid in roots)
        ps = deps.windows_process_probe_script(root_list, deps.ps_quote(json.dumps(specs)))
        try:
            probe_timeout = max(12, min(60, 6 + len(specs) // 2))
            rc, out, _ = deps.run_windows_ps(node, ps, timeout=probe_timeout, check=False)
            return out if rc == 0 else None
        except Exception:
            return None

    high_fanout_threshold = max(
        1,
        int(deps.environ.get("SCHEDULEURM_WINDOWS_PROCESS_PROBE_HIGH_FANOUT", "32")),
    )
    outputs = {}
    for node in by_node.keys():
        if len(by_node.get(node, [])) > high_fanout_threshold:
            outputs[node] = None
        else:
            outputs[node] = probe_processes(node)

    log_exit_results = {
        node: probe_log_exits(node)
        for node, out in outputs.items()
        if out is None
    }

    for node, out in outputs.items():
        if out is None:
            for task, pids in by_node[node]:
                log_result = (log_exit_results.get(node) or {}).get(task["id"])
                if log_result:
                    results[task["id"]] = log_result
                    continue
                results[task["id"]] = deps.windows_queue_accounting_fallback_result(
                    task, pids, "windows_queue_accounting")
            continue
        try:
            data = deps.normalize_windows_json_rows(out, require_nonempty=True)
        except Exception as exc:
            log_result_by_id = probe_log_exits(node)
            for task, pids in by_node[node]:
                log_result = log_result_by_id.get(task["id"])
                if log_result:
                    results[task["id"]] = log_result
                    continue
                results[task["id"]] = deps.windows_queue_accounting_fallback_result(
                    task, pids, f"windows_parse_failed: {str(exc)[:80]}")
            continue
        results.update(deps.windows_process_results_from_rows(
            by_node[node],
            data,
            local_launch_transport_alive=deps.local_launch_transport_alive,
        ))
    return results
