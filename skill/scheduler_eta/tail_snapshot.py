"""ETA log-tail snapshot helpers used outside the scheduler state lock."""

from __future__ import annotations

import shlex
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class EtaTailOutputDeps:
    node_is_windows: Callable[[str], bool]
    ps_quote: Callable[[str], str]
    windows_tail_ps: Callable[..., str]
    windows_path_for_task: Callable[[dict, str], str]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    run_on: Callable[..., tuple[int, str, str]]
    timeout_s: int = 10


@dataclass(frozen=True)
class EtaTailSnapshotDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    eta_refresh_due: Callable[[dict], bool]
    eta_tail_targets: Callable[[dict], tuple[dict, list]]
    eta_tail_outputs_by_node: Callable[[dict], dict]
    notify: Callable[..., Any]


def eta_tail_outputs_by_node(by_node: dict, *, deps: EtaTailOutputDeps) -> dict:
    if not by_node:
        return {}

    def _probe(node: str):
        entries = by_node[node]
        if deps.node_is_windows(node):
            parts = []
            for task, log_path in entries:
                tid = task["id"]
                parts.append(f"Write-Output {deps.ps_quote(f'===ETA_LOG_{tid}===')}")
                parts.append(
                    deps.windows_tail_ps(
                        deps.windows_path_for_task(task, log_path),
                        max_bytes=4096,
                    )
                )
            try:
                rc, out, _ = deps.run_windows_ps(
                    node,
                    "\n".join(parts),
                    timeout=max(1, int(deps.timeout_s)),
                    check=False,
                )
                return out if rc == 0 else None
            except Exception:
                return None

        parts = []
        for task, log_path in entries:
            tid = task["id"]
            q_log = shlex.quote(log_path)
            parts.append(f"echo '===ETA_LOG_{tid}==='")
            task_text = " ".join(
                str(task.get(k) or "") for k in ("project", "signature", "description", "cmd")
            ).lower()
            if "freqduet" in task_text or "run_freqduet_" in task_text:
                parts.append(
                    f"(grep -aE '^(Shard jobs|SKIP |DONE )' {q_log} 2>/dev/null; "
                    f"tail -c 65536 {q_log} 2>/dev/null; "
                    f"grep -a 'Results saved to:' {q_log} 2>/dev/null | tail -n 100)"
                )
            elif "bapr" in task_text:
                parts.append(
                    f"(grep -aE '^(TRAIN SUBPROCESS:|Training complete! Total time:|PAIR .*COMPLETE:)' "
                    f"{q_log} 2>/dev/null | tail -n 40; "
                    f"tail -c 65536 {q_log} 2>/dev/null; "
                    f"grep -a 'Results saved to:' {q_log} 2>/dev/null | tail -n 100)"
                )
            elif "kg_op" in task_text or "benchmark_sota.py" in task_text:
                parts.append(
                    f"(grep -aE '(\\[kg-inner\\]|\\[sota\\]|Iter |Iteration |Step |ETA |DONE|Results saved to:)' "
                    f"{q_log} 2>/dev/null | tail -n 200; "
                    f"tail -c 65536 {q_log} 2>/dev/null; "
                    f"grep -a 'Results saved to:' {q_log} 2>/dev/null | tail -n 100)"
                )
            else:
                parts.append(
                    f"(tail -c 4096 {q_log} 2>/dev/null; "
                    f"grep -a 'Results saved to:' {q_log} 2>/dev/null | tail -n 100)"
                )
        cmd = "; ".join(parts) + "; true"
        try:
            rc, out, _ = deps.run_on(
                node,
                cmd,
                timeout=max(1, int(deps.timeout_s)),
                check=False,
            )
            return out if rc == 0 else None
        except Exception:
            return None

    nodes_list = list(by_node.keys())
    with ThreadPoolExecutor(max_workers=max(1, len(nodes_list))) as ex:
        return dict(zip(nodes_list, ex.map(_probe, nodes_list)))


def eta_tail_snapshot_outside_lock(
    purpose: str = "eta-tail",
    *,
    deps: EtaTailSnapshotDeps,
) -> tuple[dict, set]:
    try:
        with deps.state_lock(shared=True, purpose=f"{purpose}:snapshot"):
            snapshot = deps.load_state()
            if not deps.eta_refresh_due(snapshot):
                return {}, set()
            by_node, _pure_ewma = deps.eta_tail_targets(snapshot)
            running_ids = {
                str(t.get("id") or "")
                for t in snapshot.get("tasks", [])
                if t.get("status") == "running" and t.get("id")
            }
            if not running_ids:
                return {}, running_ids
    except Exception as e:
        deps.notify(
            "eta_tail_snapshot_error",
            {"purpose": purpose, "error": str(e)[:200]},
            feishu_enabled=False,
        )
        return {}, set()
    try:
        return deps.eta_tail_outputs_by_node(by_node), running_ids
    except Exception as e:
        deps.notify(
            "eta_tail_error_outer",
            {"purpose": purpose, "error": str(e)[:200]},
            feishu_enabled=False,
        )
        return {}, running_ids
