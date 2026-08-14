from __future__ import annotations

import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


MAX_LOCAL_PROBE_SHELL_BYTES = 16 * 1024


@dataclass(frozen=True)
class LocalBatchProbeDeps:
    run_on: Callable[..., tuple[int, str, str]]
    task_pids: Callable[[dict], list[int]]
    descendants_of: Callable[[set[int], dict[int, int]], set[int]]
    max_workers: int = 2
    timeout_s: int = 30


def _expected_start_ticks(task: dict, pid: int) -> int | None:
    values = task.get("remote_pid_start_ticks") or {}
    if not isinstance(values, dict):
        return None
    raw = values.get(str(pid), values.get(pid))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _build_control_checks(
    task_specs: list[tuple[dict, list[int]]],
    all_pids: list[int],
) -> str:
    status_checks = []
    identity_checks = []
    legacy_pids = set(all_pids)
    for task, pids in task_specs:
        task_id = str(task.get("id") or "")
        status_path = str(task.get("exit_status_path") or "")
        if task_id and status_path and task.get("exit_status_token"):
            status_checks.append(
                f"if IFS=\"$(printf '\\t')\" read -r rc ft tok < "
                f"{shlex.quote(status_path)} 2>/dev/null; then "
                f"printf 'T\\t%s\\t%s\\t%s\\t%s\\n' {shlex.quote(task_id)} "
                "\"$rc\" \"$ft\" \"$tok\"; fi"
            )
        for pid in pids:
            expected = _expected_start_ticks(task, pid)
            if expected is None:
                continue
            legacy_pids.discard(pid)
            identity_checks.append(
                f"if kill -0 {pid} 2>/dev/null; then "
                f"st=$(awk '/^State:/{{print $2; exit}}' /proc/{pid}/status 2>/dev/null); "
                f"if [ -n \"$st\" ] && [ \"$st\" != Z ] && [ \"$st\" != X ]; then "
                f"actual=$(awk '{{print $22}}' /proc/{pid}/stat 2>/dev/null); "
                f"if [ \"$actual\" = {shlex.quote(str(expected))} ]; then "
                f"printf 'I\\t%s\\t%s\\n' {shlex.quote(task_id)} {pid}; "
                f"elif [ -n \"$actual\" ]; then "
                f"printf 'M\\t%s\\t%s\\n' {shlex.quote(task_id)} {pid}; fi; fi; fi"
            )
    # Zombie guard: kill -0 returns success for zombies too. Augment it with
    # /proc status so only non-Z/non-X roots are marked alive.
    pid_checks = "; ".join(
        f"kill -0 {pid} 2>/dev/null && "
        f"awk '/^State:/{{s=$2}} END{{if(s!=\"Z\" && s!=\"X\") print \"A{pid}\"}}' "
        f"/proc/{pid}/status 2>/dev/null"
        for pid in sorted(legacy_pids)
    )
    # The launcher atomically writes the exit-status file before its wrapper
    # exits. Check PID identity first and the sentinel last so a completion
    # racing this probe cannot be observed as PID-dead while missing its rc.
    return "; ".join(
        identity_checks + ([pid_checks] if pid_checks else []) + status_checks
    )


def _probe_node(
    node: str,
    task_specs: list[tuple[dict, list[int]]],
    all_pids: list[int],
    deps: LocalBatchProbeDeps,
):
    control_checks = _build_control_checks(task_specs, all_pids)
    cmd = (
        f"({control_checks}; true); echo '===VRAM==='; "
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null; "
        "echo '===PSALL==='; "
        "ps -eo pid= -o ppid= -o rss= -o pcpu= -o stat= 2>/dev/null; true"
    )
    try:
        timeout_s = max(1, int(deps.timeout_s or 30))
        rc, out, err = deps.run_on(node, cmd, timeout=timeout_s, check=False)
        if rc == 0:
            return out, None
        detail = " ".join(((err or out or "").strip()).split())
        msg = f"local backend ssh/proc probe failed on {node}: rc={rc}"
        if detail:
            msg = f"{msg}: {detail[:220]}"
        return None, msg
    except subprocess.TimeoutExpired:
        return None, f"local backend ssh/proc probe timed out on {node} after {max(1, int(deps.timeout_s or 30))}s"
    except Exception as exc:
        return None, (
            f"local backend ssh/proc probe exception on {node}: "
            f"{type(exc).__name__}: {str(exc)[:220]}"
        )


def _partition_node_tasks(
    task_specs: list[tuple[dict, list[int]]],
) -> list[list[tuple[dict, list[int]]]]:
    """Keep probe scripts small enough for nested ssh/sudo shell wrapping."""
    batches: list[list[tuple[dict, list[int]]]] = []
    current: list[tuple[dict, list[int]]] = []
    for spec in task_specs:
        candidate = current + [spec]
        candidate_pids = sorted({pid for _, pids in candidate for pid in pids})
        control_checks = _build_control_checks(candidate, candidate_pids)
        if current and len(control_checks.encode("utf-8")) > MAX_LOCAL_PROBE_SHELL_BYTES:
            batches.append(current)
            current = [spec]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _parse_probe_output(out: str):
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    vram_sep = lines.index("===VRAM===") if "===VRAM===" in lines else len(lines)
    ps_sep = lines.index("===PSALL===") if "===PSALL===" in lines else len(lines)
    alive_roots = {
        int(line[1:])
        for line in lines[:vram_sep]
        if line.startswith("A") and line[1:].isdigit()
    }
    exit_statuses: dict[str, dict] = {}
    identity_roots: dict[str, set[int]] = {}
    identity_mismatches: dict[str, set[int]] = {}
    for line in lines[:vram_sep]:
        parts = line.split("\t")
        if len(parts) == 5 and parts[0] == "T":
            try:
                exit_statuses[parts[1]] = {
                    "exit_code": int(parts[2]),
                    "finished_at": float(parts[3]),
                    "token": parts[4],
                }
            except (TypeError, ValueError):
                continue
        elif len(parts) == 3 and parts[0] in {"I", "M"}:
            try:
                pid = int(parts[2])
            except ValueError:
                continue
            target = identity_roots if parts[0] == "I" else identity_mismatches
            target.setdefault(parts[1], set()).add(pid)

    vram_per_pid: dict[int, int] = {}
    for probe_line in lines[vram_sep + 1 : ps_sep]:
        parts = [part.strip() for part in probe_line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mb = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        vram_per_pid[pid] = vram_per_pid.get(pid, 0) + mb

    rss_per_pid: dict[int, int] = {}
    pcpu_per_pid: dict[int, float] = {}
    ppid_of: dict[int, int] = {}
    for ps_line in lines[ps_sep + 1 :]:
        bits = ps_line.split()
        if len(bits) < 5:
            continue
        try:
            pid = int(bits[0])
            parent = int(bits[1])
            rss_kb = int(bits[2])
            pcpu = float(bits[3])
        except ValueError:
            continue
        stat = bits[4]
        if stat and stat[0] in ("Z", "X"):
            continue
        ppid_of[pid] = parent
        rss_per_pid[pid] = rss_kb // 1024
        pcpu_per_pid[pid] = pcpu
    return (
        alive_roots,
        vram_per_pid,
        rss_per_pid,
        pcpu_per_pid,
        ppid_of,
        exit_statuses,
        identity_roots,
        identity_mismatches,
    )


def _apply_probe_result(
    results: dict,
    task_specs: list[tuple[dict, list[int]]],
    probe_result,
    deps: LocalBatchProbeDeps,
) -> None:
    if isinstance(probe_result, tuple):
        out, probe_error = probe_result
    else:
        out, probe_error = probe_result, None
    if out is None:
        for task, _ in task_specs:
            results[task["id"]] = {
                "state": "unknown",
                "alive_pids": [],
                "vram_mb": 0,
                "ram_mb": 0,
                "pcpu": 0.0,
                "error": probe_error or "local backend ssh/proc probe failed",
            }
        return

    (
        alive_roots,
        vram_per_pid,
        rss_per_pid,
        pcpu_per_pid,
        ppid_of,
        exit_statuses,
        identity_roots,
        identity_mismatches,
    ) = _parse_probe_output(out)
    for task, pids in task_specs:
        task_id = str(task.get("id") or "")
        exit_status = exit_statuses.get(task_id)
        if (
            exit_status
            and str(exit_status.get("token") or "")
            == str(task.get("exit_status_token") or "")
        ):
            exit_code = int(exit_status["exit_code"])
            results[task["id"]] = {
                "state": "dead",
                "alive_pids": [],
                "vram_mb": 0,
                "ram_mb": 0,
                "pcpu": 0.0,
                "exit_code": exit_code,
                "backend_finished_at": exit_status.get("finished_at"),
                "backend_state": "LOCAL_EXIT_STATUS",
                "terminal_reason": f"launcher exit-status sentinel recorded rc={exit_code}",
                "terminal_diagnosis_deferred": True,
            }
            if exit_code != 0:
                results[task["id"]]["terminal_ok"] = False
            continue
        pid_set = set(pids)
        protected_pids = {
            pid for pid in pid_set if _expected_start_ticks(task, pid) is not None
        }
        verified_roots = identity_roots.get(task_id, set())
        legacy_roots = (pid_set - protected_pids) & alive_roots
        live_roots = verified_roots | legacy_roots
        expanded_pid_set = live_roots | deps.descendants_of(live_roots, ppid_of)
        this_alive = expanded_pid_set & (live_roots | set(rss_per_pid))
        if not this_alive:
            identity_mismatch = bool(identity_mismatches.get(task_id))
            results[task["id"]] = {
                "state": "dead",
                "alive_pids": [],
                "vram_mb": 0,
                "ram_mb": 0,
                "pcpu": 0.0,
                "backend_state": (
                    "LOCAL_PID_IDENTITY_MISMATCH"
                    if identity_mismatch
                    else "LOCAL_PID_DEAD"
                ),
                "terminal_reason": (
                    "local backend tracked PID start identity no longer matches launch"
                    if identity_mismatch
                    else "local backend pid probe found no live tracked process"
                ),
                "terminal_diagnosis_deferred": True,
            }
            continue
        results[task["id"]] = {
            "state": "alive",
            "alive_pids": sorted(this_alive),
            "vram_mb": sum(vram_per_pid.get(pid, 0) for pid in this_alive),
            "ram_mb": sum(rss_per_pid.get(pid, 0) for pid in this_alive),
            "pcpu": sum(pcpu_per_pid.get(pid, 0.0) for pid in this_alive),
        }


def local_backend_batch_probe(state: dict, *, deps: LocalBatchProbeDeps) -> dict:
    by_node: dict[str, list[tuple[dict, list[int]]]] = {}
    for task in state["tasks"]:
        if task["status"] != "running":
            continue
        node = task.get("node")
        if not node:
            continue
        pids = deps.task_pids(task)
        has_exit_sentinel = bool(
            task.get("id")
            and task.get("exit_status_path")
            and task.get("exit_status_token")
        )
        if not pids and not has_exit_sentinel:
            continue
        by_node.setdefault(node, []).append((task, pids))

    results: dict = {}
    if not by_node:
        return results

    nodes_list = list(by_node.keys())
    max_workers = max(1, min(len(nodes_list), int(deps.max_workers or 1)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outputs_by_node = dict(
            zip(
                nodes_list,
                executor.map(
                    lambda node: [
                        (
                            batch,
                            _probe_node(
                                node,
                                batch,
                                sorted({pid for _, pids in batch for pid in pids}),
                                deps,
                            ),
                        )
                        for batch in _partition_node_tasks(by_node[node])
                    ],
                    nodes_list,
                ),
            )
        )

    for node_batches in outputs_by_node.values():
        for task_specs, probe_result in node_batches:
            _apply_probe_result(results, task_specs, probe_result, deps)
    return results
