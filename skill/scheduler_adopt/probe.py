"""Remote process probes used by external task auto-adoption."""

from __future__ import annotations

import getpass
import math
import os
import shlex
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class AdoptProbeDeps:
    node_configs: dict
    node_is_windows: Callable[[str], bool]
    run_on: Callable[..., tuple[int, str, str]]
    local_user: Callable[[], str]
    task_pids: Callable[[dict], list]
    descendants_cap: int = 500


def node_processes(name: str, *, deps: AdoptProbeDeps) -> list[dict]:
    """Fetch GPU compute processes and metadata from one Linux node."""
    if deps.node_is_windows(name):
        return []
    cmd = (
        "nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null; "
        "echo '===BUS==='; "
        "nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null; "
        "echo '===META==='; "
        "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do "
        "  o=$(ps -o user= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "  c=$(readlink /proc/$p/cwd 2>/dev/null); "
        "  r=$(awk '/^VmRSS:/ {print $2}' /proc/$p/status 2>/dev/null); "
        "  pc=$(ps -o pcpu= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "  pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "  sl=0; grep -aqE 'SLURM_JOB_ID=|SLURM_JOBID=' /proc/$p/environ 2>/dev/null && sl=1; "
        "  cl=$(head -c 4096 /proc/$p/cmdline 2>/dev/null | tr '\\0' ' ' | sed 's/[[:space:]]*$//'); "
        "  echo \"${p}|${o}|${c}|${r}|${pc}|${pg}|${sl}|${cl}\"; "
        "done"
    )
    try:
        rc, out, _ = deps.run_on(name, cmd, timeout=20, check=False)
        if rc != 0:
            return []
    except Exception:
        return []
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    bus_sep = lines.index("===BUS===") if "===BUS===" in lines else 0
    meta_sep = lines.index("===META===") if "===META===" in lines else len(lines)
    proc_lines = lines[:bus_sep]
    gpu_lines = lines[bus_sep + 1:meta_sep]
    meta_lines = lines[meta_sep + 1:]

    bus_to_idx = {}
    for gpu_line in gpu_lines:
        parts = [part.strip() for part in gpu_line.split(",")]
        if len(parts) < 2:
            continue
        try:
            bus_to_idx[parts[1]] = int(parts[0])
        except ValueError:
            continue

    pid_meta = {}
    for meta_line in meta_lines:
        bits = meta_line.split("|", 7)
        if len(bits) < 3:
            continue
        owner = bits[1].strip()
        cwd = bits[2].strip()
        rss_kb = 0
        if len(bits) >= 4 and bits[3].strip().isdigit():
            rss_kb = int(bits[3].strip())
        try:
            pcpu = float(bits[4].strip()) if len(bits) >= 5 and bits[4].strip() else 0.0
        except ValueError:
            pcpu = 0.0
        pgid = None
        if len(bits) >= 6 and bits[5].strip().isdigit():
            pgid = int(bits[5].strip())
        is_slurm = len(bits) >= 7 and bits[6].strip() == "1"
        cmdline = bits[7].strip() if len(bits) >= 8 else ""
        try:
            pid_meta[int(bits[0])] = (owner, cwd, rss_kb, pcpu, pgid, is_slurm, cmdline)
        except ValueError:
            continue

    out_list = []
    for proc_line in proc_lines:
        parts = [part.strip() for part in proc_line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[1])
            used = int(parts[2])
        except ValueError:
            continue
        gpu_idx = bus_to_idx.get(parts[0])
        if gpu_idx is None:
            continue
        owner, cwd, rss_kb, pcpu, pgid, is_slurm, cmdline = pid_meta.get(
            pid, ("", "", 0, 0.0, None, False, "")
        )
        out_list.append({
            "node": name,
            "pid": pid,
            "gpu_idx": gpu_idx,
            "used_mb": used,
            "owner": owner,
            "cwd": cwd,
            "rss_mb": rss_kb // 1024,
            "pcpu": pcpu,
            "pgid": pgid or pid,
            "cmdline": cmdline,
            "is_slurm": is_slurm,
        })
    return out_list


def node_ppid_map(name: str, *, deps: AdoptProbeDeps) -> dict[int, int]:
    """Build {pid: ppid} for one node."""
    if deps.node_is_windows(name):
        return {}
    try:
        rc, out, _ = deps.run_on(name, "ps -eo pid= -o ppid=", timeout=15, check=False)
        if rc != 0:
            return {}
    except Exception:
        return {}
    ppid_of = {}
    for line in out.splitlines():
        bits = line.split()
        if len(bits) != 2:
            continue
        try:
            ppid_of[int(bits[0])] = int(bits[1])
        except ValueError:
            continue
    return ppid_of


def descendants_of(roots, ppid_of, *, cap: int = 500) -> set:
    """Return PIDs whose ancestor chain hits one of roots, capped for probe safety."""
    if not roots or not ppid_of:
        return set()
    descendants = set()
    cache = {}
    for pid in ppid_of:
        if len(descendants) >= cap:
            break
        if pid in roots:
            continue
        path = []
        cur = pid
        guard = 0
        while cur and cur != 1 and guard < 64:
            if cur in cache:
                under = cache[cur]
                break
            if cur in roots:
                under = True
                break
            path.append(cur)
            cur = ppid_of.get(cur)
            guard += 1
        else:
            under = False
        for path_pid in path:
            cache[path_pid] = under
            if under:
                descendants.add(path_pid)
            if len(descendants) >= cap:
                break
    return descendants


def node_expected_process_owner(name: str, *, deps: AdoptProbeDeps) -> str:
    info = deps.node_configs.get(name, {}) or {}
    return str(info.get("sudo_ssh_run_as") or info.get("ssh_user") or deps.local_user())


def node_auto_adopt_owners(name: str, *, deps: AdoptProbeDeps) -> set[str]:
    info = deps.node_configs.get(name, {}) or {}
    owners = []
    raw = info.get("auto_adopt_owners") or []
    if isinstance(raw, str):
        raw = [raw]
    owners.extend(str(item).strip() for item in raw if str(item).strip())
    owners.append(node_expected_process_owner(name, deps=deps))
    return {owner for owner in owners if owner}


def node_auto_adopt_roots(name: str, owner: str, *, deps: AdoptProbeDeps) -> list[str]:
    info = deps.node_configs.get(name, {}) or {}
    roots = []
    remote_root = str(info.get("remote_workspace_root") or "").strip()
    if remote_root:
        roots.append(remote_root)
    if owner:
        roots.append(f"/home/{owner}")
    local = deps.local_user()
    if local and local != owner:
        roots.append(f"/home/{local}")

    out = []
    seen = set()
    for root in roots:
        norm = os.path.normpath(root)
        if not norm or norm == "." or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def path_under_roots(path: str, roots: list[str]) -> bool:
    if not path:
        return False
    norm = os.path.normpath(path)
    for root in roots:
        root_norm = os.path.normpath(root)
        if norm == root_norm or norm.startswith(root_norm.rstrip("/") + "/"):
            return True
    return False


def infer_adopt_cwd_from_cmdline(cmdline: str, roots: list[str]) -> str:
    """Fallback when /proc/<pid>/cwd is unreadable."""
    if not cmdline:
        return ""
    try:
        parts = shlex.split(cmdline)
    except Exception:
        parts = cmdline.split()
    for part in parts:
        if not part.startswith("/"):
            continue
        path = os.path.normpath(part)
        if not path_under_roots(path, roots):
            continue
        base = os.path.basename(path)
        if base.startswith("python") and "/bin/" in path:
            continue
        if "." in base:
            return os.path.dirname(path)
        return path
    return ""


def node_cpu_processes(name: str, *, deps: AdoptProbeDeps) -> list[dict]:
    """Find user-owned CPU-burning python processes not visible in nvidia-smi."""
    if deps.node_is_windows(name):
        return []
    script = (
        f"PIDS=$(ps -eo pid,pcpu,cmd --no-headers 2>/dev/null | "
        f"awk '$3~/python/ && $2+0>=50 {{print $1}}'); "
        f"for p in $PIDS; do "
        f"  own=$(ps -o user= -p $p 2>/dev/null | tr -d ' '); "
        f"  pcpu=$(ps -o pcpu= -p $p 2>/dev/null | tr -d ' '); "
        f"  rss=$(ps -o rss= -p $p 2>/dev/null | tr -d ' '); "
        f"  pg=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' '); "
        f"  cwd=$(readlink /proc/$p/cwd 2>/dev/null); "
        f"  sl=0; grep -aqE 'SLURM_JOB_ID=|SLURM_JOBID=' /proc/$p/environ 2>/dev/null && sl=1; "
        f"  cl=$(head -c 4096 /proc/$p/cmdline 2>/dev/null | tr '\\0' ' ' | sed 's/[[:space:]]*$//'); "
        f"  echo \"${{p}}|${{own}}|${{pcpu}}|${{rss}}|${{pg}}|${{cwd}}|${{sl}}|${{cl}}\"; "
        f"done"
    )
    try:
        rc, out, _ = deps.run_on(name, script, timeout=20, check=False)
        if rc != 0:
            return []
    except Exception:
        return []
    out_list = []
    for line in out.splitlines():
        bits = line.strip().split("|", 7)
        if len(bits) < 6:
            continue
        try:
            pid = int(bits[0])
            owner = bits[1].strip()
            pcpu = float(bits[2]) if bits[2] else 0.0
            rss_kb = int(bits[3]) if bits[3].isdigit() else 0
            pgid = int(bits[4]) if bits[4].isdigit() else pid
        except ValueError:
            continue
        cwd = bits[5].strip()
        is_slurm = len(bits) >= 7 and bits[6].strip() == "1"
        cmdline = bits[7].strip() if len(bits) >= 8 else ""
        out_list.append({
            "node": name,
            "pid": pid,
            "owner": owner,
            "rss_mb": rss_kb // 1024,
            "cwd": cwd,
            "pcpu": pcpu,
            "gpu_idx": None,
            "used_mb": 0,
            "is_cpu_only": True,
            "pgid": pgid,
            "cmdline": cmdline,
            "is_slurm": is_slurm,
        })
    return out_list


def refresh_adopted_resources(state: dict, gpu_proc_lists: list, cpu_proc_lists: list, *, deps: AdoptProbeDeps) -> None:
    """Refresh running auto-adopted task CPU/RAM estimates from current probes."""
    pid_stats = {}
    for plist in gpu_proc_lists + cpu_proc_lists:
        for proc in plist:
            key = (proc["node"], proc["pid"])
            pid_stats[key] = (proc.get("pcpu", 0.0), proc.get("rss_mb", 0), proc.get("pgid"))
    for task in state["tasks"]:
        if task.get("status") != "running" or not task.get("auto_adopted"):
            continue
        node = task.get("node")
        pids = deps.task_pids(task)
        if not node or not pids:
            continue
        sum_pcpu, sum_rss = 0.0, 0
        pgids_seen = set()
        seen = 0
        for pid in pids:
            if (node, pid) not in pid_stats:
                continue
            pcpu, rss_mb, pgid = pid_stats[(node, pid)]
            sum_pcpu += pcpu
            sum_rss += rss_mb
            seen += 1
            if pgid:
                pgids_seen.add(pgid)
        if seen == 0:
            continue
        if len(pgids_seen) == 1 and not task.get("process_group"):
            task["process_group"] = next(iter(pgids_seen))
        new_cpu = max(1, math.ceil(sum_pcpu / 100.0)) if sum_pcpu > 0 else None
        if new_cpu is not None:
            cur = task.get("cpu_cores", new_cpu)
            if new_cpu > cur:
                task["cpu_cores"] = new_cpu
            elif cur > new_cpu * 2.5:
                task["cpu_cores"] = new_cpu
        if sum_rss > 0 and sum_rss < task.get("ram_mb", 10**9):
            task["ram_mb"] = sum_rss


def collect_external_task_probe_data(
    state: Optional[dict] = None,
    *,
    deps: AdoptProbeDeps,
    eligible_node_names: Optional[set[str]] = None,
    node_processes_fn: Callable[[str], list[dict]] | None = None,
    node_cpu_processes_fn: Callable[[str], list[dict]] | None = None,
    node_ppid_map_fn: Callable[[str], dict[int, int]] | None = None,
) -> dict:
    """Collect remote process tables for auto-adopt outside the state lock."""
    me = getpass.getuser()
    eligible = (
        None
        if eligible_node_names is None
        else {str(name) for name in eligible_node_names if str(name)}
    )
    adopt_node_names = [
        name for name, info in deps.node_configs.items()
        if not (info or {}).get("monitor_only")
        and not (info or {}).get("retired")
        and not (info or {}).get("disable_auto_adopt")
        and (eligible is None or name in eligible)
    ]
    adopt_owner_by_node = {
        name: node_expected_process_owner(name, deps=deps)
        for name in adopt_node_names
    }
    adopt_owners_by_node = {
        name: node_auto_adopt_owners(name, deps=deps)
        for name in adopt_node_names
    }
    adopt_roots_by_node = {
        name: node_auto_adopt_roots(name, adopt_owner_by_node.get(name) or me, deps=deps)
        for name in adopt_node_names
    }
    adopt_skip_projects_by_node = {
        name: {
            me,
            adopt_owner_by_node.get(name) or "",
            *(os.path.basename(root.rstrip("/")) for root in adopt_roots_by_node.get(name, [])),
        }
        for name in adopt_node_names
    }
    cleanup_node_names = []
    for name in adopt_node_names:
        if name not in cleanup_node_names:
            cleanup_node_names.append(name)
    for task in (state or {}).get("tasks", []):
        name = task.get("node")
        node_info = deps.node_configs.get(name, {}) if name else {}
        if (
            task.get("status") == "running"
            and name in deps.node_configs
            and not (node_info or {}).get("retired")
            and (eligible is None or name in eligible)
            and name not in cleanup_node_names
        ):
            cleanup_node_names.append(name)

    node_processes_fn = node_processes_fn or (lambda name: node_processes(name, deps=deps))
    node_cpu_processes_fn = node_cpu_processes_fn or (lambda name: node_cpu_processes(name, deps=deps))
    node_ppid_map_fn = node_ppid_map_fn or (lambda name: node_ppid_map(name, deps=deps))
    with ThreadPoolExecutor(max_workers=max(1, len(adopt_node_names) * 2)) as executor:
        gpu_proc_lists = list(executor.map(node_processes_fn, adopt_node_names))
        cpu_proc_lists = list(executor.map(node_cpu_processes_fn, adopt_node_names))
    with ThreadPoolExecutor(max_workers=max(1, len(cleanup_node_names))) as executor:
        ppid_maps = list(executor.map(node_ppid_map_fn, cleanup_node_names))
    return {
        "me": me,
        "adopt_node_names": adopt_node_names,
        "adopt_owner_by_node": adopt_owner_by_node,
        "adopt_owners_by_node": adopt_owners_by_node,
        "adopt_roots_by_node": adopt_roots_by_node,
        "adopt_skip_projects_by_node": adopt_skip_projects_by_node,
        "cleanup_node_names": cleanup_node_names,
        "gpu_proc_lists": gpu_proc_lists,
        "cpu_proc_lists": cpu_proc_lists,
        "ppid_maps": ppid_maps,
    }
