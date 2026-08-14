"""Manual adopt command for externally launched Linux GPU processes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AdoptCommandDeps:
    node_is_windows: Callable[[str], bool]
    run_on: Callable[..., tuple[int, str, str]]
    project_from_path: Callable[[Any], str]
    project_from_pid: Callable[[str, int], str]
    history_get: Callable[[str], dict | None]
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    allocate_task_id: Callable[[dict], str]
    save_state: Callable[[dict], None]
    local_user: Callable[[], str]
    default_vram_mb: int
    default_ram_mb: int
    now: Callable[[], float]


def _probe_manual_adopt_gpu(args: Any, pids: list[int], *, deps: AdoptCommandDeps) -> tuple[int, set[int]]:
    pid_checks = "; ".join(
        f"kill -0 {pid} 2>/dev/null && "
        f"awk '/^State:/{{s=$2}} END{{if(s!=\"Z\" && s!=\"X\") print \"ALIVE_{pid}\"}}' "
        f"/proc/{pid}/status 2>/dev/null"
        for pid in pids
    )
    probe = (
        f"({pid_checks}; true); echo '===PROC==='; "
        "nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory "
        "--format=csv,noheader,nounits 2>/dev/null; "
        "echo '===GPU==='; "
        "nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null"
    )
    try:
        rc, out, err = deps.run_on(args.node, probe, timeout=15, check=False)
    except Exception as exc:
        sys.exit(f"failed to probe {args.node}: {exc}")
    if rc != 0:
        sys.exit(f"probe rc={rc}: {err.strip()[:300]}")
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    proc_sep = lines.index("===PROC===") if "===PROC===" in lines else len(lines)
    gpu_sep = lines.index("===GPU===") if "===GPU===" in lines else len(lines)
    alive_pids = {
        int(line.split("_", 1)[1])
        for line in lines[:proc_sep]
        if line.startswith("ALIVE_")
    }
    dead = set(pids) - alive_pids
    if dead:
        sys.exit(f"these PIDs are not alive on {args.node}: {sorted(dead)}")

    bus_to_idx = {}
    for gpu_line in lines[gpu_sep + 1:]:
        parts = [part.strip() for part in gpu_line.split(",")]
        if len(parts) < 2:
            continue
        try:
            bus_to_idx[parts[1]] = int(parts[0])
        except ValueError:
            continue

    pid_set = set(pids)
    sum_vram = 0
    detected_gpus: set[int] = set()
    for proc_line in lines[proc_sep + 1:gpu_sep]:
        parts = [part.strip() for part in proc_line.split(",")]
        if len(parts) < 3:
            continue
        try:
            proc_pid = int(parts[1])
            used_mb = int(parts[2])
        except ValueError:
            continue
        if proc_pid not in pid_set:
            continue
        sum_vram += used_mb
        bus = parts[0]
        if bus in bus_to_idx:
            detected_gpus.add(bus_to_idx[bus])
    return sum_vram, detected_gpus


def _probe_manual_adopt_process_usage(node: str, pids: list[int], *, deps: AdoptCommandDeps) -> tuple[int, float]:
    checks = []
    for pid in pids:
        checks.append(
            f"p={int(pid)}; "
            "rss=$(awk '/VmRSS:/ {print int($2/1024)}' /proc/$p/status 2>/dev/null); "
            "pcpu=$(ps -o pcpu= -p $p 2>/dev/null | tr -d ' '); "
            "echo \"$p|${rss:-0}|${pcpu:-0}\""
        )
    try:
        rc, out, _ = deps.run_on(node, "; ".join(checks), timeout=8, check=False)
    except Exception:
        return 0, 0.0
    if rc != 0:
        return 0, 0.0
    sum_rss = 0
    sum_pcpu = 0.0
    wanted = set(pids)
    for line in (out or "").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid not in wanted:
            continue
        try:
            sum_rss += int(float(parts[1] or 0))
        except ValueError:
            pass
        try:
            sum_pcpu += float(parts[2] or 0.0)
        except ValueError:
            pass
    return sum_rss, sum_pcpu


def _manual_adopt_cwds(args: Any, pids: list[int], *, deps: AdoptCommandDeps) -> dict[int, str]:
    cwd_per_pid = {}
    for pid in pids:
        try:
            rc, out, _ = deps.run_on(args.node, f"readlink /proc/{pid}/cwd 2>/dev/null", timeout=5, check=False)
            cwd_per_pid[pid] = out.strip() if rc == 0 else ""
        except Exception:
            cwd_per_pid[pid] = ""
    return cwd_per_pid


def _validate_single_project(args: Any, pids: list[int], cwd_per_pid: dict[int, str], *, deps: AdoptCommandDeps) -> None:
    distinct_projects = {deps.project_from_path(cwd) for cwd in cwd_per_pid.values() if cwd}
    if len(distinct_projects) <= 1 or args.allow_multi_project:
        return
    groups = {}
    for pid, cwd in cwd_per_pid.items():
        groups.setdefault(deps.project_from_path(cwd) or "?", []).append(pid)
    msg_lines = ["adopt refused - these PIDs belong to different projects:"]
    for project, project_pids in groups.items():
        msg_lines.append(f"  {project}: pids={project_pids}")
    msg_lines.append("Run a separate `adopt` per project, or pass --allow-multi-project to override.")
    sys.exit("\n".join(msg_lines))


def _manual_adopt_project(args: Any, pids: list[int], *, deps: AdoptCommandDeps) -> str:
    return (
        args.project
        or deps.project_from_path(args.cwd)
        or deps.project_from_pid(args.node, pids[0])
        or (args.signature.split("/", 1)[0] if "/" in args.signature else args.signature)
    )


def _manual_adopt_task(
    args: Any,
    pids: list[int],
    *,
    project: str,
    est_vram: int,
    sum_vram: int,
    sum_rss: int,
    sum_pcpu: float,
    hist: dict,
    deps: AdoptCommandDeps,
) -> dict:
    now = deps.now()
    return {
        "status": "running",
        "description": args.description,
        "project": project,
        "cmd": "(auto-adopted manually — launched outside scheduler)",
        "cwd": args.cwd or "(unknown)",
        "signature": args.signature,
        "est_vram_mb": int(est_vram),
        "ram_mb": hist.get("ram_mb") or deps.default_ram_mb,
        "cpu_cores": hist.get("cpu_cores") or len(pids),
        "priority": "normal",
        "preferred_node": None,
        "git_repo": None,
        "ckpt_dir": args.ckpt_dir,
        "ckpt_glob": "*",
        "resume_flag": "",
        "extra_env": {},
        "origin": "manual-adopt",
        "submitted_by": deps.local_user(),
        "process_owner": deps.local_user(),
        "submitted_host": args.node,
        "scheduler_id": None,
        "shared_account_suspect": True,
        "node": args.node,
        "gpu_idx": args.gpu,
        "remote_pids": pids,
        "log_path": args.log_path,
        "submitted_at": now,
        "started_at": now,
        "finished_at": None,
        "peak_vram_mb": sum_vram,
        "peak_ram_mb": sum_rss,
        "current_vram_mb": sum_vram,
        "current_ram_mb": sum_rss,
        "current_pcpu": sum_pcpu,
        "resume_from": None,
        "adopted": True,
        "auto_adopted": True,
        "notified_launch": True,
    }


def cmd_adopt(args: Any, *, deps: AdoptCommandDeps) -> None:
    """Register externally launched process(es) as a tracked task."""
    if deps.node_is_windows(args.node):
        sys.exit("adopt is Linux/GPU-only for now; Windows jtl110cpu tasks must be launched through scheduler.py")
    pids = list(args.pids)
    if not pids:
        sys.exit("--pid requires at least one PID")

    sum_vram, detected_gpus = _probe_manual_adopt_gpu(args, pids, deps=deps)
    if detected_gpus and args.gpu not in detected_gpus:
        actual = sorted(detected_gpus)
        if len(actual) == 1:
            print(f"NOTE: detected your PIDs are on GPU{actual[0]} (you said GPU{args.gpu}). Using GPU{actual[0]}.")
            args.gpu = actual[0]
        else:
            print(
                f"WARNING: PIDs span GPUs {actual} (multi-GPU job). "
                f"Tagging as GPU{args.gpu}, but VRAM tracking still sums all of them."
            )
    if sum_vram == 0 and not detected_gpus:
        print(
            f"WARNING: PIDs alive but not visible as nvidia-smi compute apps. "
            f"Adopting with --gpu {args.gpu}; VRAM will be 0 until they allocate."
        )

    cwd_per_pid = _manual_adopt_cwds(args, pids, deps=deps)
    _validate_single_project(args, pids, cwd_per_pid, deps=deps)
    sum_rss, sum_pcpu = _probe_manual_adopt_process_usage(args.node, pids, deps=deps)
    est = args.est_vram or sum_vram or deps.default_vram_mb
    project = _manual_adopt_project(args, pids, deps=deps)
    hist = deps.history_get(args.signature) or {}

    with deps.state_lock():
        state = deps.load_state()
        task_id = deps.allocate_task_id(state)
        task = _manual_adopt_task(
            args,
            pids,
            project=project,
            est_vram=est,
            sum_vram=sum_vram,
            sum_rss=sum_rss,
            sum_pcpu=sum_pcpu,
            hist=hist,
            deps=deps,
        )
        task["id"] = task_id
        state["tasks"].append(task)
        deps.save_state(state)
    print(
        f"adopted {task['id']}: {len(pids)} pid(s)={pids} on {args.node}:GPU{args.gpu}  "
        f"vram_now={sum_vram}MB est={est}MB sig={args.signature}"
    )
    print(f"  task is marked 'done' only when ALL {len(pids)} PIDs have exited.")
