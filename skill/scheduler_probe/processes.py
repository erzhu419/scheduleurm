"""Read-only workload snapshots, separate from scheduler-owned task state."""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

from .memory import PROCESS_MEMORY_SCRIPT


PROCESS_MARKER = "===SCHEDULEURM_PROCESSES==="

# The HPC system interpreter is Python 2.7; GPU hosts also provide Python 3.
PROCESS_SCRIPT = r'''
# scheduleurm-process-snapshot
from __future__ import print_function
import json, os, pwd, subprocess, sys, time
''' + PROCESS_MEMORY_SCRIPT + r'''
sampled_at = time.time()
table = subprocess.check_output([
    "ps", "-eo", "user:64,pid,ppid,pgid,etimes,pcpu,rss,comm,args", "--no-headers"
]).decode("utf-8", "replace")
rows = {}
for line in table.splitlines():
    fields = line.split(None, 8)
    if len(fields) < 8:
        continue
    owner, pid, ppid, pgid, elapsed, pcpu, rss, comm = fields[:8]
    rows[int(pid)] = dict(pid=int(pid), ppid=int(ppid), pgid=int(pgid),
        owner=owner, elapsed_s=int(elapsed), pcpu=float(pcpu),
        rss_mb=int(rss)//1024, comm=comm, cmdline=fields[8][:2048] if len(fields)>8 else "")
gpu_pids = {}
nvidia_smi = sys.argv[1]
try:
    gpu_text = subprocess.check_output([
        nvidia_smi, "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"
    ], stderr=subprocess.PIPE).decode("utf-8", "replace")
    gpu_names = subprocess.check_output([
        nvidia_smi, "--query-gpu=uuid,index", "--format=csv,noheader,nounits"
    ], stderr=subprocess.PIPE).decode("utf-8", "replace")
    indices = dict((a.strip(), int(b.strip())) for a,b in
        (line.split(",",1) for line in gpu_names.splitlines() if "," in line))
    for line in gpu_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts)==3 and parts[0] in indices and parts[1].isdigit():
            gpu_pids.setdefault(int(parts[1]), []).append(dict(
                idx=indices[parts[0]], used_mb=int(parts[2]) if parts[2].isdigit() else 0))
except (OSError, ValueError, subprocess.CalledProcessError):
    pass

skip = set([os.getpid()])
parent = os.getpid()
while parent in rows and rows[parent]["ppid"] > 1:
    parent = rows[parent]["ppid"]
    skip.add(parent)
uids = {}
processes = []
session_services = set(["Xorg", "Xwayland", "gnome-shell", "mutter-x11-fram",
    "xdg-desktop-por", "update-manager", "snap", "sshd", "bash", "sh", "ps", "systemd"])
for pid, row in rows.items():
    owner = row["owner"]
    if owner not in uids:
        try:
            uids[owner] = pwd.getpwnam(owner).pw_uid
        except KeyError:
            uids[owner] = -1
    runtime = row["comm"].lower().startswith(("python", "matlab", "fluent", "cortex", "comsol", "ansys"))
    user_work = 1000 <= uids[owner] < 65534 or (uids[owner]==0 and runtime)
    if pid in skip or row["comm"] in session_services:
        continue
    if "# scheduleurm-process-snapshot" in row["cmdline"]:
        continue
    if pid not in gpu_pids and not (user_work and (
            row["pcpu"] >= 1 or row["rss_mb"] >= 100 or runtime)):
        continue
    try:
        row["cwd"] = os.readlink("/proc/%s/cwd" % pid)
    except OSError:
        row["cwd"] = ""
    ancestors = []
    parent = row["ppid"]
    while parent > 1 and parent in rows and parent not in ancestors:
        ancestors.append(parent)
        parent = rows[parent]["ppid"]
    row["ancestors"] = ancestors
    row["gpus"] = gpu_pids.get(pid, [])
    pss = process_pss_kb(pid)
    if pss is not None:
        row["pss_kb"] = pss
    processes.append(row)
print(json.dumps(dict(sampled_at=sampled_at, processes=processes)))
'''


def process_probe_command(nvidia_smi: str = "nvidia-smi") -> str:
    return (
        f"printf '\\n{PROCESS_MARKER}\\n'; "
        "_sched_python=$(command -v python3 || command -v python); "
        f'"$_sched_python" -c {shlex.quote(PROCESS_SCRIPT)} {shlex.quote(nvidia_smi)}; '
        "true"
    )


def parse_process_snapshot(output: str) -> dict:
    _, marker, body = output.partition(PROCESS_MARKER)
    if not marker:
        return {}
    try:
        snapshot = json.loads(body.strip())
        snapshot["received_at"] = time.time()
        return {"process_snapshot": snapshot}
    except ValueError:
        return {"process_probe_error": "process snapshot unavailable"}


def _local_project_name(cwd: str) -> str:
    if not cwd:
        return ""
    directory = Path(cwd)
    home = Path.home()
    for parent in (directory, *directory.parents):
        if parent in (home, Path(parent.anchor), Path("/tmp"), Path("/var/tmp")):
            break
        if any((parent / marker).exists() for marker in (".git", "pyproject.toml", "setup.py")):
            return parent.name
    return ""


def _local_compute_process(proc: dict) -> bool:
    if proc.get("gpus"):
        return True
    runtime = str(proc.get("comm") or "").lower()
    if not runtime.startswith(("python", "matlab", "fluent", "cortex", "comsol", "ansys")):
        return False
    if runtime.startswith("python"):
        try:
            args = shlex.split(proc.get("cmdline") or "")
        except ValueError:
            return False
        if any(arg in ("-", "-c") for arg in args[1:3]):
            return False
        if "-m" in args:
            module_args = args[args.index("-m") + 1:]
            if module_args and module_args[0] in ("pytest", "coverage", "pip"):
                return False
    return True


def external_display_tasks(state: dict, nodes: list, *, is_control_process, now=None) -> list:
    """Make display-only rows; never insert observed processes into queue.json."""
    now = time.time() if now is None else now
    tracked, groups = {}, {}
    for task in state.get("tasks", []):
        if task.get("status") not in ("running", "launching"):
            continue
        node = task.get("node")
        tracked.setdefault(node, set()).update(int(pid) for pid in task.get("remote_pids") or [])
        if task.get("remote_pid"):
            tracked[node].add(int(task["remote_pid"]))
        if task.get("process_group"):
            groups.setdefault(node, set()).add(int(task["process_group"]))

    observed = []
    local_projects = {}
    for node in nodes:
        name = node.get("name")
        snapshot = node.get("process_snapshot") or {}
        sampled_at = float(snapshot.get("received_at") or snapshot.get("sampled_at") or now)
        by_group = {}
        for proc in snapshot.get("processes", []):
            pid, pgid = proc["pid"], proc["pgid"]
            known = tracked.get(name, set())
            if pid in known or known.intersection(proc.get("ancestors") or []):
                continue
            if pgid in groups.get(name, set()):
                continue
            if is_control_process(proc.get("cwd") or "", proc.get("cmdline") or ""):
                continue
            if name == "local" and not _local_compute_process(proc):
                continue
            by_group.setdefault((proc["owner"], pgid), []).append(proc)
        for (owner, pgid), procs in by_group.items():
            pids = [proc["pid"] for proc in procs]
            leader = min(procs, key=lambda proc: proc["pid"])
            cwd = leader.get("cwd") or ""
            command = leader.get("cmdline") or leader.get("comm") or ""
            try:
                args = shlex.split(command)
            except ValueError:
                args = command.split()
            script = next((arg for arg in args[1:] if arg.endswith(".py")), "")
            project = os.path.basename(cwd.rstrip("/")) or os.path.basename(script or leader.get("comm") or "external")
            if name == "local":
                if not any(proc.get("gpus") for proc in procs) and sum(proc["pcpu"] for proc in procs) < 50:
                    continue
                if cwd not in local_projects:
                    local_projects[cwd] = _local_project_name(cwd)
                project = local_projects[cwd]
                if not project:
                    continue
            gpu_indices = sorted({g["idx"] for proc in procs for g in proc.get("gpus", [])})
            vram = sum(g["used_mb"] for proc in procs for g in proc.get("gpus", []))
            ram = sum(proc.get("pss_kb", proc["rss_mb"] * 1024) for proc in procs) // 1024
            ram_measurement = "pss" if all("pss_kb" in proc for proc in procs) else "rss"
            elapsed = max(proc["elapsed_s"] for proc in procs)
            task = {
                "id": f"ext-{name}-{owner}-{pgid}", "status": "running", "node": name,
                "project": project, "description": f"{leader.get('comm', '')} ({len(pids)} procs)",
                "cmd": command, "cwd": cwd, "remote_pids": pids, "process_group": pgid,
                "origin": "external", "submitted_by": owner, "process_owner": owner,
                "observed_only": True, "started_at": sampled_at - elapsed,
                "gpu_idx": gpu_indices[0] if gpu_indices else None, "gpu_indices": gpu_indices,
                "current_vram_mb": vram, "current_ram_mb": ram,
                "current_ram_measurement": ram_measurement,
                "current_pcpu": sum(proc["pcpu"] for proc in procs),
            }
            if node.get("alive") is not True or now - sampled_at > 180:
                task["probe_unknown_since"] = min(sampled_at, now - 180)
            observed.append(task)
    return observed
