"""Single-node resource probe helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ProbeNodeDeps:
    node_configs: dict
    canonical_node_name: Callable[[str], str]
    node_is_windows: Callable[[str], bool]
    probe_windows_node: Callable[[str], dict]
    path_exists: Callable[[str], bool]
    run_on: Callable[..., tuple]
    probe_windows_host_extras: Callable[[], dict | None]
    sleep: Callable[[float], Any]


def _nvidia_smi_path(name: str, info: dict, deps: ProbeNodeDeps) -> str:
    nvsmi = str(info.get("nvidia_smi_path") or info.get("nvidia_smi") or "nvidia-smi")
    if name == "local":
        win_nvsmi = "/mnt/c/WINDOWS/system32/nvidia-smi.exe"
        if deps.path_exists(win_nvsmi):
            nvsmi = win_nvsmi
    return nvsmi


def _probe_command(nvsmi: str) -> str:
    return (
        f"{nvsmi} --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null; echo '===SEP==='; "
        "awk '/^MemTotal:/{t=int($2/1024)} /^MemAvailable:/{a=int($2/1024)} END{print a; print t}' /proc/meminfo; echo '===SEP==='; "
        "nproc; echo '===SEP==='; "
        "awk '{print $1}' /proc/loadavg; echo '===SEP==='; "
        f"for i in 1 2; do sleep 0.1; "
        f"{nvsmi} --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; "
        "echo '---SAMPLE---'; done; echo '===SEP==='; "
        "u=$(id -un 2>/dev/null || whoami); "
        "ut=$(ps -eLo user= 2>/dev/null | awk -v u=\"$u\" '$1==u{c++} END{print c+0}'); "
        "ul=$(ulimit -u 2>/dev/null || echo 0); "
        "cg=$(cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0); "
        "cm=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo 0); "
        "printf '%s %s %s %s\\n' \"$ut\" \"$ul\" \"$cg\" \"$cm\""
    )


def _parse_probe_int(values: list[str], idx: int) -> int | None:
    if idx >= len(values):
        return None
    raw = str(values[idx]).strip()
    if not raw or raw.lower() in ("max", "unlimited"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_gpus(primary: str, samples: str) -> list[dict]:
    gpus = []
    for line in primary.strip().splitlines():
        bits = [x.strip() for x in line.split(",")]
        if len(bits) < 5:
            continue
        gpus.append({
            "idx": int(bits[0]),
            "used_mb": int(bits[1]),
            "total_mb": int(bits[2]),
            "free_mb": int(bits[3]),
            "util_pct": int(bits[4]),
        })
    if samples:
        extra = {}
        for chunk in samples.split("---SAMPLE---"):
            for line in chunk.strip().splitlines():
                bits = [x.strip() for x in line.split(",")]
                if len(bits) < 2:
                    continue
                try:
                    extra.setdefault(int(bits[0]), []).append(int(bits[1]))
                except ValueError:
                    continue
        for gpu in gpus:
            util_samples = [gpu["util_pct"]] + extra.get(gpu["idx"], [])
            if len(util_samples) > 1:
                gpu["util_pct"] = sum(util_samples) // len(util_samples)
    return gpus


def _merge_local_windows_extras(result: dict, total_cpu: int, loadavg: float,
                                deps: ProbeNodeDeps) -> None:
    extras = deps.probe_windows_host_extras()
    if not extras:
        return
    host_free = extras.get("host_free_ram_mb")
    if host_free is not None:
        try:
            host_free = max(0, int(host_free))
            result["wsl_free_ram_mb"] = result.get("free_ram_mb")
            result["wsl_actual_free_ram_mb"] = result.get("actual_free_ram_mb")
            result["wsl_total_ram_mb"] = result.get("total_ram_mb")
            result["host_free_ram_mb"] = host_free
            result["free_ram_mb"] = min(int(result.get("free_ram_mb") or 0), host_free)
        except Exception:
            pass
    result["host_total_ram_mb"] = extras.get("host_total_ram_mb")
    host_cpu = extras.get("host_cpu_load_pct")
    if host_cpu is not None:
        host_cpu = max(0, min(100, int(host_cpu)))
        host_used = int(round(total_cpu * host_cpu / 100.0))
        result["host_cpu_load_pct"] = host_cpu
        result["host_cpu_used_cores"] = host_used
        result["wsl_free_cpu"] = result["free_cpu"]
        result["wsl_loadavg"] = loadavg
        result["free_cpu"] = min(result["free_cpu"], max(0, total_cpu - host_used))
    compute_util = extras.get("gpu_compute_util_pct")
    if compute_util is not None:
        for gpu in result["gpus"]:
            gpu["util_pct_compute"] = compute_util


def parse_probe_output(name: str, info: dict, out: str, *, deps: ProbeNodeDeps) -> dict:
    parts = out.split("===SEP===")
    gpus = _parse_gpus(parts[0] if parts else "", parts[4] if len(parts) > 4 else "")
    mem_lines = (
        [int(x) for x in parts[1].strip().split() if x.isdigit()]
        if len(parts) > 1 else []
    )
    free_ram = mem_lines[0] if len(mem_lines) >= 1 else 0
    probed_total_ram = mem_lines[1] if len(mem_lines) >= 2 else 0
    cores = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else 0
    try:
        loadavg = float(parts[3].strip()) if len(parts) > 3 else 0.0
    except ValueError:
        loadavg = 0.0
    thread_vals = parts[5].strip().split() if len(parts) > 5 else []
    user_threads = _parse_probe_int(thread_vals, 0)
    user_thread_limit = _parse_probe_int(thread_vals, 1)
    cgroup_pids_current = _parse_probe_int(thread_vals, 2)
    cgroup_pids_max = _parse_probe_int(thread_vals, 3)

    declared = info.get("ram_mb", 0)
    if declared and probed_total_ram:
        total_ram = min(declared, probed_total_ram)
    elif probed_total_ram:
        total_ram = probed_total_ram
    elif declared:
        total_ram = declared
    else:
        total_ram = free_ram + 1
    sched_free_ram = min(free_ram, total_ram) if total_ram else free_ram
    total_cpu = info.get("cpu_cores", cores) or cores or 1
    free_cpu = max(0, total_cpu - int(round(loadavg)))
    result = {
        "name": name,
        "alive": True,
        "gpus": gpus,
        "free_ram_mb": sched_free_ram,
        "actual_free_ram_mb": free_ram,
        "total_ram_mb": total_ram,
        "free_cpu": free_cpu,
        "total_cpu": total_cpu,
        "loadavg": loadavg,
        "cores": cores,
    }
    if user_threads is not None:
        result["user_threads"] = user_threads
    if user_thread_limit is not None:
        result["user_thread_limit"] = user_thread_limit
        if user_threads is not None:
            result["free_user_threads"] = max(0, user_thread_limit - user_threads)
    if cgroup_pids_current is not None:
        result["cgroup_pids_current"] = cgroup_pids_current
    if cgroup_pids_max is not None:
        result["cgroup_pids_max"] = cgroup_pids_max
        if cgroup_pids_current is not None:
            result["free_cgroup_pids"] = max(0, cgroup_pids_max - cgroup_pids_current)
    if name == "local":
        _merge_local_windows_extras(result, total_cpu, loadavg, deps)
    return result


def probe_node(name: str, *, deps: ProbeNodeDeps) -> dict:
    name = deps.canonical_node_name(name)
    if deps.node_is_windows(name):
        return deps.probe_windows_node(name)
    info = deps.node_configs.get(name, {})
    cmd = _probe_command(_nvidia_smi_path(name, info, deps))
    probe_attempts = int(info.get("probe_attempts") or (4 if info.get("sudo_ssh_host") else 1))
    last_error = ""
    out = ""
    for attempt in range(max(1, probe_attempts)):
        try:
            rc, out, err = deps.run_on(name, cmd, timeout=15, check=False)
            if rc == 0:
                break
            last_error = (err or f"rc={rc}")[:160]
        except Exception as exc:
            rc = 1
            last_error = str(exc)[:160]
        if attempt + 1 < probe_attempts:
            deps.sleep(0.4 * (attempt + 1))
    else:
        return {"name": name, "alive": False, "error": f"ssh/cmd failed: {last_error}"}
    return parse_probe_output(name, info, out, deps=deps)
