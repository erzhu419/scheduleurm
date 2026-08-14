from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SudoCpuBatchProbeDeps:
    node_configs: dict
    route_jump_cache: dict
    outer_route_cache: dict
    route_cache_ttl_s: float
    login_prefight_timeout_s: float
    now: Callable[[], float]
    outer_ssh_route_key: Callable[[str], str]
    configured_proxy_jumps: Callable[[dict], list]
    probe_ssh_proxy_jump: Callable[[str, str], bool]
    remote_bash_command_for_node: Callable[..., str]
    ssh_base_args_with_proxy: Callable[[str, str], list]
    run_ssh_subprocess: Callable[..., Any]


def cpu_only_sudo_probe_cmd() -> str:
    return (
        "awk '/^MemTotal:/{t=int($2/1024)} /^MemAvailable:/{a=int($2/1024)} "
        "END{print a; print t}' /proc/meminfo; echo '__SEP__'; "
        "nproc; echo '__SEP__'; "
        "awk '{print $1}' /proc/loadavg; echo '__SEP__'; "
        "u=$(id -un 2>/dev/null || whoami); "
        "ut=$(ps -eLo user= 2>/dev/null | awk -v u=\"$u\" '$1==u{c++} END{print c+0}'); "
        "ul=$(ulimit -u 2>/dev/null || echo 0); "
        "cg=$(cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0); "
        "cm=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo 0); "
        "printf '%s %s %s %s\\n' \"$ut\" \"$ul\" \"$cg\" \"$cm\""
    )


def _parse_probe_int(values: list[str], idx: int):
    if idx >= len(values):
        return None
    raw = str(values[idx]).strip()
    if not raw or raw.lower() in ("max", "unlimited"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_cpu_only_sudo_probe(name: str, body: str, *, node_configs: dict) -> dict:
    info = node_configs.get(name, {}) or {}
    parts = str(body or "").split("__SEP__")
    mem_lines = [int(x) for x in parts[0].strip().split() if x.isdigit()] if parts else []
    free_ram = mem_lines[0] if len(mem_lines) >= 1 else 0
    probed_total_ram = mem_lines[1] if len(mem_lines) >= 2 else 0
    cores = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 0
    try:
        loadavg = float(parts[2].strip()) if len(parts) > 2 else 0.0
    except ValueError:
        loadavg = 0.0
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
    result = {
        "name": name,
        "alive": True,
        "gpus": [],
        "free_ram_mb": sched_free_ram,
        "actual_free_ram_mb": free_ram,
        "total_ram_mb": total_ram,
        "free_cpu": max(0, total_cpu - int(round(loadavg))),
        "total_cpu": total_cpu,
        "loadavg": loadavg,
        "cores": cores,
        "probe_source": "sudo_batch",
    }
    thread_vals = parts[3].strip().split() if len(parts) > 3 else []
    user_threads = _parse_probe_int(thread_vals, 0)
    user_thread_limit = _parse_probe_int(thread_vals, 1)
    cgroup_pids_current = _parse_probe_int(thread_vals, 2)
    cgroup_pids_max = _parse_probe_int(thread_vals, 3)
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
    return result


def _eligible_sudo_cpu_names(names: list[str], deps: SudoCpuBatchProbeDeps) -> list[str]:
    return [
        name for name in names
        if name in deps.node_configs
        and (deps.node_configs.get(name, {}) or {}).get("sudo_ssh_host")
        and int((deps.node_configs.get(name, {}) or {}).get("max_vram_per_task") or 0) == 0
    ]


def _select_proxy_jump(names: list[str], deps: SudoCpuBatchProbeDeps) -> str:
    first_info = deps.node_configs.get(names[0], {}) or {}
    route_key = deps.outer_ssh_route_key(names[0])
    cached = deps.route_jump_cache.get(route_key) if route_key else None
    jumps = deps.configured_proxy_jumps(first_info)
    if cached:
        jump, ts = cached
        if jump in jumps and deps.now() - float(ts or 0) <= deps.route_cache_ttl_s:
            return jump
    for jump in jumps:
        for name in names:
            if deps.probe_ssh_proxy_jump(
                name,
                jump,
                timeout=max(3, deps.login_prefight_timeout_s + 2),
            ):
                if route_key:
                    deps.route_jump_cache[route_key] = (jump, deps.now())
                    deps.outer_route_cache[route_key] = (True, "", deps.now())
                return jump
    return ""


def probe_sudo_cpu_nodes_batch(names: list[str], *, deps: SudoCpuBatchProbeDeps) -> dict:
    names = _eligible_sudo_cpu_names(names, deps)
    if not names or "zhengliang-hpc" not in deps.node_configs:
        return {}
    selected_jump = _select_proxy_jump(names, deps)
    if not selected_jump:
        return {
            name: {"name": name, "alive": False, "error": "batch cpu probe found no working sudo-hop jump"}
            for name in names
        }

    probe_cmd = cpu_only_sudo_probe_cmd()
    pieces = []
    for name in names:
        remote = deps.remote_bash_command_for_node(name, probe_cmd, timeout=8)
        pieces.append(
            f"printf '\\n__SCHED_NODE_BEGIN__ {name}\\n'; "
            f"{remote}; rc=$?; "
            f"printf '\\n__SCHED_NODE_END__ {name} %s\\n' \"$rc\""
        )
    batch_cmd = "; ".join(pieces)
    remote_cmd = f"bash -lc {shlex.quote(batch_cmd)}"
    try:
        proc = deps.run_ssh_subprocess(
            deps.ssh_base_args_with_proxy("zhengliang-hpc", selected_jump) + [remote_cmd],
            capture_output=True,
            timeout=max(20, len(names) * 9),
            text=True,
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = exc.output or ""
        err = exc.stderr or f"batch probe timed out via {selected_jump}"
    if rc != 0 and not out:
        reason = (err or f"batch probe rc={rc}")[:160]
        return {name: {"name": name, "alive": False, "error": reason} for name in names}

    results = {}
    pattern = re.compile(
        r"__SCHED_NODE_BEGIN__\s+(\S+)\n(.*?)\n__SCHED_NODE_END__\s+\1\s+(\d+)",
        re.S,
    )
    seen = set()
    for match in pattern.finditer(out or ""):
        name, body, rc_s = match.group(1), match.group(2), match.group(3)
        if name not in names:
            continue
        seen.add(name)
        if int(rc_s) != 0:
            results[name] = {"name": name, "alive": False, "error": f"batch cpu probe rc={rc_s}"}
            continue
        try:
            results[name] = parse_cpu_only_sudo_probe(name, body, node_configs=deps.node_configs)
        except Exception as exc:
            results[name] = {
                "name": name,
                "alive": False,
                "error": f"batch cpu probe parse failed: {str(exc)[:100]}",
            }
    for name in names:
        if name not in seen:
            results[name] = {"name": name, "alive": False, "error": "batch cpu probe missing result"}
    return results
