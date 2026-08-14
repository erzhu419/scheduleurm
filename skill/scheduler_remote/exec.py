from __future__ import annotations

import base64
import os
import shlex
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from .subprocess import (
        cleanup_proxy_jump_children,
        run_ssh_subprocess,
        ssh_arg_value,
        ssh_target_from_args,
    )
except ModuleNotFoundError:
    from .subprocess import (
        cleanup_proxy_jump_children,
        run_ssh_subprocess,
        ssh_arg_value,
        ssh_target_from_args,
    )


@dataclass(frozen=True)
class RemoteExecDeps:
    node_configs: dict
    canonical_node_name: Callable[[str], str]
    node_is_windows: Callable[[str], bool]
    raise_if_watcher_shutdown: Callable[[], None]
    register_control_subprocess: Callable[[Any], Any]
    unregister_control_subprocess: Callable[[Any], Any]
    terminate_control_process_group: Callable[..., Any]
    ssh_route_cache_ttl_s: int
    ssh_login_preflight_timeout_s: int
    ssh_login_failure_cache_ttl_s: int
    ssh_disable_mux: bool
    ssh_proxy_jump_cache: dict
    ssh_route_jump_cache: dict
    ssh_outer_route_cache: dict
    probe_ssh_proxy_jump_callback: Optional[Callable[..., bool]] = None
    probe_outer_ssh_route_callback: Optional[Callable[..., tuple[bool, str]]] = None
    subprocess_popen: Callable[..., Any] = subprocess.Popen
    subprocess_run: Callable[..., Any] = subprocess.run
    subprocess_check_output: Callable[..., Any] = subprocess.check_output
    kill: Callable[[int, int], Any] = os.kill
    now: Callable[[], float] = time.time
    expanduser: Callable[[str], str] = os.path.expanduser
    pipe: Any = subprocess.PIPE
    devnull: Any = subprocess.DEVNULL
    sigkill: int = signal.SIGKILL
    sigterm: int = signal.SIGTERM
    route_probe_max_workers: int = 2


def route_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def configured_proxy_jumps(info: dict) -> list[str]:
    jumps = route_list(info.get("ssh_proxy_jumps") or info.get("proxy_jumps"))
    if jumps:
        return jumps
    return route_list(info.get("ssh_proxy_jump") or info.get("proxy_jump"))


def retired_node_reason(node: str, *, deps: RemoteExecDeps) -> str:
    node = deps.canonical_node_name(node)
    info = deps.node_configs.get(node, {}) or {}
    if info.get("retired"):
        return f"{node}: node is retired; remote execution is disabled"
    return ""


def outer_ssh_route_key(node: str, *, deps: RemoteExecDeps):
    node = deps.canonical_node_name(node)
    info = deps.node_configs.get(node, {}) or {}
    host = str(info.get("host") or "").strip()
    jumps = tuple(configured_proxy_jumps(info))
    if not host or deps.node_is_windows(node):
        return None
    identity = deps.expanduser(str(info.get("ssh_identity") or ""))
    options = tuple(
        str(x or "").strip()
        for x in (info.get("ssh_options") or [])
        if str(x or "").strip()
    )
    return (
        str(info.get("ssh_user") or ""),
        host,
        str(info.get("ssh_port") or ""),
        identity,
        jumps,
        options,
    )


def ssh_base_args_with_proxy(
    node: str,
    proxy_jump: Optional[str] = None,
    *,
    deps: RemoteExecDeps,
) -> list:
    node = deps.canonical_node_name(node)
    retired_reason = retired_node_reason(node, deps=deps)
    if retired_reason:
        raise RuntimeError(retired_reason)
    info = deps.node_configs[node]
    host = info["host"]
    disable_mux = bool(deps.ssh_disable_mux or proxy_jump or info.get("disable_control_master"))
    connect_timeout = max(1, int(deps.ssh_login_preflight_timeout_s or 5))
    args = [
        "ssh",
        "-o", f"ConnectTimeout={connect_timeout}",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        "-o", "BatchMode=yes",
    ]
    if disable_mux:
        args.extend([
            "-o", "ControlMaster=no",
            "-o", "ControlPath=none",
            "-o", "ControlPersist=no",
        ])
    if info.get("ssh_identity"):
        args.extend(["-i", deps.expanduser(str(info["ssh_identity"]))])
    if info.get("ssh_port"):
        args.extend(["-p", str(info["ssh_port"])])
    if proxy_jump:
        proxy_cmd = (
            f"ssh -o ConnectTimeout={connect_timeout} -o ServerAliveInterval=5 "
            "-o ServerAliveCountMax=3 -o BatchMode=yes "
            "-o ControlMaster=no -o ControlPath=none -o ControlPersist=no "
            f"-W %h:%p {shlex.quote(str(proxy_jump))}"
        )
        args.extend(["-o", f"ProxyCommand={proxy_cmd}"])
    for opt in (info.get("ssh_options") or []):
        opt = str(opt or "").strip()
        if opt:
            args.extend(["-o", opt])
    user = info.get("ssh_user")
    target = f"{user}@{host}" if user and "@" not in str(host) else str(host)
    args.append(target)
    return args


def remote_bash_command_for_node(
    node: str,
    shell_cmd: str,
    *,
    deps: RemoteExecDeps,
    timeout: Optional[int] = None,
) -> str:
    node = deps.canonical_node_name(node)
    info = deps.node_configs.get(node, {}) or {}
    inner_host = str(info.get("sudo_ssh_host") or "").strip()
    if not inner_host:
        return f"bash -lc {shlex.quote(shell_cmd)}"
    inner = f"bash -c {shlex.quote(shell_cmd)}"
    run_as = str(info.get("sudo_ssh_run_as") or "").strip()
    if run_as and run_as != "root":
        inner = f"su {shlex.quote(run_as)} -s /bin/bash -c {shlex.quote(inner)}"
    ssh_bin = str(info.get("sudo_ssh_bin") or "/usr/bin/ssh")
    inner_ssh_args = [ssh_bin, inner_host, inner]
    sudo_cmd = "sudo -n " + " ".join(shlex.quote(arg) for arg in inner_ssh_args) + " </dev/null"
    try:
        remote_timeout = max(1, int(timeout) - 1) if timeout else 0
    except Exception:
        remote_timeout = 0
    wrapped = f"timeout -k 2s {remote_timeout}s {sudo_cmd}" if remote_timeout else sudo_cmd
    return f"bash -c {shlex.quote(wrapped)}"


def probe_ssh_proxy_jump(
    node: str,
    proxy_jump: str,
    *,
    deps: RemoteExecDeps,
    timeout: int = 8,
) -> bool:
    node = deps.canonical_node_name(node)
    if retired_node_reason(node, deps=deps):
        return False
    try:
        remote_cmd = remote_bash_command_for_node(node, "true", deps=deps, timeout=timeout)
        proc = run_ssh_subprocess(
            ssh_base_args_with_proxy(node, proxy_jump, deps=deps) + [remote_cmd],
            stdout=deps.devnull,
            stderr=deps.devnull,
            timeout=timeout,
            deps=deps,
        )
        return proc.returncode == 0
    except Exception:
        return False


def probe_outer_ssh_route(
    node: str,
    *,
    deps: RemoteExecDeps,
    timeout: Optional[int] = None,
) -> tuple[bool, str]:
    node = deps.canonical_node_name(node)
    retired_reason = retired_node_reason(node, deps=deps)
    if retired_reason:
        return False, retired_reason
    key = outer_ssh_route_key(node, deps=deps)
    if key:
        cached = deps.ssh_outer_route_cache.get(key)
        if cached:
            ok, reason, ts = cached
            ttl = deps.ssh_route_cache_ttl_s if ok else deps.ssh_login_failure_cache_ttl_s
            if ttl and deps.now() - float(ts or 0) <= ttl:
                return bool(ok), str(reason or "")
            deps.ssh_outer_route_cache.pop(key, None)
    info = deps.node_configs.get(node, {}) or {}
    jumps = configured_proxy_jumps(info)
    timeout_s = int(timeout or deps.ssh_login_preflight_timeout_s)
    errors = []
    routes: list[Optional[str]] = list(jumps) or [None]
    for jump in routes:
        route_label = jump or "direct"
        try:
            proc = run_ssh_subprocess(
                ssh_base_args_with_proxy(node, jump, deps=deps) + ["true"],
                stdout=deps.devnull,
                stderr=deps.pipe,
                timeout=timeout_s,
                text=True,
                deps=deps,
            )
            if proc.returncode == 0:
                if jump is not None:
                    deps.ssh_proxy_jump_cache[node] = (jump, deps.now())
                if key:
                    if jump is not None:
                        deps.ssh_route_jump_cache[key] = (jump, deps.now())
                    deps.ssh_outer_route_cache[key] = (True, "", deps.now())
                return True, ""
            err = (proc.stderr or "").strip() or f"rc={proc.returncode}"
            errors.append(f"{route_label}: {err[:120]}")
        except subprocess.TimeoutExpired:
            errors.append(f"{route_label}: outer ssh route timed out after {timeout_s}s")
        except Exception as exc:
            errors.append(f"{route_label}: {str(exc)[:120]}")
    reason = "; ".join(errors)[:240] or "outer ssh route failed"
    hard_failure = any("timed out" in str(err).lower() for err in errors)
    if not hard_failure:
        return True, ""
    if key and deps.ssh_login_failure_cache_ttl_s:
        deps.ssh_outer_route_cache[key] = (False, reason, deps.now())
    return False, reason


def cached_outer_route_failure(node: str, *, deps: RemoteExecDeps) -> str:
    key = outer_ssh_route_key(node, deps=deps)
    if not key or not deps.ssh_login_failure_cache_ttl_s:
        return ""
    cached = deps.ssh_outer_route_cache.get(key)
    if not cached:
        return ""
    ok, reason, ts = cached
    if ok:
        return ""
    if deps.now() - float(ts or 0) <= deps.ssh_login_failure_cache_ttl_s:
        return str(reason or "outer ssh route failed")
    deps.ssh_outer_route_cache.pop(key, None)
    return ""


def probe_all_route_failures(*, deps: RemoteExecDeps) -> dict:
    reps = {}
    for name, info in deps.node_configs.items():
        if (info or {}).get("retired") or (info or {}).get("monitor_only"):
            continue
        key = outer_ssh_route_key(name, deps=deps)
        if not key:
            continue
        cur = reps.get(key)
        if cur is None:
            reps[key] = name
    def probe_route(item):
        key, name = item
        cached = deps.ssh_outer_route_cache.get(key)
        if cached:
            ok, _reason, ts = cached
            if ok and deps.now() - float(ts or 0) <= deps.ssh_route_cache_ttl_s:
                return key, ""
        info = deps.node_configs.get(name, {}) or {}
        if info.get("sudo_ssh_host"):
            route_ok = False
            for jump in configured_proxy_jumps(info):
                timeout = max(3, deps.ssh_login_preflight_timeout_s + 2)
                if deps.probe_ssh_proxy_jump_callback:
                    jump_ok = deps.probe_ssh_proxy_jump_callback(name, jump, timeout=timeout)
                else:
                    jump_ok = probe_ssh_proxy_jump(name, jump, deps=deps, timeout=timeout)
                if jump_ok:
                    deps.ssh_proxy_jump_cache[name] = (jump, deps.now())
                    deps.ssh_route_jump_cache[key] = (jump, deps.now())
                    deps.ssh_outer_route_cache[key] = (True, "", deps.now())
                    route_ok = True
                    break
            if route_ok:
                return key, ""
        if deps.probe_outer_ssh_route_callback:
            ok, reason = deps.probe_outer_ssh_route_callback(name)
        else:
            ok, reason = probe_outer_ssh_route(name, deps=deps)
        if not ok:
            return key, reason
        return key, ""

    items = list(reps.items())
    workers = max(1, min(len(items), int(deps.route_probe_max_workers or 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(probe_route, items))
    failures = {key: reason for key, reason in results if reason}
    return failures


def select_proxy_jump(node: str, *, deps: RemoteExecDeps) -> Optional[str]:
    node = deps.canonical_node_name(node)
    info = deps.node_configs.get(node, {}) or {}
    jumps = configured_proxy_jumps(info)
    if not jumps:
        return None
    if len(jumps) == 1:
        return jumps[0]
    now = deps.now()
    cached = deps.ssh_proxy_jump_cache.get(node)
    if cached:
        jump, ts = cached
        if jump in jumps and now - float(ts or 0) <= deps.ssh_route_cache_ttl_s:
            return jump
    route_key = outer_ssh_route_key(node, deps=deps)
    route_cached = deps.ssh_route_jump_cache.get(route_key) if route_key else None
    if route_cached:
        jump, ts = route_cached
        if jump in jumps and now - float(ts or 0) <= deps.ssh_route_cache_ttl_s:
            deps.ssh_proxy_jump_cache[node] = (jump, now)
            return jump
    for jump in jumps:
        if probe_ssh_proxy_jump(node, jump, deps=deps):
            deps.ssh_proxy_jump_cache[node] = (jump, now)
            if route_key:
                deps.ssh_route_jump_cache[route_key] = (jump, now)
            return jump
    deps.ssh_proxy_jump_cache.pop(node, None)
    if route_key:
        deps.ssh_route_jump_cache.pop(route_key, None)
    return jumps[0]


def ssh_base_arg_variants(node: str, *, deps: RemoteExecDeps) -> list:
    node = deps.canonical_node_name(node)
    info = deps.node_configs.get(node, {}) or {}
    jumps = configured_proxy_jumps(info)
    if not jumps:
        return [ssh_base_args_with_proxy(node, None, deps=deps)]
    selected = select_proxy_jump(node, deps=deps)
    ordered = []
    for jump in [selected] + jumps:
        if jump and jump not in ordered:
            ordered.append(jump)
    return [ssh_base_args_with_proxy(node, jump, deps=deps) for jump in ordered]


def ssh_base_args(node: str, *, deps: RemoteExecDeps) -> list:
    return ssh_base_arg_variants(deps.canonical_node_name(node), deps=deps)[0]


def ssh_no_stdin_args(node: str, *, deps: RemoteExecDeps) -> list:
    args = ssh_base_args(node, deps=deps)
    return args[:1] + ["-n"] + args[1:]


def ssh_no_stdin_arg_variants(node: str, *, deps: RemoteExecDeps) -> list:
    variants = []
    for args in ssh_base_arg_variants(node, deps=deps):
        variants.append(args[:1] + ["-n"] + args[1:])
    return variants


def run_windows_ps(
    node: str,
    ps_script: str,
    *,
    deps: RemoteExecDeps,
    timeout=15,
    check=True,
    input_data: bytes | None = None,
):
    retired_reason = retired_node_reason(node, deps=deps)
    if retired_reason:
        if check:
            raise RuntimeError(retired_reason)
        return 255, "", retired_reason
    encoded = base64.b64encode((ps_script or "").encode("utf-16le")).decode("ascii")
    base_cmd = ssh_base_args(node, deps=deps) + [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
    ]
    if len(encoded) < 7000:
        proc = run_ssh_subprocess(
            base_cmd + ["-EncodedCommand", encoded],
            input=input_data,
            capture_output=True,
            timeout=timeout,
            deps=deps,
        )
    else:
        if input_data is not None:
            raise RuntimeError("cannot pass stdin to oversized Windows PowerShell script")
        runner = r'''
$p = Join-Path $env:TEMP ("scheduleurm_" + [guid]::NewGuid().ToString() + ".ps1")
$enc = New-Object System.Text.UTF8Encoding $false
[IO.File]::WriteAllText($p, [Console]::In.ReadToEnd(), $enc)
& powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $p
$ec = $LASTEXITCODE
Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue
if ($null -eq $ec) { $ec = 0 }
exit ([int]$ec)
'''
        runner_encoded = base64.b64encode(runner.encode("utf-16le")).decode("ascii")
        proc = run_ssh_subprocess(
            base_cmd + ["-EncodedCommand", runner_encoded],
            input=(ps_script or "").encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            deps=deps,
        )
    stdout_raw = proc.stdout or b""
    stderr_raw = proc.stderr or b""
    stdout = stdout_raw.decode("utf-8", "replace") if isinstance(stdout_raw, bytes) else str(stdout_raw)
    stderr = stderr_raw.decode("utf-8", "replace") if isinstance(stderr_raw, bytes) else str(stderr_raw)
    if check and proc.returncode != 0:
        raise RuntimeError(f"[{node}] powershell failed (rc={proc.returncode}): {stderr.strip()[:300]}")
    return proc.returncode, stdout, stderr


def run_on(
    node,
    shell_cmd,
    *,
    deps: RemoteExecDeps,
    timeout=15,
    check=True,
):
    node = deps.canonical_node_name(node)
    retired_reason = retired_node_reason(node, deps=deps)
    if retired_reason:
        if check:
            raise RuntimeError(retired_reason)
        return 255, "", retired_reason
    if deps.node_is_windows(node):
        raise RuntimeError(f"run_on() bash path called for Windows node {node}; use _run_windows_ps")
    host = deps.node_configs[node]["host"]
    if host is None:
        proc = deps.subprocess_run(
            ["bash", "-lc", shell_cmd],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
    else:
        cached_failure = cached_outer_route_failure(node, deps=deps)
        if cached_failure:
            err = f"ssh login route down: {cached_failure}"
            if check:
                raise RuntimeError(f"[{node}] {err}")
            return 255, "", err
        remote_cmd = remote_bash_command_for_node(node, shell_cmd, deps=deps, timeout=timeout)
        proc = None
        variants = ssh_no_stdin_arg_variants(node, deps=deps)
        for idx, ssh_args in enumerate(variants):
            try:
                proc = run_ssh_subprocess(
                    ssh_args + [remote_cmd],
                    capture_output=True,
                    timeout=timeout,
                    text=True,
                    deps=deps,
                )
            except subprocess.TimeoutExpired:
                deps.ssh_proxy_jump_cache.pop(node, None)
                if idx + 1 < len(variants):
                    continue
                raise
            if proc.returncode == 255 and idx + 1 < len(variants):
                deps.ssh_proxy_jump_cache.pop(node, None)
                continue
            break
    if check and proc.returncode != 0:
        raise RuntimeError(f"[{node}] cmd failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")
    return proc.returncode, proc.stdout, proc.stderr


def ssh_rsync_shell_for_node(node: str, *, deps: RemoteExecDeps) -> str:
    return " ".join(shlex.quote(a) for a in ssh_base_args(node, deps=deps)[:-1])


def ssh_target_for_node(node: str, *, deps: RemoteExecDeps) -> str:
    return ssh_base_args(node, deps=deps)[-1]
