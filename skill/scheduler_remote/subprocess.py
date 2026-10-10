from __future__ import annotations

import subprocess
from typing import Any, Optional

try:
    from scheduler_transport.budget import transport_child_env
except ModuleNotFoundError:
    from ..scheduler_transport.budget import transport_child_env


_KILL_DRAIN_TIMEOUT_S = 2.0


def _drain_killed_process(proc: Any) -> tuple[Any, Any]:
    """Reap a killed transport without allowing pipe cleanup to block forever."""
    try:
        return proc.communicate(timeout=_KILL_DRAIN_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(proc, name, None)
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=_KILL_DRAIN_TIMEOUT_S)
        except Exception:
            pass
        return exc.output, exc.stderr
    except Exception:
        return None, None


def ssh_arg_value(args: list, opt: str) -> Optional[str]:
    for idx, arg in enumerate(args):
        if arg == opt and idx + 1 < len(args):
            return str(args[idx + 1])
    return None


def ssh_target_from_args(args: list) -> tuple[str, str]:
    port = ssh_arg_value(args, "-p") or "22"
    opts_with_values = {
        "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L",
        "-l", "-m", "-O", "-o", "-p", "-Q", "-R", "-S", "-W", "-w",
    }
    skip = False
    for arg in list(args)[1:]:
        text = str(arg)
        if skip:
            skip = False
            continue
        if text in opts_with_values:
            skip = True
            continue
        if text.startswith("-"):
            continue
        host = text.rsplit("@", 1)[-1].strip("[]")
        return host, str(port)
    return "", str(port)


def cleanup_proxy_jump_children(args: list, *, deps: Any) -> None:
    jump = ssh_arg_value(args, "-J")
    if not jump:
        return
    host, port = ssh_target_from_args(args)
    if not host:
        return
    jumps = [j.strip() for j in str(jump).split(",") if j.strip()]
    prefixes = tuple(f"ssh -W [{host}]:{port} {j}" for j in jumps) + tuple(
        f"ssh -W {host}:{port} {j}" for j in jumps
    )
    try:
        out = deps.subprocess_check_output(["ps", "-eo", "pid=,args="], text=True)
    except Exception:
        return
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, cmdline = parts
        if not cmdline.startswith(prefixes):
            continue
        try:
            deps.kill(int(pid_s), deps.sigterm)
        except Exception:
            pass


def _run_admitted_subprocess(
    args: list,
    *,
    timeout,
    deps: Any,
    input=None,
    capture_output=False,
    stdout=None,
    stderr=None,
    text=None,
    **popen_kwargs,
):
    deps.raise_if_watcher_shutdown()
    if capture_output:
        stdout = deps.pipe
        stderr = deps.pipe
    stdin = popen_kwargs.pop("stdin", deps.pipe if input is not None else None)
    child_env = transport_child_env(popen_kwargs.get("env"))
    if child_env is not None:
        popen_kwargs["env"] = child_env
    proc = deps.subprocess_popen(
        args,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        text=text,
        start_new_session=True,
        **popen_kwargs,
    )
    pgid = deps.register_control_subprocess(proc)
    try:
        try:
            out, err = proc.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            deps.terminate_control_process_group(pgid, sig=deps.sigkill)
            try:
                proc.kill()
            except Exception:
                pass
            out, err = _drain_killed_process(proc)
            cleanup_proxy_jump_children(args, deps=deps)
            raise subprocess.TimeoutExpired(args, timeout, output=out, stderr=err)
        except BaseException:
            deps.terminate_control_process_group(pgid, sig=deps.sigkill)
            try:
                proc.kill()
            except Exception:
                pass
            _drain_killed_process(proc)
            cleanup_proxy_jump_children(args, deps=deps)
            raise
    finally:
        deps.unregister_control_subprocess(pgid)
    cleanup_proxy_jump_children(args, deps=deps)
    deps.raise_if_watcher_shutdown()
    return subprocess.CompletedProcess(args, proc.returncode, out, err)


def run_ssh_subprocess(args: list, *, timeout, deps: Any, **kwargs):
    with deps.transport_guard(args):
        return _run_admitted_subprocess(args, timeout=timeout, deps=deps, **kwargs)
