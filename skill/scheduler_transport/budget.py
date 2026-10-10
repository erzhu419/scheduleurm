"""Shared transport admission for SSH commands and file transfers."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


TOTAL_SLOTS = 8
CONTROL_SLOTS = 2
BULK_SLOTS = 2
_CONTROL = ContextVar("transport_control", default=False)
_BULK = ContextVar("transport_bulk", default=False)
_STATE_LOCK_DEPTH = ContextVar("transport_state_lock_depth", default=0)
_ADMITTED = ContextVar("transport_admitted", default=False)


def transport_child_env(env=None):
    if not _ADMITTED.get():
        return env
    return {**(os.environ if env is None else env), "SCHEDULEURM_TRANSPORT_ADMITTED": "1"}


class TransportBusy(RuntimeError):
    def __init__(self, routes):
        super().__init__(f"TRANSPORT_WAIT: waiting for transport capacity on {', '.join(routes)}")


def transport_waiting(message) -> bool:
    return "TRANSPORT_WAIT:" in str(message)


@contextmanager
def control_transport():
    token = _CONTROL.set(True)
    try:
        yield
    finally:
        _CONTROL.reset(token)


@contextmanager
def bulk_transport():
    token = _BULK.set(True)
    try:
        yield
    finally:
        _BULK.reset(token)


@contextmanager
def transport_state_lock():
    token = _STATE_LOCK_DEPTH.set(_STATE_LOCK_DEPTH.get() + 1)
    try:
        yield
    finally:
        _STATE_LOCK_DEPTH.reset(token)


def _route_name(host: str) -> str:
    host = host.rsplit("@", 1)[-1].strip("[]")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", host)


def _ssh_parts(args: list[str]) -> tuple[str, list[str]]:
    if len(args) > 1 and Path(str(args[1])).name == "transport_exec.py":
        args = args[args.index("--") + 1:]
    target = ""
    jumps = []
    i = 1
    while i < len(args):
        arg = str(args[i])
        if arg in {"-o", "-J", "-p", "-P", "-i", "-F", "-l", "-W", "-S"}:
            value = str(args[i + 1])
            if arg == "-J":
                jumps.extend(value.split(","))
            elif arg == "-o" and value.startswith("ProxyCommand="):
                _target, nested = _ssh_parts(shlex.split(value.split("=", 1)[1]))
                jumps.extend([_target, *nested])
            elif arg == "-o" and value.startswith("ProxyJump="):
                jumps.extend(value.split("=", 1)[1].split(","))
            i += 2
        elif arg.startswith("-"):
            i += 1
        else:
            target = arg
            break
    return target, jumps


def transport_routes(args: list[str]) -> list[str]:
    """Extract the destination and actual jump from our SSH/scp/rsync commands."""
    if not args:
        return []
    program = Path(str(args[0])).name
    if program == "ssh":
        target, jumps = _ssh_parts(args)
        routes = [target, *jumps]
    elif program in {"rsync", "scp"}:
        shell = ["ssh"]
        operands = []
        i = 1
        while i < len(args):
            arg = str(args[i])
            if arg in {"-e", "--rsh"}:
                shell = shlex.split(str(args[i + 1]))
                i += 2
            elif arg in {"-o", "-i", "-P"}:
                shell.extend([arg, str(args[i + 1])])
                i += 2
            elif not arg.startswith("-"):
                operands.append(arg)
                i += 1
            else:
                i += 1
        routes = []
        for operand in operands:
            if ":" in operand and not operand.startswith(("/", "./")):
                host = operand.split(":", 1)[0]
                if len(host) > 1:
                    target, jumps = _ssh_parts(shell + [host])
                    routes.extend([target, *jumps])
    else:
        return []
    return sorted({_route_name(route) for route in routes if route})


def _open_slot(root: Path, route: str, lane: str, indices):
    for index in indices:
        path = root / f"{route}.{lane}.{index}.lock"
        handle = path.open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        return handle
    return None


def _release(handles):
    for handle in reversed(handles):
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def transport_load(root: Path, route: str) -> int:
    route = _route_name(route)
    if not root.exists():
        return 0
    busy = 0
    for index in range(TOTAL_SLOTS):
        path = root / f"{route}.total.{index}.lock"
        if not path.exists():
            continue
        with path.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy += 1
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
    return busy


@contextmanager
def transport_guard(args, *, root: Path, check_shutdown=lambda: None, wait_s=60):
    routes = transport_routes(args)
    if not routes or os.environ.get("SCHEDULEURM_TRANSPORT_ADMITTED") == "1":
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    bulk = _BULK.get() or Path(str(args[0])).name in {"rsync", "scp"}
    indices = list(range(TOTAL_SLOTS - CONTROL_SLOTS))
    if _CONTROL.get() and not bulk:
        indices = list(range(TOTAL_SLOTS - CONTROL_SLOTS, TOTAL_SLOTS)) + indices
    started = time.monotonic()
    handles = []
    while True:
        check_shutdown()
        for route in routes:
            if bulk:
                handle = _open_slot(root, route, "bulk", range(BULK_SLOTS))
                if handle is None:
                    break
                handles.append(handle)
            handle = _open_slot(root, route, "total", indices)
            if handle is None:
                break
            handles.append(handle)
        else:
            break
        _release(handles)
        handles = []
        if _STATE_LOCK_DEPTH.get() or time.monotonic() - started >= wait_s:
            raise TransportBusy(routes)
        time.sleep(0.1)
    waited = time.monotonic() - started
    admitted = _ADMITTED.set(True)
    try:
        yield
    finally:
        _ADMITTED.reset(admitted)
        _release(handles)
        if waited >= 1:
            try:
                with (root / "waits.jsonl").open("a") as output:
                    output.write(json.dumps({"ts": time.time(), "pid": os.getpid(),
                                             "routes": routes, "bulk": bulk,
                                             "wait_s": round(waited, 3)}) + "\n")
            except OSError:
                pass
