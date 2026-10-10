#!/usr/bin/env python3
"""Shared transport entrypoint for project SSH/rsync commands and rsync -e."""

import argparse
import ctypes
import signal
import shlex
import subprocess
import sys
from pathlib import Path

from scheduler_transport.budget import _ssh_parts, bulk_transport, transport_child_env, transport_guard


def prepare_command(command):
    command = list(command)
    program = Path(command[0]).name
    if program == "ssh" and not any(str(arg).startswith("ControlMaster=") for arg in command):
        target, jumps = _ssh_parts(command)
        via = "-".join(jumps) or "direct"
        control_path = (f"~/.ssh/scheduleurm-%r@%h-%p-{via}" if jumps else
                        f"~/.ssh/scheduleurm-jump-{target.rsplit('@', 1)[-1]}")
        if len(jumps) == 1 and "-J" in command:
            index = command.index("-J")
            proxy = shlex.join(["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
                                "-o", "ControlPersist=60", "-o",
                                f"ControlPath=~/.ssh/scheduleurm-jump-{jumps[0]}",
                                "-W", "%h:%p", jumps[0]])
            command[index:index + 2] = ["-o", f"ProxyCommand={proxy}"]
        command[1:1] = ["-o", "ControlMaster=auto", "-o", "ControlPersist=60",
                        "-o", f"ControlPath={control_path}"]
    elif program == "rsync":
        shell_index = next((i + 1 for i, arg in enumerate(command)
                            if arg in {"-e", "--rsh"}), None)
        shell = command[shell_index] if shell_index is not None else "ssh"
        shell_args = shlex.split(shell)
        if len(shell_args) > 1 and Path(shell_args[1]).name == "transport_exec.py":
            return command
        runner = shlex.join([sys.executable, str(Path(__file__).resolve()), "--bulk", "--"])
        wrapped = f"{runner} {shell}"
        if shell_index is None:
            command[1:1] = ["-e", wrapped]
        else:
            command[shell_index] = wrapped
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a transport command is required")
    command = prepare_command(command)
    # rsync callers can time out without terminating their rsh child.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    from contextlib import nullcontext
    with bulk_transport() if args.bulk else nullcontext():
        with transport_guard(command, root=Path.home() / ".claude/scheduler/transport_coord",
                             wait_s=float("inf")):
            proc = subprocess.Popen(command, env=transport_child_env())
            signal.signal(signal.SIGTERM, lambda *_: proc.terminate())
            try:
                rc = proc.wait()
                return rc if rc >= 0 else 128 - rc
            except BaseException:
                proc.kill()
                proc.wait()
                raise


if __name__ == "__main__":
    raise SystemExit(main())
