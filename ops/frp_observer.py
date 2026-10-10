#!/usr/bin/env python3
"""Independent fresh-handshake observations, delivered through existing SSH routes."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from frp_health import probe, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill"))
from scheduler_transport.budget import control_transport, transport_guard
from scheduler_remote.subprocess import run_ssh_subprocess


PUBLISH_CODE = """import json, os, sys
from pathlib import Path
os.umask(0o077)
directory = Path.home() / '.cache/scheduleurm-frp'
directory.mkdir(parents=True, exist_ok=True)
temporary = directory / 'external.tmp'
temporary.write_text(json.dumps(json.load(sys.stdin)))
temporary.replace(directory / 'external.json')
"""


def ssh_settings(node):
    result = subprocess.run(["ssh", "-G", node], check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, timeout=5)
    return dict(line.split(None, 1) for line in result.stdout.splitlines() if " " in line)


def ssh_args(node):
    # Reuse an existing master, but do not leave a new monitoring connection
    # alive: that would itself keep the host's active-session veto set.
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=1", "-o", "ControlMaster=no",
            "-o", "ControlPersist=no", "-o", "ControlPath=~/.ssh/scheduleurm-jump-" + node]


def relay_args(target, peer, settings):
    node = target["node"]
    hostkey = settings.get("hostkeyalias", "none")
    if hostkey == "none":
        hostkey = settings["hostname"]
        if settings["port"] != "22":
            hostkey = "[{}]:{}".format(hostkey, settings["port"])
    proxy = " ".join(shlex.quote(arg) for arg in ssh_args(peer["node"]) +
                     ["-W", "%h:%p", peer["node"]])
    return ssh_args(node)[:-2] + [
        "-o", "ControlPath=none", "-o", "Hostname=" + target["internal_host"],
        "-o", "HostKeyAlias=" + hostkey, "-o", "ProxyCommand=" + proxy,
        "-p", "22", node]


def publish(target, peer, settings, report, root):
    remote = "/usr/bin/python3 -c " + shlex.quote(PUBLISH_CODE)
    routes = [("direct", ssh_args(target["node"]) + [target["node"]])]
    if peer:
        routes.append(("via_" + peer["node"], relay_args(target, peer, settings)))
    failures = []
    for route, args in routes:
        args = args + [remote]
        try:
            deps = SimpleNamespace(
                transport_guard=lambda command: transport_guard(command, root=root, wait_s=3),
                raise_if_watcher_shutdown=lambda: None, register_control_subprocess=lambda proc: proc.pid,
                unregister_control_subprocess=lambda pid: None,
                terminate_control_process_group=lambda pid, sig: os.killpg(pid, sig),
                subprocess_popen=subprocess.Popen, pipe=subprocess.PIPE, sigkill=signal.SIGKILL)
            with control_transport():
                result = run_ssh_subprocess(args, deps=deps, input=json.dumps(report),
                                            capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                return {"ok": True, "route": route}
            failures.append({"route": route, "error": result.stderr[-500:]})
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            failures.append({"route": route, "error": str(exc)})
    return {"ok": False, "attempts": failures}


def run_once(config):
    root = Path.home() / ".claude/scheduler/transport_coord"
    control = probe(config["control_host"], config["control_port"], ssh=False)
    records = []
    targets = config["targets"]
    for target in targets:
        node = target["node"]
        try:
            settings = ssh_settings(node)
            with control_transport(), transport_guard(["ssh", node], root=root, wait_s=3):
                public = probe(settings["hostname"], int(settings["port"]))
            report = {"time": time.time(), "reporter": socket.gethostname(),
                      "public": public, "control": control}
            peer = next((other for other in targets if other["node"] != node), None)
            record = {"node": node, "observation": report,
                      "delivery": publish(target, peer, settings, report, root)}
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            record = {"node": node, "error": str(exc)}
        records.append(record)
    directory = Path.home() / ".cache/scheduleurm-frp"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "observer-latest.json", {"time": time.time(), "targets": records})
    print(json.dumps(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_once(json.loads(Path(args.config).read_text()))
