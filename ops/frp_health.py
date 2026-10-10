#!/usr/bin/env python3
"""Host-local FRP end-to-end checks; Python 3.6+, standard library only."""

import argparse
import array
import fcntl
import glob
import json
import logging
import logging.handlers
import os
from pathlib import Path
import signal
import socket
import subprocess
import termios
import time


FAILURES = 3
COOLDOWN = 15 * 60
HOURLY_RESTARTS = 2
SNAPSHOT_BYTES = 64 * 1024
RESTART_EVIDENCE_DAYS = 7
EXTERNAL_MAX_AGE = 120
# Linux command number; GPU1's Python 3.6 does not expose F_GETPIPE_SZ.
LINUX_F_GETPIPE_SZ = 1032


def command(args, timeout=8):
    # Write command output to a bounded tail, without loading a whole journal.
    import tempfile

    with tempfile.TemporaryFile() as out:
        try:
            process = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT)
        except OSError as exc:
            return {"rc": -1, "output": str(exc)}
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        length = out.tell()
        out.seek(max(0, length - SNAPSHOT_BYTES))
        return {"rc": process.returncode,
                "output": out.read().decode("utf-8", "replace")}


def probe(host, port, ssh=True):
    start = time.monotonic()
    result = {"host": host, "port": port, "ok": False, "tcp_ok": False}

    def expired(signum, frame):
        raise TimeoutError("DNS/connect/banner deadline exceeded")

    # DNS failures have occurred on these hosts; socket timeouts alone do not
    # bound getaddrinfo(), so the entire probe needs a deadline.
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 8)
    try:
        with socket.create_connection((host, port), timeout=5) as connection:
            result["tcp_ok"] = True
            result["peer"] = connection.getpeername()[0]
            if not ssh:
                result["ok"] = True
            else:
                received = b""
                while len(received) < 4096:
                    chunk = connection.recv(512)
                    if not chunk:
                        break
                    received += chunk
                    banners = [line for line in received.splitlines()
                               if line.startswith((b"SSH-2.0-", b"SSH-1.99-"))]
                    if banners:
                        result.update(ok=True, banner=banners[0].decode("ascii", "replace"))
                        break
                if not result["ok"]:
                    result["error"] = "no SSH banner: " + repr(received[:160])
    except (OSError, TimeoutError) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result["seconds"] = round(time.monotonic() - start, 3)
    return result


def read_tail(path):
    try:
        with open(str(path), "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - SNAPSHOT_BYTES))
            data = stream.read(SNAPSHOT_BYTES)
            return {"bytes": size, "nul_bytes": data.count(b"\x00"),
                    "allocated_bytes": os.fstat(stream.fileno()).st_blocks * 512,
                    "text": data.replace(b"\x00", b"").decode("utf-8", "replace")}
    except OSError as exc:
        return {"error": str(exc)}


def read_proc_file(path):
    try:
        with open(str(path), "rb") as stream:
            data = stream.read(SNAPSHOT_BYTES)
        return {"bytes": len(data), "text": data.decode("utf-8", "replace")}
    except OSError as exc:
        return {"error": str(exc)}


def frp_registration_status(path, now):
    tail = read_tail(path)
    if "error" in tail:
        return {"status": "unknown", "error": tail["error"]}
    latest = {"status": "unknown"}
    for line in tail["text"].splitlines():
        if " start proxy success" in line:
            status = "registered"
        elif " start error: " in line:
            error = line.split(" start error: ", 1)[1]
            status = ("provider_registration_failed"
                      if any(message in error for message in (
                          "port unavailable", "send NewProxy request to plugin error"))
                      else "registration_failed")
        else:
            continue
        try:
            timestamp = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        latest = {"status": status, "time": timestamp,
                  "age_s": round(now - timestamp, 1), "line": line}
    # FRPC retries registration every ~33 seconds on these installed clients.
    # Old errors must not veto recovery after a later, unrelated failure.
    if "time" in latest and not -15 <= now - latest["time"] <= EXTERNAL_MAX_AGE:
        latest["event_status"] = latest["status"]
        latest["status"] = "stale"
    return latest


def process_metrics():
    records = []
    for path in glob.glob("/proc/[0-9]*/comm"):
        try:
            name = Path(path).read_text().strip()
            if name not in ("frpc", "frpembed"):
                continue
            parent = Path(path).parent
            record = {
                "pid": int(parent.name), "name": name,
                "fd_count": len(list((parent / "fd").iterdir())),
                "status": (parent / "status").read_text(),
                "limits": (parent / "limits").read_text(),
                "stdout": os.readlink(str(parent / "fd/1")),
                "stderr": os.readlink(str(parent / "fd/2")),
            }
            # FRPEMBED pipes FRPC stdout/stderr. Measure unread bytes without
            # consuming them, so a full logging pipe can be identified later.
            record["pipes"] = {}
            for number in ("1", "2"):
                if not os.readlink(str(parent / "fd" / number)).startswith("pipe:"):
                    continue
                fd = os.open(str(parent / "fd" / number), os.O_RDONLY | os.O_NONBLOCK)
                try:
                    count = array.array("i", [0])
                    fcntl.ioctl(fd, termios.FIONREAD, count, True)
                    record["pipes"][number] = {"unread": count[0],
                                               "capacity": fcntl.fcntl(fd, LINUX_F_GETPIPE_SZ)}
                finally:
                    os.close(fd)
            records.append(record)
        except OSError:
            continue
    return records


def ssh_status():
    ssh_processes = command(["ps", "-C", "sshd", "-o", "pid,ppid,user,stat,args"])
    rows = [line.split(None, 4) for line in ssh_processes["output"].splitlines()]
    rows = [row for row in rows if len(row) == 5 and row[0].isdigit()]
    # Privileged and network children belong to one handshake. Forward-only
    # authenticated sessions have a bare username, without @notty or @pts.
    preauth = {row[0] for row in rows if "[accepted]" in row[4]}
    preauth.update(row[1] for row in rows if "[net]" in row[4])
    authenticated = sum(row[4].startswith("sshd: ") and "[" not in row[4] and
                        not row[4].split()[1].startswith("/") for row in rows)
    settings = command(["/usr/sbin/sshd", "-T"])
    limits = dict(line.split(None, 1) for line in settings["output"].splitlines()
                  if line.startswith(("maxstartups ", "maxsessions ", "logingracetime ")))
    start_limit = limits.get("maxstartups", "").split(":", 1)[0]
    return {
        "ssh_processes": ssh_processes,
        "ssh_load": {
            "available": ssh_processes["rc"] == 0 and settings["rc"] == 0 and start_limit.isdigit(),
            "preauth_children": len(preauth),
            "authenticated_sessions": authenticated,
            "maxstartups_begin": int(start_limit) if start_limit.isdigit() else None,
            "limits": limits,
        },
    }


def sample(config):
    public = probe(config["public_host"], config["public_port"])
    local = probe("127.0.0.1", 22)
    return {
        "time": time.time(), "hostname": socket.gethostname(),
        "public": public, "local": local,
        "service": command(["systemctl", "is-active", "frpembed.service"])["output"].strip(),
        "processes": process_metrics(),
        "sockets": command(["ss", "-s"]),
        "links": command(["ip", "-s", "-j", "link"]),
        **ssh_status(),
        "tcp_counters": read_proc_file("/proc/net/snmp"),
        "tcp_ext_counters": read_proc_file("/proc/net/netstat"),
        "host_load": {name: read_proc_file("/proc/" + name)
                      for name in ("loadavg", "pressure/cpu", "pressure/memory", "pressure/io")},
    }


def external_status(config, now, hostname):
    try:
        report = json.loads(Path(config["external_probe_path"]).read_text())
        public = report["public"]
        age = now - report["time"]
        if (report["reporter"] == hostname or not report["reporter"] or
                public["host"] != config["public_host"] or public["port"] != config["public_port"]):
            status = "external_wrong_source"
        elif age < -15 or age > EXTERNAL_MAX_AGE:
            status = "external_stale"
        elif public["ok"]:
            status = "external_healthy"
        elif "maxstartups" in public.get("error", "").lower():
            status = "ssh_throttled"
        elif not report["control"]["ok"]:
            status = "external_upstream_unreachable"
        elif "gaierror" in public.get("error", ""):
            status = "external_dns_failure"
        else:
            status = "external_failed"
        return {"status": status, "age_s": round(age, 1), "report": report}
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {"status": "external_unconfirmed", "error": str(exc)}


def assess_failure(config, current):
    if current["public"]["ok"]:
        return
    current["frp_registration"] = frp_registration_status(
        config.get("frpc_log", "/var/log/scheduleurm-frp/frpc.log"), time.time())
    current["control"] = probe(config["control_host"], config["control_port"], ssh=False)
    ssh_log = command(["journalctl", "-u", "ssh.service", "--since", "3 minutes ago",
                       "-n", "1500", "--no-pager", "-o", "cat"])
    text = ssh_log["output"].lower()
    load = current["ssh_load"]
    errors = " ".join(current[key].get("error", "").lower() for key in ("public", "local"))
    # OpenSSH 7.6 silently drops throttled handshakes: no journal keyword or banner.
    current["ssh_throttled"] = (
        any(term in text or term in errors for term in ("maxstartups", "drop connection", "throttl")) or
        (load["available"] and load["preauth_children"] >= load["maxstartups_begin"]))
    current["ssh_journal_counts"] = {
        "invalid_user_lines": text.count("invalid user"),
        "failed_password_lines": text.count("failed password"),
        "accepted_key_lines": text.count("accepted publickey"),
        "maxstartups_lines": text.count("maxstartups"),
    }
    current["external"] = external_status(config, time.time(), current["hostname"])


def restart_block_reason(current):
    if current["public"]["ok"]:
        return "healthy"
    if current["service"] != "active":
        return "service_not_active"
    if current.get("ssh_throttled"):
        return "ssh_throttled"
    if not current["local"]["ok"]:
        return "local_ssh_unhealthy"
    if not current["control"]["ok"]:
        return "upstream_unreachable"
    if not current["ssh_load"]["available"]:
        return "ssh_load_unavailable"
    status = current["external"]["status"]
    if status != "external_failed":
        return status
    if current["ssh_load"]["authenticated_sessions"]:
        return "active_ssh_sessions"
    if current.get("frp_registration", {}).get("status") == "provider_registration_failed":
        return "provider_registration_failed"
    return None


def decide(state, current, now):
    state["restarts"] = [t for t in state.get("restarts", []) if now - t < 3600]
    blocked = restart_block_reason(current)
    if blocked:
        state["failures"] = 0
        state.pop("last_external_failure_at", None)
        if blocked == "healthy":
            state.pop("incident", None)
        return blocked
    external_time = current["external"]["report"]["time"]
    if external_time <= state.get("last_external_failure_at", 0):
        return "awaiting_external_probe"
    state["last_external_failure_at"] = external_time
    state["failures"] = state.get("failures", 0) + 1
    if state["failures"] < FAILURES:
        return "confirming_failure"
    if state["restarts"] and now - state["restarts"][-1] < COOLDOWN:
        return "restart_cooldown"
    if len(state["restarts"]) >= HOURLY_RESTARTS:
        return "restart_budget_exhausted"
    return "restart"


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(path)


def capture(config, directory, current, restart=False):
    commands = {
        "frp_unit": ["systemctl", "show", "frpembed.service"],
        "frp_journal": ["journalctl", "-u", "frpembed.service", "--since", "15 minutes ago", "-n", "250", "--no-pager", "-o", "short-iso"],
        "ssh_journal": ["journalctl", "-u", "ssh.service", "--since", "5 minutes ago", "-n", "500", "--no-pager", "-o", "short-iso"],
        "kernel": ["journalctl", "-k", "--since", "15 minutes ago", "-n", "300", "--no-pager", "-o", "short-iso"],
        "network_manager": ["journalctl", "-u", "NetworkManager.service", "--since", "15 minutes ago", "-n", "200", "--no-pager", "-o", "short-iso"],
        "wpa_supplicant": ["journalctl", "-u", "wpa_supplicant.service", "--since", "15 minutes ago", "-n", "300", "--no-pager", "-o", "short-iso"],
        "wireless": ["iwconfig"],
        "sockets": ["ss", "-tinp", "( sport = :22 or dport = :7000 )"],
        "routes": ["ip", "-j", "route"],
        "threads": ["ps", "-C", "frpc,frpembed", "-L", "-o", "pid,tid,comm,wchan:32"],
        "memory": ["free", "-m"],
        "ssh_config": ["/usr/sbin/sshd", "-T"],
    }
    outputs = {name: command(args) for name, args in commands.items()}
    outputs["ssh_config"]["output"] = "\n".join(
        line for line in outputs["ssh_config"]["output"].splitlines()
        if line.split(" ", 1)[0] in ("maxstartups", "maxsessions", "logingracetime"))
    record = {"sample": current,
              "commands": outputs,
              "frpc_log": read_tail(config["frpc_log"]),
              "supervisor_log": read_tail(config["supervisor_log"])}
    prefix = "restart-" if restart else "incident-"
    path = directory / (prefix + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + ".json")
    write_json(path, record)
    for old in sorted(directory.glob("incident-*.json"))[:-12]:
        old.unlink()
    cutoff = time.time() - RESTART_EVIDENCE_DAYS * 86400
    for old in directory.glob("restart-*.json"):
        if old.stat().st_mtime < cutoff:
            old.unlink()
    return str(path)


def run_once(config):
    directory = Path(config["state_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("frp_health")
    handler = logging.handlers.RotatingFileHandler(
        str(directory / "health.jsonl"), maxBytes=4 * 1024 * 1024, backupCount=2)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    current = sample(config)
    assess_failure(config, current)
    action = decide(state, current, current["time"])
    current["action"] = action
    if not current["public"]["ok"] and not state.get("incident"):
        state["incident"] = capture(config, directory, current)
    if action == "restart":
        # Capturing journals takes time: recheck every restart prerequisite,
        # including new users and a newer independent observation.
        before = capture(config, directory, current, restart=True)
        check = sample(config)
        assess_failure(config, check)
        current["recheck"] = check
        blocked = restart_block_reason(check)
        if blocked:
            current["action"] = ("recovered_before_restart" if blocked == "healthy"
                                 else blocked + "_before_restart")
            state["failures"] = 0
            state.pop("last_external_failure_at", None)
        else:
            state["restarts"].append(time.time())
            state["failures"] = 0
            # Save the cooldown before restart, including an unsuccessful attempt.
            write_json(state_path, state)
            current["restart_result"] = command(["systemctl", "restart", "frpembed.service"], timeout=25)
            after = []
            for attempt in range(4):
                time.sleep(5)
                after.append(probe(config["public_host"], config["public_port"]))
                if after[-1]["ok"]:
                    break
            current["after_restart"] = after
        evidence = json.loads(Path(before).read_text())
        evidence["recovery"] = current
        write_json(Path(before), evidence)
    write_json(state_path, state)
    write_json(directory / "latest.json", current)
    logger.info(json.dumps(current, ensure_ascii=True))
    logger.removeHandler(handler)
    handler.close()
    print(json.dumps({"action": current["action"], "failures": state.get("failures", 0),
                      "incident": state.get("incident"), "public": current["public"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/etc/scheduleurm/frp-health.json")
    args = parser.parse_args()
    run_once(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
