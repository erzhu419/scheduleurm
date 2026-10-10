#!/usr/bin/env python3
"""Install the small FRP health bundle as root on one of the shared GPU hosts."""

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def update_policy(bundle):
    """Update an installed collector without touching the shared FRP service."""
    if os.geteuid() != 0:
        raise RuntimeError("installation requires root")
    script = Path("/usr/local/lib/scheduleurm/frp_health.py")
    config_path = Path("/etc/scheduleurm/frp-health.json")
    config = json.loads(config_path.read_text())
    config["external_probe_path"] = bundle["external_probe_path"]
    compile(bundle["script"], str(script), "exec")
    timer = "scheduleurm-frp-health.timer"
    subprocess.check_call(["systemctl", "stop", timer])
    try:
        deadline = time.monotonic() + 180
        while True:
            active = subprocess.check_output(
                ["systemctl", "show", "scheduleurm-frp-health.service", "-p", "ActiveState", "--value"]
            ).decode().strip()
            if active in ("inactive", "failed"):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("collector is still running; policy update aborted")
            time.sleep(0.5)
        for path in (script, config_path):
            backup = path.with_name(path.name + ".before-independent-observer")
            if not backup.exists():
                shutil.copy2(str(path), str(backup))
        for path, text in ((script, bundle["script"]),
                           (config_path, json.dumps(config, indent=2) + "\n")):
            temporary = path.with_suffix(".tmp")
            temporary.write_text(text)
            temporary.chmod(path.stat().st_mode & 0o777)
            temporary.replace(path)
        state_path = Path(config["state_dir"]) / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            state["failures"] = 0
            state.pop("last_external_failure_at", None)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state) + "\n")
            temporary.replace(state_path)
    finally:
        subprocess.check_call(["systemctl", "start", timer])
    print("UPDATED health policy only; FRP service and configuration were not changed")


def install(bundle):
    if os.geteuid() != 0:
        raise RuntimeError("installation requires root")
    frp_dir = Path(bundle["frp_dir"])
    config_path = frp_dir / "frpc.toml"
    original = config_path.read_text()
    prefix = ('log.to = "/var/log/scheduleurm-frp/frpc.log"\n'
              'log.level = "debug"\nlog.maxDays = 3\nlog.disablePrintColor = true\n')
    updated = original if original.startswith(prefix) else prefix + original
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", dir=str(frp_dir)) as candidate:
        candidate.write(updated)
        candidate.flush()
        subprocess.check_call([str(frp_dir / "frpc"), "verify", "-c", candidate.name])
    destinations = {
        "script": "/usr/local/lib/scheduleurm/frp_health.py",
        "service": "/etc/systemd/system/scheduleurm-frp-health.service",
        "timer": "/etc/systemd/system/scheduleurm-frp-health.timer",
    }
    for key, destination in destinations.items():
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(bundle[key])
        path.chmod(0o644)
    directory = Path("/etc/scheduleurm")
    directory.mkdir(parents=True, exist_ok=True)
    configuration = dict(bundle["config"])
    configuration.update(state_dir="/var/lib/scheduleurm-frp-health",
                         frpc_log="/var/log/scheduleurm-frp/frpc.log",
                         supervisor_log=str(frp_dir / "frpembed.log"))
    (directory / "frp-health.json").write_text(json.dumps(configuration, indent=2) + "\n")
    for destination in (configuration["state_dir"], "/var/log/scheduleurm-frp"):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    # FRPC already rotates daily. copytruncate leaves holes in its open log file.
    external_rotation = Path("/etc/logrotate.d/scheduleurm-frp")
    if external_rotation.exists():
        external_rotation.unlink()
    backup = config_path.with_name("frpc.toml.before-frp-health")
    if not backup.exists():
        shutil.copy2(str(config_path), str(backup))
    config_path.write_text(updated)
    dropin = Path("/etc/systemd/system/frpembed.service.d/90-scheduleurm-logging.conf")
    dropin.parent.mkdir(parents=True, exist_ok=True)
    dropin.write_text("[Service]\nStandardOutput=journal\nStandardError=journal\n")
    subprocess.check_call(["systemd-analyze", "verify", destinations["service"], destinations["timer"]])
    subprocess.check_call(["systemctl", "daemon-reload"])
    subprocess.check_call(["systemctl", "enable", "--now", "scheduleurm-frp-health.timer"])
    print("INSTALLED; FRP service was not restarted")


if __name__ == "__main__":
    if sys.argv[1] == "--update-policy":
        update_policy(json.loads(base64.b64decode(sys.argv[2])))
    else:
        install(json.loads(base64.b64decode(sys.argv[1])))
