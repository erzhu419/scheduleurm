import importlib.util
import json
from pathlib import Path
import subprocess
import sys

OPS = Path(__file__).parents[1]
sys.path.insert(0, str(OPS))
SPEC = importlib.util.spec_from_file_location("frp_observer", OPS / "frp_observer.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def test_fresh_socket_probes_are_not_replaced_by_mux_login(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(observer.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(observer, "ssh_settings", lambda node: {"hostname": "public", "port": "23035"})

    def probe(host, port, ssh=True):
        calls.append((host, port, ssh))
        return {"ok": True, "host": host, "port": port}

    monkeypatch.setattr(observer, "probe", probe)
    delivered = []
    monkeypatch.setattr(observer, "publish", lambda target, peer, settings, report, root:
                        delivered.append(report) or {"ok": True})
    observer.run_once({"control_host": "control", "control_port": 7000,
                       "targets": [{"node": "gpu"}]})
    assert calls == [("control", 7000, False), ("public", 23035, True)]
    assert delivered[0]["public"]["ok"]
    assert json.loads((tmp_path / ".cache/scheduleurm-frp/observer-latest.json").read_text())["targets"][0]["delivery"]["ok"]


def test_failed_direct_delivery_uses_peer_lan_and_existing_host_key(monkeypatch, tmp_path):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert json.loads(kwargs["input"])["public"]["ok"] is False
        assert kwargs["timeout"] == 15
        return subprocess.CompletedProcess(args, 255 if len(calls) == 1 else 0, "", "closed")

    monkeypatch.setattr(observer, "run_ssh_subprocess", run)
    result = observer.publish(
        {"node": "gpu2", "internal_host": "192.0.2.21"}, {"node": "gpu1"},
        {"hostname": "public", "port": "23035", "hostkeyalias": "none"},
        {"public": {"ok": False}}, tmp_path)
    assert result == {"ok": True, "route": "via_gpu1"}
    assert "Hostname=192.0.2.21" in calls[1]
    assert "HostKeyAlias=[public]:23035" in calls[1]
    assert "ControlPath=none" in calls[1]
    assert "StrictHostKeyChecking=yes" in calls[1]
    assert "-p" in calls[1] and calls[1][calls[1].index("-p") + 1] == "22"
    proxy = next(arg for arg in calls[1] if arg.startswith("ProxyCommand="))
    assert "gpu1" in proxy and "-W %h:%p" in proxy
    assert "ControlPersist=no" in proxy
    assert observer.transport_guard.__module__ == "scheduler_transport.budget"


def test_both_delivery_failures_are_logged_without_recovery_commands(monkeypatch, tmp_path):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, 15)

    monkeypatch.setattr(observer, "run_ssh_subprocess", run)
    result = observer.publish({"node": "gpu"}, None, {}, {"public": {"ok": False}}, tmp_path)
    assert not result["ok"]
    assert len(result["attempts"]) == 1
    assert not any("systemctl" in arg for args in calls for arg in args)


def test_published_observation_is_small_atomic_user_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    report = {"time": 123, "reporter": "wsl", "public": {"ok": False}}
    result = subprocess.run([sys.executable, "-c", observer.PUBLISH_CODE],
                            input=json.dumps(report), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    path = tmp_path / ".cache/scheduleurm-frp/external.json"
    assert json.loads(path.read_text()) == report
    assert path.stat().st_mode & 0o777 == 0o600
    assert not path.with_suffix(".tmp").exists()
