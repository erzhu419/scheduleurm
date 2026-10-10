import importlib.util
import json
import os
from pathlib import Path
import socket
import threading
import time

import pytest


SPEC = importlib.util.spec_from_file_location("frp_health", Path(__file__).parents[1] / "frp_health.py")
health = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(health)


def broken(**kwargs):
    current = {"public": {"ok": False}, "local": {"ok": True},
               "control": {"ok": True}, "service": "active", "hostname": "gpu",
               "time": 10000,
               "ssh_load": {"available": True, "preauth_children": 0,
                            "maxstartups_begin": 10, "authenticated_sessions": 0},
               "external": {"status": "external_failed", "report": {"time": kwargs.get("time", 10000)}}}
    current.update(kwargs)
    return current


def test_active_but_broken_needs_three_samples():
    state = {}
    assert health.decide(state, broken(), 10000) == "confirming_failure"
    assert health.decide(state, broken(time=10060), 10060) == "confirming_failure"
    assert health.decide(state, broken(time=10120), 10120) == "restart"


@pytest.mark.parametrize("changes,expected", [
    ({"local": {"ok": False}}, "local_ssh_unhealthy"),
    ({"control": {"ok": False}}, "upstream_unreachable"),
    ({"service": "inactive"}, "service_not_active"),
    ({"service": "activating"}, "service_not_active"),
    ({"ssh_throttled": True}, "ssh_throttled"),
    ({"external": {"status": "external_healthy"}}, "external_healthy"),
    ({"external": {"status": "external_stale"}}, "external_stale"),
    ({"external": {"status": "external_unconfirmed"}}, "external_unconfirmed"),
    ({"external": {"status": "external_upstream_unreachable"}}, "external_upstream_unreachable"),
    ({"ssh_load": {"available": False}}, "ssh_load_unavailable"),
    ({"ssh_load": {"available": True, "authenticated_sessions": 1}}, "active_ssh_sessions"),
    ({"frp_registration": {"status": "provider_registration_failed"}}, "provider_registration_failed"),
])
def test_other_faults_do_not_restart_frp(changes, expected):
    state = {"failures": 2}
    assert health.decide(state, broken(**changes), 10000) == expected
    assert state["failures"] == 0


def test_recovered_link_resets_failure_streak():
    state = {"failures": 2, "incident": "old.json"}
    assert health.decide(state, broken(public={"ok": True}), 10000) == "healthy"
    assert state["failures"] == 0
    assert "incident" not in state


def test_restart_cooldown_and_hourly_budget():
    state = {"failures": 2, "restarts": [9950]}
    assert health.decide(state, broken(), 10000) == "restart_cooldown"
    state = {"failures": 2, "restarts": [7000, 9000]}
    assert health.decide(state, broken(), 10000) == "restart_budget_exhausted"
    assert health.decide(state, broken(time=13000), 13000) == "restart"


def test_same_external_failure_is_not_counted_three_times():
    state = {}
    for now in (10000, 10060, 10120):
        expected = "confirming_failure" if now == 10000 else "awaiting_external_probe"
        assert health.decide(state, broken(), now) == expected
    assert state["failures"] == 1


@pytest.mark.parametrize("reply,ok", [(b"SSH-2.0-test\r\n", True), (b"", False), (b"not SSH\r\n", False)])
def test_tcp_open_alone_is_not_a_healthy_forward(reply, ok):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def serve():
            connection, _ = listener.accept()
            with connection:
                connection.sendall(reply)

        thread = threading.Thread(target=serve)
        thread.start()
        result = health.probe("127.0.0.1", listener.getsockname()[1])
        thread.join(timeout=2)
    assert result["tcp_ok"]
    assert result["ok"] is ok


def test_restart_captures_before_and_after_and_preserves_budget(tmp_path, monkeypatch):
    config = {"state_dir": str(tmp_path), "public_host": "public", "public_port": 23035,
              "control_host": "control", "control_port": 7000}
    (tmp_path / "state.json").write_text(json.dumps({"failures": 2}))
    current = broken(time=10000)
    monkeypatch.setattr(health, "sample", lambda config: dict(current))
    monkeypatch.setattr(health, "external_status", lambda *args: current["external"])
    events = []

    def capture(config, directory, snapshot, restart=False):
        events.append("capture")
        path = directory / ("incident-%d.json" % len(events))
        health.write_json(path, {"sample": snapshot})
        return str(path)

    def command(args, **kwargs):
        if "restart" in args:
            assert "capture" in events
            assert json.loads((tmp_path / "state.json").read_text())["restarts"]
            events.append("restart")
        return {"rc": 0, "output": "active" if "is-active" in args else ""}

    def probe(host, port, ssh=True):
        return {"ok": host == "control" or "restart" in events}

    monkeypatch.setattr(health, "capture", capture)
    monkeypatch.setattr(health, "command", command)
    monkeypatch.setattr(health, "probe", probe)
    monkeypatch.setattr(health.time, "sleep", lambda seconds: None)
    health.run_once(config)
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert events == ["capture", "capture", "restart"]
    assert latest["after_restart"][-1]["ok"]
    assert json.loads((tmp_path / "incident-2.json").read_text())["recovery"]["after_restart"][-1]["ok"]


def test_recovery_during_capture_avoids_restart(tmp_path, monkeypatch):
    config = {"state_dir": str(tmp_path), "public_host": "public", "public_port": 23035,
              "control_host": "control", "control_port": 7000}
    (tmp_path / "state.json").write_text(json.dumps({"failures": 2}))
    monkeypatch.setattr(health, "external_status", lambda *args: broken()["external"])
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        return {"rc": 0, "output": "active" if "is-active" in args else ""}

    def capture(config, directory, current, restart=False):
        path = directory / "restart-test.json"
        health.write_json(path, {"sample": current})
        return str(path)

    monkeypatch.setattr(health, "command", command)
    monkeypatch.setattr(health, "capture", capture)
    monkeypatch.setattr(health, "sample", lambda config: next(samples))
    samples = iter([broken(), broken(public={"ok": True})])
    monkeypatch.setattr(health, "probe", lambda *args, **kwargs: {"ok": True})
    health.run_once(config)
    assert not any("restart" in args for args in commands)
    assert json.loads((tmp_path / "latest.json").read_text())["action"] == "recovered_before_restart"
    evidence = json.loads((tmp_path / "restart-test.json").read_text())
    assert evidence["recovery"]["action"] == "recovered_before_restart"


@pytest.mark.parametrize("path,heading", [
    ("/proc/net/snmp", "Tcp:"),
    ("/proc/net/netstat", "TcpExt:"),
])
def test_proc_counters_are_read_without_seeking(path, heading):
    result = health.read_proc_file(path)
    assert "error" not in result
    assert heading in result["text"]
    assert result["bytes"] > 0


def test_sparse_logging_tail_reports_holes_without_expanding_json(tmp_path):
    path = tmp_path / "frpc.log"
    with path.open("wb") as stream:
        stream.seek(2 * health.SNAPSHOT_BYTES)
        stream.write(b"latest log\n")
    result = health.read_tail(path)
    assert result["text"] == "latest log\n"
    assert result["nul_bytes"] == health.SNAPSHOT_BYTES - len(b"latest log\n")
    assert result["allocated_bytes"] < result["bytes"]


@pytest.mark.parametrize("error", [
    "port unavailable", "send NewProxy request to plugin error",
])
def test_provider_registration_error_vetoes_restart(tmp_path, error):
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    path = tmp_path / "frpc.log"
    path.write_text(stamp + ".051 [W] [client/control.go:166] [id] [gpu] start error: " + error + "\n")
    registration = health.frp_registration_status(path, now)
    assert registration["status"] == "provider_registration_failed"
    state = {"failures": 2}
    assert health.decide(state, broken(frp_registration=registration), now) == "provider_registration_failed"
    assert state["failures"] == 0
    assert error in registration["line"]


def test_later_registration_success_clears_provider_error(tmp_path):
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    path = tmp_path / "frpc.log"
    path.write_text(
        stamp + ".051 [W] [client/control.go:166] [id] [gpu] start error: port unavailable\n" +
        stamp + ".262 [I] [client/control.go:168] [id] [gpu] start proxy success\n")
    registration = health.frp_registration_status(path, now)
    assert registration["status"] == "registered"
    assert health.decide({"failures": 2}, broken(frp_registration=registration), now) == "restart"


def test_stale_provider_error_does_not_disable_future_recovery(tmp_path):
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - 180))
    path = tmp_path / "frpc.log"
    path.write_text(stamp + ".051 [W] [client/control.go:166] [id] [gpu] start error: port unavailable\n")
    registration = health.frp_registration_status(path, now)
    assert registration["status"] == "stale"
    assert registration["event_status"] == "provider_registration_failed"
    assert health.decide({"failures": 2}, broken(frp_registration=registration), now) == "restart"


def test_missing_registration_log_does_not_disable_existing_recovery(tmp_path):
    registration = health.frp_registration_status(tmp_path / "missing.log", time.time())
    assert registration["status"] == "unknown"
    assert health.decide({"failures": 2}, broken(frp_registration=registration), time.time()) == "restart"


def test_failed_sample_records_provider_registration_error(tmp_path, monkeypatch):
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    path = tmp_path / "frpc.log"
    path.write_text(stamp + ".051 [W] [client/control.go:166] [id] [gpu] start error: port unavailable\n")
    monkeypatch.setattr(health, "command", lambda *args: {"rc": 0, "output": ""})
    monkeypatch.setattr(health, "probe", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(health, "external_status", lambda *args: broken()["external"])
    current = broken()
    health.assess_failure({"control_host": "control", "control_port": 7000,
                           "frpc_log": str(path)}, current)
    assert current["frp_registration"]["status"] == "provider_registration_failed"
    assert health.decide({"failures": 2}, current, now) == "provider_registration_failed"


def test_restart_evidence_survives_transient_incidents_and_expires_after_a_week(tmp_path, monkeypatch):
    config = {"frpc_log": str(tmp_path / "frpc.log"),
              "supervisor_log": str(tmp_path / "frpembed.log")}
    monkeypatch.setattr(health, "command", lambda *args: {"rc": 0, "output": ""})
    stamps = iter("20261002T%06d" % n for n in range(20))
    monkeypatch.setattr(health.time, "strftime", lambda *args: next(stamps))
    restart_path = Path(health.capture(config, tmp_path, broken(time=10000), restart=True))
    expired = tmp_path / "restart-expired.json"
    expired.write_text("{}")
    old = time.time() - 8 * 86400
    os.utime(str(expired), (old, old))
    for _ in range(15):
        health.capture(config, tmp_path, broken(time=10000))
    assert len(list(tmp_path.glob("incident-*.json"))) == 12
    assert restart_path.exists()
    assert not expired.exists()


def test_sample_counts_handshakes_not_authenticated_sessions(monkeypatch):
    output = "\n".join([
        "PID PPID USER STAT COMMAND",
        "1 0 root Ss sshd: user [priv]",
        "2 1 sshd S sshd: unknown [net]",
        "3 0 root Ss sshd: [accepted]",
        "4 1 user S sshd: user@notty",
        "5 1 user S sshd: user@pts/1",
    ])
    monkeypatch.setattr(health, "command", lambda args: {"rc": 0, "output":
                        "maxstartups 10:30:100\nmaxsessions 10\nlogingracetime 120" if "-T" in args else output})
    monkeypatch.setattr(health, "probe", lambda *args: {"ok": True})
    monkeypatch.setattr(health, "process_metrics", lambda: [])
    result = health.sample({"public_host": "public", "public_port": 1})
    assert result["ssh_load"]["preauth_children"] == 2
    assert result["ssh_load"]["authenticated_sessions"] == 2
    assert result["ssh_load"]["maxstartups_begin"] == 10
    assert result["ssh_load"]["available"]


def test_samples_keep_pressure_and_incidents_keep_wireless_disconnect_reasons(tmp_path, monkeypatch):
    reads = []
    commands = []

    def read_proc(path):
        reads.append(path)
        return {"bytes": 1, "text": "0"}

    def command(args):
        commands.append(args)
        text = "CTRL-EVENT-BEACON-LOSS\nCTRL-EVENT-DISCONNECTED reason=4 locally_generated=1"
        return {"rc": 0, "output": text if "wpa_supplicant.service" in args else ""}

    monkeypatch.setattr(health, "read_proc_file", read_proc)
    monkeypatch.setattr(health, "command", command)
    monkeypatch.setattr(health, "probe", lambda *args: {"ok": True})
    monkeypatch.setattr(health, "process_metrics", lambda: [])
    monkeypatch.setattr(health, "ssh_status", lambda: {})
    current = health.sample({"public_host": "public", "public_port": 23035})
    assert set(current["host_load"]) == {"loadavg", "pressure/cpu", "pressure/memory", "pressure/io"}
    assert "/proc/pressure/cpu" in reads
    config = {"frpc_log": str(tmp_path / "frpc.log"),
              "supervisor_log": str(tmp_path / "frpembed.log")}
    incident = json.loads(Path(health.capture(config, tmp_path, current)).read_text())
    assert "BEACON-LOSS" in incident["commands"]["wpa_supplicant"]["output"]
    assert "reason=4 locally_generated=1" in incident["commands"]["wpa_supplicant"]["output"]
    assert ["iwconfig"] in commands
    assert all(not set(args).intersection({"restart", "reapply", "disconnect", "down"})
               for args in commands)


def test_forward_only_and_root_sessions_are_protected_and_handshakes_deduplicated(monkeypatch):
    output = "\n".join([
        "PID PPID USER STAT COMMAND",
        "1 0 root Ss /usr/sbin/sshd -D",
        "2 1 root Ss sshd: [accepted]",
        "3 2 sshd S sshd: [net]",
        "4 1 root Ss sshd: user [priv]",
        "5 4 user S sshd: user",
        "6 1 root Ss sshd: root [priv]",
        "7 6 root S sshd: root",
        "8 1 root Ss sshd: unknown [priv]",
        "9 8 sshd S sshd: unknown [net]",
        "10 1 root Ss sshd: /usr/sbin/sshd -D [listener] 2 of 10-100 startups",
    ])
    monkeypatch.setattr(health, "command", lambda args: {"rc": 0, "output":
                        "maxstartups 10:30:100\n" if "-T" in args else output})
    load = health.ssh_status()["ssh_load"]
    assert load["preauth_children"] == 2
    assert load["authenticated_sessions"] == 2
    assert health.decide({}, broken(ssh_load=load), 10000) == "active_ssh_sessions"


def test_old_openssh_silent_throttle_does_not_restart(monkeypatch):
    current = broken(local={"ok": False, "error": "no SSH banner: b''"})
    current["ssh_load"]["preauth_children"] = 10
    monkeypatch.setattr(health, "command", lambda *args: {"rc": 0, "output": "Invalid user admin"})
    monkeypatch.setattr(health, "probe", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(health, "external_status", lambda *args: broken()["external"])
    health.assess_failure({"control_host": "control", "control_port": 7000}, current)
    assert health.decide({"failures": 2}, current, 10000) == "ssh_throttled"
    assert current["ssh_journal_counts"]["invalid_user_lines"] == 1


@pytest.mark.parametrize("change,expected", [
    ({}, "external_failed"),
    ({"time": 9879}, "external_stale"),
    ({"reporter": "gpu"}, "external_wrong_source"),
    ({"public": {"host": "other", "port": 23035, "ok": False}}, "external_wrong_source"),
    ({"public": {"host": "public", "port": 23035, "ok": True}}, "external_healthy"),
    ({"control": {"ok": False}}, "external_upstream_unreachable"),
    ({"public": {"host": "public", "port": 23035, "ok": False, "error": "gaierror"}}, "external_dns_failure"),
    ({"public": {"host": "public", "port": 23035, "ok": False, "error": "Exceeded MaxStartups"}}, "ssh_throttled"),
])
def test_external_confirmation_must_be_fresh_and_independent(tmp_path, change, expected):
    path = tmp_path / "external.json"
    report = {"time": 10000, "reporter": "wsl", "public": {"host": "public", "port": 23035, "ok": False},
              "control": {"ok": True}}
    report.update(change)
    health.write_json(path, report)
    config = {"external_probe_path": str(path), "public_host": "public", "public_port": 23035}
    assert health.external_status(config, 10000, "gpu")["status"] == expected
    path.unlink()
    assert health.external_status(config, 10000, "gpu")["status"] == "external_unconfirmed"


@pytest.mark.parametrize("change,expected", [
    ({"external": {"status": "external_healthy"}}, "external_healthy_before_restart"),
    ({"ssh_throttled": True}, "ssh_throttled_before_restart"),
    ({"ssh_load": {"available": True, "authenticated_sessions": 1}}, "active_ssh_sessions_before_restart"),
    ({"service": "inactive"}, "service_not_active_before_restart"),
    ({"control": {"ok": False}}, "upstream_unreachable_before_restart"),
    ({"frp_registration": {"status": "provider_registration_failed"}}, "provider_registration_failed_before_restart"),
])
def test_new_fault_or_user_during_capture_vetoes_restart(tmp_path, monkeypatch, change, expected):
    health.write_json(tmp_path / "state.json", {"failures": 2})
    snapshots = iter([broken(), broken(**change)])
    monkeypatch.setattr(health, "sample", lambda config: next(snapshots))
    monkeypatch.setattr(health, "assess_failure", lambda *args: None)

    def capture(config, directory, current, restart=False):
        path = directory / ("restart.json" if restart else "incident.json")
        health.write_json(path, {"sample": current})
        return str(path)

    monkeypatch.setattr(health, "capture", capture)
    commands = []
    monkeypatch.setattr(health, "command", lambda args, **kwargs: commands.append(args))
    health.run_once({"state_dir": str(tmp_path)})
    assert not commands
    assert json.loads((tmp_path / "latest.json").read_text())["action"] == expected


def test_healthy_collection_does_not_manage_services_or_truncate_logs(tmp_path, monkeypatch):
    config = {"state_dir": str(tmp_path)}
    monkeypatch.setattr(health, "sample", lambda config: {"time": 10000, "public": {"ok": True}})
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        return {"rc": 0, "output": ""}

    monkeypatch.setattr(health, "command", command)
    health.run_once(config)
    assert commands == []
    assert json.loads((tmp_path / "latest.json").read_text())["action"] == "healthy"


def test_logging_pipe_occupancy_does_not_consume_output():
    import array
    import fcntl
    import os
    import termios

    reader, writer = os.pipe()
    try:
        os.write(writer, b"pending log")
        count = array.array("i", [0])
        fcntl.ioctl(reader, termios.FIONREAD, count, True)
        assert count[0] == 11
        assert fcntl.fcntl(reader, health.LINUX_F_GETPIPE_SZ) >= count[0]
        assert os.read(reader, 11) == b"pending log"
    finally:
        os.close(reader)
        os.close(writer)
