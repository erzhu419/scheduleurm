import importlib.util
import json
from pathlib import Path

import pytest
import tomli


SPEC = importlib.util.spec_from_file_location("install_frp_health", Path(__file__).parents[1] / "install_frp_health.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def setup_install(tmp_path, monkeypatch):
    frp_dir = tmp_path / "frpembed"
    frp_dir.mkdir()
    original = 'serverAddr = "frp.example"\nserverPort = 7000\n[[proxies]]\nname = "SSH"\ntype = "tcp"\nlocalPort = 22\nremotePort = 23035\n'
    (frp_dir / "frpc.toml").write_text(original)
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)

    def path(value):
        if str(value).startswith(("/etc/", "/var/", "/usr/local/")):
            return tmp_path / "root" / str(value).lstrip("/")
        return Path(value)

    monkeypatch.setattr(installer, "Path", path)
    bundle = {"frp_dir": str(frp_dir), "script": "script", "service": "service",
              "timer": "timer",
              "config": {"public_host": "public", "public_port": 23035,
                         "control_host": "control", "control_port": 7000}}
    return frp_dir, original, bundle


def test_install_preserves_frp_settings_and_does_not_restart_shared_gateway(tmp_path, monkeypatch):
    frp_dir, original, bundle = setup_install(tmp_path, monkeypatch)
    commands = []

    def check_call(args, **kwargs):
        commands.append(args)
        if "verify" in args and "-c" in args:
            parsed = tomli.loads(Path(args[-1]).read_text())
            assert parsed["log"]["level"] == "debug"
            del parsed["log"]
            assert parsed == tomli.loads(original)

    monkeypatch.setattr(installer.subprocess, "check_call", check_call)
    rotation = tmp_path / "root/etc/logrotate.d/scheduleurm-frp"
    rotation.parent.mkdir(parents=True)
    rotation.write_text("copytruncate")
    installer.install(bundle)
    first = (frp_dir / "frpc.toml").read_text()
    installer.install(bundle)
    assert (frp_dir / "frpc.toml").read_text() == first
    assert (frp_dir / "frpc.toml.before-frp-health").read_text() == original
    assert not rotation.exists()
    assert not any("restart" in args or "stop" in args for args in commands)
    assert not any("logrotate" in args[0] for args in commands)
    assert ["systemctl", "enable", "--now", "scheduleurm-frp-health.timer"] in commands


def test_invalid_config_is_rejected_before_installation(tmp_path, monkeypatch):
    frp_dir, original, bundle = setup_install(tmp_path, monkeypatch)

    def check_call(args, **kwargs):
        raise installer.subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(installer.subprocess, "check_call", check_call)
    with pytest.raises(installer.subprocess.CalledProcessError):
        installer.install(bundle)
    assert (frp_dir / "frpc.toml").read_text() == original
    assert not (tmp_path / "root").exists()


def test_policy_update_only_pauses_collector_and_preserves_gateway(tmp_path, monkeypatch):
    frp_dir, original, bundle = setup_install(tmp_path, monkeypatch)
    root = tmp_path / "root"
    config_path = root / "etc/scheduleurm/frp-health.json"
    config_path.parent.mkdir(parents=True)
    state_dir = root / "var/lib/scheduleurm-frp-health"
    state_dir.mkdir(parents=True)
    config = dict(bundle["config"], state_dir=str(state_dir))
    config_path.write_text(json.dumps(config))
    script = root / "usr/local/lib/scheduleurm/frp_health.py"
    script.parent.mkdir(parents=True)
    script.write_text("old = True\n")
    script.chmod(0o644)
    state_path = state_dir / "state.json"
    state_path.write_text(json.dumps({"failures": 2, "restarts": [123], "last_external_failure_at": 100}))
    commands = []
    monkeypatch.setattr(installer.subprocess, "check_call", lambda args: commands.append(args))
    monkeypatch.setattr(installer.subprocess, "check_output", lambda args: commands.append(args) or b"inactive\n")
    installer.update_policy({"script": "new = True\n", "external_probe_path": "/home/user/external.json"})
    assert script.read_text() == "new = True\n"
    assert script.with_name(script.name + ".before-independent-observer").read_text() == "old = True\n"
    assert json.loads(config_path.read_text()) == dict(config, external_probe_path="/home/user/external.json")
    assert json.loads(state_path.read_text()) == {"failures": 0, "restarts": [123]}
    assert (frp_dir / "frpc.toml").read_text() == original
    assert commands[0] == ["systemctl", "stop", "scheduleurm-frp-health.timer"]
    assert commands[-1] == ["systemctl", "start", "scheduleurm-frp-health.timer"]
    assert not any("frpembed.service" in args or "restart" in args for args in commands)


def test_invalid_policy_never_pauses_running_monitor(tmp_path, monkeypatch):
    _, _, bundle = setup_install(tmp_path, monkeypatch)
    config_path = tmp_path / "root/etc/scheduleurm/frp-health.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(bundle["config"]))
    commands = []
    monkeypatch.setattr(installer.subprocess, "check_call", lambda args: commands.append(args))
    with pytest.raises(SyntaxError):
        installer.update_policy({"script": "if\n", "external_probe_path": "/home/user/external.json"})
    assert commands == []
