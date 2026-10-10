from __future__ import annotations

import base64
import subprocess
from dataclasses import replace

import pytest

from skill.scheduler_remote_exec import (
    RemoteExecDeps,
    configured_proxy_jumps,
    outer_ssh_route_key,
    probe_all_route_failures,
    probe_outer_ssh_route,
    remote_bash_command_for_node,
    run_on,
    run_windows_ps,
    ssh_base_args_with_proxy,
    select_proxy_jump,
)
from skill.scheduler_remote.subprocess import run_ssh_subprocess


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False

    def communicate(self, input=None, timeout=None):
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True


def _deps(*, nodes=None, calls=None, proc_factory=None):
    nodes = nodes or {
        "local": {"host": None},
        "node1": {
            "host": "node-host",
            "ssh_user": "user",
            "ssh_identity": "~/.ssh/sched",
            "ssh_port": 2201,
            "ssh_proxy_jump": "jump1",
        },
        "hpc": {
            "host": "login",
            "ssh_proxy_jump": "jump1",
            "sudo_ssh_host": "node001",
            "sudo_ssh_run_as": "hpcuser",
        },
        "win": {"host": "win-host", "os": "windows"},
    }
    calls = calls if calls is not None else []

    def popen(args, **kwargs):
        calls.append(("popen", list(args), kwargs))
        if proc_factory:
            return proc_factory(args, **kwargs)
        text = kwargs.get("text")
        out = "ok" if text else b"ok"
        err = "" if text else b""
        return _Proc(0, out, err)

    def subprocess_run(args, **kwargs):
        calls.append(("run", list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, "local-out", "")

    return RemoteExecDeps(
        node_configs=nodes,
        canonical_node_name=lambda name: {"alias": "node1"}.get(name, name),
        node_is_windows=lambda name: name == "win",
        raise_if_watcher_shutdown=lambda: None,
        register_control_subprocess=lambda proc: 123,
        unregister_control_subprocess=lambda pgid: calls.append(("unregister", pgid)),
        terminate_control_process_group=lambda pgid, **kwargs: calls.append(("terminate", pgid, kwargs)),
        ssh_route_cache_ttl_s=30,
        ssh_login_preflight_timeout_s=8,
        ssh_login_failure_cache_ttl_s=5,
        ssh_disable_mux=True,
        ssh_proxy_jump_cache={},
        ssh_route_jump_cache={},
        ssh_outer_route_cache={},
        subprocess_popen=popen,
        subprocess_run=subprocess_run,
        subprocess_check_output=lambda *args, **kwargs: "",
        kill=lambda *args, **kwargs: None,
        now=lambda: 100.0,
        expanduser=lambda path: path.replace("~", "/home/researcher"),
    )


def test_configured_proxy_jumps_prefers_plural_list():
    assert configured_proxy_jumps({"ssh_proxy_jumps": ["a", " b "], "ssh_proxy_jump": "c"}) == ["a", "b"]
    assert configured_proxy_jumps({"proxy_jump": "legacy"}) == ["legacy"]


def test_ssh_base_args_with_proxy_builds_unattended_transport():
    args = ssh_base_args_with_proxy("alias", "jump1", deps=_deps())

    assert args[:2] == ["ssh", "-o"]
    assert "ConnectTimeout=8" in args
    assert "BatchMode=yes" in args
    assert "ControlMaster=no" in args
    assert ["-i", "/home/researcher/.ssh/sched"] == args[args.index("-i"):args.index("-i") + 2]
    assert ["-p", "2201"] == args[args.index("-p"):args.index("-p") + 2]
    assert any(str(arg).startswith("ProxyCommand=ssh ") for arg in args)
    assert any("ProxyCommand=ssh -o ConnectTimeout=8 " in str(arg) for arg in args)
    assert args[-1] == "user@node-host"


def test_mux_is_enabled_on_both_hops_without_sharing_other_route_socket():
    deps = replace(_deps(), ssh_disable_mux=False)
    first = ssh_base_args_with_proxy("node1", "jump1", deps=deps)
    second = ssh_base_args_with_proxy("node1", "jump2", deps=deps)
    assert "ControlMaster=auto" in first
    assert "ControlPath=~/.ssh/scheduleurm-%r@%h-%p-jump1" in first
    assert "ControlPath=~/.ssh/scheduleurm-%r@%h-%p-jump2" in second
    proxy = next(arg for arg in first if arg.startswith("ProxyCommand="))
    assert "ControlMaster=auto" in proxy
    assert "ControlPath=~/.ssh/scheduleurm-jump-jump1" in proxy
    assert "%r@%h" not in proxy


def test_busy_cached_jump_yields_to_less_loaded_healthy_jump():
    nodes = {"hpc": {"host": "login", "ssh_proxy_jumps": ["jump1", "jump2"]}}
    deps = replace(_deps(nodes=nodes), transport_load=lambda route: 6 if route == "jump1" else 0)
    deps.ssh_proxy_jump_cache["hpc"] = ("jump1", 99)
    assert select_proxy_jump("hpc", deps=deps) == "jump2"


def test_direct_jump_and_proxy_use_the_same_budget_key():
    from scheduler_transport.budget import transport_routes

    nodes = {"hpc": {"host": "login", "ssh_proxy_jumps": ["jump1", "jump2"]},
             "jump1": {"host": "frp-host", "ssh_port": 22916, "ssh_user": "user"},
             "jump2": {"host": "frp-host", "ssh_port": 23035, "ssh_user": "user"}}
    deps = _deps(nodes=nodes)
    direct = ssh_base_args_with_proxy("jump2", None, deps=deps)
    proxied = ssh_base_args_with_proxy("hpc", "jump2", deps=deps)
    assert "Hostname=frp-host" in direct
    assert transport_routes(direct) == ["jump2"]
    assert transport_routes(proxied) == ["jump2", "login"]


def test_jump_alias_keeps_its_ssh_config_hostname():
    nodes = {"hpc": {"host": "login", "ssh_proxy_jump": "jump1"},
             "jump1": {"host": "jump1"}}
    args = ssh_base_args_with_proxy("jump1", None, deps=_deps(nodes=nodes))
    assert "Hostname=jump1" not in args
    assert args[-1] == "jump1"


def test_busy_preferred_jump_uses_other_route_without_poisoning_cache():
    from contextlib import contextmanager
    from scheduler_transport.budget import TransportBusy

    @contextmanager
    def guard(args):
        if any("-W %h:%p jump1" in arg for arg in args):
            raise TransportBusy(["jump1"])
        yield

    nodes = {"hpc": {"host": "login", "ssh_proxy_jumps": ["jump1", "jump2"]}}
    deps = replace(_deps(nodes=nodes), transport_guard=guard)
    assert select_proxy_jump("hpc", deps=deps) == "jump2"
    deps.ssh_proxy_jump_cache["hpc"] = ("jump1", 99)
    assert run_on("hpc", "true", deps=deps) == (0, "ok", "")
    assert probe_outer_ssh_route("hpc", deps=deps) == (True, "")
    assert all(value[0] is True for value in deps.ssh_outer_route_cache.values())


def test_busy_route_is_not_cached_as_down():
    from scheduler_transport.budget import TransportBusy
    from contextlib import contextmanager

    @contextmanager
    def full(args):
        raise TransportBusy(["jump1"])
        yield

    deps = replace(_deps(), transport_guard=full)
    with pytest.raises(TransportBusy):
        probe_outer_ssh_route("node1", deps=deps)
    assert deps.ssh_outer_route_cache == {}


def test_remote_bash_command_for_sudo_hop_wraps_inner_command():
    cmd = remote_bash_command_for_node("hpc", "echo hi", timeout=15, deps=_deps())

    assert cmd.startswith("bash -c ")
    assert "sudo -n /usr/bin/ssh node001" in cmd
    assert "su hpcuser -s /bin/bash" in cmd
    assert "timeout -k 2s 14s" in cmd


def test_run_on_remote_uses_no_stdin_ssh_and_remote_sudo_hop():
    calls = []

    rc, out, err = run_on("hpc", "echo hi", timeout=15, check=False, deps=_deps(calls=calls))

    assert (rc, out, err) == (0, "ok", "")
    popen_args = calls[0][1]
    assert popen_args[0:2] == ["ssh", "-n"]
    assert "ProxyCommand=ssh" in " ".join(popen_args)
    assert "sudo -n" in popen_args[-1]
    assert calls[-1] == ("unregister", 123)


def test_run_on_local_uses_local_bash_without_ssh():
    calls = []

    rc, out, err = run_on("local", "pwd", deps=_deps(calls=calls))

    assert (rc, out, err) == (0, "local-out", "")
    assert calls == [("run", ["bash", "-lc", "pwd"], {"capture_output": True, "timeout": 15, "text": True})]


def test_run_windows_ps_uses_encoded_command_and_decodes_bytes():
    calls = []

    rc, out, err = run_windows_ps("win", "Write-Output OK", deps=_deps(calls=calls), check=False)

    assert (rc, out, err) == (0, "ok", "")
    popen_args = calls[0][1]
    assert popen_args[-2] == "-EncodedCommand"
    script = base64.b64decode(popen_args[-1]).decode("utf-16le")
    assert script == "Write-Output OK"


def test_remote_control_interrupt_kills_process_group_before_unregister():
    calls = []

    class InterruptProc(_Proc):
        def __init__(self):
            super().__init__()
            self.communicate_count = 0

        def communicate(self, input=None, timeout=None):
            self.communicate_count += 1
            if self.communicate_count == 1:
                raise KeyboardInterrupt()
            return "", ""

    proc = InterruptProc()
    with pytest.raises(KeyboardInterrupt):
        run_on(
            "node1",
            "sleep 60",
            deps=_deps(calls=calls, proc_factory=lambda *args, **kwargs: proc),
        )

    assert proc.killed is True
    terminate_idx = calls.index(("terminate", 123, {"sig": 9}))
    unregister_idx = calls.index(("unregister", 123))
    assert terminate_idx < unregister_idx


def test_timed_out_ssh_cleanup_never_uses_unbounded_communicate():
    calls = []

    class StuckAfterKillProc(_Proc):
        def __init__(self):
            super().__init__()
            self.communicate_timeouts = []

        def communicate(self, input=None, timeout=None):
            self.communicate_timeouts.append(timeout)
            raise subprocess.TimeoutExpired(
                ["ssh"],
                timeout,
                output="partial",
                stderr="timed out",
            )

    proc = StuckAfterKillProc()
    deps = _deps(calls=calls, proc_factory=lambda *args, **kwargs: proc)

    with pytest.raises(subprocess.TimeoutExpired):
        run_ssh_subprocess(
            ["ssh", "host"],
            timeout=0.01,
            capture_output=True,
            text=True,
            deps=deps,
        )

    assert proc.communicate_timeouts == [0.01, 2.0]
    assert proc.killed is True
    assert calls[-1] == ("unregister", 123)


def test_probe_all_route_failures_reuses_fresh_positive_route_cache():
    calls = []
    deps = _deps(calls=calls)
    key = outer_ssh_route_key("node1", deps=deps)
    deps.ssh_outer_route_cache[key] = (True, "", 95.0)
    deps = replace(
        deps,
        probe_ssh_proxy_jump_callback=lambda node, jump, **kwargs: True,
        probe_outer_ssh_route_callback=lambda node: (
            calls.append(("outer_probe", node)) or (True, "")
        ),
        route_probe_max_workers=2,
    )

    assert probe_all_route_failures(deps=deps) == {}
    assert ("outer_probe", "node1") not in calls


def test_probe_all_route_failures_skips_retired_and_monitor_only_nodes():
    calls = []
    deps = _deps(
        nodes={
            "local": {"host": None},
            "retired": {"host": "old-host", "retired": True},
            "monitor": {"host": "monitor-host", "monitor_only": True},
        },
        calls=calls,
    )
    deps = replace(
        deps,
        probe_outer_ssh_route_callback=lambda node: (
            calls.append(("outer_probe", node)) or (True, "")
        ),
    )

    assert probe_all_route_failures(deps=deps) == {}
    assert not any(call[0] == "outer_probe" for call in calls)


def test_retired_node_transport_is_fail_closed_without_subprocesses():
    calls = []
    deps = _deps(nodes={"retired": {"host": "old-host", "retired": True}}, calls=calls)

    ok, reason = probe_outer_ssh_route("retired", deps=deps)
    assert not ok
    assert "node is retired" in reason

    rc, out, err = run_on("retired", "true", check=False, deps=deps)
    assert (rc, out) == (255, "")
    assert "node is retired" in err

    with pytest.raises(RuntimeError, match="node is retired"):
        run_on("retired", "true", check=True, deps=deps)
    with pytest.raises(RuntimeError, match="node is retired"):
        ssh_base_args_with_proxy("retired", None, deps=deps)

    assert calls == []


def test_direct_linux_route_is_keyed_but_windows_route_is_not():
    deps = _deps()
    deps.node_configs["direct"] = {"host": "direct-host", "ssh_user": "user"}

    assert outer_ssh_route_key("direct", deps=deps) is not None
    assert outer_ssh_route_key("win", deps=deps) is None


def test_direct_linux_preflight_timeout_is_reused_by_run_on():
    calls = []

    class TimeoutProc(_Proc):
        def __init__(self):
            super().__init__()
            self.communicate_count = 0

        def communicate(self, input=None, timeout=None):
            self.communicate_count += 1
            if self.communicate_count == 1:
                raise subprocess.TimeoutExpired(["ssh"], timeout)
            return "", ""

    proc = TimeoutProc()
    deps = _deps(calls=calls, proc_factory=lambda *args, **kwargs: proc)
    deps.node_configs["direct"] = {"host": "direct-host", "ssh_user": "user"}

    ok, reason = probe_outer_ssh_route("direct", deps=deps)

    assert not ok
    assert "direct: outer ssh route timed out" in reason
    assert sum(call[0] == "popen" for call in calls) == 1

    rc, _out, err = run_on("direct", "true", check=False, deps=deps)

    assert rc == 255
    assert "ssh login route down" in err
    assert sum(call[0] == "popen" for call in calls) == 1
