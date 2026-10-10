from skill.transport_exec import prepare_command
from scheduler_transport.budget import transport_routes


def test_common_ssh_entry_reuses_scheduler_jump_socket():
    command = prepare_command(["ssh", "gpu_b", "hostname"])
    assert "ControlMaster=auto" in command
    assert "ControlPath=~/.ssh/scheduleurm-jump-gpu_b" in command
    assert transport_routes(command) == ["gpu_b"]


def test_common_hpc_entry_reuses_both_hops_and_keeps_jump_budget():
    command = prepare_command(["ssh", "-J", "gpu_b", "user@hpc", "hostname"])
    assert "ControlMaster=auto" in command
    proxy = next(arg for arg in command if arg.startswith("ProxyCommand="))
    assert "ControlPath=~/.ssh/scheduleurm-jump-gpu_b" in proxy
    assert transport_routes(command) == sorted(["hpc", "gpu_b"])


def test_rsync_entry_routes_its_rsh_through_common_entry_once():
    command = prepare_command(["rsync", "-a", "-e", "ssh -J gpu_b",
                               "/source/", "user@hpc:/dest/"])
    assert "transport_exec.py" in command[command.index("-e") + 1]
    assert transport_routes(command) == sorted(["hpc", "gpu_b"])
    assert prepare_command(command) == command


def test_explicit_mux_disable_is_not_overridden():
    command = ["ssh", "-o", "ControlMaster=no", "gpu_b", "hostname"]
    assert prepare_command(command) == command
