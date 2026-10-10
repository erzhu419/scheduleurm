from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import multiprocessing
import time

import pytest

from scheduler_transport.budget import (
    TransportBusy, bulk_transport, control_transport, transport_child_env,
    transport_guard, transport_load, transport_routes,
)
from skill.scheduler_state.locks import state_lock
from skill.scheduler_claim.records import release_task_claims_and_intents


def _worker(root, bulk, ready, start, current, peak, lock):
    with bulk_transport() if bulk else ExitStack():
        ready.put(True)
        start.wait(10)
        with transport_guard(["ssh", "-J", "gpu2", "hpc"], root=Path(root), wait_s=10):
            with lock:
                current.value += 1
                peak.value = max(peak.value, current.value)
            time.sleep(0.15)
            with lock:
                current.value -= 1


@pytest.mark.parametrize("bulk, expected", [(False, 6), (True, 2)])
def test_ten_processes_share_one_route_budget(tmp_path, bulk, expected):
    ctx = multiprocessing.get_context("spawn")
    ready, start, lock = ctx.Queue(), ctx.Event(), ctx.Lock()
    current, peak = ctx.Value("i", 0), ctx.Value("i", 0)
    children = [ctx.Process(target=_worker, args=(str(tmp_path), bulk, ready, start,
                                                 current, peak, lock)) for _ in range(10)]
    try:
        for child in children:
            child.start()
        for _ in children:
            ready.get(timeout=15)
        start.set()
        for child in children:
            child.join(15)
            assert child.exitcode == 0
        assert peak.value == expected
        assert transport_load(tmp_path, "gpu2") == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)


def test_probe_and_cancel_have_two_reserved_slots(tmp_path):
    command = ["ssh", "gpu2"]
    with ExitStack() as active:
        for _ in range(6):
            active.enter_context(transport_guard(command, root=tmp_path, wait_s=0))
        with pytest.raises(TransportBusy):
            with transport_guard(command, root=tmp_path, wait_s=0):
                pass
        with control_transport():
            active.enter_context(transport_guard(command, root=tmp_path, wait_s=0))
            active.enter_context(transport_guard(command, root=tmp_path, wait_s=0))
            assert transport_load(tmp_path, "gpu2") == 8
            with pytest.raises(TransportBusy):
                with transport_guard(command, root=tmp_path, wait_s=0):
                    pass


def test_claim_release_can_use_reserved_capacity(tmp_path):
    def release(node, task_id):
        with transport_guard(["ssh", node], root=tmp_path, wait_s=0):
            return True

    with ExitStack() as active:
        for _ in range(6):
            active.enter_context(transport_guard(["ssh", "gpu2"], root=tmp_path, wait_s=0))
        assert release_task_claims_and_intents(
            {"id": "t1", "node": "gpu2"}, claim_enabled_for=lambda _: True,
            claim_release=release) == 1


def test_transport_never_waits_for_capacity_under_real_state_lock(tmp_path):
    command = ["ssh", "gpu2"]
    with ExitStack() as active:
        for _ in range(6):
            active.enter_context(transport_guard(command, root=tmp_path / "transport", wait_s=0))
        with state_lock(lock_file=tmp_path / ".lock", log_dir=tmp_path / "logs"):
            started = time.monotonic()
            with pytest.raises(TransportBusy):
                with transport_guard(command, root=tmp_path / "transport", wait_s=60):
                    pass
            assert time.monotonic() - started < 0.5


def test_failed_multi_route_admission_releases_partial_slots(tmp_path):
    with ExitStack() as active:
        for _ in range(6):
            active.enter_context(transport_guard(["ssh", "z-gpu2"], root=tmp_path, wait_s=0))
        with pytest.raises(TransportBusy):
            with transport_guard(["ssh", "-J", "z-gpu2", "a-hpc"], root=tmp_path, wait_s=0):
                pass
        assert transport_load(tmp_path, "a-hpc") == 0


def test_external_rsync_shell_identifies_actual_jump():
    shell = "/usr/bin/python3 /repo/skill/transport_exec.py --bulk -- ssh -o 'ProxyCommand=ssh -W %h:%p gpu2'"
    assert transport_routes(["rsync", "-az", "-e", shell, "user@hpc:/source/", "/dest/"]) == ["gpu2", "hpc"]
    assert transport_routes(["scp", "-o", "ProxyCommand=ssh -W %h:%p gpu2", "/source", "user@hpc:/dest"]) == ["gpu2", "hpc"]


def test_admitted_rsync_child_does_not_acquire_the_same_budget_again(tmp_path, monkeypatch):
    command = ["rsync", "-e", "ssh", "gpu2:/source", "/dest"]
    with ExitStack() as active:
        active.enter_context(transport_guard(command, root=tmp_path, wait_s=0))
        active.enter_context(transport_guard(command, root=tmp_path, wait_s=0))
        env = transport_child_env()
        assert env["SCHEDULEURM_TRANSPORT_ADMITTED"] == "1"
        monkeypatch.setenv("SCHEDULEURM_TRANSPORT_ADMITTED", "1")
        with bulk_transport(), transport_guard(["ssh", "gpu2"], root=tmp_path, wait_s=0):
            assert transport_load(tmp_path, "gpu2") == 2


def test_budget_is_released_when_command_raises(tmp_path):
    with pytest.raises(RuntimeError):
        with transport_guard(["ssh", "gpu2"], root=tmp_path):
            raise RuntimeError("command failed")
    assert transport_load(tmp_path, "gpu2") == 0
