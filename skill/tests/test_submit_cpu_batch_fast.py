from __future__ import annotations

import argparse
from contextlib import contextmanager

import pytest


def test_submit_cpu_batch_uses_one_bulk_state_transaction(monkeypatch, capsys, sch):
    args = argparse.Namespace(
        items=4,
        item_multiplier=1,
        nodes="jtl110cpu,jtl110cpu2",
        use_total_cores=True,
        json=False,
        dry_run=False,
        allow_env_only_shard=False,
        cmd_template="python eval.py --start {start} --end {end} --workers {workers}",
        cwd="/tmp/cpu-batch-work",
        signature="Demo/cpu/{node}",
        description="demo {node}",
        ram_mb=128,
        priority="normal",
        project="Demo",
        env=[],
        result_dir_template="/tmp/cpu-batch-results/{node}",
        local_result_dir_template="/tmp/cpu-batch-local-results/{node}",
        wait_for_file_template=[],
        allow_cpu_training=False,
        cpu_training_justification="",
        allow_no_ckpt=False,
        allow_no_resume=False,
        allow_shared_result_dir=False,
        allow_remote_large_data=False,
        allow_duplicate=False,
        node_down_requeue_s=0,
        env_spec="none",
        image="",
        stage_exclude=[],
    )

    def forbidden_cmd_submit(_args):
        raise AssertionError("submit-cpu-batch must not call cmd_submit per shard")

    real_state_lock = sch.state_lock
    submit_lock_count = 0

    @contextmanager
    def counting_state_lock(*lock_args, **lock_kwargs):
        nonlocal submit_lock_count
        if lock_kwargs.get("purpose") == "submit-cpu-batch":
            submit_lock_count += 1
        with real_state_lock(*lock_args, **lock_kwargs):
            yield

    monkeypatch.setattr(sch, "cmd_submit", forbidden_cmd_submit)
    monkeypatch.setattr(sch, "state_lock", counting_state_lock)

    sch.cmd_submit_cpu_batch(args)
    out = capsys.readouterr().out
    state = sch.load_state()

    assert submit_lock_count == 1
    assert "via one state-lock transaction" in out
    assert [t["id"] for t in state["tasks"]] == ["t0001", "t0002"]
    assert [t["preferred_node"] for t in state["tasks"]] == ["jtl110cpu", "jtl110cpu2"]
    assert [t["cpu_parallel_start"] for t in state["tasks"]] == [0, 2]
    assert [t["cpu_parallel_end"] for t in state["tasks"]] == [2, 4]


def test_submit_cpu_batch_rejects_shared_result_destination_before_partial_submit(sch):
    args = argparse.Namespace(
        items=4,
        item_multiplier=1,
        nodes="jtl110cpu,jtl110cpu2",
        use_total_cores=True,
        json=False,
        dry_run=False,
        allow_env_only_shard=True,
        cmd_template="python eval.py --start {start} --end {end}",
        cwd="/tmp/cpu-batch-work",
        signature="Demo/cpu/{node}",
        description="demo {node}",
        ram_mb=128,
        priority="normal",
        project="Demo",
        env=[],
        result_dir_template="/tmp/shared-result",
        local_result_dir_template="/tmp/shared-local-result",
        wait_for_file_template=[],
        allow_cpu_training=False,
        cpu_training_justification="",
        allow_no_ckpt=False,
        allow_no_resume=False,
        allow_shared_result_dir=False,
        allow_remote_large_data=False,
        allow_duplicate=False,
        node_down_requeue_s=0,
        env_spec="none",
        image="",
        stage_exclude=[],
    )

    with pytest.raises(SystemExit, match="share a local result destination"):
        sch.cmd_submit_cpu_batch(args)

    assert sch.load_state()["tasks"] == []
