from __future__ import annotations

import argparse

import pytest

from skill.scheduler_submit_preflight import (
    SubmitPreflightDeps,
    SubmitPreflightRefusal,
    run_submit_preflight,
)


def _args(**overrides):
    values = {
        "cmd": "python train.py --resume",
        "cwd": "/work",
        "description": "demo train",
        "ckpt_dir": "",
        "result_dir": "",
        "resume_flag": "",
        "require_node": None,
        "preferred_node": None,
        "allowed_nodes": None,
        "vram": 1024,
        "allow_remote_large_data": False,
        "allow_cpu_training": False,
        "cpu_training_justification": "",
        "allow_no_ckpt": False,
        "allow_no_resume": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _deps(**overrides):
    values = {
        "submit_policy_text": lambda cmd, cwd: cmd,
        "infer_checkpoint_from_submit": lambda cmd, cwd: None,
        "canonical_node_name": lambda node: str(node).replace("n1", "node001"),
        "canonicalize_node_list": lambda nodes: [
            str(node).replace("n1", "node001") for node in (nodes or [])
        ],
        "simple_sac_large_data_reason": lambda cmd, cwd: "",
        "cpu_training_policy_reason": lambda cmd, desc, allow, vram: None,
        "task_looks_like_training": lambda cmd, desc: "train" in cmd,
        "cmd_looks_like_training": lambda cmd: "train" in cmd,
        "resume_capability_reason": lambda cmd, ckpt, flag, allow: "",
        "checkpoint_contract_reason": lambda raw, cwd, ckpt, flag, allow: "",
        "same_declared_path": lambda a, b: str(a).rstrip("/") == str(b).rstrip("/"),
    }
    values.update(overrides)
    return SubmitPreflightDeps(**values)


def test_submit_preflight_infers_checkpoint_and_canonicalizes_nodes():
    args = _args(
        ckpt_dir="",
        result_dir="",
        require_node="n1",
        preferred_node="n1",
        allowed_nodes=["n1", "node002"],
    )

    result = run_submit_preflight(
        args,
        deps=_deps(
            infer_checkpoint_from_submit=lambda cmd, cwd: {
                "ckpt_dir": "/work/ckpt",
                "result_dir": "/work/results",
                "resume_managed_by_cmd": True,
                "source": "cmd",
            }
        ),
        default_vram_mb=3500,
    )

    assert args.ckpt_dir == "/work/ckpt"
    assert args.result_dir == "/work/results"
    assert args.require_node == "node001"
    assert args.preferred_node == "node001"
    assert args.allowed_nodes == ["node001", "node002"]
    assert result.ckpt_dir_was_inferred is True
    assert result.inferred_resume_managed is True
    assert result.inferred_ckpt_source == "cmd"


def test_submit_preflight_forces_simple_sac_large_data_to_local():
    args = _args(require_node=None, allow_no_ckpt=True)

    result = run_submit_preflight(
        args,
        deps=_deps(simple_sac_large_data_reason=lambda cmd, cwd: "large data"),
        default_vram_mb=3500,
    )

    assert args.require_node == "local"
    assert result.notes == ["NOTE: forcing --require-node local: large data"]


def test_submit_preflight_refuses_remote_simple_sac_large_data():
    args = _args(require_node="node001")

    with pytest.raises(SubmitPreflightRefusal) as exc:
        run_submit_preflight(
            args,
            deps=_deps(simple_sac_large_data_reason=lambda cmd, cwd: "large data"),
            default_vram_mb=3500,
        )

    assert exc.value.code == 2
    assert "large local-only SimpleSAC data" in exc.value.lines[0]
    assert "large data" in exc.value.lines[1]


def test_submit_preflight_refuses_cpu_training_policy_mismatch():
    args = _args(vram=0)

    with pytest.raises(SubmitPreflightRefusal) as exc:
        run_submit_preflight(
            args,
            deps=_deps(
                cpu_training_policy_reason=lambda cmd, desc, allow, vram: "training on CPU"
            ),
            default_vram_mb=3500,
        )

    assert "scheduler CPU/GPU policy is inconsistent" in exc.value.lines[0]
    assert "training on CPU" in "\n".join(exc.value.lines)


def test_submit_preflight_refuses_short_cpu_training_justification():
    args = _args(
        allow_cpu_training=True,
        allow_no_ckpt=True,
        cpu_training_justification="too short",
    )

    with pytest.raises(SubmitPreflightRefusal) as exc:
        run_submit_preflight(args, deps=_deps(), default_vram_mb=3500)

    assert "requires --cpu-training-justification" in exc.value.lines[0]
    assert "too short" in "\n".join(exc.value.lines)


def test_submit_preflight_refuses_training_without_ckpt():
    args = _args(ckpt_dir="", allow_no_ckpt=False)

    with pytest.raises(SubmitPreflightRefusal) as exc:
        run_submit_preflight(args, deps=_deps(), default_vram_mb=3500)

    assert "cmd looks like training but --ckpt-dir is not set" in exc.value.lines[0]


def test_submit_preflight_refuses_unwired_resume_and_contract_errors():
    args = _args(ckpt_dir="/work/ckpt", allow_no_resume=False)

    with pytest.raises(SubmitPreflightRefusal) as exc:
        run_submit_preflight(
            args,
            deps=_deps(resume_capability_reason=lambda cmd, ckpt, flag, allow: "no resume flag"),
            default_vram_mb=3500,
        )
    assert "resume is not wired up" in exc.value.lines[0]

    with pytest.raises(SubmitPreflightRefusal) as exc2:
        run_submit_preflight(
            args,
            deps=_deps(checkpoint_contract_reason=lambda raw, cwd, ckpt, flag, allow: "bad contract"),
            default_vram_mb=3500,
        )
    assert "contract is not verifiable" in exc2.value.lines[0]


def test_submit_preflight_refuses_ckpt_or_result_at_cwd_root():
    with pytest.raises(SubmitPreflightRefusal) as exc:
        run_submit_preflight(
            _args(ckpt_dir="/work", allow_no_resume=True),
            deps=_deps(),
            default_vram_mb=3500,
        )
    assert "--ckpt-dir must not equal --cwd" in exc.value.lines[0]

    with pytest.raises(SubmitPreflightRefusal) as exc2:
        run_submit_preflight(
            _args(ckpt_dir="/work/ckpt", result_dir="/work", allow_no_resume=True),
            deps=_deps(),
            default_vram_mb=3500,
        )
    assert "--result-dir must not equal --cwd" in exc2.value.lines[0]
