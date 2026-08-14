from __future__ import annotations

from types import SimpleNamespace

import pytest

from skill.scheduler_submit_cpu_batch import (
    CpuBatchSubmitValidationDeps,
    validate_cpu_batch_submit_spec,
)


def _deps(**overrides):
    values = dict(
        submit_policy_text=lambda cmd, cwd: f"{cmd} cwd={cwd}",
        simple_sac_large_data_reason=lambda cmd, cwd: "large data" if "large-data" in cmd else "",
        cpu_training_policy_reason=lambda policy_cmd, desc, allow, vram: (
            "training cannot run as CPU shard" if "policy-block" in policy_cmd else ""
        ),
        task_looks_like_training=lambda policy_cmd, desc: "train" in policy_cmd or "train" in (desc or ""),
        cmd_looks_like_training=lambda policy_cmd: "train" in policy_cmd,
        resume_capability_reason=lambda policy_cmd, ckpt_dir, resume_flag, allow_no_resume: (
            "missing resume flag" if "needs-resume" in policy_cmd and not allow_no_resume else ""
        ),
        checkpoint_contract_reason=lambda raw_cmd, cwd, ckpt_dir, resume_flag, allow_no_resume: (
            "contract mismatch" if "bad-contract" in raw_cmd else ""
        ),
        same_declared_path=lambda left, right: str(left or "").rstrip("/") == str(right or "").rstrip("/"),
    )
    values.update(overrides)
    return CpuBatchSubmitValidationDeps(**values)


def _args(**overrides):
    values = dict(
        allow_remote_large_data=False,
        allow_cpu_training=False,
        cpu_training_justification="",
        allow_no_ckpt=False,
        allow_no_resume=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(spec, args=None, deps=None):
    return validate_cpu_batch_submit_spec(spec, args or _args(), deps=deps or _deps())


def test_validate_cpu_batch_submit_allows_non_training_eval_spec():
    _validate({
        "cmd": "python eval.py",
        "cwd": "/work/proj",
        "description": "evaluation shard",
        "preferred_node": "node001",
        "result_dir": "/tmp/result",
    })


def test_validate_cpu_batch_submit_rejects_remote_large_local_data():
    with pytest.raises(SystemExit) as exc:
        _validate({
            "cmd": "python eval.py --large-data",
            "cwd": "/work/proj",
            "preferred_node": "node001",
        })

    assert "large local-only SimpleSAC data" in str(exc.value)

    _validate(
        {
            "cmd": "python eval.py --large-data",
            "cwd": "/work/proj",
            "preferred_node": "node001",
        },
        _args(allow_remote_large_data=True),
    )


def test_validate_cpu_batch_submit_rejects_training_policy_and_short_justification():
    with pytest.raises(SystemExit) as exc:
        _validate({"cmd": "python policy-block.py", "cwd": "/work/proj"})
    assert "policy is inconsistent" in str(exc.value)

    with pytest.raises(SystemExit) as exc:
        _validate(
            {"cmd": "python train.py", "cwd": "/work/proj", "ckpt_dir": "/ckpt"},
            _args(allow_cpu_training=True, cpu_training_justification="too short"),
        )
    assert "--cpu-training-justification" in str(exc.value)


def test_validate_cpu_batch_submit_rejects_training_without_ckpt_unless_overridden():
    with pytest.raises(SystemExit) as exc:
        _validate({"cmd": "python train.py", "cwd": "/work/proj"})
    assert "--ckpt-dir is not set" in str(exc.value)

    _validate(
        {"cmd": "python train.py", "cwd": "/work/proj"},
        _args(allow_no_ckpt=True),
    )


def test_validate_cpu_batch_submit_rejects_resume_and_contract_failures():
    with pytest.raises(SystemExit) as exc:
        _validate({"cmd": "python needs-resume.py", "cwd": "/work/proj"})
    assert "resume is not wired up" in str(exc.value)

    with pytest.raises(SystemExit) as exc:
        _validate({"cmd": "python bad-contract.py", "cwd": "/work/proj"})
    assert "checkpoint contract is not verifiable" in str(exc.value)


def test_validate_cpu_batch_submit_rejects_paths_equal_to_cwd():
    with pytest.raises(SystemExit) as exc:
        _validate({"cmd": "python eval.py", "cwd": "/work/proj", "ckpt_dir": "/work/proj/"})
    assert "--ckpt-dir must not equal --cwd" in str(exc.value)

    with pytest.raises(SystemExit) as exc:
        _validate({"cmd": "python eval.py", "cwd": "/work/proj", "result_dir": "/work/proj"})
    assert "--result-dir must not equal --cwd" in str(exc.value)
