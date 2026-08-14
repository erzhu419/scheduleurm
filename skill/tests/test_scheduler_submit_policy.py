from __future__ import annotations

import argparse

from skill import scheduler_submit_policy as policy


class _EtaTracker:
    @staticmethod
    def _extract_current_only_from_tail(text):
        return 1105

    @staticmethod
    def parse_progress(text, cmd=None):
        return None


def test_split_bapr_seed_batch_rewrites_each_seed():
    args = argparse.Namespace(
        cmd=("bash -lc 'set -euo pipefail; for seed in 0 1 2; do "
             "bash /tmp/bapr/run_seed.sh sac Walker2d-v2 \"$seed\" 1500 60 min paper; done'"),
        cwd="/work/BAPR",
        signature="BAPR/paper/Walker2d/sac",
        project="BAPR",
        description="BAPR seeds",
        ckpt_dir="/runs/$seed/ckpt",
        result_dir="/runs/${seed}/out",
        local_result_dir=None,
        wait_for_files=["/ready/$seed.ok"],
        allow_seed_batch=False,
    )

    expanded = policy.split_bapr_seed_batch_submit_args(args)

    assert len(expanded) == 3
    assert [item.signature.rsplit("/", 1)[-1] for item in expanded] == ["s0", "s1", "s2"]
    assert "for seed in" not in expanded[1].cmd
    assert 'Walker2d-v2 "1" 1500' in expanded[1].cmd
    assert expanded[2].ckpt_dir == "/runs/2/ckpt"
    assert expanded[2].result_dir == "/runs/2/out"
    assert expanded[2].wait_for_files == ["/ready/2.ok"]


def test_infer_bapr_run_seed_checkpoint_and_result_dirs():
    cmd = ("bash -lc 'set -euo pipefail; : --resume; "
           "for seed in 0 1; do bash /tmp/bapr_v15/run_seed.sh "
           "sac Hopper-v2 \"$seed\" 1500 60 min paper_negative_v2; done'")

    dirs = policy.infer_bapr_result_dirs_from_cmd(cmd, "/work/BAPR")
    ckpt = policy.infer_bapr_run_seed_ckpt(
        "bash -lc 'bash /tmp/bapr_v15/run_seed.sh sac Hopper-v2 1 1500 60 min paper_negative_v2'",
        "/work/BAPR",
    )

    assert dirs == [
        "/tmp/bapr_v15/jax_experiments/results_paper/paper_negative_v2_sac_Hopper_dw60_s0",
        "/tmp/bapr_v15/jax_experiments/results_paper/paper_negative_v2_sac_Hopper_dw60_s1",
    ]
    assert ckpt["resume_managed_by_cmd"] is True
    assert ckpt["ckpt_dir"].endswith("paper_negative_v2_sac_Hopper_dw60_s1/checkpoints")


def test_h2oplus_dispatch_runner_is_recognized_as_command_managed_resume(tmp_path):
    reproducibility = tmp_path / "reproducibility"
    reproducibility.mkdir()
    dispatch = reproducibility / "run_training_dispatch_v7.py"
    delegated = reproducibility / "run_training_dispatch_v6.py"
    matrix = reproducibility / "run_training_matrix_v4.py"
    dispatch.write_text(
        "from reproducibility.run_training_dispatch_v6 import main\n"
    )
    delegated.write_text(
        "from reproducibility.run_training_matrix_v4 import load_plan\n"
    )
    matrix.write_text("# resumable matrix implementation\n")

    cmd = "python reproducibility/run_training_dispatch_v7.py --dispatch plan.json"
    assert policy.cmd_has_resume_flag(cmd) is True
    assert policy.cmd_has_resume_flag(
        "python reproducibility/run_training_dispatch_preview.py"
    ) is False
    assert policy.candidate_training_source_paths(cmd, str(tmp_path)) == [
        str(dispatch),
        str(delegated),
        str(matrix),
    ]
    assert policy.infer_checkpoint_from_submit(cmd, str(tmp_path)) == {
        "resume_managed_by_cmd": True,
        "source": "command-managed:run_training_dispatch_v7.py",
    }


def test_epoch_and_total_updates_satisfy_full_resume_progress_contract(tmp_path):
    runner = tmp_path / "run_training_matrix_v4.py"
    runner.write_text(
        """
def run(path, trainer, optimizer, replay_buffer):
    if path.exists():
        state = torch.load(path)
        start_epoch = state['epoch']
        trainer.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        replay_buffer = pickle.loads(state['replay_buffer'])
        total_updates = state['total_updates']
    for epoch in range(start_epoch, 10):
        total_updates += 1
    torch.save({'epoch': epoch + 1, 'total_updates': total_updates}, path)
"""
    )

    cmd = "python run_training_matrix_v4.py"
    assert policy.checkpoint_contract_reason(
        cmd,
        str(tmp_path),
        str(tmp_path / "checkpoints"),
        "",
        False,
    ) is None


def test_bapr_batch_projection_counts_completed_seeds():
    cmd = ("bash -lc 'set -euo pipefail; for seed in 0 1 2 3 4; do "
           "bash /tmp/bapr_v15/run_seed.sh sac Ant-v2 \"$seed\" 1500 60 min paper_full; done'")
    tail = (
        "Iter 1105 | Reward: 123.4\n"
        "Results saved to: /tmp/bapr_v15/jax_experiments/results_paper/"
        "paper_full_sac_Ant_dw60_s0/logs\n"
        "Results saved to: /tmp/bapr_v15/jax_experiments/results_paper/"
        "paper_full_sac_Ant_dw60_s1/logs\n"
    )

    projection = policy.bapr_batch_projection(
        {"cmd": cmd, "cwd": "/work/BAPR"},
        tail,
        elapsed_s=82000,
        load_eta_tracker_module=lambda: _EtaTracker,
        clean_result_path=lambda raw: raw.rstrip(".,;)]}"),
    )

    assert projection["source"] == "bapr_seed_batch"
    assert projection["completed_seeds"] == 2
    assert projection["seed_count"] == 5
    assert projection["current"] == 4105
    assert projection["total_units"] == 7500
    assert projection["eta_s"] > 0


def test_cpu_training_guard_requires_scheduler_level_intent():
    assert policy.cpu_training_policy_reason(
        "python train_iql_bus.py --device cpu",
        "train",
        False,
        0,
    ).startswith("training-looking task has vram=0")
    assert policy.cpu_training_policy_reason(
        "python train_iql_bus.py --device cpu",
        "train",
        True,
        0,
    ) is None
    assert "would reserve a GPU" in policy.cpu_training_policy_reason(
        "python train_iql_bus.py --device cpu",
        "train",
        True,
        1000,
    )


def test_training_description_detection_does_not_match_constrained_algorithm_names():
    description = (
        "SC-OLH structural backend "
        "backends/common_sobol_constrained_ei seed=0"
    )

    assert policy.task_looks_like_training("python run_lodo_manifest_shard.py", description) is False
    assert policy.cpu_training_policy_reason(
        "python run_lodo_manifest_shard.py",
        description,
        False,
        0,
    ) is None
    assert policy.task_looks_like_training("python worker.py", "CPU training seed=0") is True
