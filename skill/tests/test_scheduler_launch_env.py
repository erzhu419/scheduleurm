from __future__ import annotations

import pytest

from skill import scheduler_launch_env as launch_env


def test_project_from_path_and_pid_are_defensive():
    assert launch_env.project_from_path("/work/foo/") == "foo"
    assert launch_env.project_from_path("") == ""
    assert launch_env.project_from_pid(
        "win",
        123,
        node_is_windows=lambda node: True,
        run_on=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    ) == ""
    assert launch_env.project_from_pid(
        "node001",
        123,
        node_is_windows=lambda node: False,
        run_on=lambda *args, **kwargs: (0, "/repo/project\n", ""),
    ) == "project"
    assert launch_env.project_from_pid(
        "node001",
        123,
        node_is_windows=lambda node: False,
        run_on=lambda *args, **kwargs: (1, "", "missing"),
    ) == ""


def test_parse_env_accepts_valid_pairs_and_rejects_unsafe_keys():
    assert launch_env.parse_env(["TOKEN=a=b", "_X=1"]) == {"TOKEN": "a=b", "_X": "1"}

    with pytest.raises(SystemExit, match="KEY=VALUE"):
        launch_env.parse_env(["NO_EQUALS"])
    with pytest.raises(SystemExit, match="not a valid POSIX"):
        launch_env.parse_env(["BAD KEY=x"])
    with pytest.raises(SystemExit, match="reserved"):
        launch_env.parse_env(["CUDA_VISIBLE_DEVICES=2"])


def test_safe_extra_env_items_filters_legacy_invalid_and_reserved_keys():
    legacy = {
        "OK": "1",
        "BAD KEY": "2",
        "CUDA_VISIBLE_DEVICES": "3",
        "SCHEDULEURM_TASK_ID": "t1",
        "SCHEDULEURM_LAUNCH_TOKEN": "user-token",
        "_ALSO_OK": "4",
    }

    assert dict(launch_env.safe_extra_env_items(legacy)) == {"OK": "1", "_ALSO_OK": "4"}
    assert list(launch_env.safe_extra_env_items(None)) == []


def test_apply_node_cmd_rewrites_uses_canonical_node_name():
    configs = {
        "node": {
            "cmd_rewrites": [
                ("/old/python", "/new/python"),
                ("", "/ignored"),
            ],
        }
    }

    rewritten = launch_env.apply_node_cmd_rewrites(
        "alias",
        "/old/python train.py",
        node_configs=configs,
        canonical_node_name=lambda node: "node" if node == "alias" else node,
    )

    assert rewritten == "/new/python train.py"


def test_node_launch_extra_env_keeps_node_cuda_env_only_for_jax_gpu_tasks():
    configs = {
        "gpu": {
            "launch_extra_env": {
                "LD_LIBRARY_PATH": "/local/cuda",
                "PATH": "/local/bin",
            }
        }
    }
    rewrite_calls = []

    def rewrite(node, value):
        rewrite_calls.append((node, value))
        return value.replace("/local", "/remote")

    non_jax = launch_env.node_launch_extra_env(
        {
            "node": "gpu",
            "project": "bapr_v15",
            "cmd": "python torch_train.py",
            "est_vram_mb": 1024,
            "extra_env": {"USER_FLAG": "1"},
        },
        node_configs=configs,
        rewrite_command_paths_for_node=rewrite,
        default_vram_mb=512,
        task_launch_cpu_mode=lambda task: False,
    )
    assert non_jax == {"USER_FLAG": "1"}

    jax = launch_env.node_launch_extra_env(
        {
            "node": "gpu",
            "project": "RE-SAC",
            "cmd": "python train.py",
            "cwd": "/local/repo",
            "est_vram_mb": 1024,
        },
        node_configs=configs,
        rewrite_command_paths_for_node=rewrite,
        default_vram_mb=512,
        task_launch_cpu_mode=lambda task: False,
    )

    assert jax["LD_LIBRARY_PATH"] == "/remote/cuda"
    assert jax["PATH"] == "/remote/bin"
    assert jax["PYTHONPATH"] == "/remote/repo"
    assert jax["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert ("gpu", "/local/cuda") in rewrite_calls


def test_launch_extra_env_skips_jax_defaults_for_cpu_mode():
    env = launch_env.launch_extra_env(
        {
            "project": "RE-SAC",
            "cmd": "python jax_train.py",
            "cwd": "/repo",
            "est_vram_mb": 1024,
            "extra_env": {"KEEP": "yes"},
        },
        default_vram_mb=512,
        task_launch_cpu_mode=lambda task: True,
    )

    assert env == {"KEEP": "yes"}
