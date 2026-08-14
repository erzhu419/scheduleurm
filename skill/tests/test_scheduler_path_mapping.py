from __future__ import annotations

from skill.scheduler_transport.path_mapping import (
    WindowsPrepareCommandDeps,
    remote_path_for_node,
    rewrite_command_paths_for_node,
    windows_env_spec_error,
    windows_path_for_node,
    windows_prepare_command,
    windows_rewrite_token,
)


def test_windows_path_for_node_maps_workspace_paths_and_keeps_native_paths():
    nodes = {"win": {"windows_workspace_root": r"F:\work"}}

    def is_windows(node):
        return node == "win"

    mapped = windows_path_for_node(
        "win",
        "/home/erzhu419/mine_code/Proj/sub",
        node_configs=nodes,
        node_is_windows=is_windows,
    )

    assert mapped == r"F:\work\Proj\sub"
    assert windows_path_for_node(
        "win",
        r"C:\native\file.py",
        node_configs=nodes,
        node_is_windows=is_windows,
    ) == r"C:\native\file.py"
    assert windows_path_for_node(
        "node001",
        "/home/erzhu419/mine_code/Proj/sub",
        node_configs=nodes,
        node_is_windows=is_windows,
    ) == "/home/erzhu419/mine_code/Proj/sub"


def test_remote_path_for_node_uses_configured_workspace_root():
    nodes = {
        "node001": {
            "remote_workspace_root": "/remote/work",
            "remote_path_prefixes": ["/home/erzhu419/mine_code"],
        }
    }

    def mapper(node, path):
        return windows_path_for_node(
            node,
            path,
            node_configs=nodes,
            node_is_windows=lambda n: False,
        )

    assert remote_path_for_node(
        "node001",
        "/home/erzhu419/mine_code/Proj/train.py",
        node_configs=nodes,
        canonical_node_name=lambda node: node,
        node_is_windows=lambda node: False,
        windows_path_mapper=mapper,
    ) == "/remote/work/Proj/train.py"
    assert remote_path_for_node(
        "node001",
        "relative/path.py",
        node_configs=nodes,
        canonical_node_name=lambda node: node,
        node_is_windows=lambda node: False,
        windows_path_mapper=mapper,
    ) == "relative/path.py"


def test_rewrite_command_paths_for_node_rewrites_configured_prefixes_only():
    nodes = {
        "node001": {
            "remote_workspace_root": "/remote/work",
            "remote_path_prefixes": ["/home/erzhu419/mine_code"],
        }
    }

    def remote_mapper(node, path):
        return remote_path_for_node(
            node,
            path,
            node_configs=nodes,
            canonical_node_name=lambda n: n,
            node_is_windows=lambda n: False,
            windows_path_mapper=lambda n, p: p,
        )

    rewritten = rewrite_command_paths_for_node(
        "node001",
        "python /home/erzhu419/mine_code/Proj/train.py --out /home/erzhu419/mine_code/out",
        node_configs=nodes,
        node_is_windows=lambda node: False,
        remote_path_mapper=remote_mapper,
    )

    assert rewritten == "python /remote/work/Proj/train.py --out /remote/work/out"


def test_windows_prepare_command_rewrites_pythonpath_python_paths_and_cpu_device():
    nodes = {
        "win": {
            "windows_workspace_root": r"F:\work",
            "windows_python": r"C:\Python\python.exe",
        }
    }

    def win_path(node, path):
        return windows_path_for_node(
            node,
            path,
            node_configs=nodes,
            node_is_windows=lambda n: n == "win",
        )

    deps = WindowsPrepareCommandDeps(
        node_configs=nodes,
        inject_python_u=lambda cmd: cmd.replace("python ", "python -u ", 1),
        task_launch_cpu_mode=lambda task: True,
        windows_path_for_node=win_path,
        windows_rewrite_token=lambda node, token: windows_rewrite_token(
            node,
            token,
            node_configs=nodes,
            windows_path_mapper=win_path,
        ),
    )
    task = {
        "node": "win",
        "cmd": (
            "PYTHONPATH=/home/erzhu419/mine_code/lib:/home/erzhu419/mine_code/shared "
            "python /home/erzhu419/mine_code/Proj/train.py --device cuda"
        ),
    }

    prepared = windows_prepare_command(task, deps=deps)

    assert prepared["env"]["PYTHONPATH"] == r"F:\work\lib;F:\work\shared"
    assert prepared["argv"] == [
        r"C:\Python\python.exe",
        "-u",
        r"F:\work\Proj\train.py",
        "--device",
        "cpu",
    ]


def test_windows_env_spec_error_rejects_explicit_conda_or_docker():
    assert windows_env_spec_error({"env_spec": "none"}) is None
    assert windows_env_spec_error({"env_spec": "auto"}) is None
    assert "does not support explicit env_spec" in windows_env_spec_error({"env_spec": "conda:foo"})
    assert "does not support explicit env_spec" in windows_env_spec_error({"env_spec": "docker:img"})
    assert "unsupported env_spec" in windows_env_spec_error({"env_spec": "module:foo"})
