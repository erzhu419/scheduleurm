from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def inventory(home, nodes_file=None):
    env = dict(os.environ, HOME=str(home), PYTHONPATH=str(ROOT / "skill"))
    env.pop("SCHEDULEURM_NODES_FILE", None)
    if nodes_file is not None:
        env["SCHEDULEURM_NODES_FILE"] = str(nodes_file)
    return subprocess.run(
        [sys.executable, "-c", "import json; from scheduler_node.inventory import NODES; print(json.dumps(NODES))"],
        env=env, capture_output=True, text=True, timeout=10,
    )


def test_clean_install_enables_only_local(tmp_path):
    result = inventory(tmp_path)
    assert result.returncode == 0, result.stderr
    nodes = json.loads(result.stdout)
    assert list(nodes) == ["local"]
    assert nodes["local"]["host"] is None
    assert nodes["local"]["cpu_cores"] == 1


def test_default_private_file_replaces_inventory(tmp_path):
    path = tmp_path / ".claude" / "scheduler" / "nodes.json"
    path.parent.mkdir(parents=True)
    expected = {"compute-a": {"host": "compute-a", "cpu_cores": 32, "ram_mb": 0}}
    path.write_text(json.dumps(expected))
    result = inventory(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_explicit_nodes_file_overrides_default(tmp_path):
    default = tmp_path / ".claude" / "scheduler" / "nodes.json"
    default.parent.mkdir(parents=True)
    default.write_text('{"unused": {"host": "unused"}}')
    path = tmp_path / "cluster.json"
    expected = {"compute-b": {"host": "compute-b", "cpu_cores": 16}}
    path.write_text(json.dumps(expected))
    result = inventory(tmp_path, path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_explicit_missing_file_is_not_silent_local_fallback(tmp_path):
    result = inventory(tmp_path, tmp_path / "missing.json")
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr


def test_invalid_nodes_file_reports_configuration_error(tmp_path):
    path = tmp_path / "bad.json"
    for value in ([], {}, {"compute-a": "compute-a"}):
        path.write_text(json.dumps(value))
        result = inventory(tmp_path, path)
        assert result.returncode != 0
        assert "ValueError" in result.stderr
