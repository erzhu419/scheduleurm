from __future__ import annotations

import argparse

import pytest


def _adopt_args(**overrides):
    data = {
        "node": "node001",
        "pids": [11, 12],
        "gpu": 0,
        "allow_multi_project": False,
        "est_vram": None,
        "project": "ProjectA",
        "cwd": "/work/ProjectA",
        "signature": "ProjectA/manual",
        "description": "manual adopted job",
        "ckpt_dir": None,
        "log_path": None,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_cmd_adopt_records_manual_process_resources(monkeypatch, capsys, sch):
    calls = []

    def fake_run_on(node, cmd, **kwargs):
        calls.append(cmd)
        if "nvidia-smi --query-compute-apps" in cmd:
            return (
                0,
                "\n".join([
                    "ALIVE_11",
                    "ALIVE_12",
                    "===PROC===",
                    "0000:01:00.0, 11, 1500",
                    "0000:01:00.0, 12, 500",
                    "===GPU===",
                    "0, 0000:01:00.0",
                ]),
                "",
            )
        if "readlink /proc/11/cwd" in cmd:
            return 0, "/work/ProjectA\n", ""
        if "readlink /proc/12/cwd" in cmd:
            return 0, "/work/ProjectA\n", ""
        if "VmRSS" in cmd and "pcpu" in cmd:
            return 0, "11|300|120.5\n12|100|20.0\n", ""
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(sch, "run_on", fake_run_on)
    monkeypatch.setattr(sch, "_local_user", lambda: "alice")

    sch.cmd_adopt(_adopt_args())

    out = capsys.readouterr().out
    assert "adopted t0001" in out
    task = sch.load_state()["tasks"][0]
    assert task["status"] == "running"
    assert task["origin"] == "manual-adopt"
    assert task["auto_adopted"] is True
    assert task["remote_pids"] == [11, 12]
    assert task["gpu_idx"] == 0
    assert task["peak_vram_mb"] == 2000
    assert task["current_vram_mb"] == 2000
    assert task["peak_ram_mb"] == 400
    assert task["current_ram_mb"] == 400
    assert task["current_pcpu"] == pytest.approx(140.5)
    assert task["submitted_by"] == "alice"
    assert any("VmRSS" in cmd for cmd in calls)


def test_cmd_adopt_rejects_multi_project_pid_bundle(monkeypatch, sch):
    def fake_run_on(node, cmd, **kwargs):
        if "nvidia-smi --query-compute-apps" in cmd:
            return (
                0,
                "\n".join([
                    "ALIVE_11",
                    "ALIVE_12",
                    "===PROC===",
                    "0000:01:00.0, 11, 1500",
                    "0000:01:00.0, 12, 500",
                    "===GPU===",
                    "0, 0000:01:00.0",
                ]),
                "",
            )
        if "readlink /proc/11/cwd" in cmd:
            return 0, "/work/ProjectA\n", ""
        if "readlink /proc/12/cwd" in cmd:
            return 0, "/work/ProjectB\n", ""
        return 0, "", ""

    monkeypatch.setattr(sch, "run_on", fake_run_on)

    with pytest.raises(SystemExit, match="different projects"):
        sch.cmd_adopt(_adopt_args(project="", cwd=""))

    assert sch.load_state()["tasks"] == []
