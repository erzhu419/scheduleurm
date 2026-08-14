from __future__ import annotations

import argparse
import json


class FakeLocalBackend:
    def batch_probe(self, state):
        return {
            "profile-local": {
                "state": "alive",
                "vram_mb": 12,
                "ram_mb": 34,
                "pcpu": 120.0,
            }
        }


def test_cmd_profile_local_records_log_and_history(tmp_path, monkeypatch, capsys, sch):
    log_path = tmp_path / "profile.log"
    monkeypatch.setattr(sch, "LocalBackend", FakeLocalBackend)
    monkeypatch.setattr(
        sch,
        "_runtime_profile_from_log",
        lambda *args, **kwargs: {"source": "test", "total_s": 7, "eta_s": 0},
    )

    args = argparse.Namespace(
        cwd=str(tmp_path),
        project="Demo",
        signature="Demo/profile",
        log_path=str(log_path),
        env=[],
        cmd="python3 -c 'print(123)'",
        description="profile smoke",
        sample_interval=1,
        timeout=0,
        json=True,
    )

    sch.cmd_profile_local(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["log_path"] == str(log_path)
    text = log_path.read_text()
    assert "scheduleurm profile-local start" in text
    assert "scheduleurm profile-local end rc=0" in text
    hist = sch.load_history()
    assert hist["Demo/profile"]["vram_mb"] == 12
    assert hist["Demo/profile"]["ram_mb"] == 34
    assert hist["Demo/profile"]["cpu_cores"] == 2
    runtime = sch.load_runtime_history()
    actual_duration = max(1, int(payload["duration_s"]))
    assert any(
        rec.get("source") == "duration" and rec.get("total_s") == actual_duration
        for rec in runtime.values()
    )
