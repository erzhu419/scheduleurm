import json

from algorithm.experiments import critical_gpu_completion_series as series
from algorithm.experiments.critical_gpu_completion_campaign import (
    PROTOCOL,
    campaign_cells,
)


def _pass_report(node: str, wave: int):
    return {
        "measurement_protocol": PROTOCOL,
        "campaign_id": f"campaign-{node}-{wave}",
        "status": "PASS",
        "pass": True,
        "rows": [
            {**cell, "ready": True, "capacity_boundary": False}
            for cell in campaign_cells(node=node, wave=wave)
        ],
    }


def test_completed_wave_requires_exact_registered_cell_set(tmp_path, monkeypatch):
    monkeypatch.setattr(series, "ARTIFACT_ROOT", tmp_path)
    path = series.wave_artifact("jtl110gpu", 1)
    path.write_text(json.dumps(_pass_report("jtl110gpu", 1)), encoding="utf-8")
    assert series.completed_wave("jtl110gpu", 1) is not None

    report = _pass_report("jtl110gpu", 1)
    report["rows"].pop()
    path.write_text(json.dumps(report), encoding="utf-8")
    assert series.completed_wave("jtl110gpu", 1) is None


def test_series_skips_only_strict_pass_and_stops_on_first_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(series, "ARTIFACT_ROOT", tmp_path)
    path = series.wave_artifact("jtl110gpu", 1)
    path.write_text(json.dumps(_pass_report("jtl110gpu", 1)), encoding="utf-8")
    calls = []

    def fake_campaign(*, node, wave, allow_launch):
        calls.append((node, wave, allow_launch))
        return {"campaign_id": f"c{wave}", "status": "INCOMPLETE", "ready_row_count": 4}

    monkeypatch.setattr(series, "build_critical_gpu_completion_campaign", fake_campaign)
    report = series.run_series(node="jtl110gpu", first_wave=1, last_wave=3, allow_launch=True)

    assert calls == [("jtl110gpu", 2, True)]
    assert [row["status"] for row in report["waves"]] == [
        "SKIPPED_ALREADY_PASS",
        "INCOMPLETE",
    ]
    assert report["pass"] is False
