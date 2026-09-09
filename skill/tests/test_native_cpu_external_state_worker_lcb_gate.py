import json
from pathlib import Path

from algorithm.experiments.native_cpu_external_state_worker_lcb_gate import (
    build_native_cpu_external_state_worker_lcb_gate,
)


def _row(regime, workers, replicate, wall_s, node):
    return {
        "node": node,
        "workers": workers,
        "replicate": replicate,
        "target_regime": regime,
        "measurement_valid": True,
        "completion_model_ready": True,
        "dispatch_preflight": {
            "ready": True,
            "dispatch_external_cpu_regime": regime,
        },
        "observed_resource_state": {"external_cpu_core_equiv": 24.0},
        "completion_model": {
            "completion_model_ready": True,
            "natural_exit": True,
            "stopped_on_stable": False,
            "total_wall_s": wall_s,
            "startup_overhead_s": 1.0,
            "completion_unit_s": 0.04,
            "terminal_overhead_s": 1.0,
            "checkpoint_observed_s": 0.4,
            "save_observed_s": 0.5,
            "progress_observation_count": 10,
        },
    }


def test_external_state_gate_uses_disjoint_split_and_exact_regime(tmp_path: Path):
    rows = []
    for regime, node, scale in (
        ("cpu_external_light", "node006", 1.0),
        ("cpu_external_moderate", "node004", 1.2),
    ):
        for workers in (2, 4):
            base = scale * (10.0 + workers)
            for replicate in range(13):
                rows.append(
                    _row(
                        regime,
                        workers,
                        replicate,
                        base * (1.0 + 0.002 * (replicate % 5)),
                        node,
                    )
                )
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"status": "PASS", "rows": rows}))

    result = build_native_cpu_external_state_worker_lcb_gate(
        campaign_paths=(campaign,),
        profiles=(2, 4),
    )

    assert result["status"] == "PASS"
    assert result["sample_counts_ready"] is True
    assert result["all_lower_service_valid_on_holdout"] is True
    assert {tuple(row["measured_nodes"]) for row in result["rows"]} == {
        ("node004",),
        ("node006",),
    }
