from __future__ import annotations

import pytest

from algorithm.experiments import critical_gpu_loaded_completion_campaign as campaign


@pytest.fixture(autouse=True)
def _ready_environment(monkeypatch):
    monkeypatch.setattr(
        campaign,
        "_node_environment_gate",
        lambda _node_spec: {"ready": True, "reason": "test"},
    )


def test_loaded_campaign_is_finite_policy_reachable_and_no_touch(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)

    report = campaign.build_loaded_completion_campaign(node="node007", wave=1)

    assert report["status"] == "MANIFEST_ONLY"
    assert report["allow_launch"] is False
    assert report["ordinary_running_tasks_touched"] is False
    assert report["legacy_scheduler_limits_bypassed"] is True
    assert report["natural_completion_required"] is True
    assert report["task_native_progress_required"] is True
    assert report["full_resident_target_overlap_required"] is True
    assert report["terminate_on_stable"] is False
    assert report["scenario_parallelism"] == 1
    assert report["scenario_parallelism_scope"] == (
        "sequential_independent_action_measurement"
    )
    assert report["cross_node_parallelism_allowed"] is True
    assert len(report["measurement_code_manifest"]["sha256"]) == 64
    assert report["pre_registered_split"] == {
        "training": [1, 2, 3],
        "calibration": list(range(4, 13)),
        "holdout": 13,
    }
    assert [row["scenario_id"] for row in report["rows"]] == [
        "cnn_after_llm",
        "llm_after_cnn",
        "rl_after_cnn",
        "cnn_after_rl",
    ]
    assert all(row["resource_state"] == "mixed_colocation" for row in report["rows"])
    assert [row["resident_total_units"] for row in report["rows"]] == [
        12000,
        600,
        18000,
        80,
    ]
    assert [row["target_total_units"] for row in report["rows"]] == [
        300,
        600,
        40,
        3000,
    ]
    assert [row["resident_mix"] for row in report["rows"]] == [
        "resident_llm_target_cnn",
        "resident_cnn_target_llm",
        "resident_cnn_target_hybrid_rl",
        "resident_hybrid_rl_target_cnn",
    ]
    assert all(row["status"] == "planned" for row in report["rows"])


def test_loaded_campaign_preregisters_node_specific_overlap_duration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)

    report = campaign.build_loaded_completion_campaign(
        node="jtl110gpu2", wave=1
    )
    rows = {row["scenario_id"]: row for row in report["rows"]}

    assert report["registered_resident_total_units"]["rl_after_cnn"] == 30_000
    assert rows["rl_after_cnn"]["resident_total_units"] == 30_000
    assert rows["cnn_after_llm"]["resident_total_units"] == 12_000
    assert campaign._resident_total_units(
        "jtl110gpu", next(row for row in campaign.SCENARIOS if row.scenario_id == "rl_after_cnn")
    ) == 18_000


def test_llm_template_isolates_node_specific_pythonpath():
    template = (campaign.TEMPLATE_ROOT / "torch_llm_distilgpt2.cmd.tpl").read_text(
        encoding="utf-8"
    )

    assert 'export PYTHONPATH="{workload_pythonpath}"' in template
    assert "--local-files-only" in template
    assert "--require-transformers" in template


def test_loaded_campaign_rejects_unknown_scenario(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="unknown scenarios"):
        campaign.build_loaded_completion_campaign(
            node="node007",
            wave=1,
            scenario_ids=["not_registered"],
        )


def test_filtered_smoke_cannot_overwrite_full_wave_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)

    full = campaign.build_loaded_completion_campaign(node="node007", wave=1)
    filtered = campaign.build_loaded_completion_campaign(
        node="node007",
        wave=1,
        scenario_ids=["cnn_after_llm"],
    )

    assert full["campaign_id"] != filtered["campaign_id"]
    assert filtered["selection"] == {
        "scenario_ids": ["cnn_after_llm"],
        "filtered": True,
    }
    assert (tmp_path / f"{full['campaign_id']}.json").is_file()
    assert (tmp_path / f"{filtered['campaign_id']}.json").is_file()


def test_marker_parser_preserves_remote_exit_and_overlap_evidence():
    markers = campaign._markers(
        "\n".join(
            (
                "__RESIDENT_RC__ 0",
                "__TARGET_RC__ 139",
                "__RESIDENT_ALIVE_AT_TARGET_START__ 1",
                "__RESIDENT_ALIVE_AT_TARGET_END__ 1",
                "__RESIDENT_READY_OBSERVATIONS__ 7",
            )
        )
    )

    assert markers == {
        "RESIDENT_RC": 0,
        "TARGET_RC": 139,
        "RESIDENT_ALIVE_AT_TARGET_START": 1,
        "RESIDENT_ALIVE_AT_TARGET_END": 1,
        "RESIDENT_READY_OBSERVATIONS": 7,
    }


def test_resident_overlap_service_is_conservative_and_marker_bounded():
    result = campaign._resident_overlap_service(
        "\n".join(
            (
                "ScheduleurmProgress Step 10/100 rate=20 step/s",
                "ScheduleurmColocation event=target_start epoch=100.0",
                "ScheduleurmProgress Step 20/100 rate=18 step/s",
                "ScheduleurmProgress Step 35/100 rate=16 step/s",
                "ScheduleurmProgress Step 50/100 rate=14 step/s",
                "ScheduleurmColocation event=target_end epoch=102.0",
                "ScheduleurmProgress Step 80/100 rate=30 step/s",
            )
        )
    )

    assert result["ready"] is True
    assert result["progress_observation_count"] == 3
    assert result["completed_units_lower"] == 30
    assert result["conservative_rate_units_per_s"] == 15.0
    assert result["reported_rate_median_units_per_s"] == 16.0


def test_resident_overlap_service_uses_remote_line_and_nanosecond_bounds():
    result = campaign._resident_overlap_service(
        "\n".join(
            (
                "initialization",
                "ScheduleurmProgress Step 10/100 rate=20 step/s",
                "ScheduleurmProgress Step 20/100 rate=18 step/s",
                "ScheduleurmProgress Step 35/100 rate=16 step/s",
                "ScheduleurmProgress Step 50/100 rate=14 step/s",
                "ScheduleurmProgress Step 80/100 rate=30 step/s",
            )
        ),
        start_line=2,
        end_line=5,
        start_epoch_s=100.0,
        end_epoch_s=102.0,
    )

    assert result["ready"] is True
    assert result["boundary_source"] == "remote_line_count_and_nanosecond_bounds"
    assert result["progress_observation_count"] == 3
    assert result["completed_units_lower"] == 30
    assert result["conservative_rate_units_per_s"] == 15.0


def test_sequential_scenarios_rotate_physical_gpu_between_waves():
    node007_w1 = campaign._scenario_gpu_assignments(
        campaign.SCENARIOS, (0, 1, 2, 3), wave=1
    )
    node007_w2 = campaign._scenario_gpu_assignments(
        campaign.SCENARIOS, (0, 1, 2, 3), wave=2
    )
    jtl311_w1 = campaign._scenario_gpu_assignments(
        campaign.SCENARIOS, (0, 1), wave=1
    )
    jtl311_w2 = campaign._scenario_gpu_assignments(
        campaign.SCENARIOS, (0, 1), wave=2
    )

    assert [gpu for _, gpu in node007_w1] == [0, 1, 2, 3]
    assert [gpu for _, gpu in node007_w2] == [1, 2, 3, 0]
    assert [gpu for _, gpu in jtl311_w1] == [0, 1, 0, 1]
    assert [gpu for _, gpu in jtl311_w2] == [1, 0, 1, 0]


def test_split_contract_matches_pre_registered_conformal_roles():
    assert [campaign._split_role(wave) for wave in campaign.TRAINING_WAVES] == [
        "training"
    ] * 3
    assert [campaign._split_role(wave) for wave in campaign.CALIBRATION_WAVES] == [
        "calibration"
    ] * 9
    assert campaign._split_role(campaign.HOLDOUT_WAVE) == "holdout"


def test_remote_script_requires_natural_exit_and_observed_overlap():
    script = campaign._remote_script(
        gpu=0,
        remote_logs="/tmp/scheduleurm-loaded-test",
        resident_cmd="python resident.py",
        target_cmd="python target.py",
        resident_timeout_s=100,
        target_timeout_s=50,
        readiness_timeout_s=20,
        readiness_observations=3,
    )

    assert 'wait "$RESIDENT_PID"' in script
    assert "RESIDENT_ALIVE_AT_TARGET_START" in script
    assert "RESIDENT_ALIVE_AT_TARGET_END" in script
    assert "RESIDENT_READY_OBSERVATIONS" in script
    assert "TARGET_START_NS" in script
    assert "TARGET_END_NS" in script
    assert "TARGET_START_LINE" in script
    assert "TARGET_END_LINE" in script
    assert "__HOST_SAMPLE__" in script
    assert "__USER_PROC_SNAPSHOT__" in script
    assert "__USER_PROC__" in script
    assert "MONITOR_SAMPLE_INDEX % 5" in script
    assert "MemAvailable:" in script
    assert "sleep 0.25" in script
    assert "ScheduleurmColocation event=target_start" not in script
    assert "ScheduleurmColocation event=target_end" not in script
    assert "kill -TERM" not in script
