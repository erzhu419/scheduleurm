from algorithm.experiments import critical_gpu_completion_campaign as campaign_module
from algorithm.experiments.critical_gpu_completion_campaign import (
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    NODE_SPECS,
    TRAINING_WAVES,
    WORKLOAD_SPECS,
    _admission_audit,
    _coordination_audit,
    build_critical_gpu_completion_campaign,
    campaign_cells,
    workload_profiles,
    wave_role,
)


def test_pre_registered_split_is_disjoint_and_complete():
    assert set(TRAINING_WAVES).isdisjoint(CALIBRATION_WAVES)
    assert HOLDOUT_WAVE not in set(TRAINING_WAVES) | set(CALIBRATION_WAVES)
    assert {wave_role(i) for i in range(14)} == {
        "smoke_excluded",
        "training",
        "calibration",
        "holdout",
    }


def test_representative_nodes_have_exact_nine_policy_reachable_actions():
    for node in ("jtl110gpu", "jtl311linux", "node007"):
        cells = campaign_cells(node=node, wave=1)
        assert len(cells) == 9
        assert sum(cell["total_task_count"] for cell in cells) > 0
        assert all(cell["profile_axis"] == "tasks_per_gpu" for cell in cells)


def test_hardware_local_llm_profiles_respect_measured_capacity():
    llm = next(spec for spec in WORKLOAD_SPECS if spec.workload_key == "gpu_llm_distilgpt2")
    assert workload_profiles("jtl110gpu", llm) == (1, 3, 8)
    assert workload_profiles("jtl110gpu2", llm) == (1, 3, 8)
    assert workload_profiles("node007", llm) == (1, 3, 10)
    assert workload_profiles("jtl311linux", llm) == (1, 3, 5)


def test_hardware_classes_are_not_collapsed():
    assert NODE_SPECS["jtl110gpu"].node_bucket != NODE_SPECS["node007"].node_bucket
    assert NODE_SPECS["jtl311linux"].node_bucket != NODE_SPECS["node007"].node_bucket
    assert NODE_SPECS["jtl110gpu2"].role == "homogeneous_equivalence"


def test_every_workload_naturally_completes_and_saves():
    assert {spec.workload_key for spec in WORKLOAD_SPECS} == {
        "gpu_heavy_jax_matmul",
        "gpu_cnn_torch_resnet50",
        "gpu_llm_distilgpt2",
        "hybrid_rl_resac_ant",
    }
    assert all(spec.max_iters > spec.stable_skip_samples for spec in WORKLOAD_SPECS)
    assert all(spec.timeout_s >= 1800 for spec in WORKLOAD_SPECS)
    assert all(spec.coordinated_start_barrier for spec in WORKLOAD_SPECS[:3])
    assert WORKLOAD_SPECS[3].coordinated_start_barrier is False
    assert WORKLOAD_SPECS[3].max_iters == 40
    assert WORKLOAD_SPECS[3].stable_cycle_units == 5
    assert WORKLOAD_SPECS[3].admission_stagger_s == 30.0
    assert WORKLOAD_SPECS[3].admission_stagger_axis == "global_index"
    assert WORKLOAD_SPECS[3].admission_stagger_min_profile == 5


def test_coordination_audit_requires_both_markers_per_child(tmp_path):
    spec = WORKLOAD_SPECS[0]
    logs = []
    for index in range(2):
        path = tmp_path / f"child-{index}.log"
        path.write_text(
            "ScheduleurmPhase name=coordination_barrier event=start\n"
            "ScheduleurmPhase name=coordination_barrier event=end\n",
            encoding="utf-8",
        )
        logs.append({"log_path": str(path)})

    assert _coordination_audit(spec, {"rows": logs}, expected_tasks=2)["ready"]
    (tmp_path / "child-1.log").write_text(
        "ScheduleurmPhase name=coordination_barrier event=start\n",
        encoding="utf-8",
    )
    assert not _coordination_audit(spec, {"rows": logs}, expected_tasks=2)["ready"]


def test_admission_audit_requires_delay_to_be_present_in_completion_jct():
    spec = WORKLOAD_SPECS[3]
    summary = {
        "rows": [
            {"global_index": 0, "idx": 0, "completion_model": {"phase_durations_s": {}}},
            {"global_index": 1, "idx": 1, "completion_model": {"phase_durations_s": {"admission_delay": [30.1]}}},
        ]
    }
    assert _admission_audit(spec, summary, profile=5)["ready"]
    summary["rows"][1]["completion_model"]["phase_durations_s"]["admission_delay"] = [2.0]
    assert not _admission_audit(spec, summary, profile=5)["ready"]


def test_filtered_smoke_artifact_cannot_overwrite_full_wave_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_module, "ARTIFACT_ROOT", tmp_path)
    full = build_critical_gpu_completion_campaign(node="jtl110gpu", wave=0)
    filtered = build_critical_gpu_completion_campaign(
        node="jtl110gpu",
        wave=0,
        workload_keys=["gpu_llm_distilgpt2"],
        profiles=[1],
    )

    assert full["campaign_id"] != filtered["campaign_id"]
    assert filtered["selection"] == {
        "workload_keys": ["gpu_llm_distilgpt2"],
        "profiles": [1],
        "filtered": True,
    }
