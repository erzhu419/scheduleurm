from algorithm.experiments.native_cpu_allocation_worker_lcb_gate import (
    _idle_state_ready,
    _index_rows,
    _phase_model_ready,
)


def _row(profile=2, replicate=0):
    return {
        "workers": profile,
        "replicate": replicate,
        "measurement_valid": True,
        "completion_model_ready": True,
        "target_regime": "cpu_external_idle",
        "dispatch_preflight": {
            "ready": True,
            "dispatch_external_cpu_regime": "cpu_external_idle",
        },
        "observed_resource_state": {"external_cpu_core_equiv": 0.5},
        "completion_model": {
            "completion_model_ready": True,
            "natural_exit": True,
            "stopped_on_stable": False,
            "total_wall_s": 10.0,
            "checkpoint_observed_s": 0.2,
            "save_observed_s": 0.1,
            "progress_observation_count": 10,
        },
    }


def test_phase_and_idle_state_require_full_natural_completion_contract():
    row = _row()
    assert _phase_model_ready(row)
    assert _idle_state_ready(row)
    row["completion_model"]["save_observed_s"] = 0.0
    assert not _phase_model_ready(row)


def test_idle_state_uses_same_two_percent_boundary_as_dispatch_classifier():
    row = _row()
    row["observed_resource_state"]["external_cpu_core_equiv"] = 3.0
    assert _idle_state_ready(row)
    row["observed_resource_state"]["external_cpu_core_equiv"] = 3.85
    assert not _idle_state_ready(row)


def test_duplicate_profile_replicate_is_reported_not_overwritten():
    indexed, duplicates = _index_rows([_row(), _row()])
    assert len(indexed) == 1
    assert duplicates == [{"profile": 2, "replicate": 0}]
