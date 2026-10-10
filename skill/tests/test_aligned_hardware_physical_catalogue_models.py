import json

import pytest

from algorithm.experiments import unified_hardware_or_replay as replay
from algorithm.experiments.aligned_hardware_frame_model import WORKLOADS, load_model
from algorithm.experiments.aligned_hardware_physical_catalogue_models import load_catalogues


def test_original_catalogue_is_unchanged_field_for_field():
    model = next(model for model in load_catalogues() if model.node == "gpu_b")
    assert model.snapshot() == load_model().snapshot()


def test_all_measured_nodes_are_included_with_natural_widths_and_lane_counts():
    models = {model.node: model for model in load_catalogues()}
    assert set(models) == {"gpu_b", "gpu_a", "cpu_a", "node007"}
    assert {node: (model.lanes, len(model.options)) for node, model in models.items()} == {
        "gpu_b": (2, 9), "gpu_a": (2, 9), "cpu_a": (2, 8), "node007": (4, 9),
    }
    assert {node: max(option.requirements[2] for option in model.options)
            for node, model in models.items()} == {
                "gpu_b": 8, "gpu_a": 8, "cpu_a": 5, "node007": 10,
            }
    assert all(model.workloads == WORKLOADS and model.scales == (60.0, 40.0, 120.0)
               for model in models.values())
    assert len({tuple(option.requirements for option in model.options)
                for model in models.values()}) == 3


def test_loaded_phase_accounting_includes_measured_nonzero_resident_drain():
    drained = []
    for model in load_catalogues():
        assert len(model.excluded_actions) == 1
        for option in model.options:
            assert tuple(map(len, option.completion_offsets)) == option.requirements
            assert all(0 < time <= option.duration for times in option.completion_offsets for time in times)
            assert all(model.node in source for source in option.source_action_ids)
            if not option.loaded:
                assert option.duration == pytest.approx(sum(dict(option.phases).values()))
                continue
            phases = dict(option.phases)
            assert option.duration == pytest.approx(sum(phases[key] for key in (
                "resident_initialization_s", "target_overlap_s",
                "resident_dedicated_drain_s", "resident_finalization_s",
            )))
            resident_id = option.source_action_ids[1]
            resident = next(item for item in model.options if item.action_id == resident_id)
            coordinate = resident.requirements.index(1)
            assert phases["resident_remaining_native_units"] == pytest.approx(max(
                0.0, model.scales[coordinate]
                - phases["resident_overlap_rate_native_per_s"] * phases["target_overlap_s"],
            ))
            assert phases["resident_dedicated_drain_s"] == pytest.approx(
                phases["resident_remaining_native_units"] * dict(resident.phases)["loop_s"]
                / model.scales[coordinate]
            )
            if phases["resident_dedicated_drain_s"] > 0:
                drained.append((model.node, option.action_id))
    assert drained == [("cpu_a", "gpu_loaded:cpu_a:llm_after_cnn")]


def _cache(tmp_path, mutate):
    payload = json.loads(replay.DEFAULT_CACHE.read_text())
    mutate(payload["records"])
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(payload))
    return path


def _cnn_singleton(row):
    return (row["node_bucket"] == "gpu_2080_8gb_dual:cpu_a"
            and row["workload_key"] == WORKLOADS[0]
            and row["resource_state"] == "empty" and row["colocation_count"] == 1)


def test_missing_local_resident_p1_is_rejected_even_when_other_nodes_have_one(tmp_path):
    path = _cache(tmp_path, lambda rows: rows.__setitem__(slice(None),
                  [row for row in rows if not _cnn_singleton(row)]))
    with pytest.raises(ValueError, match="cpu_a.*requires one dedicated p1"):
        load_catalogues(path)


def test_a_node_uses_its_own_rates_without_changing_other_catalogues(tmp_path):
    before = {model.node: model.snapshot() for model in load_catalogues()}

    def mutate(rows):
        for row in rows:
            if _cnn_singleton(row):
                row["completion_unit_s"] *= 2

    path = _cache(tmp_path, mutate)
    after = {model.node: model.snapshot() for model in load_catalogues(path)}
    assert before["cpu_a"] != after["cpu_a"]
    assert all(before[node] == after[node] for node in before if node != "cpu_a")


@pytest.mark.parametrize("field,value", [
    ("workload_env", "different_native_workload"), ("unit", "different_native_unit"),
])
def test_loaded_and_dedicated_environments_and_units_must_agree(tmp_path, field, value):
    def mutate(rows):
        for row in rows:
            if _cnn_singleton(row):
                row[field] = value

    path = _cache(tmp_path, mutate)
    with pytest.raises(ValueError, match="environment or native unit mismatch"):
        load_catalogues(path)


def test_native_job_size_is_not_renormalized_to_fit_an_inconsistent_profile(tmp_path):
    def mutate(rows):
        for row in rows:
            if (row["node_bucket"] == "gpu_2080_8gb_dual:cpu_a"
                    and row["workload_key"] == WORKLOADS[2]
                    and row["resource_state"] == "empty" and row["colocation_count"] == 5):
                row["completion_group_total_units"] += 5

    path = _cache(tmp_path, mutate)
    with pytest.raises(ValueError, match="dedicated native job sizes disagree"):
        load_catalogues(path)
