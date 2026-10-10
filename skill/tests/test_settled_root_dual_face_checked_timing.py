"""Checked auditing keeps the original data registration and eight-arm design."""
from copy import deepcopy
import json

import pytest

from algorithm.experiments import aligned_hardware_settled_root_dual_face_checked_timing as timing
from algorithm.experiments import aligned_hardware_settled_root_dual_face_timing as original
from algorithm.experiments import settled_root_dual_face_checked_development_summary as auditor
from algorithm.experiments import settled_root_dual_face_checked_timing_summary as summary
from skill.tests.test_settled_root_dual_face_timing import frozen_sources


def test_checked_outputs_and_auditor_are_separate_from_original_data():
    assert timing.DEFAULT_OUTPUT.name == "aligned_hardware_settled_root_dual_face_timing_20260928_checked.json"
    assert timing.DEVELOPMENT_OUTPUT.name == "aligned_hardware_settled_root_dual_face_development_20260928_checked.json"
    assert timing.RAW_DEVELOPMENT_OUTPUT == original.DEVELOPMENT_OUTPUT
    assert timing.SOURCES == (
        "algorithm/experiments/aligned_hardware_settled_root_dual_face_checked_timing.py",
        "algorithm/experiments/settled_root_dual_face_checked_timing_summary.py",
    )
    assert summary._audit_result is auditor._audit_settled_root_result
    assert summary._work is auditor._work


@pytest.fixture
def checked_sources(frozen_sources, monkeypatch, tmp_path):
    source = frozen_sources
    path = tmp_path / "quality_checked.json"
    development = deepcopy(source.development)
    development["summary_reanalysis_registered_at_utc"] = "checked analysis"
    timing._protocol_path(path).write_text(json.dumps(development))
    path.write_text(json.dumps(source.quality))
    for name in ("ROOT", "PREVIOUS_OUTPUT", "PREVIOUS_DEVELOPMENT_OUTPUT", "SOURCES"):
        monkeypatch.setattr(timing, name, getattr(original, name))
    monkeypatch.setattr(timing, "DEVELOPMENT_OUTPUT", path)
    monkeypatch.setattr(timing, "RAW_DEVELOPMENT_OUTPUT", source.dev_path)
    return source, path, development


def test_checked_decision_borrows_v1_data_at_its_original_registration(checked_sources):
    source, path, _ = checked_sources
    checked, old = timing.build_protocol(source.models), original.build_protocol(source.models)
    assert checked["experiment"] == "settled_root_dual_face_eight_arm_checked_development_timing_v1"
    assert checked["reference_sources"] == {
        "settled_root_development": {"path": source.dev_path.name,
                                     "registered_at_utc": source.development["registered_at_utc"]},
    }
    assert checked["reference_sources"]["settled_root_development"]["path"] != path.name
    for key in ("labels", "repeats", "warmups_per_label", "order_seed", "cells", "models",
                "expected_state_count", "expected_group_counts", "expected_timed_calls",
                "expected_warmup_calls", "expected_selector_calls", "focus_keys", "gate",
                "nonregression_limit", "trigger_speedup_limit", "initial_lp_budget", "lp_node_budget",
                "integer_state_budget", "long_lp_budget", "control_integer_state_budget"):
        assert checked[key] == old[key]


def test_checked_analysis_cannot_replace_original_data_registration(checked_sources):
    source, path, development = checked_sources
    development["registered_at_utc"] = "new unrelated registration"
    timing._protocol_path(path).write_text(json.dumps(development))
    with pytest.raises(ValueError, match="original development registration"):
        timing.build_protocol(source.models)
