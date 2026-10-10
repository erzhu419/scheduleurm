"""Audit repair reuses actual saved records and never runs a selector."""
from copy import deepcopy
import json

import pytest

from algorithm.experiments import aligned_hardware_settled_root_dual_face_development_reanalysis as reanalysis


@pytest.fixture
def saved_run(monkeypatch, tmp_path):
    original = tmp_path / "original.json"
    protocol = {"registered_at_utc": "original run", "cells": [dict(node="actual", lanes=8, states=[])],
                "execution_sources": {"kernel.py": "frozen selector\n"}}
    result = dict(decision="INVALID_EXECUTION", state_count=2802, executed_interface_calls=2802,
        references_complete=True, source_native_behavior_preserved=True, nonclosures=[],
        failures=[dict(errors=[reanalysis.DIAGNOSTIC]) for _ in range(3)])
    original.write_text(json.dumps(result))
    reanalysis._protocol_path(original).write_text(json.dumps(protocol))
    (tmp_path / "kernel.py").write_text("frozen selector\n")
    (tmp_path / "audit.py").write_text("corrected audit\n")
    monkeypatch.setattr(reanalysis, "ROOT", tmp_path)
    monkeypatch.setattr(reanalysis, "ORIGINAL_OUTPUT", original)
    monkeypatch.setattr(reanalysis, "SOURCES", ("audit.py",))
    return original, result


def test_saved_registration_and_data_are_preserved_without_new_calls(saved_run):
    protocol = reanalysis.build_protocol()
    protocol["summary_reanalysis_registered_at_utc"] = "repair date"
    calls, actual_row = [], dict(registered_at_utc="original run", states=[])

    def loader(path, node, lanes, stamp):
        calls.append((path, node, lanes, stamp))
        return actual_row

    def summarize(registered, rows):
        assert registered is protocol and rows == [actual_row]
        return dict(executed_interface_calls=2802, decision="ADVANCE")

    result = reanalysis.reanalyze(protocol, loader=loader, summarizer=summarize)
    assert calls == [(saved_run[0], "actual", 8, "original run")]
    assert protocol["registered_at_utc"] == "original run"
    assert result["actual_selector_calls_during_reanalysis"] == result["actual_native_advance_calls_during_reanalysis"] == 0
    assert result["inherited_selector_calls"] == 2802
    assert json.loads(saved_run[0].read_text())["decision"] == "INVALID_EXECUTION"


@pytest.mark.parametrize("change", ("kernel", "audit", "original_failure"))
def test_correction_cannot_change_frozen_algorithm_or_hide_other_failures(saved_run, change):
    original, result = saved_run
    protocol = reanalysis.build_protocol()
    protocol["summary_reanalysis_registered_at_utc"] = "repair date"
    if change == "original_failure":
        result["failures"][0]["errors"] = ["wrong returned action"]
        original.write_text(json.dumps(result))
    else:
        (reanalysis.ROOT / ("kernel.py" if change == "kernel" else "audit.py")).write_text("changed\n")
    with pytest.raises(ValueError):
        reanalysis.verify_protocol(protocol)
