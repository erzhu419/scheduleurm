from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_threshold_canonical_catalogue_screen as screen


@pytest.fixture(scope="module")
def models():
    return screen.load_catalogues()


def test_known_real_reversed_dominance_action_is_restored(models):
    model = next(model for model in models if model.node == "node007")
    queue = (1, 0, 4)
    expected = screen.oracle(screen.enumerate_frames(model.options, 2), queue)
    assert expected[2] == (0, 8)
    selected = screen.select_threshold_canonical_requirement_dominance_frame(
        model.options, queue, 2, screen.LP_BUDGET)
    assert not any(screen._check_selection(selected.selection, expected, model.options, queue).values())


def test_oracle_gate_rejects_score_preserving_wrong_original_action(models, monkeypatch):
    model = next(model for model in models if model.node == "node007")
    select = screen.select_threshold_canonical_requirement_dominance_frame

    def wrong_order(*args, **kwargs):
        result = select(*args, **kwargs)
        return SimpleNamespace(selection=replace(result.selection, option_indices=(1, 8)),
                               preprocessing=result.preprocessing)

    monkeypatch.setattr(screen, "select_threshold_canonical_requirement_dominance_frame", wrong_order)
    row = screen.screen_catalogue(model, ((1, 0, 4),), lanes=2)
    assert row["full_failures"] == 0
    assert row["threshold_failures"] == 1
    assert row["first_failure"]["checks"] == {
        "score_mismatch": False, "duration_mismatch": False,
        "action_mismatch": True, "count_mismatch": False,
        "service_mismatch": False, "uncertified": False,
    }


def test_matching_action_without_exact_certificate_is_rejected(models):
    model = models[0]
    queue = (0, 0, 0)
    expected = screen.oracle(screen.enumerate_frames(model.options, model.lanes), queue)
    result = screen.select_threshold_canonical_requirement_dominance_frame(
        model.options, queue, model.lanes, screen.LP_BUDGET)
    checks = screen._check_selection(replace(result.selection, canonical_exact=False),
                                     expected, model.options, queue)
    assert checks["uncertified"]
    assert not checks["action_mismatch"]


def test_registered_model_drift_stops_before_search(tmp_path, monkeypatch):
    output = tmp_path / "screen.json"
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    protocol_path = tmp_path / "screen_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["models"][0]["options"][0]["duration"] += 1
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output)])
    with pytest.raises(ValueError, match="registered models or protocol changed"):
        screen.main()
    assert not output.exists()


def test_cli_reproduction_writes_external_output(tmp_path, monkeypatch):
    output = tmp_path / "screen.json"
    monkeypatch.setattr(screen, "QUEUE_LEVELS", (0,))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output)])
    screen.main()
    result = json.loads(output.read_text())
    assert result["oracle_checks_passed"]
    assert result["decision"] == "MULTI_CATALOGUE_SCALING_ADVANCE"
    assert result["queue_count"] == 4
    assert result["selector_calls"] == 10
    assert result["protocol_path"] == str(tmp_path / "screen_protocol.json")
