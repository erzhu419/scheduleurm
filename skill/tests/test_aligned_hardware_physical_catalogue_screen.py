import json

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.canonical_count_space_bnb_requirement_dominance_frame_candidates import (
    prepare_canonical_count_space_bnb_requirement_dominance_catalogue,
)
from algorithm.experiments import aligned_hardware_physical_catalogue_screen as screen
from algorithm.requirement_dominance_frame_candidates import reduce_requirement_dominated_options


def test_real_reversed_dominance_preserves_score_but_changes_canonical_action():
    model = next(model for model in screen.load_catalogues() if model.node == "node007")
    reduced = reduce_requirement_dominated_options(model.options)
    queue = (1, 0, 4)
    full = screen.oracle(screen.enumerate_frames(model.options, 2), queue)
    quotient = screen.oracle(screen.enumerate_frames(reduced.options, 2), queue)
    witness = screen._witness(model, reduced, queue, 2, full, quotient)
    assert full[:2] == quotient[:2]
    assert witness["full_option_indices"] == (0, 8)
    assert witness["reduced_mapped_option_indices"] == (1, 8)
    for options, expected in ((model.options, full), (reduced.options, quotient)):
        selected = select_canonical_count_space_bnb_frame(options, queue, 2, screen.LP_BUDGET)
        assert selected.canonical_exact
        assert selected.option_indices == expected[2]
    with pytest.raises(ValueError, match="earlier retained"):
        prepare_canonical_count_space_bnb_requirement_dominance_catalogue(model.options)


def test_registered_model_drift_is_rejected_before_running_screen(tmp_path, monkeypatch):
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


def test_cli_can_write_a_reproduction_outside_the_repository(tmp_path, monkeypatch):
    output = tmp_path / "screen.json"
    monkeypatch.setattr(screen, "QUEUE_LEVELS", (0,))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output)])
    screen.main()
    result = json.loads(output.read_text())
    assert result["oracle_checks_passed"]
    assert result["protocol_path"] == str(tmp_path / "screen_protocol.json")
