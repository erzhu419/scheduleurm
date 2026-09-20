import json

import pytest

from algorithm.experiments import aligned_hardware_lookahead_confirmation as confirmation


def test_changed_model_is_rejected_before_generating_new_arrivals(tmp_path, monkeypatch):
    (tmp_path / "registration.json").write_text(json.dumps({"model": {"node": "changed"}}))
    (tmp_path / "source_snapshot.json").write_text(json.dumps(confirmation._source_texts()))

    def forbidden_arrivals(*args, **kwargs):
        raise AssertionError("arrivals used before detecting the changed model")

    monkeypatch.setattr(confirmation, "poisson_arrivals", forbidden_arrivals)
    with pytest.raises(ValueError, match="cached model changed; new seeds remain unused"):
        confirmation.run_registered(tmp_path)
    assert not (tmp_path / "runs.jsonl").exists()


def test_changed_terminal_rates_are_rejected_before_generating_arrivals(tmp_path, monkeypatch):
    registration = confirmation.register(tmp_path)
    registration["terminal_service_rates"][0] *= 2
    (tmp_path / "registration.json").write_text(json.dumps(registration))

    def forbidden_arrivals(*args, **kwargs):
        raise AssertionError("arrivals used before detecting the changed terminal reference")

    monkeypatch.setattr(confirmation, "poisson_arrivals", forbidden_arrivals)
    with pytest.raises(ValueError, match="terminal service rates changed; new seeds remain unused"):
        confirmation.run_registered(tmp_path)
    assert not (tmp_path / "runs.jsonl").exists()


def test_registration_preserves_all_controls_and_uses_fresh_seeds(tmp_path):
    registration = confirmation.register(tmp_path)
    protocol = registration["protocol"]
    assert len(protocol["seeds"]) == 20
    assert not set(protocol["seeds"]).intersection(registration["previously_used_seeds"])
    assert (len(protocol["loads"]) * len(protocol["seeds"])
            * len(protocol["budgets"]) * len(registration["policies"])) == 960
    assert registration["policy_terminal_weights"] == {"frame_area_cap_1pct": 0.0, "lookahead_cap_1pct": 1.0}
    assert registration["max_relative_support_loss"] == 0.01
    saved = (tmp_path / "registration.json").read_text()
    with pytest.raises(FileExistsError):
        confirmation.register(tmp_path)
    assert (tmp_path / "registration.json").read_text() == saved
