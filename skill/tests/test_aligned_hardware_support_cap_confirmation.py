import json

import pytest

from algorithm.experiments import aligned_hardware_support_cap_confirmation as confirmation


def test_changed_model_is_rejected_before_using_registered_seeds(tmp_path, monkeypatch):
    (tmp_path / "registration.json").write_text(json.dumps({"model": {"node": "changed"}}))
    (tmp_path / "source_snapshot.json").write_text(json.dumps(confirmation._source_texts()))

    def forbidden_arrivals(*args, **kwargs):
        raise AssertionError("arrival tape generated before the model mismatch was rejected")

    monkeypatch.setattr(confirmation, "poisson_arrivals", forbidden_arrivals)
    with pytest.raises(ValueError, match="new seeds remain unused"):
        confirmation.run_registered(tmp_path)
    assert not (tmp_path / "runs.jsonl").exists()


def test_registration_keeps_fixed_roster_and_existing_results(tmp_path):
    registration = confirmation.register(tmp_path)
    seeds = registration["protocol"]["seeds"]
    assert len(seeds) == 20
    assert not set(seeds).intersection(registration["previously_used_seeds"])
    assert (len(registration["protocol"]["loads"]) * len(seeds)
            * len(registration["protocol"]["budgets"])
            * len(registration["policy_max_relative_support_loss"])) == 720
    assert registration["policy_max_relative_support_loss"][registration["main_policy"]] == 0.01
    saved = (tmp_path / "registration.json").read_text()
    with pytest.raises(FileExistsError):
        confirmation.register(tmp_path)
    assert (tmp_path / "registration.json").read_text() == saved
