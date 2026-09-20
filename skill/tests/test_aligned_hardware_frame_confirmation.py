import json

import pytest

from algorithm.experiments import aligned_hardware_frame_confirmation as confirmation


def test_changed_cached_model_is_rejected_before_running_any_arrival_seed(tmp_path, monkeypatch):
    # The service cache is data, so matching Python source alone does not freeze
    # its execution model. This must fail before build_experiment can use seeds.
    (tmp_path / "registration.json").write_text(json.dumps({"model": {"node": "changed"}}))
    (tmp_path / "source_snapshot.json").write_text(json.dumps(confirmation._source_texts()))

    def forbidden_experiment(*args, **kwargs):
        raise AssertionError("an arrival seed was used before the model mismatch was rejected")

    monkeypatch.setattr(confirmation, "build_experiment", forbidden_experiment)
    with pytest.raises(ValueError, match="holdout seeds remain unused"):
        confirmation.run_registered(tmp_path)
    assert not (tmp_path / "v1.json").exists()
    assert not (tmp_path / "v4.json").exists()
