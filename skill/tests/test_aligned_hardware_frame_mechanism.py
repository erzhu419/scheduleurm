from types import SimpleNamespace

import pytest

from algorithm.experiments.aligned_hardware_frame_mechanism import Recorder, _expected_barrier_area


def _action(score, duration, count):
    action = SimpleNamespace(score=score, duration=duration, service=(count,), option_indices=(count,))
    description = {"departures": (count,), "duration_s": duration,
                   "completion_offsets": ((duration,) * count,)}
    return action, description


class Pairs:
    def get(self, queues):
        exact, exact_description = _action(2.0, 2.0, 2)
        if queues[0] == 0:
            small, description = exact, exact_description
        elif queues[0] == 1:
            small, description = _action(1.0, 1.5, 1)  # Strict score loss.
        else:
            small, description = _action(2.0, 1.5, 1)  # Different action, same support.
        return small, exact, (description, exact_description)


def test_one_switch_uses_shared_prefix_and_returns_to_exact_after_one_difference():
    recorder = Recorder(Pairs(), "first_difference_once")
    assert recorder(None, (0,), 1, 120).duration == 2.0
    assert recorder.intervention is None
    assert recorder(None, (1,), 1, 120).duration == 1.5
    assert recorder.intervention["frame_index"] == 1
    assert recorder.intervention["start_s"] == 2.0
    assert recorder.intervention["queue"] == (1,)
    assert recorder(None, (2,), 1, 120).duration == 2.0
    assert recorder(None, (1,), 1, 120).duration == 2.0
    assert recorder.substituted_frames == 1
    assert recorder.frames[-1]["start_s"] == 5.5


def test_tie_only_and_strict_loss_only_interventions_are_complementary():
    strict = Recorder(Pairs(), "k26_on_strict_loss")
    ties = Recorder(Pairs(), "k26_on_equal_score")
    assert strict(None, (1,), 1, 120).score == 1.0
    assert ties(None, (1,), 1, 120).score == 2.0
    assert strict(None, (2,), 1, 120).duration == 2.0
    assert ties(None, (2,), 1, 120).duration == 1.5
    assert strict.substituted_frames == ties.substituted_frames == 1


def test_expected_arrival_barrier_area_truncates_exposure_at_horizon():
    # Arrivals have rate 2 over [1,3] inside a frame ending at 5:
    # integral_1^3 2*(5-t) dt = 12. The next frame has no arrival exposure.
    frames = [{"start_s": 1.0, "duration_s": 4.0}, {"start_s": 5.0, "duration_s": 2.0}]
    assert _expected_barrier_area(frames, 3.0, (0.5, 1.5)) == pytest.approx(12.0)
