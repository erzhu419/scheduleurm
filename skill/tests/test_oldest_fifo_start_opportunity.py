"""Physical lane-release examples for the cached action description."""
from algorithm.experiments.oldest_fifo_start_opportunity import start_opportunity


def test_pending_reservation_and_immediate_service():
    context = dict(available=[10., 19.], now=10., lane=0, waiting_ages=[[8., 3.], [5.]])
    omitted = start_opportunity([0, 1], 4., **context)
    served = start_opportunity([1, 0], 12., **context)
    assert omitted["oldest_coordinates"] == (0,)
    assert omitted["earliest_start_per_coordinate"] == (14., 10.)
    assert omitted["oldest_start_sum"] == 4.
    assert served["oldest_start_sum"] == 0.


def test_other_idle_lane_preserves_opportunity_and_tied_heads_are_retained():
    row = start_opportunity([0, 0, 1], 20., [10., 9.], 10., 0, [[7.], [7.], [1.]])
    assert row["oldest_coordinates"] == (0, 1)
    assert row["earliest_start_per_coordinate"] == (10., 10., 10.)
    assert row["oldest_start_sum"] == 0.


if __name__ == "__main__":
    test_pending_reservation_and_immediate_service()
    test_other_idle_lane_preserves_opportunity_and_tied_heads_are_retained()
    print("2 passed", flush=True)
