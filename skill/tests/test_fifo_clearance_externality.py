"""Hand-calculated clearance costs and unavailable frozen-cache handling."""
from fractions import Fraction

from algorithm.experiments.fifo_clearance_externality import clearance_externality


def test_last_H_admission_sets_class_cost_and_committed_jobs_do_not():
    jobs = [
        dict(id=["visible", 0, 0], coordinate=0, start=[12, 1], completion=[20, 1]),
        dict(id=["background", 0], coordinate=0, start=[15, 1], completion=[21, 1]),
        dict(id=["prefix", 1], coordinate=1, start=[11, 1], completion=[19, 1]),
        dict(id=["committed", 0], coordinate=0, start=[1, 1], completion=[100, 1]),
    ]
    row = clearance_externality(jobs,
        [[["visible", 0, 0], ["background", 0]], [["prefix", 1]]],
        [.5, Fraction(3, 4)], [10, 1])
    # The queue clearance delays are five and one seconds.  Integrating
    # arrival rates 1/2 and 3/4 over their triangular waits gives 25/4+3/8.
    assert row["externality"] == Fraction(53, 8)
    assert [part["externality"] for part in row["per_coordinate"]] == [Fraction(25, 4), Fraction(3, 8)]
    assert [part["clearance"] for part in row["per_coordinate"]] == [15, 11]
    assert [part["waiting_jobs_at_H"] for part in row["per_coordinate"]] == [2, 1]
    assert all(isinstance(part["externality"], Fraction) for part in row["per_coordinate"])


def test_empty_FIFO_zero_rate_and_H_boundary_have_no_externality():
    jobs = [
        dict(id=["committed", 0], coordinate=0, start=9., completion=100.),
        dict(id=["prefix", 0], coordinate=1, start=100., completion=110.),
        dict(id=["prefix", 1], coordinate=2, start=10., completion=12.),
    ]
    row = clearance_externality(jobs, [[], [["prefix", 0]], [["prefix", 1]]], [.5, 0., .25], 10.)
    assert row["externality"] == 0
    assert [part["delay"] for part in row["per_coordinate"]] == [0, 90, 0]
    assert row["per_coordinate"][0]["clearance"] == Fraction(10)


def test_unfinished_or_missing_H_job_cannot_be_silently_ignored():
    waiting = [[["prefix", 0]]]
    for jobs in ([], [dict(id=["prefix", 0], coordinate=0, start=None, completion=None)]):
        try:
            clearance_externality(jobs, waiting, [Fraction(1, 2)], Fraction(10))
        except ValueError as error:
            assert "no complete cached admission" in str(error)
        else:
            raise AssertionError("an unavailable H FIFO cache was assigned a score")


if __name__ == "__main__":
    test_last_H_admission_sets_class_cost_and_committed_jobs_do_not()
    test_empty_FIFO_zero_rate_and_H_boundary_have_no_externality()
    test_unfinished_or_missing_H_job_cannot_be_silently_ignored()
    print("3 passed", flush=True)
