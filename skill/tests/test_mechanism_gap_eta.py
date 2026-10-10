"""ETA must not confuse calibration updates with simulator iterations."""
import pytest

from skill import eta_tracker
from skill.scheduler_runtime.history import runtime_history_best
from skill.tests.test_scheduler_history import _runtime_deps


MODULE = "python -u -m jax_experiments.analysis.run_mechanism_gap_v34 "


def test_legacy_bapr_elapsed_log_uses_measured_window():
    log = "\n".join(f"Iter {iteration} | steps=123 | elapsed={seconds}s | metrics={{}}"
                    for iteration, seconds in ((0, 47.6), (50, 547.6), (100, 1047.6)))
    cmd = "SCHEDULEURM_ETA_TOTAL_UNITS=2100 " + MODULE + "train --kind hc --arm reference"
    projection = eta_tracker.runtime_projection(log, elapsed_s=1100, cmd=cmd)
    assert projection["source"] == "iter_time_window"
    assert projection["unit_s"] == pytest.approx(10)
    assert projection["eta_s"] == pytest.approx(20000, abs=1)


def test_resumed_bapr_window_excludes_restored_iterations_and_startup():
    log = "\n".join(f"Iter {iteration}/2100 | Time: {seconds}s | rollout=3s"
                    for iteration, seconds in ((1401, 50), (1402, 60), (1403, 70)))
    cmd = "SCHEDULEURM_ETA_TOTAL_UNITS=2100 " + MODULE + "train --kind hc --arm reference"
    projection = eta_tracker.runtime_projection(log, elapsed_s=600, cmd=cmd)
    assert projection["unit_s"] == pytest.approx(10)
    assert projection["eta_s"] == 6970
    assert projection["total_s"] == 7570


def test_sparse_elapsed_window_does_not_count_log_silence_as_slow_training():
    log = "Iter 0 | elapsed=47.6s\nIter 50 | elapsed=742.9s"
    cmd = "SCHEDULEURM_ETA_TOTAL_UNITS=2100 " + MODULE + "train --kind hc --arm reference"
    projection = eta_tracker.runtime_projection(log, elapsed_s=1200, cmd=cmd)
    assert projection["source"] == "iter_time_window"
    assert projection["unit_s"] == pytest.approx(13.906)
    assert projection["eta_s"] == pytest.approx(28507, abs=1)
    assert eta_tracker.iter_time_window_projection(
        "Iter 1/2100 | Time: 47s\nIter 2/2100 | Time: 100s", elapsed_s=1200, cmd=cmd) is None


@pytest.mark.parametrize("other", [
    "smoke --resume", "fit --seed 94103 --resume",
    "train --kind reacher --arm reference --seed 92003 --resume",
    "train --kind hc --arm conditioned --seed 92003 --resume",
    "audit --kind hc --seed 92003 --resume",
])
def test_bapr_history_never_crosses_subcommand_environment_or_arm(other):
    task = {"cmd": MODULE + "train --kind hc --arm reference --seed 92021 --resume",
            "project": "BAPR", "cwd": "/tmp/BAPR"}
    history = {"wrong": {**task, "cmd": MODULE + other, "total_s": 750000}}
    assert runtime_history_best(task, task_runtime_payload=lambda row: row,
                               deps=_runtime_deps(runtime_history=history)) == (None, None, None)


def test_bapr_history_reuses_same_workload_other_seed():
    task = {"cmd": MODULE + "train --kind hc --arm reference --seed 92021 --resume",
            "project": "BAPR", "cwd": "/tmp/BAPR"}
    history = {"same": {**task, "cmd": task["cmd"].replace("92021", "92003"), "total_s": 12000}}
    rec, key, kind = runtime_history_best(task, task_runtime_payload=lambda row: row,
                                        deps=_runtime_deps(runtime_history=history))
    assert (rec["total_s"], key, kind) == (12000, "closest:same", "closest")
