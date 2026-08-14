from __future__ import annotations

from skill.scheduler_runtime_profile import (
    RuntimeProfileDeps,
    apply_runtime_projection,
    apply_test_log_runtime_profile,
    read_text_tail,
    runtime_profile_from_log,
    runtime_total_units_from_cmd,
)


class _EtaTracker:
    @staticmethod
    def _extract_total_from_cmd(cmd):
        return 321 if "--steps 321" in (cmd or "") else 0

    @staticmethod
    def runtime_projection_from_log(text, *, cmd="", observed_duration_s=0):
        if "PROGRESS" not in text:
            return None
        return {
            "source": "test_profile",
            "total_s": 123,
            "eta_s": 45,
            "current": 5,
            "total_units": 10,
            "unit_s": 12.3,
            "observed": observed_duration_s,
        }


def test_runtime_total_units_from_cmd_uses_eta_tracker_hook():
    assert runtime_total_units_from_cmd(
        "python train.py --steps 321",
        load_eta_tracker_module_fn=lambda: _EtaTracker,
    ) == 321
    assert runtime_total_units_from_cmd(
        "python train.py",
        load_eta_tracker_module_fn=lambda: None,
    ) == 0


def test_eta_tracker_extracts_kg_benchmark_budget_units():
    from skill import eta_tracker

    assert eta_tracker._extract_total_from_cmd(
        "python performance/benchmark_sota.py --problem P --d 10000 --N 160 --n0 8"
    ) == 160


def test_apply_runtime_projection_updates_expected_fields():
    task = {}

    apply_runtime_projection(
        task,
        {
            "source": "progress",
            "total_s": 100,
            "eta_s": 20,
            "current": 8,
            "total_units": 10,
            "unit_s": 12.5,
        },
        now=lambda: 999.0,
    )

    assert task["runtime_total_s_est"] == 100
    assert task["runtime_eta_s_est"] == 20
    assert task["runtime_current_unit"] == 8
    assert task["runtime_total_units"] == 10
    assert task["runtime_unit_s_est"] == 12.5
    assert task["runtime_progress_at"] == 999


def test_runtime_profile_from_log_reads_tail_and_swallows_parse_failures(tmp_path):
    path = tmp_path / "task.log"
    path.write_text("old\n" + "x" * 100 + "\nPROGRESS 5/10\n", encoding="utf-8")

    profile = runtime_profile_from_log(
        str(path),
        observed_duration_s=7,
        load_eta_tracker_module_fn=lambda: _EtaTracker,
        read_text_tail_fn=lambda path: read_text_tail(path, max_bytes=64),
    )

    assert profile["source"] == "test_profile"
    assert profile["observed"] == 7


def test_apply_test_log_runtime_profile_records_history(tmp_path):
    log_path = tmp_path / "test.log"
    log_path.write_text("PROGRESS 1/2\n", encoding="utf-8")
    saved = []
    task = {"cmd": "python train.py"}

    ok = apply_test_log_runtime_profile(
        task,
        str(log_path),
        deps=RuntimeProfileDeps(
            runtime_profile_from_log=lambda path, cmd="": {
                "source": "from_log",
                "total_s": 55,
                "eta_s": 10,
            },
            apply_runtime_projection=lambda task, profile: apply_runtime_projection(
                task, profile, now=lambda: 111.0
            ),
            runtime_history_record=lambda task: saved.append(dict(task)),
        ),
    )

    assert ok is True
    assert task["runtime_total_s_est"] == 55
    assert task["test_log_path"] == str(log_path)
    assert task["runtime_est_source"].startswith("from_log:")
    assert saved and saved[0]["runtime_total_s_est"] == 55
