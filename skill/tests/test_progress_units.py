from algorithm.experiments.progress_units import (
    CompletionTimingTracker,
    completion_eta_seconds,
    completion_model_line,
    parse_completion_model,
    parse_progress_line,
    parse_progress_observation,
    task_progress_observation,
)
from algorithm.experiments.remote_workload_selected_profile_probe import _row_from_output
from algorithm.experiments.progress_wrapper import (
    _cycle_stable_rate_decision,
    _observation_allowed_in_phases,
    _stable_rate_decision,
    _update_active_phases,
    build_progress_line,
    build_tqdm_line,
)
from algorithm.experiments.sweetspot_ab_validation import _summarize_phase
from simulation.service_cache import ProfileRecord, records_from_summary_file


def test_natural_completion_model_separates_startup_loop_and_terminal_save(tmp_path):
    tracker = CompletionTimingTracker(total_units=4, unit="step")
    tracker.observe_phase_line(
        "ScheduleurmPhase name=initialization event=start", elapsed_s=0.0
    )
    tracker.observe_phase_line(
        "ScheduleurmPhase name=initialization event=end", elapsed_s=2.0
    )
    for current, elapsed in ((1, 3.0), (2, 4.0), (3, 5.0), (4, 6.0)):
        obs = parse_progress_line(f"Step {current}/4 rate=1.0 step/s")
        assert obs is not None
        tracker.observe_progress(obs, elapsed_s=elapsed)
    tracker.observe_phase_line(
        "ScheduleurmPhase name=final_save event=start index=4", elapsed_s=6.0
    )
    tracker.observe_phase_line(
        "ScheduleurmPhase name=final_save event=end index=4", elapsed_s=7.0
    )
    model = tracker.finalize(
        elapsed_s=7.0,
        child_returncode=0,
        stopped_on_stable=False,
    )

    assert model["completion_model_ready"] is True
    assert model["startup_overhead_s"] == 2.0
    assert model["completion_unit_s"] == 1.0
    assert model["terminal_overhead_s"] == 1.0
    assert model["save_observed_s"] == 1.0
    assert completion_eta_seconds(model, include_startup=True) == 7.0
    assert completion_eta_seconds(model, current_unit=2) == 3.0
    assert parse_completion_model(completion_model_line(model)) == model

    row = _row_from_output(
        gpu=0,
        local_index=0,
        rendered={"values": {"index": 0}, "seed": 7, "run_name": "natural"},
        returncode=0,
        output="rate=1.0 step/s\n" + completion_model_line(model) + "\n",
        raw_dir=tmp_path,
        unit="step",
    )
    assert row["completion_model_ready"] is True
    assert row["completion_model"]["startup_overhead_s"] == 2.0


def test_stable_stopped_probe_never_becomes_completion_model():
    tracker = CompletionTimingTracker(total_units=100, unit="step")
    for current, elapsed in ((1, 1.0), (2, 2.0), (3, 3.0)):
        obs = parse_progress_line(f"Step {current}/100 rate=1.0 step/s")
        assert obs is not None
        tracker.observe_progress(obs, elapsed_s=elapsed)
    model = tracker.finalize(
        elapsed_s=3.1,
        child_returncode=143,
        stopped_on_stable=True,
    )
    assert model["completion_model_ready"] is False
    assert model["readiness_reason"] == "stable_service_probe_not_natural_completion"


def test_service_cache_preserves_phase_aware_completion_model(tmp_path):
    model = {
        "completion_model_ready": True,
        "startup_overhead_s": 2.0,
        "completion_unit_s": 1.25,
        "terminal_overhead_s": 3.0,
        "checkpoint_observed_s": 0.5,
        "save_observed_s": 1.0,
        "total_wall_s": 10.0,
        "model_relative_error": 0.01,
    }
    summary = {
        "profile": 1,
        "node": "jtl110gpu",
        "running_count": 1,
        "returncode_valid_count": 1,
        "returncode_accepted_count": 1,
        "measurement_valid": True,
        "placement_valid": True,
        "rate_units": ["step"],
        "aggregate_stable_rate_unit_s": 0.8,
        "stable_rates_unit_s": [0.8],
        "all_stable_rate_ready": True,
        "stable_rate_ready_count": 1,
        "rows": [{"completion_model": model}],
    }
    path = tmp_path / "profile_1_per_gpu_summary.json"
    import json
    path.write_text(json.dumps(summary), encoding="utf-8")
    record = records_from_summary_file(
        path,
        workload_key="cnn",
        command_fingerprint="fp",
        resource_kind="gpu",
        total_units=4,
        node_bucket="jtl110gpu",
    )[0]

    assert record.completion_model_ready is True
    assert record.completion_eta_s(include_startup=True) == 10.0
    restored = ProfileRecord.from_snapshot(record.snapshot())
    assert restored == record


def test_progress_units_parse_benchmark_step_rate(check, sch):
    obs = parse_progress_line(
        "Step 124/2400 label=x rate=10.353019 step/s elapsed=12.0s"
    )
    check("progress parser reads benchmark current/total",
          obs is not None
          and obs.current == 124
          and obs.total == 2400
          and obs.unit == "step",
          diag=str(obs))
    check("progress parser reads benchmark step/s rate",
          obs is not None and abs(float(obs.rate_per_s) - 10.353019) < 1e-9,
          diag=str(obs))


def test_progress_parser_uses_cumulative_rate_not_window_rate():
    obs = parse_progress_line(
        "BENCH_PROGRESS Step 60/60 rate=11.1395714 step/s "
        "window_rate=12.682061 step/s ETA 0.0s"
    )

    assert obs is not None
    assert abs(float(obs.rate_per_s) - 11.1395714) < 1e-9


def test_progress_units_parse_rl_seconds_per_iter(check, sch):
    obs = parse_progress_line(
        "Iter 1989 | Reward: 4739.1 | Time: 32142s | 10.6s/iter",
        cmd="python train.py --max_iters 2000",
    )
    check("progress parser reads RL iter and cmd total",
          obs is not None
          and obs.current == 1989
          and obs.total == 2000
          and obs.unit == "iter",
          diag=str(obs))
    check("progress parser converts s/iter to iter/s",
          obs is not None
          and abs(float(obs.rate_per_s) - (1.0 / 10.6)) < 1e-12
          and abs(float(obs.seconds_per_unit) - 10.6) < 1e-12,
          diag=str(obs))


def test_progress_wrapper_emits_scheduler_stable_iter_line(check, sch):
    child_obs = parse_progress_line(
        "Iter    14 | Reward: 123.0 | Time: 456s | 21.0s/iter",
        cmd="python train.py --max_iters 1500",
    )
    line = build_progress_line(child_obs, total_override=1500, unit_override="iter")
    obs = parse_progress_line(line)
    check("progress wrapper emits line with current total and rate",
          line is not None
          and "ScheduleurmProgress Iter 14/1500" in line
          and "ETA " in line
          and obs is not None
	          and obs.current == 14
	          and obs.total == 1500
	          and abs(float(obs.rate_per_s) - (1.0 / 21.0)) < 1e-9,
	          diag=f"line={line}, obs={obs}")


def test_progress_wrapper_emits_persistent_tqdm_line(check, sch):
    child_obs = parse_progress_line(
        "Step 20/100 loss=0.2 rate=2.5 step/s",
        cmd="python bench.py --steps 100",
    )
    line = build_tqdm_line(child_obs, total_override=100, unit_override="step", elapsed_s=8.0)
    obs = parse_progress_line(line)
    check("progress wrapper emits persistent tqdm-style step line",
          line is not None
          and "ScheduleurmTqdm step:" in line
          and "20/100" in line
          and "[00:08<00:32, 2.5step/s]" in line
          and obs is not None
          and obs.current == 20
          and obs.total == 100
          and obs.unit == "step",
          diag=f"line={line}, obs={obs}")


def test_progress_wrapper_rejects_library_tqdm_outside_outer_loop():
    loading = parse_progress_line(
        "Loading weights: 45%|#### | 34/76 [00:01<00:01, 30.7it/s]"
    )
    assert loading is not None and loading.source == "per_second"
    phases = set()
    _update_active_phases(
        "ScheduleurmPhase name=initialization event=start", phases
    )
    assert not _observation_allowed_in_phases(loading, phases)
    _update_active_phases(
        "ScheduleurmPhase name=initialization event=end", phases
    )
    _update_active_phases("ScheduleurmPhase name=outer_loop event=start", phases)
    assert _observation_allowed_in_phases(loading, phases)
    _update_active_phases("ScheduleurmPhase name=checkpoint event=start", phases)
    assert not _observation_allowed_in_phases(loading, phases)


def test_progress_wrapper_cycle_average_stabilizes_periodic_rl_eta(check, sch):
    cycle_rates = [1 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 12.0]
    rates = cycle_rates * 4
    raw = _stable_rate_decision(
        rates,
        windows=5,
        min_samples=8,
        max_cv=0.08,
        max_last_two_rel_delta=0.05,
        skip_samples=0,
    )
    cycle = _cycle_stable_rate_decision(
        rates,
        cycle_units=5,
        windows=3,
        min_samples=15,
        max_cv=0.08,
        max_last_two_rel_delta=0.05,
        skip_samples=0,
    )

    expected_cycle_rate = 5.0 / (4.0 * 6.0 + 12.0)
    check("raw RL train/eval samples are correctly treated as non-stationary",
          not bool(raw.get("ready")),
          diag=str(raw))
    check("cycle-average RL train/eval rate is stable over complete cycles",
          bool(cycle.get("ready"))
          and abs(float(cycle.get("mean_rate") or 0.0) - expected_cycle_rate) < 1e-12
          and int(cycle.get("cycle_units") or 0) == 5,
          diag=str(cycle))


def test_progress_parser_ignores_units_per_iter_banner(check, sch):
    obs = parse_progress_line(
        "  Updates/iter: 250  Samples/iter: 4000",
        cmd="python train.py --max_iters 1500",
    )
    child_obs = parse_progress_line("Iter 4000 | bogus | 1.0s/iter", cmd="python train.py --max_iters 1500")
    line = build_progress_line(child_obs, total_override=1500, unit_override="iter") if child_obs else None
    check("progress parser ignores banner unit metadata and impossible current totals",
          obs is None and child_obs is None and line is None,
          diag=f"obs={obs}, child_obs={child_obs}, line={line}")


def test_progress_units_latest_line_wins_and_task_fallbacks(check, sch):
    tail = "\n".join([
        "Iter 10 | Reward: 0.0 | 50.0s/iter",
        "Iter 11 | Reward: 1.0 | 25.0s/iter",
    ])
    obs = parse_progress_observation(tail, cmd="python train.py --max_iters 100")
    check("progress parser uses latest RL progress line",
          obs is not None
          and obs.current == 11
          and obs.total == 100
          and abs(float(obs.rate_per_s) - 0.04) < 1e-12,
          diag=str(obs))

    task_obs = task_progress_observation({
        "last_progress_line": "rate=12.5 step/s",
        "runtime_current_unit": 33,
        "runtime_total_units": 100,
    })
    check("task progress parser fills scheduler runtime counters",
          task_obs is not None
          and task_obs.current == 33
          and task_obs.total == 100
          and task_obs.unit == "step",
          diag=str(task_obs))


def test_sweetspot_summary_counts_rl_iter_rate(check, sch):
    tasks = [
        {
            "id": "t1",
            "status": "running",
            "gpu_idx": 3,
            "cmd": "python train.py --max_iters 2000",
            "last_progress_line": "Iter 1105 | Reward: 123.4 | 25.6s/iter",
        },
        {
            "id": "t2",
            "status": "running",
            "gpu_idx": 3,
            "cmd": "python train.py --max_iters 2000",
            "last_progress_line": "Iter 1110 | Reward: 120.0 | 12.8s/iter",
        },
    ]
    summary = _summarize_phase("rl_2_per_gpu", tasks)
    check("sweetspot summary counts RL iter/s rates",
          summary["running_with_rate_count"] == 2
          and summary["rate_units"] == ["iter"]
          and abs(summary["aggregate_active_rate_unit_s"] - (1 / 25.6 + 1 / 12.8)) < 1e-12,
          diag=str(summary))
