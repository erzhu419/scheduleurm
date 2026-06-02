from algorithm.experiments.service_curve_validation import build_service_curve_verdict


def _summary(count, mean_rate, aggregate_rate=None, evictions=0, blocked=0):
    running = 2 * count
    return {
        "running_count": running,
        "running_with_rate_count": running,
        "per_gpu_running": {"0": count, "1": count},
        "mean_active_rate_step_s": mean_rate,
        "aggregate_active_rate_step_s": aggregate_rate if aggregate_rate is not None else mean_rate * running,
        "eviction_count": evictions,
        "blocked_count": blocked,
    }


def test_service_curve_verdict_identifies_two_per_gpu_flow_sweetspot(check, sch):
    verdict = build_service_curve_verdict(
        run_id="toy",
        node="node",
        steps=2400,
        total_jobs_for_proxy=6,
        expected_sweetspot_count=2,
        min_two_vs_three_gain=1.05,
        summaries={
            1: _summary(1, 2.30),
            2: _summary(2, 2.00),
            3: _summary(3, 1.35),
        },
    )
    check("service curve verdict passes two-per-GPU flow sweetspot",
          verdict["pass"]
          and verdict["best_flow_time_count_per_gpu"] == 2
          and verdict["best_makespan_count_per_gpu"] == 3
          and verdict["two_vs_three_mean_rate_gain"] > 1.05,
          diag=str(verdict))


def test_service_curve_verdict_fails_unmeasured_or_blocked_profiles(check, sch):
    missing_rate = _summary(2, 2.0)
    missing_rate["running_with_rate_count"] = 3
    blocked = _summary(3, 1.4, blocked=1)
    verdict = build_service_curve_verdict(
        run_id="bad",
        node="node",
        steps=2400,
        total_jobs_for_proxy=6,
        expected_sweetspot_count=2,
        min_two_vs_three_gain=1.05,
        summaries={
            1: _summary(1, 2.30),
            2: missing_rate,
            3: blocked,
        },
    )
    check("service curve verdict fails incomplete/blocked profiles",
          not verdict["pass"]
          and any("did not measure every running task" in x for x in verdict["failure_reasons"])
          and any("placement blocks" in x for x in verdict["failure_reasons"]),
          diag=str(verdict))


def test_service_curve_verdict_can_report_measured_best_without_fixed_expectation(check, sch):
    verdict = build_service_curve_verdict(
        run_id="measured",
        node="node",
        steps=2400,
        total_jobs_for_proxy=6,
        expected_sweetspot_count=0,
        min_two_vs_three_gain=1.05,
        summaries={
            1: _summary(1, 10.0),
            2: _summary(2, 4.8),
            3: _summary(3, 3.2),
        },
    )
    check("service curve verdict reports one-per-GPU measured best without failing fixed expectation",
          verdict["pass"]
          and verdict["best_flow_time_count_per_gpu"] == 1
          and verdict["best_makespan_count_per_gpu"] == 1
          and not verdict["expected_sweetspot_checked"],
          diag=str(verdict))
