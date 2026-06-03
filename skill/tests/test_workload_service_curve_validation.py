from algorithm.experiments.service_curve_validation import (
    annotate_expected_placement,
    build_service_curve_verdict,
)
from algorithm.experiments.workload_service_curve_validation import render_workload_command


def test_workload_command_template_renders_unique_identity(check, sch):
    rendered = render_workload_command(
        (
            "python -m train --seed {seed} --max_iters {max_iters} "
            "--save_root {output_root}/{phase} --run_name {run_name} --gpu {gpu_idx}"
        ),
        run_id="runA",
        phase="profile_5_per_gpu",
        count_per_gpu=5,
        index=7,
        gpu_idx=2,
        seed_base=100,
        max_iters=80,
        output_root="scratch/curves",
        node="jtl110gpu2",
    )
    cmd = rendered["cmd"]
    check("workload template renders deterministic seed",
          rendered["seed"] == 107 and "--seed 107" in cmd,
          diag=str(rendered))
    check("workload template renders unique run name",
          rendered["run_name"] == "runA_profile_5_per_gpu_gpu2_7"
          and "--run_name runA_profile_5_per_gpu_gpu2_7" in cmd,
          diag=str(rendered))
    check("workload template passes max_iters and output root",
          "--max_iters 80" in cmd and "scratch/curves/profile_5_per_gpu" in cmd,
          diag=cmd)


def test_workload_command_template_rejects_unknown_placeholder(check, sch):
    try:
        render_workload_command(
            "python train.py --bad {missing}",
            run_id="r",
            phase="profile_1_per_gpu",
            count_per_gpu=1,
            index=0,
            gpu_idx=0,
            seed_base=1,
            max_iters=10,
            output_root="out",
            node="node",
        )
    except ValueError as e:
        check("workload template rejects unknown placeholder",
              "missing" in str(e),
              diag=str(e))
    else:
        check("workload template rejects unknown placeholder", False)


def _iter_summary(count, mean_rate):
    running = count
    return {
        "running_count": running,
        "running_with_rate_count": running,
        "per_gpu_running": {"0": count},
        "rate_units": ["iter"],
        "mean_active_rate_unit_s": mean_rate,
        "aggregate_active_rate_unit_s": mean_rate * running,
        "eviction_count": 0,
        "blocked_count": 0,
    }


def test_workload_service_curve_accepts_iter_units(check, sch):
    verdict = build_service_curve_verdict(
        run_id="rl",
        node="node",
        steps=80,
        total_jobs_for_proxy=5,
        proxy_job_counts=[5, 10],
        expected_sweetspot_count=5,
        min_two_vs_three_gain=0.0,
        summaries={
            1: _iter_summary(1, 0.10),
            2: _iter_summary(2, 0.098),
            3: _iter_summary(3, 0.096),
            4: _iter_summary(4, 0.095),
            5: _iter_summary(5, 0.094),
        },
    )
    check("workload service curve can certify RL iter sweetspot",
          verdict["pass"]
          and verdict["best_flow_time_count_per_gpu"] == 5
          and verdict["rows"][-1]["rate_units"] == ["iter"],
          diag=str(verdict))


def test_workload_service_curve_rejects_cross_gpu_profile_contamination(check, sch):
    contaminated = _iter_summary(15, 0.20)
    contaminated["per_gpu_running"] = {"0": 1, "1": 14}
    contaminated = annotate_expected_placement(contaminated, [1] * 15)
    verdict = build_service_curve_verdict(
        run_id="rl",
        node="node",
        steps=80,
        total_jobs_for_proxy=15,
        proxy_job_counts=[15],
        expected_sweetspot_count=0,
        min_two_vs_three_gain=0.0,
        summaries={
            2: annotate_expected_placement(_iter_summary(2, 0.10), [0, 0]),
            3: annotate_expected_placement(_iter_summary(3, 0.09), [0, 0, 0]),
            15: contaminated,
        },
    )
    check("cross-GPU profile contamination fails validation",
          not verdict["pass"]
          and any("placement invalid" in reason for reason in verdict["failure_reasons"]),
          diag=str(verdict))
    check("invalid contaminated profile is excluded from best-count selection",
          verdict["best_flow_time_count_per_gpu"] != 15,
          diag=str(verdict))


def test_workload_service_curve_records_capacity_boundary_without_failing_curve(check, sch):
    boundary = _iter_summary(14, 0.01)
    boundary["queued_count"] = 1
    boundary["blocked_count"] = 1
    boundary["per_gpu_running"] = {"1": 14}
    boundary = annotate_expected_placement(boundary, [1] * 15)
    boundary["capacity_boundary"] = True
    boundary["boundary_reasons"] = ["gpu1 post-claim free 288MB < margin 500MB"]
    verdict = build_service_curve_verdict(
        run_id="rl",
        node="node",
        steps=80,
        total_jobs_for_proxy=15,
        proxy_job_counts=[15],
        expected_sweetspot_count=0,
        min_two_vs_three_gain=0.0,
        summaries={
            2: annotate_expected_placement(_iter_summary(2, 0.10), [0, 0]),
            3: annotate_expected_placement(_iter_summary(3, 0.09), [0, 0, 0]),
            14: annotate_expected_placement(
                _iter_summary(14, 0.02) | {"per_gpu_running": {"1": 14}},
                [1] * 14,
            ),
            15: boundary,
        },
    )
    check("capacity boundary is reported without failing validated curve",
          verdict["pass"]
          and verdict["capacity_boundaries"]
          and verdict["capacity_boundaries"][0]["count_per_gpu"] == 15,
          diag=str(verdict))
    check("capacity boundary is excluded from best-count selection",
          verdict["best_flow_time_count_per_gpu"] != 15,
          diag=str(verdict))
