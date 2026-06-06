import tempfile
from pathlib import Path

from simulation.defaults import build_default_cache
from simulation.trace_benchmark import (
    build_task_trace,
    load_trace,
    replay_trace_matrix,
    replay_trace_suite,
    save_trace,
)


def _candidate_row(report):
    return next(row for row in report["results"] if row["policy"].startswith("calibrated_"))


def _row(report, policy_name):
    return next(row for row in report["results"] if row["policy"] == policy_name)


def _aggregate_row(matrix, policy_key):
    return next(row for row in matrix["aggregate_by_policy"] if row["policy"] == policy_key)


def test_trace_builder_produces_explicit_q01_task_list(check, sch):
    trace = build_task_trace("q01_gpu_bound_compute", arrival_mode="static", seed=42)
    specs = trace.workload_specs()
    check("q01 trace is an explicit 48-job static task list",
          len(trace.jobs) == 48 and all(job.arrival_s == 0.0 for job in trace.jobs),
          diag=str(trace.snapshot()))
    check("q01 trace preserves workload and resource-pool metadata",
          len(specs) == 1
          and specs[0].workload_key == "gpu_heavy_jax_matmul"
          and specs[0].task_count == 48
          and specs[0].resource_count == 2,
          diag=str(specs))
    check("q01 trace job ids are stable and reviewer-readable",
          trace.jobs[0].job_id.startswith("q01_gpu_bound_compute:gpu_heavy_jax_matmul:"),
          diag=trace.jobs[0].job_id)


def test_trace_roundtrip_preserves_task_list(check, sch):
    trace = build_task_trace("q11_cpu_gpu_coupled", arrival_mode="poisson", seed=43)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace.json"
        save_trace(trace, path)
        loaded = load_trace(path)
    check("trace JSON roundtrip preserves task identities and arrivals",
          loaded.snapshot() == trace.snapshot(),
          diag=f"loaded={loaded.snapshot()} expected={trace.snapshot()}")


def test_q01_task_list_replay_compares_legacy_candidate_and_sota(check, sch):
    cache = build_default_cache()
    trace = build_task_trace("q01_gpu_bound_compute", arrival_mode="static", seed=42)
    report = replay_trace_suite(cache, trace, seed=7)
    candidate = _candidate_row(report)
    legacy_rel = report["relative_to_legacy"][candidate["policy"]]
    throughput = _row(report, "sota_gavel_pollux_sia_table_goodput")
    delay = _row(report, "sota_srpt_gittins_mean_flow_oracle")
    sota = report["sota_tasklist_comparison"]
    throughput_cmp = next(
        row for row in sota["rows"]
        if row["baseline"]["name"] == "throughput_table_goodput"
    )
    delay_cmp = next(
        row for row in sota["rows"]
        if row["baseline"]["name"] == "delay_oracle"
    )

    check("q01 task-list replay includes legacy, candidate, and SOTA-style policies",
          {
              "legacy_fixed_caps",
              candidate["policy"],
              "sota_gavel_pollux_sia_table_goodput",
              "sota_srpt_gittins_mean_flow_oracle",
              "sota_iadeep_salus_interference_guard",
              "sota_quadrant_composite",
          }.issubset({row["policy"] for row in report["results"]}),
          diag=str(report))
    check("q01 task-list replay candidate completes all jobs at the Pareto knee",
          candidate["completed_jobs"] == len(trace.jobs)
          and candidate["profiles"] == {"gpu_heavy_jax_matmul": 4},
          diag=str(candidate))
    check("q01 task-list replay candidate improves all-job time and mean flow over legacy",
          legacy_rel["makespan_improvement"] > 1.07
          and legacy_rel["mean_flow_improvement"] > 1.04,
          diag=str(report["relative_to_legacy"]))
    check("q01 task-list replay keeps the expected SOTA endpoint tradeoff",
          throughput["profiles"] == {"gpu_heavy_jax_matmul": 8}
          and delay["profiles"] == {"gpu_heavy_jax_matmul": 1}
          and candidate["makespan_s"] > throughput["makespan_s"]
          and candidate["mean_flow_s"] < throughput["mean_flow_s"]
          and candidate["makespan_s"] < delay["makespan_s"]
          and candidate["mean_flow_s"] > delay["mean_flow_s"],
          diag=str(report["results"]))
    check("q01 task-list replay candidate is not Pareto-dominated by SOTA-style baselines",
          sota["candidate_not_pareto_dominated"]
          and sota["candidate_pareto_dominated_by"] == [],
          diag=str(sota))
    check("q01 SOTA comparison reports throughput and delay tradeoffs explicitly",
          throughput_cmp["candidate_vs_baseline_makespan"] < 1.0
          and throughput_cmp["candidate_vs_baseline_mean_flow"] > 1.10
          and delay_cmp["candidate_vs_baseline_makespan"] > 1.0
          and delay_cmp["candidate_vs_baseline_mean_flow"] < 1.0,
          diag=str(sota))


def test_static_trace_replay_passes_four_quadrants_and_portfolio(check, sch):
    cache = build_default_cache()
    expected_profiles = {
        "q00_light_control": {"light_control_local": 13},
        "q01_gpu_bound_compute": {"gpu_heavy_jax_matmul": 4},
        "q10_cpu_host_bound": {"cpu_heavy_local_bench": 8},
        "q11_cpu_gpu_coupled": {"hybrid_rl_resac_ant": 10},
        "hybrid_research_portfolio": {
            "cpu_heavy_local_bench": 8,
            "gpu_heavy_jax_matmul": 1,
            "hybrid_rl_resac_ant": 10,
        },
    }
    for taskset_name, profiles in expected_profiles.items():
        trace = build_task_trace(taskset_name, arrival_mode="static", seed=42)
        report = replay_trace_suite(cache, trace, seed=7)
        candidate = _candidate_row(report)
        rel = report["relative_to_legacy"][candidate["policy"]]
        sota = report["sota_tasklist_comparison"]
        completed = all(row["completed_jobs"] == len(trace.jobs) for row in report["results"])
        check(f"{taskset_name} trace replay completes every listed job",
              completed,
              diag=str(report["results"]))
        check(f"{taskset_name} candidate profile matches the validated module choice",
              candidate["profiles"] == profiles,
              diag=str(candidate))
        check(f"{taskset_name} candidate beats legacy on all-job time and mean flow",
              rel["makespan_improvement"] >= 1.0 and rel["mean_flow_improvement"] >= 1.0,
              diag=str(report["relative_to_legacy"]))
        check(f"{taskset_name} candidate is not Pareto-dominated by SOTA-style baselines",
              sota["candidate_not_pareto_dominated"]
              and sota["candidate_pareto_dominated_by"] == [],
              diag=str(sota))


def test_fixed_sota_policy_matrix_runs_each_algorithm_on_all_tasksets(check, sch):
    cache = build_default_cache()
    tasksets = [
        "q00_light_control",
        "q01_gpu_bound_compute",
        "q10_cpu_host_bound",
        "q11_cpu_gpu_coupled",
        "hybrid_research_portfolio",
    ]
    matrix = replay_trace_matrix(cache, tasksets, arrival_mode="static", trace_seed=42, replay_seed=7)
    candidate = _aggregate_row(matrix, "scheduleurm_candidate")
    throughput = _aggregate_row(matrix, "sota_gavel_pollux_sia_table_goodput")
    delay = _aggregate_row(matrix, "sota_srpt_gittins_mean_flow_oracle")
    interference = _aggregate_row(matrix, "sota_iadeep_salus_interference_guard")
    composite = _aggregate_row(matrix, "sota_quadrant_composite")

    check("fixed-policy matrix aggregates the candidate as one method across all tasksets",
          candidate["completed_jobs"] == 1376
          and candidate["policies"] == ["calibrated_global_guarded", "calibrated_guarded_knee"],
          diag=str(candidate))
    check("every individual SOTA-style algorithm completes the same full task-list suite",
          all(row["completed_jobs"] == 1376 for row in (throughput, delay, interference, composite)),
          diag=str(matrix["aggregate_by_policy"]))
    check("throughput-table SOTA no longer dominates after real q10 promotion",
          throughput["candidate_vs_policy_sum_makespan"] >= 1.0
          and throughput["candidate_vs_policy_job_weighted_mean_flow"] >= 1.0,
          diag=str(throughput))
    check("delay-oracle SOTA is a fixed-policy aggregate tradeoff",
          delay["candidate_vs_policy_sum_makespan"] > 1.0
          and delay["candidate_vs_policy_job_weighted_mean_flow"] < 1.0,
          diag=str(delay))
    check("interference and composite SOTA policies do not beat candidate mean-flow in aggregate",
          interference["candidate_vs_policy_job_weighted_mean_flow"] >= 1.0
          and composite["candidate_vs_policy_job_weighted_mean_flow"] >= 1.0,
          diag=str(matrix["aggregate_by_policy"]))

    dominators = []
    for row in matrix["aggregate_by_policy"]:
        if row["policy_family"] != "sota_style":
            continue
        ms = row["candidate_vs_policy_sum_makespan"]
        flow = row["candidate_vs_policy_job_weighted_mean_flow"]
        if ms <= 1.0 and flow <= 1.0 and (ms < 1.0 or flow < 1.0):
            dominators.append(row["policy"])
    check("no individual fixed SOTA policy Pareto-dominates candidate in aggregate",
          dominators == []
          and matrix["aggregate_candidate_not_pareto_dominated"]
          and matrix["aggregate_candidate_pareto_dominated_by"] == [],
          diag=f"manual={dominators} report={matrix['aggregate_candidate_pareto_dominated_by']}")

    q01_makespan = next(
        row for row in matrix["winner_transfer"]
        if row["source_taskset"] == "q01_gpu_bound_compute"
        and row["objective"] == "best_all_job_makespan"
    )
    q01_flow = next(
        row for row in matrix["winner_transfer"]
        if row["source_taskset"] == "q01_gpu_bound_compute"
        and row["objective"] == "best_mean_flow"
    )
    check("q01's best-throughput SOTA winner is run on the full suite, not just q01",
          q01_makespan["winner_policy"] == "sota_gavel_pollux_sia_table_goodput"
          and q01_makespan["aggregate_completed_jobs"] == 1376,
          diag=str(q01_makespan))
    check("q01's best-delay SOTA winner is run on the full suite, not just q01",
          q01_flow["winner_policy"] == "sota_srpt_gittins_mean_flow_oracle"
          and q01_flow["aggregate_completed_jobs"] == 1376,
          diag=str(q01_flow))
