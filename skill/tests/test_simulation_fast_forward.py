from simulation.defaults import (
    build_default_cache,
    calibrated_candidate_policy,
    calibrated_makespan_policy,
    calibrated_policy,
    default_workload_specs,
    legacy_policy,
)
from simulation.fast_forward import compare_policies
from simulation.service_cache import (
    ServiceRateCache,
    cache_needs_probe,
    deterministic_makespan_s,
    deterministic_mean_flow_s,
    missing_exact_profiles,
)
from simulation.sota_baselines import compare_against_sota_suite
from simulation.tasksets import taskset_by_name


def test_simulation_cache_has_cpu_gpu_hybrid_workloads(check, sch):
    cache = build_default_cache()
    workloads = set(cache.available_workloads())
    check("simulation cache includes hybrid RL workload",
          "hybrid_rl_resac_ant" in workloads,
          diag=str(workloads))
    check("simulation cache includes GPU-heavy workload",
          "gpu_heavy_jax_matmul" in workloads,
          diag=str(workloads))
    check("simulation cache includes CPU-heavy workload",
          "cpu_heavy_protocol" in workloads,
          diag=str(workloads))
    check("simulation cache includes real local CPU-heavy q10 probe",
          "cpu_heavy_local_bench" in workloads,
          diag=str(workloads))
    check("simulation cache includes light-control workload",
          "light_control_local" in workloads,
          diag=str(workloads))


def test_simulation_cache_reuses_existing_eta_profile(check, sch):
    cache = build_default_cache()
    check("new live RE-SAC profile-10 boundary forces a new scheduler probe",
          cache_needs_probe(cache, "hybrid_rl_resac_ant", 10))
    check("robust colocated RE-SAC profiles below the live boundary have exact measurements",
          missing_exact_profiles(cache, "hybrid_rl_resac_ant", range(1, 10)) == [])
    check("unknown RE-SAC profile still needs measurement",
          cache_needs_probe(cache, "hybrid_rl_resac_ant", 99))
    check("RE-SAC profile 13 is treated as unusable after runtime OOM",
          cache_needs_probe(cache, "hybrid_rl_resac_ant", 13),
          diag=str(cache.get("hybrid_rl_resac_ant", 13).snapshot()))


def test_calibrated_policy_selects_replay_makespan_profile(check, sch):
    cache = build_default_cache()
    makespan_record = cache.best_profile_for_makespan(
        "hybrid_rl_resac_ant",
        task_count=120,
        total_units=80,
        resource_count=1,
    )
    guarded_record = cache.best_profile_for_guarded_mean_flow(
        "gpu_heavy_jax_matmul",
        task_count=48,
        total_units=2400,
        resource_count=2,
        max_makespan_regret=0.05,
    )
    legacy = cache.get("hybrid_rl_resac_ant", 5)
    calibrated_ms = deterministic_makespan_s(
        task_count=120,
        total_units=80,
        resource_count=1,
        profile=makespan_record.profile,
        aggregate_rate=makespan_record.aggregate_rate,
    )
    legacy_ms = deterministic_makespan_s(
        task_count=120,
        total_units=80,
        resource_count=1,
        profile=legacy.profile,
        aggregate_rate=legacy.aggregate_rate,
    )
    check("calibrated RE-SAC profile stays below the live capacity boundary",
          makespan_record.profile == 3,
          diag=str(makespan_record.snapshot()))
    check("calibrated RE-SAC deterministic makespan beats legacy cap",
          legacy_ms / calibrated_ms > 1.15,
          diag=f"legacy={legacy_ms}, calibrated={calibrated_ms}")
    check("guarded GPU-heavy selector can prefer lower congestion within makespan slack",
          guarded_record.profile == 1,
          diag=str(guarded_record.snapshot()))
    check("deterministic mean-flow proxy is finite for measured GPU profiles",
          deterministic_mean_flow_s(
              task_count=48,
              total_units=2400,
              resource_count=2,
              profile=guarded_record.profile,
              aggregate_rate=guarded_record.aggregate_rate,
          ) > 0.0,
          diag=str(guarded_record.snapshot()))


def test_fast_forward_replay_candidate_beats_legacy_portfolio(check, sch):
    cache = build_default_cache()
    comparison = compare_policies(
        cache,
        default_workload_specs(),
        baseline=legacy_policy(),
        candidate=calibrated_candidate_policy(cache, default_workload_specs()),
        trials=31,
        seed=42,
    )
    check("trace-driven replay improves total makespan over legacy caps",
          comparison.makespan_improvement > 1.05,
          diag=str(comparison.snapshot()))
    check("trace-driven replay improves empirical GPU and hybrid classes",
          comparison.per_workload_improvements["hybrid_rl_resac_ant"]["makespan_improvement"] > 1.05
          and comparison.per_workload_improvements["gpu_heavy_jax_matmul"]["makespan_improvement"] > 1.03,
          diag=str(comparison.snapshot()))
    check("guarded statewise replay fixes GPU-heavy mean-flow regression",
          comparison.per_workload_improvements["gpu_heavy_jax_matmul"]["mean_flow_improvement"] > 1.03,
          diag=str(comparison.snapshot()))
    check("trace-driven replay keeps workload class decisions explicit",
          {
              row.workload_key: row.selected_profile
              for row in comparison.candidate.workloads
          } == {
              "hybrid_rl_resac_ant": 3,
              "gpu_heavy_jax_matmul": 1,
              "cpu_heavy_local_bench": 8,
          },
          diag=str(comparison.snapshot()))
    makespan_only = compare_policies(
        cache,
        default_workload_specs(),
        baseline=legacy_policy(),
        candidate=calibrated_makespan_policy(),
        trials=31,
        seed=42,
    )
    check("guarded statewise replay keeps portfolio mean-flow at least makespan-only",
          comparison.mean_flow_improvement >= makespan_only.mean_flow_improvement,
          diag=f"guarded={comparison.snapshot()} makespan_only={makespan_only.snapshot()}")


def test_sota_style_baselines_do_not_pareto_dominate_candidate(check, sch):
    cache = build_default_cache()
    for taskset_name, specs in (
        ("q00", taskset_by_name("q00_light_control").workload_specs()),
        ("q01", taskset_by_name("q01_gpu_bound_compute").workload_specs()),
        ("q10", taskset_by_name("q10_cpu_host_bound").workload_specs()),
        ("q11", taskset_by_name("q11_cpu_gpu_coupled").workload_specs()),
        ("portfolio", default_workload_specs()),
    ):
        report = compare_against_sota_suite(cache, specs, trials=31, seed=42)
        check(f"{taskset_name} candidate is not Pareto-dominated by SOTA-style suite",
              report["candidate_not_pareto_dominated"],
              diag=str(report))
    portfolio = compare_against_sota_suite(cache, default_workload_specs(), trials=31, seed=42)
    delay = next(row for row in portfolio["baselines"] if row["baseline"]["name"] == "delay_oracle")
    check("real-q10 portfolio keeps expected makespan/flow tradeoff against delay oracle",
          delay["candidate_vs_baseline_makespan"] > 1.0
          and delay["candidate_vs_baseline_mean_flow"] < 1.0,
          diag=str(portfolio))


def test_q01_default_candidate_uses_pareto_knee(check, sch):
    cache = build_default_cache()
    report = compare_against_sota_suite(
        cache,
        taskset_by_name("q01_gpu_bound_compute").workload_specs(),
        trials=31,
        seed=42,
    )
    throughput = next(row for row in report["baselines"] if row["baseline"]["name"] == "throughput_table_goodput")
    delay = next(row for row in report["baselines"] if row["baseline"]["name"] == "delay_oracle")
    check("q01 default candidate selects the guarded Pareto knee",
          throughput["candidate_profiles"] == {"gpu_heavy_jax_matmul": 4},
          diag=str(report))
    check("q01 knee trades less than 2% makespan for more than 10% mean-flow vs throughput SOTA",
          throughput["candidate_vs_baseline_makespan"] > 0.98
          and throughput["candidate_vs_baseline_mean_flow"] > 1.10,
          diag=str(report))
    check("q01 knee is a real tradeoff against the delay oracle",
          delay["candidate_vs_baseline_makespan"] > 1.0
          and delay["candidate_vs_baseline_mean_flow"] < 1.0,
          diag=str(report))


def test_fast_forward_refuses_unmeasured_colocation_profile(check, sch):
    cache = ServiceRateCache([build_default_cache().get("hybrid_rl_resac_ant", 1)])
    comparison_error = ""
    try:
        compare_policies(
            cache,
            default_workload_specs()[:1],
            baseline=legacy_policy(),
            candidate=calibrated_policy(),
            trials=1,
            seed=1,
        )
    except KeyError as exc:
        comparison_error = str(exc)
    check("fast-forward refuses to interpolate missing multi-task colocated ETA",
          "missing exact service profile" in comparison_error
          or "no service cache entries" in comparison_error,
          diag=comparison_error)
