import tempfile
from pathlib import Path

from simulation.defaults import build_default_cache
from simulation.trace_benchmark import (
    build_task_trace,
    load_trace,
    replay_trace_suite,
    save_trace,
)


def _candidate_row(report):
    return next(row for row in report["results"] if row["policy"].startswith("calibrated_"))


def _row(report, policy_name):
    return next(row for row in report["results"] if row["policy"] == policy_name)


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


def test_static_trace_replay_passes_four_quadrants_and_portfolio(check, sch):
    cache = build_default_cache()
    expected_profiles = {
        "q00_light_control": {"light_control_local": 8},
        "q01_gpu_bound_compute": {"gpu_heavy_jax_matmul": 4},
        "q10_cpu_host_bound": {"cpu_heavy_protocol": 16},
        "q11_cpu_gpu_coupled": {"hybrid_rl_resac_ant": 10},
        "hybrid_research_portfolio": {
            "cpu_heavy_protocol": 16,
            "gpu_heavy_jax_matmul": 1,
            "hybrid_rl_resac_ant": 1,
        },
    }
    for taskset_name, profiles in expected_profiles.items():
        trace = build_task_trace(taskset_name, arrival_mode="static", seed=42)
        report = replay_trace_suite(cache, trace, seed=7)
        candidate = _candidate_row(report)
        rel = report["relative_to_legacy"][candidate["policy"]]
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
