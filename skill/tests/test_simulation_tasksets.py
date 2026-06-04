from simulation.defaults import build_default_cache, default_workload_specs
from simulation.tasksets import all_missing_measurements, benchmark_tasksets, empirical_replay_taskset, taskset_by_name


def test_taskset_registry_has_four_quadrants_and_hybrid(check, sch):
    tasksets = benchmark_tasksets()
    expected = {
        "q00_light_control",
        "q01_gpu_bound_compute",
        "q10_cpu_host_bound",
        "q11_cpu_gpu_coupled",
        "hybrid_research_portfolio",
    }
    check("taskset registry exposes four quadrants plus hybrid portfolio",
          expected.issubset(set(tasksets)),
          diag=str(sorted(tasksets)))
    quadrants = {
        member.quadrant
        for name in expected
        for member in tasksets[name].members
    }
    check("taskset quadrants cover the CPU/GPU pressure surface",
          {
              "low_cpu_low_gpu",
              "low_cpu_high_gpu",
              "high_cpu_low_gpu",
              "high_cpu_high_gpu",
          }.issubset(quadrants),
          diag=str(sorted(quadrants)))


def test_empirical_replay_taskset_matches_default_specs(check, sch):
    taskset = empirical_replay_taskset()
    expected = [(x.workload_key, x.task_count, x.resource_count) for x in taskset.workload_specs()]
    actual = [(x.workload_key, x.task_count, x.resource_count) for x in default_workload_specs()]
    check("default replay specs come from the hybrid research taskset",
          actual == expected,
          diag=f"actual={actual} expected={expected}")


def test_hybrid_research_portfolio_has_no_missing_empirical_gpu_profiles(check, sch):
    cache = build_default_cache()
    missing = taskset_by_name("hybrid_research_portfolio").missing_measurements(cache)
    check("hybrid replay portfolio has exact measured GPU/hybrid profiles required for validation",
          missing == {},
          diag=str(missing))


def test_full_quadrant_tasksets_surface_probe_obligations(check, sch):
    cache = build_default_cache()
    missing = all_missing_measurements(cache)
    check("light control remains a real-probe obligation",
          missing.get("q00_light_control", {}).get("light_control_protocol") == [1, 2, 4, 8, 16],
          diag=str(missing))
    check("GPU-bound full saturation set refuses to invent profiles beyond measured curve",
          missing.get("q01_gpu_bound_compute", {}).get("gpu_heavy_jax_matmul") == [4, 5, 6, 7, 8],
          diag=str(missing))
    check("coupled RL set records remaining saturation probes",
          missing.get("q11_cpu_gpu_coupled", {}).get("hybrid_rl_resac_ant") == [14, 15, 16],
          diag=str(missing))


def test_replayable_only_filters_unmeasured_members(check, sch):
    cache = build_default_cache()
    light_specs = taskset_by_name("q00_light_control").workload_specs(replayable_only=True, cache=cache)
    hybrid_specs = taskset_by_name("hybrid_research_portfolio").workload_specs(replayable_only=True, cache=cache)
    check("unmeasured control taskset is not replayable yet",
          light_specs == [],
          diag=str(light_specs))
    check("validated hybrid portfolio remains replayable",
          len(hybrid_specs) == 3
          and {spec.workload_key for spec in hybrid_specs}
          == {"hybrid_rl_resac_ant", "gpu_heavy_jax_matmul", "cpu_heavy_protocol"},
          diag=str(hybrid_specs))
