from simulation.defaults import build_default_cache, default_workload_specs
from simulation.tasksets import all_missing_measurements, benchmark_tasksets, empirical_replay_taskset, taskset_by_name


def test_taskset_registry_has_four_quadrants_and_hybrid(check, sch):
    tasksets = benchmark_tasksets()
    expected = {
        "q00_light_control",
        "q01_gpu_bound_compute",
        "q10_cpu_host_bound",
        "q10_cpu_host_bound_local_real_probe",
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
    check("light control has measured local curve closed by profile 14 boundary",
          "q00_light_control" not in missing,
          diag=str(missing))
    check("GPU-bound full saturation set has measured profiles 1-8",
          "q01_gpu_bound_compute" not in missing,
          diag=str(missing))
    check("coupled RL set is closed by measured profile 13 capacity boundary",
          "q11_cpu_gpu_coupled" not in missing,
          diag=str(missing))
    check("local real q10 probe has measured profiles 1-9 and capacity boundary 10",
          "q10_cpu_host_bound_local_real_probe" not in missing,
          diag=str(missing))


def test_replayable_only_filters_unmeasured_members(check, sch):
    cache = build_default_cache()
    light_specs = taskset_by_name("q00_light_control").workload_specs(replayable_only=True, cache=cache)
    q10_local_specs = taskset_by_name("q10_cpu_host_bound_local_real_probe").workload_specs(replayable_only=True, cache=cache)
    hybrid_specs = taskset_by_name("hybrid_research_portfolio").workload_specs(replayable_only=True, cache=cache)
    check("measured light-control taskset is replayable",
          len(light_specs) == 1 and light_specs[0].workload_key == "light_control_local",
          diag=str(light_specs))
    check("validated hybrid portfolio remains replayable",
          len(hybrid_specs) == 3
          and {spec.workload_key for spec in hybrid_specs}
          == {"hybrid_rl_resac_ant", "gpu_heavy_jax_matmul", "cpu_heavy_local_bench"},
          diag=str(hybrid_specs))
    check("local real q10 probe is replayable without protocol curve",
          len(q10_local_specs) == 1
          and q10_local_specs[0].workload_key == "cpu_heavy_local_bench",
          diag=str(q10_local_specs))


def test_capacity_boundary_closes_higher_required_profiles(check, sch):
    cache = build_default_cache()
    snapshot = taskset_by_name("q11_cpu_gpu_coupled").snapshot(cache)
    member = snapshot["members"][0]
    check("q11 snapshot records profile 13 as the closing boundary",
          member["closed_by_capacity_boundary_profile"] == 13,
          diag=str(snapshot))
    check("q11 missing-measurement list stops at the measured boundary",
          snapshot["missing_measurements"] == {},
          diag=str(snapshot))

    q00 = taskset_by_name("q00_light_control").snapshot(cache)
    check("q00 snapshot records profile 14 as the local light-control boundary",
          q00["members"][0]["closed_by_capacity_boundary_profile"] == 14,
          diag=str(q00))

    q10 = taskset_by_name("q10_cpu_host_bound").snapshot(cache)
    check("q10 real local snapshot records profile 10 as the CPU capacity boundary",
          q10["members"][0]["closed_by_capacity_boundary_profile"] == 10,
          diag=str(q10))
