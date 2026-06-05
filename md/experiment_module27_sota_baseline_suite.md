# Module27 SOTA-Style Baseline Suite

Date: 2026-06-05

This module adds a SOTA-style replay baseline suite. It is stronger than
Scheduleurm legacy comparison, but still not a direct binary-to-binary execution
of external schedulers. Every baseline uses the same Scheduleurm measured
service cache, so the comparison isolates policy semantics from hardware,
profiling, and trace mismatch.

## Baselines

No single published scheduler cleanly covers all four Scheduleurm quadrants:
low/low control, GPU-heavy co-location, CPU-heavy host saturation, and coupled
CPU/GPU RL. The suite therefore uses multiple representative policy families.

| Baseline | Representative systems | Main objective | Coverage |
|---|---|---|---|
| `throughput_table_goodput` | Gavel, Pollux, Sia | measured throughput/goodput table; minimize all-task makespan proxy | q00, q10, q11 |
| `delay_oracle` | SRPT, Gittins, SERPT | minimize deterministic mean completion time proxy | q00, q01, finite-batch delay |
| `interference_guard` | IADeep, Salus | interference-aware guarded co-location with statewise drain | q01, q11 |
| `quadrant_composite` | mixed | per-quadrant composite where no single SOTA system applies | all quadrants |

These are intentionally SOTA-style baselines. Direct SOTA execution remains a
future layer unless we run external scheduler code or faithfully reproduce its
full allocation semantics on the same trace.

Reference anchors:

```text
Gavel: https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak
Pollux: https://www.usenix.org/system/files/osdi21-qiao.pdf
IADeep: https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap290.html
```

## Candidate Upgrade

Module26 used per-workload guarded statewise selection. Module27 adds a
multi-workload global guarded selector:

```text
preserve portfolio total makespan within a global guard;
inside that guard, minimize weighted mean flow.
```

For a single-workload taskset, module28 changes q01 from the module26 delay
endpoint to the guarded knee:

```text
q01 GPU-heavy: guarded knee profile 4
q11 hybrid RL: throughput/support profile 10
```

For the mixed portfolio after q10 promotion, the real local CPU bucket remains
the host-pressure member. The selector spends slack on the hybrid RL job while
using the same measured local q10 curve as the standalone q10 taskset:

```text
hybrid_rl_resac_ant: 10/GPU
gpu_heavy_jax_matmul: 1/GPU
cpu_heavy_local_bench: 8 local CPU workers
```

This is not a weakening of the math route. It is a better implementation of the
same support-plus-bounded-penalty idea at portfolio scope.

## Commands

```bash
python3 -m simulation.sota_cli --taskset q00_light_control --trials 31 --seed 42
python3 -m simulation.sota_cli --taskset q01_gpu_bound_compute --trials 31 --seed 42
python3 -m simulation.sota_cli --taskset q10_cpu_host_bound --trials 31 --seed 42
python3 -m simulation.sota_cli --taskset q11_cpu_gpu_coupled --trials 31 --seed 42
python3 -m simulation.sota_cli --taskset hybrid_research_portfolio --trials 31 --seed 42
```

The pass condition is Pareto-based:

```text
pass = no SOTA-style baseline has both makespan <= candidate and mean-flow <= candidate,
       with one strict improvement.
```

`candidate_beats_all_sota_style_makespan` and
`candidate_beats_all_sota_style_mean_flow` are still reported, but they are not
required simultaneously because throughput-only and delay-only baselines encode
different objectives.

## Results

Standalone quadrants:

| Taskset | Candidate profile | SOTA-style Pareto status |
|---|---|---|
| `q00_light_control` | `light_control_local=8` | not dominated |
| `q01_gpu_bound_compute` | `gpu_heavy_jax_matmul=4` | not dominated |
| `q10_cpu_host_bound` | `cpu_heavy_local_bench=8` | not dominated |
| `q11_cpu_gpu_coupled` | `hybrid_rl_resac_ant=10` | not dominated |

q01 details:

| Baseline | Candidate vs baseline makespan | Candidate vs baseline mean-flow |
|---|---:|---:|
| throughput-table goodput | 0.985x | 1.111x |
| delay oracle | 1.015x | 0.910x |
| interference guard | 1.015x | 0.910x |
| quadrant composite | 1.000x | 1.000x |

q01 therefore loses about 1.5% makespan to a pure throughput-table objective,
but wins about 11% mean-flow. Against the delay oracle it wins makespan and
loses mean-flow. This is the Pareto-knee version of the tradeoff, not a module
failure.

q11 details:

| Baseline | Candidate vs baseline makespan | Candidate vs baseline mean-flow |
|---|---:|---:|
| throughput-table goodput | 1.000x | 1.000x |
| delay oracle | 1.028x | 0.983x |
| interference guard | 1.063x | 1.044x |
| quadrant composite | 1.000x | 1.000x |

q11 wins makespan against the delay oracle but gives up about 1.7% mean-flow.
Again, this is a tradeoff between throughput support and finite-batch delay.

Portfolio details:

| Baseline | Candidate vs baseline makespan | Candidate vs baseline mean-flow |
|---|---:|---:|
| throughput-table goodput | 1.001x | 0.999x |
| delay oracle | 1.028x | 0.977x |
| interference guard | 1.000x | 1.000x |
| quadrant composite | 1.000x | 1.000x |

The global guarded selector is not Pareto-dominated by any SOTA-style baseline
on the mixed portfolio.

Against Scheduleurm legacy on the same portfolio:

| Metric | Legacy | Candidate | Improvement |
|---|---:|---:|---:|
| Total makespan | 33043.785 s | 26911.926 s | 1.228x |
| Weighted mean flow | 7599.635 s | 5924.731 s | 1.283x |

## Module-Correctness Check

The SOTA suite did not expose a broken algorithm module. It exposed objective
boundaries:

```text
q01: throughput-only makespan beats the guarded knee, but loses mean-flow.
q11: delay-only mean-flow beats throughput policy, but loses makespan.
portfolio: after real q10 promotion, throughput and delay endpoints are
           explicit tradeoffs rather than Pareto dominators.
```

Targeted tests:

```text
q10-promotion targeted suite: checks=81 failed=0
```

The next unresolved baseline step is direct external-policy reproduction, not
fixing a failing Scheduleurm module. The reviewer-facing exact task-list matrix
is recorded in `md/experiment_module29_task_list_benchmark.md`.
