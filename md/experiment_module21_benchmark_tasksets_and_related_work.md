# Benchmark Tasksets And Related Scheduler Experiments

Date: 2026-06-04

This note fixes the next experiment shape before more scheduler modules are
enabled. The goal is to keep Scheduleurm's implementation, replay validation,
and `math.md` route aligned: measure real service curves where the theorem needs
constants, then replay large traces instead of waiting for every long RL job to
finish.

## What Prior Scheduler Papers Actually Do

The common pattern is not full real execution for every large-scale policy
sweep. The pattern is:

1. profile representative jobs on real hardware;
2. build throughput or goodput tables;
3. run larger trace-driven or discrete-event simulations across seeds, arrival
   rates, and cluster sizes;
4. keep a smaller physical-cluster experiment for validation and system
   overhead.

Examples:

| System | Environment pattern | Reusable part for Scheduleurm |
|---|---|---|
| Gavel | Simulator plus physical cluster. Its public repo says Gavel can be evaluated in simulation or physical cluster, and its sweep script generates continuous traces from seeds and Poisson arrival rates. | Use measured throughput tables, static/continuous traces, seed sweeps, policy runtime logs. Do not import its throughputs as Scheduleurm co-location curves. |
| Pollux | 64-GPU testbed plus a discrete-time simulator. The paper builds the simulator from measured model performance and gradient statistics, then replays workload traces. | Use goodput/service-cache replay and periodic re-optimization. Its profiling/replay split matches our fast-forward layer. |
| Tiresias | 60-GPU testbed plus large-scale trace-driven simulation using Microsoft production traces. | Use testbed only for profile/validation; use trace replay for scale and workload variation. |
| THEMIS | Testbed and simulator workloads derived from public Microsoft traces, with CV/NLP/Speech model mixes and synthetic network-intensive workload sweeps. | Add placement/interference mixes, not only homogeneous RL batches. |
| Sia | Physical 44-GPU experiments, but most broader results use the Pollux simulator extended for heterogeneity and production-derived traces. | Use category mapping from real traces to representative jobs, plus larger simulator sweeps after the local service cache is calibrated. |
| IADeep | Real 20 RTX 3090-GPU cluster evaluation for GPU multiplexing. | Closest multiplexing benchmark family. Its benchmark model list is useful for future real probes: BERT, LSTM, ResNet50, SqueezeNet, VGG16, YOLOv5, NeuMF, ADGCL. |

Local reference repos already cloned under `reference/repos/`:

- `reference/repos/gavel`
- `reference/repos/adaptdl_pollux`
- `reference/repos/iadeep`
- `reference/repos/decima_sim`
- `reference/repos/salus`

## Scheduleurm Tasksets

The tasksets now live in `simulation/tasksets.py`. They use a CPU/GPU pressure
surface rather than ad hoc names.

| Taskset | Quadrant | Purpose | Current measurement status |
|---|---|---|---|
| `q00_light_control` | low CPU, low GPU | Short/light control for scheduler overhead and queue churn. | Probe required. |
| `q01_gpu_bound_compute` | low CPU, high GPU | Pure GPU compute saturation curve. | `gpu_heavy_jax_matmul` profiles 1-8 are clean real measurements on `jtl110gpu2`. |
| `q10_cpu_host_bound` | high CPU, low GPU | CPU/host saturation separate from GPU placement. | Protocol replay curve only; needs a real CPU/data-loader benchmark before theorem-grade claims. |
| `q11_cpu_gpu_coupled` | high CPU, high GPU | RE-SAC/BAPR-like coupled RL where 4-5/GPU can remain near solo ETA. | `hybrid_rl_resac_ant` profiles 1-12 are clean real measurements; profile 13 hit runtime OOM/invalid placement and closes the higher-profile measurement obligation for this node bucket. |
| `hybrid_research_portfolio` | mixed | Post-module portfolio similar to Gavel/Pollux/Sia trace replay. | Replay-ready for current validation: RE-SAC hybrid, JAX GPU-heavy, CPU protocol. |

This is deliberately stricter than a pure simulator. GPU and hybrid co-location
profiles are exact-cache only: `5/GPU` is not inferred from `1/GPU`, and missing
profiles remain measurement obligations.

## External Benchmarks To Reuse

Reusable benchmark ideas:

- Gavel's model naming and trace machinery: ResNet-18/50, Transformer, A3C,
  language model, recommendation models, static traces, continuous Poisson
  arrivals, and measured-throughput JSONs.
- Pollux/Sia's workload construction: map trace jobs by GPU-hour category
  (small, medium, large, XL) to representative jobs, then replay several seeds.
- THEMIS's mix dimensions: CV/NLP/Speech categories and explicit
  compute-intensive versus network-intensive mixes.
- IADeep's multiplexing models: BERT, LSTM, ResNet50, SqueezeNet, VGG16,
  YOLOv5, NeuMF, ADGCL.

Not reusable without remeasurement:

- Gavel/Pollux/Sia throughput values. They measure GPU type and allocation
  throughput, not Scheduleurm's same-GPU co-location interference curve.
- Themis production trace labels where model/runtime details are absent.
- IADeep's full stack overheads, because it uses Kubernetes middleware and GPU
  multiplexing controls not currently present in Scheduleurm.

## Immediate Validation Workflow

1. Use the four quadrant tasksets to validate one module at a time.
2. For each workload key and co-location profile, run a real probe only until
   task-native progress has a stable service estimate.
3. Write the measured service profile to the cache.
4. Replay the full taskset with bootstrap variation.
5. Compare legacy fixed caps against the calibrated policy by total completion
   of all `n` tasks, not by first-task completion.
6. Only push a live scheduling module after performance validation passes.

The current replay-ready portfolio is still:

```bash
python3 -m simulation.cli \
  --taskset hybrid_research_portfolio \
  --trials 101 \
  --seed 42 \
  --min-makespan-improvement 1.05 \
  --min-class-improvement 1.03
```

For `q01_gpu_bound_compute`, the calibrated replay objective is all-task
makespan. With profiles 1-8 measured, the q01 standalone replay selects 8/GPU
against the legacy 3/GPU cap and improves makespan, while mean flow is worse.
Report that as a makespan result, not a generic latency result.

The CLI refuses incomplete tasksets by default. For example, `q01_gpu_bound_compute`
will not run as a full benchmark until profiles 4-8 are measured. Exploratory
partial replay requires `--allow-partial-taskset`, and should not be reported as
the quadrant result.

The next real probes should fill:

- `q01_gpu_bound_compute`: profiles 1-8 are now measured; next only repeat on a second node/GPU if fabric calibration needs replication;
- `q11_cpu_gpu_coupled`: profile 13 is already a measured capacity boundary on
  this node bucket; do not spend GPU time on 14-16 unless we intentionally
  change memory settings, task template, or node bucket;
- `q00_light_control`: one tiny real command family;
- `q10_cpu_host_bound`: a real CPU-heavy or data-loader-heavy command family.

## Source Links

- Gavel OSDI page: <https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak>
- Gavel public repo: <https://github.com/stanford-futuredata/gavel>
- Pollux paper PDF: <https://www.pdl.cmu.edu/PDL-FTP/CloudComputing/osdi21-pollux.pdf>
- Tiresias NSDI page: <https://www.usenix.org/conference/nsdi19/presentation/gu>
- THEMIS paper PDF: <https://utns.cs.utexas.edu/assets/papers/nsdi20-themis.pdf>
- Sia paper PDF: <https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sia_sosp23-final.pdf>
- IADeep SC23 abstract page: <https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap290.html>
