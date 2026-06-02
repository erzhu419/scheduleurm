# Reference Collection and Experiment Readiness Notes

Date: 2026-05-29, Asia/Shanghai.

This note records the current non-experimental boundary, the GPU resources needed for the first Scheduleurm experiments, and the external papers/repos collected from `md/GPT.md`.

## 1. Non-Experimental Boundary

Without running new measurements, the math/proof/algorithm interface is mostly at the point where further progress becomes calibration-limited.

What can still be done without experiments:

- tighten documentation and artifact maps;
- run synthetic/offline tests for the algorithm modules;
- compare against external simulator designs;
- prepare trace schemas and plotting/reporting scripts.

What cannot be honestly closed without experiments:

- numerical values for `L`, `rho`, `epsilon_est`, `beta`, `alpha0`, `alpha1`, `delta`, and empirical second-moment bounds;
- the actual GPU co-location service curve for Scheduleurm workloads;
- ETA error envelopes, especially the early-round overestimate/underestimate behavior;
- end-to-end comparison of baseline Scheduleurm versus candidate-set robust MaxWeight/admission-control variants.

So the answer is: theory and code scaffolding are not literally impossible to extend, but the important remaining claims are now measurement-limited. Further non-experimental changes should support experiments, not pretend to replace them.

## 2. Current Scheduler State

Snapshot from `python3 skill/scheduler.py status --json` / queue state:

| Node | GPU | Running tasks |
| --- | ---: | ---: |
| `jtl110gpu` | 0 | 4 |
| `jtl110gpu` | 1 | 4 |
| `jtl110gpu` | node-level / unknown GPU | 1 |
| `jtl110gpu2` | 0 | 4 |
| `jtl110gpu2` | 1 | 4 |
| `local` | 0 | 3 |
| `node007-direct` | 0 | 4 |
| `node007-direct` | 1 | 4 |
| `node007-direct` | 2 | 4 |
| `node007-direct` | 3 | 4 |

There were also 206 queued tasks. This is useful for passive observation but not a clean intervention experiment state.

## 3. Minimum Free Resources

Absolute minimum:

- 1 idle GPU can run sequential solo/co-location microprofiling.
- This is enough to collect some service-rate samples, but it is slow and weak because baseline and treatment are separated in time.

Practical minimum:

- 2 idle GPUs on the same node.
- Use one GPU as control/solo baseline and the other as treatment/co-location.
- This is the minimum I would use for theorem-facing calibration of service degradation, ETA error, and rollback/admission effects.

Recommended:

- one whole 2-GPU node, or at least 4 idle GPUs if using `node007-direct`;
- preferably two nodes if we want to test statewise/dynamic feasible-family effects across node types.

For the current cluster, the first clean experiment should wait until at least two GPUs on the same node are drained. A whole idle node is better because CPU/env contention and host-level bandwidth are part of the fabric metric.

## 4. Do Jobs Need To Finish?

No, not for most calibration.

For service curves, the unit should be measured progress per slot, such as iterations/sec, environment steps/sec, updates/sec, samples/sec, or a reliable log-derived progress counter. A typical design is:

- warm up for several slots and discard early ETA spikes;
- measure stable windows under fixed co-location profiles;
- stop or roll back jobs after enough samples;
- use the measured service model to simulate completion in offline replay.

ETA alone is not a theorem-grade service signal unless we also estimate a conservative ETA error envelope. It is especially risky in the first few rounds, where Scheduleurm already shows exaggerated ETA behavior. ETA should be logged, audited, and possibly used as a secondary signal, not as the primary throughput curve.

Some completed jobs are still useful later:

- to validate completion-time/JCT predictions;
- to measure checkpoint/restart/rollback overhead;
- to show end-to-end benefit against the baseline.

But the first service-curve and robust-scheduler experiments do not need a large batch of 12-18 hour jobs to run to completion.

## 5. Reference Download State

Directory layout:

- papers: `reference/papers/`
- saved landing pages: `reference/pages/`
- repository clones: `reference/repos/`
- manifests: `reference/metadata/`

Extraction from `md/GPT.md` found 30 unique reference links.

Downloaded or saved:

- 26 PDF files in `reference/papers/`;
- 6 saved landing/metadata pages in `reference/pages/`;
- manifests:
  - `reference/metadata/gpt_references.tsv`
  - `reference/metadata/download_manifest.tsv`
  - `reference/metadata/download_manifest_supplemental.tsv`
  - `reference/metadata/download_manifest_retry_curl.tsv`
  - `reference/metadata/repo_manifest.tsv`

Git tracking boundary:

- `reference/README.md` and `reference/metadata/*.tsv` are intended to be tracked.
- `reference/papers/`, `reference/pages/`, and `reference/repos/` are local artifact caches and are ignored by git.
- This keeps the 264MB PDF/repo cache available locally without embedding third-party binaries and nested git clones into the Scheduleurm repository.

Manifest validation passed on 2026-06-03:

- `gpt_references.tsv`: 30 extracted links from `md/GPT.md`;
- `download_manifest.tsv`: 30 rows, 26 local artifact paths checked;
- `download_manifest_supplemental.tsv`: 7 rows, 4 local artifact paths checked;
- `download_manifest_retry_curl.tsv`: 2 rows, 1 local artifact path checked;
- `repo_manifest.tsv`: 6 cloned repositories, local commits match the manifest.

Important access notes:

- IADeep SC23 full paper appears closed through ACM/SC proceedings; the SC abstract page and SC23 slides were saved, and the official code repo was cloned.
- RIFLING Wiley returned 403 for both article and PDF from this environment; no open PDF was found in this pass.
- The JPDC topology-aware GPU scheduling paper appears closed; ScienceDirect metadata was found but no open PDF/repo was found in this pass.
- The INFORMS Stochastic Systems PDF endpoints returned 403 in Python/curl, but open arXiv alternatives were downloaded for the state-dependent LPS paper and queueing-network DRL paper.
- The Cambridge admission-control page did not return a PDF directly, but a CWI report PDF for the same title was downloaded.
- The DPS survey final Springer PDF did not download; Springer/CWI metadata pages were saved, and CiteSeerX returned 404 from this environment.
- OPRE submission guidelines are not a research paper and returned 403 from this environment.

## 6. Cloned Repositories

The following were cloned under `reference/repos/` with depth 1:

| Local path | Source | Why it matters |
| --- | --- | --- |
| `reference/repos/iadeep` | `https://github.com/buzy-coder/IADeep.git` | Closest to our co-located GPU multiplexing problem; includes Kubernetes scheduler extender, device plugin, local coordinator, tuner, benchmark workloads, and evaluation scripts for JCT/makespan/utilization. |
| `reference/repos/adaptdl_pollux` | `https://github.com/petuum/adaptdl.git` | Pollux/AdaptDL code path for goodput/speedup modeling and cluster-wide allocation. Useful for how to convert profiling into a scheduling objective. |
| `reference/repos/gavel` | `https://github.com/stanford-futuredata/gavel.git` | Best experimental template: simulator, measured-throughput JSONs, Poisson-arrival trace sweeps, policy sweeps, physical-cluster driver, and one-GPU overhead microbenchmarks. |
| `reference/repos/decima_sim` | `https://github.com/hongzimao/decima-sim.git` | Clean RL scheduling simulator with Poisson arrivals and train/test split. Useful for Scheduleurm offline replay and queue-control baselines. |
| `reference/repos/salus` | `https://github.com/SymbioticLab/Salus.git` | Fine-grained GPU sharing primitives and benchmark scripts. Less directly usable because it depends on old/custom TensorFlow, but useful for measuring GPU sharing overhead concepts. |
| `reference/repos/awesome_dl_scheduling_papers` | `https://github.com/S-Lab-SystemGroup/Awesome-DL-Scheduling-Papers.git` | Survey companion list with paper/code links; useful for follow-up repo discovery. |

Salus references `SymbioticLab/tensorflow-salus`; I did not clone that TensorFlow fork in this pass because it is likely large and is not needed for reading the experiment design.

## 7. External Experiment Patterns To Reuse

From Gavel:

- use measured throughput tables as simulator input;
- sweep Poisson arrival rates and seeds;
- separate continuous-arrival and static-trace experiments;
- include policy runtime scaling;
- allow in-progress log parsing instead of waiting for every overloaded trace to finish;
- explicitly stop sweeps whose arrival rate exceeds cluster capacity.

From IADeep:

- compare against GPU sharing baselines by JCT, makespan, and SM/memory utilization;
- record scheduling/tuning/search-round overhead separately;
- treat task configuration tuning and placement as coupled decisions.

From Pollux/AdaptDL:

- define goodput/speedup functions from observed performance parameters;
- optimize allocations periodically rather than only at submission time;
- keep the calibration model separate from the policy.

From Decima:

- maintain a simulator with stochastic arrivals and much longer test traces than training traces;
- separate policy-learning experiments from deterministic replay/evaluation.

From Salus:

- measure GPU sharing overhead at the iteration/memory-management level;
- do not rely on aggregate GPU utilization alone.

## 8. Immediate Experiment Implication

The first Scheduleurm experiment should not start by dispatching hundreds of long jobs and waiting for all to finish.

A better first experiment is:

1. Drain at least two GPUs on the same node.
2. Pick a small set of representative RL task classes already supported by Scheduleurm.
3. Run solo baseline slots and fixed co-location-profile slots.
4. Export progress/ETA/resource snapshots through the new `algorithm/experiments` modules.
5. Fit service-rate, penalty, ETA-error, and oracle-error bounds.
6. Replay arrivals offline to compare baseline Scheduleurm against the robust candidate-set policy.
7. Only after that, run a smaller live end-to-end validation with real completions.

This keeps proof, math route, and experiment aligned: experiments estimate the constants and service model required by the theorem-facing artifacts, rather than becoming a disconnected benchmark.
