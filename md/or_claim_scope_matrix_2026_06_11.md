# OR Claim Scope Matrix

Date: 2026-06-11, Asia/Shanghai.

This note is the submission-facing claim boundary for the current Scheduleurm
theory and experiment package.  It is meant to prevent reviewer-facing wording
from mixing theorem certificates, replay baselines, local bucket measurements,
and future extensions.

## Main Safe Claim

```text
The current completed-active Scheduleurm-controlled production population has
full strict measured-bucket coverage, positive mapped-slice capacity slack, and
a service-map robust MaxWeight lower-service oracle bridge with alpha0=alpha1=0.
```

Primary artifacts:

```text
md/experiment_artifacts/module51_production_coverage_drilldown.json
md/experiment_artifacts/module49_production_load_strict.json
md/experiment_artifacts/module100_production_theorem_oracle_bridge.json
md/experiment_artifacts/module101_live_scheduler_oracle_closure.json
md/lean_verification_submission.md
```

## Claim Boundaries

| Topic | Current status | Safe wording | Do not claim |
|---|---|---|---|
| Production population | Module51 strict completed-active view is 3437/3437 mapped. | `completed_active_production` is the theorem-facing production population. | Raw history or attempted production is globally closed. |
| Raw scheduler history | Module49 raw-window global coverage is intentionally false. | Raw history is operational telemetry and a source for bucket discovery. | Raw history is a clean arrival stream for the theorem. |
| Service-map oracle bridge | Module100 is `SERVICE_MAP_THEOREM_ORACLE_PASS`. | Measured service-map lower-service oracle bridge closes alpha0=alpha1=0 for the completed-active population. | Module100 is a live dispatch trace. |
| Live scheduler trace | Module101 is `LIVE_SCHEDULER_THEOREM_ORACLE_PASS` for two local CPU control-plane slots. | One emitted live scheduler trace has been enriched and audited through Module50 -> Module55 -> Module52. | Every future dispatch is automatically theorem-grade. |
| SOTA comparison | Module27/29 are SOTA-style policy semantics on the same measured service cache. | Scheduleurm is not Pareto-dominated by the SOTA-style replay baselines under this cache. | Directly beats Gavel/Pollux/Sia/IADeep binaries or full external stacks. |
| q00 local control | Profiles 1-13 measured; profile 14 boundary. | q00 is closed for the declared local light-control bucket. | q00 represents every low-resource control or data-loader workload. |
| q10 local CPU | Profiles 1-9 measured; profile 10 boundary. | q10 is closed for the declared local CPU-heavy bucket. | q10 represents all remote CPU-node or data-loader-heavy deployments. |
| General fabric cover \(L,\rho\) | Lean theorem is complete; broad empirical calibration is not complete. | Generalized fabric-cover claims require a named feature map, envelope \(L\), and cover radius \(\rho\). | Small \(L\rho\) is established for all feasible cluster states. |
| Exact measured finite slices | Module48/100 use measured finite service maps with zero modeled cover/oracle loss where exact action rows are enumerated. | Exact finite-slice certificates can use the measured service cache directly. | Exact finite-slice certificates prove a global Lipschitz fabric metric. |
| Active-bucket learning | Lean gives conditional/high-probability event lifting. | Active-bucket learning is an extension theorem conditional on sampler and feedback certificates. | Complete online learning theorem for the current scheduler sampler. |
| Hidden regimes | Lean gives dwell/switching backlog-budget theorem. | Hidden-regime stability is an extension requiring dwell/delay/switching evidence. | Average-regime stability is part of the main theorem. |

## Required Paper Wording

Use:

```text
SOTA-style replay baselines reproduce representative policy semantics on the
same measured Scheduleurm service cache.
```

Do not use:

```text
Scheduleurm directly outperforms Gavel, Pollux, Sia, or IADeep.
```

Use:

```text
The q00 and q10 results are declared local-bucket certificates.
```

Do not use:

```text
The q00 and q10 results generalize to all CPU/data-loader workloads.
```

Use:

```text
The live oracle bridge is per emitted trace and must be rerun for future online
oracle claims.
```

Do not use:

```text
The scheduler is now permanently theorem-grade for every future dispatch.
```

## General Fabric Calibration Gate

A broad fabric-cover claim is allowed only after a table provides:

```text
feature map Phi(a)
feature weights w_r
metric d_Phi(a,a')
candidate projection pi(a)
cover population: fixed / statewise / regimewise / sampled / all feasible
rho = max_a d_Phi(a, pi(a))
service sensitivity samples
L = certified worst-case or confidence-envelope slope
Lrho = L * rho
failure probability, if the envelope is statistical
```

Until then, paper claims should distinguish exact measured finite-slice
certificates from generalized fabric-cover calibration.
