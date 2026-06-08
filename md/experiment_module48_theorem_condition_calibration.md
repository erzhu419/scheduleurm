# Module48 Theorem-Condition Calibration

Date: 2026-06-08

This module closes the first theorem-condition accounting pass for the measured
`hybrid_research_portfolio` finite service-action slice.  It is deliberately
not a production-arrival claim: the load vector is declared from the measured
candidate service vector,

```text
lambda_i = 0.80 * selected_service_i.
```

That gives a load-certified finite-support model for the proof theorem without
pretending that the finite static replay trace has a stationary arrival rate.

## Artifacts

```text
algorithm/experiments/empirical_slack_certificate.py
algorithm/experiments/slack_accounting.py
md/experiment_artifacts/module48_portfolio_slack_certificate.json
md/experiment_artifacts/module48_portfolio_slack_certificate.md
```

## Certificate Scope

```text
taskset = hybrid_research_portfolio
policy = calibrated_global_guarded
load_fraction = 0.80
selected profiles:
  cpu_heavy_local_bench = 8
  gpu_heavy_jax_matmul  = 1
  hybrid_rl_resac_ant   = 3
full measured action count = 243
candidate action count = 243
```

The candidate family equals the measured finite action slice for this
certificate, so the cover radius is exactly zero.  This uses the exact-oracle
special case of the robust candidate MaxWeight theorem; it does not claim that
a future greedy/local-search scheduler audit has zero oracle error.

## Slack Accounting

| Quantity | Value | Meaning |
|---|---:|---|
| `delta` | 0.066921842 | capacity slack |
| `L` | 47.897952 | finite service Lipschitz envelope |
| `rho` | 0.000000000 | candidate cover radius |
| `Lrho` | 0.000000000 | candidate support loss |
| `epsilon_est` | 0.000000000 | lower-service estimation loss for this measured map |
| `beta` | 0.000000000 | queue-scaled penalty slope |
| `alpha1` | 0.000000000 | queue-scaled oracle error |
| `eta` | 0.066921842 | remaining drift margin |
| `B` | 6361.354514 | finite-support second-moment bound |
| `P0` | 0.000000000 | fixed penalty term |
| `alpha0` | 0.000000000 | fixed oracle-error term |
| `finite_set_threshold_N` | 95072 | Foster finite-set threshold |

The theorem-facing inequality is therefore:

```text
delta > Lrho + epsilon_est + beta + alpha1
0.066921842 > 0
```

## Load Certificate

| Class | lambda | Selected service |
|---|---:|---:|
| `cpu_heavy_local_bench` | 43.823682800 | 54.779603500 |
| `gpu_heavy_jax_matmul` | 55.175835200 | 68.969794000 |
| `hybrid_rl_resac_ant` | 0.267687369 | 0.334609211 |

## What This Does And Does Not Claim

This module proves that the current measured Scheduleurm service cache can
instantiate the positive-slack theorem on a finite-support load model.  It also
checks the exact arithmetic needed by reviewers: `eta` is positive, all
linear-loss terms are explicit, and the finite-set threshold is finite.

It does not certify a production arrival process, unmeasured node buckets, or a
greedy scheduler's oracle error.  Those require arrival/load measurement,
additional service-map replication, and live action-log oracle audits.
