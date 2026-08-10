# MMRCPSP renewal-stream experiment v1

- Status: `MMRCPSP_RENEWAL_STREAM_PROTOCOL_PASS`
- Registered classes: `25`
- Runs: `105/105` protocol PASS
- Ours exact registered-family oracle runs: `21`
- Ours Pareto-nondominated scenario/seed rows: `0/21`
- Ours strictly improves every baseline: `0/21`
- Costs are final backlog and time-average queue; lower is better.
- Static instance outcomes were known before this stream protocol; arrival tapes and stream outcomes were not.
- Finite-sample drift remains diagnostic and is not a positive-recurrence proof.

| Scenario | Seed | Ours final backlog | Ours average queue | Disposition |
|---|---:|---:|---:|---|
| bursty_nominal | 11 | 0 | 8.471041666667 | baseline_dominates |
| bursty_nominal | 23 | 4 | 4.670833333333 | baseline_dominates |
| bursty_nominal | 37 | 0 | 3.791666666667 | baseline_dominates |
| load_035 | 11 | 0 | 2.912708333333 | baseline_dominates |
| load_035 | 23 | 1 | 3.149166666667 | baseline_dominates |
| load_035 | 37 | 1 | 3.005416666667 | baseline_dominates |
| load_065 | 11 | 1 | 4.859166666667 | baseline_dominates |
| load_065 | 23 | 1 | 7.692083333333 | baseline_dominates |
| load_065 | 37 | 0 | 9.952708333333 | baseline_dominates |
| load_090 | 11 | 23 | 17.631458333333 | baseline_dominates |
| load_090 | 23 | 4 | 16.990833333333 | baseline_dominates |
| load_090 | 37 | 21 | 24.94125 | baseline_dominates |
| load_105 | 11 | 33 | 26.174583333333 | baseline_dominates |
| load_105 | 23 | 9 | 12.648958333333 | baseline_dominates |
| load_105 | 37 | 28 | 21.368958333333 | baseline_dominates |
| poisson_nominal | 11 | 1 | 5.482083333333 | baseline_dominates |
| poisson_nominal | 23 | 0 | 6.222708333333 | baseline_dominates |
| poisson_nominal | 37 | 1 | 7.42125 | baseline_dominates |
| static_backlog | 11 | 0 | 31.384166666667 | baseline_dominates |
| static_backlog | 23 | 0 | 31.384166666667 | baseline_dominates |
| static_backlog | 37 | 0 | 31.384166666667 | baseline_dominates |

## Claim Boundary

{'registered_finite_library_only': True, 'variable_duration_queue_recurrence_executed': True, 'workload_conservation_pathwise_certified': True, 'exact_oracle_only_over_registered_feasible_family': True, 'finite_sample_drift_is_diagnostic_not_positive_recurrence_proof': True, 'global_mmrcpsp_optimality_claim': False, 'psplib_wide_generalization_claim': False, 'production_arrival_model_claim': False, 'stochastic_stability_claim_ready': False, 'reason': 'A finite replay cannot prove irreducibility, recurrence, or a domain-wide service model. The capacity and drift fields audit the registered renewal stream and expose, rather than hide, capacity-exterior load points.'}
