# Lean artifact map for Scheduleurm theory

This file maps the paper-level claims in `math.md` to the split Lean source files and to theorem names that are searchable in the consolidated upload file `ScheduleurmUpload.lean`.

Use this map to avoid the artifact inconsistency flagged in `gpt_revise_round2.md`: if only the upload file is sent to a reviewer, every theorem name below should be searchable directly in that file.

## Main theorem spine

| Paper claim | Split Lean source | Searchable theorem name |
|---|---|---|
| Full-action stationary-mix slack implies support-function slack | `<LEAN_PROOF_SOURCE>/Scheduleurm/CapacityRegion.lean` | `capacity_slack_implies_support_slack` |
| Downward-closed capacity slack implies support-function slack | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_downward_capacity_support_slack` |
| Fabric-cover candidate support approximation \(H^{full}\le H^{cand}+L\rho\|q\|_1\) | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_candidate_restricted_capacity_approximation` |
| Fabric-cover support approximation from a concrete projection and feature-sensitivity calibration | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_candidate_restricted_capacity_approximation_from_calibration` |
| Constructive coordinate-Hausdorff capacity-set approximation | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_candidate_restricted_capacity_coordinate_hausdorff` |
| Constructive coordinate-Hausdorff capacity-set approximation from calibration | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_candidate_restricted_capacity_coordinate_hausdorff_from_calibration` |
| Robust candidate MaxWeight drift with slack \(\delta-(\epsilon_{cand}+\epsilon_{est}+\beta)\) | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_robust_candidate_maxweight_drift` |
| Robust candidate MaxWeight drift with approximate oracle slack \(\delta-(\epsilon_{cand}+\epsilon_{est}+\beta+\alpha_1)\) | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_robust_candidate_maxweight_drift_approx_oracle` |
| Concrete finite-support stochastic stability from coordinate moments | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_concrete_finite_support_stochastic_stability_from_coordinate_moments` |
| Robust candidate stability under bounded conditional second-order moment | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound` |
| Robust candidate stability under bounded conditional second-order moment with approximate oracle | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound_approx_oracle` |
| Bounded-sample fabric-cover robust candidate stability | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_concrete_fabric_cover_robust_candidate_stochastic_stability_from_bounded_samples'` |
| One-statement paper theorem combining cover, robust drift, bounded finite-support stochastic model, and recurrence | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_under_fabric_cover` |
| One-statement paper theorem from calibrated projection and feature-sensitivity certificates | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric` |
| Calibrated one-statement theorem with bounded conditional second-order moment | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound` |
| Calibrated one-statement theorem with bounded conditional second-order moment and approximate oracle | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle` |
| Statewise/dynamic feasible-family calibrated theorem with bounded second moment and approximate oracle | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle` |
| Zero-slack operational necessity from a conservation law | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_operational_conservation_law_necessity` |
| Operational capacity sandwich: positive slack sufficiency plus zero-slack necessity | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_operational_capacity_sandwich` |

## Operational load binding internals

| Obligation | Split Lean source | Searchable theorem/name |
|---|---|---|
| A concrete finite-support arrival/service model explicitly encodes the offered load vector and induces the same transition kernel | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalCapacity.lean` | `ModelEncodesLoad` |
| Constructor tying a finite-support model's exact conditional mean arrivals to `lam` | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalCapacity.lean` | `finite_support_model_encodes_load` |
| Operational stabilizability uses a model bundled with its load certificate | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalCapacity.lean` | `LoadCertifiedNatQueueModel` |
| Load-certified operational stabilizability predicate | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalCapacity.lean` | `OperationallyStabilizesIntegerLoad` |

## Extension theorem spine

| Extension | Split Lean source | Searchable theorem name |
|---|---|---|
| Dwell/switching backlog budget for hidden regimes | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_hidden_regime_dwell_switching_drift` |
| Active-bucket deterministic regret bound depending on `active.card` | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_active_bucket_lcb_learning_regret` |
| Active-bucket high-probability lifting | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_active_bucket_lcb_learning_regret_high_probability` |
| Active-bucket finite local failure union bound | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_active_bucket_local_failure_union_bound` |
| Generic confidence/certificate event implies high-probability stability certificate | `<LEAN_PROOF_SOURCE>/Scheduleurm/MainTheorems.lean` | `main_high_probability_stability_from_certificate_event` |

## Fabric calibration internals

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Candidate generator projection implies fabric cover radius \(\rho\) | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalMetric.lean` | `FabricCandidateProjection.covers` |
| Feature sensitivity envelope plus coefficient domination implies service Lipschitzness | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalMetric.lean` | `fabric_service_lipschitz_of_feature_sensitivity` |
| Calibrated projection plus sensitivity gives candidate support gap \(L\rho\) | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalMetric.lean` | `calibrated_fabric_cover_support_gap` |
| State/regime-indexed calibrated candidate support gap | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalMetric.lean` | `indexed_calibrated_fabric_cover_support_gap` |
| Uniform constant state/regime-indexed calibrated candidate support gap | `<LEAN_PROOF_SOURCE>/Scheduleurm/OperationalMetric.lean` | `indexed_calibrated_fabric_cover_support_gap_uniform` |

## Approximate oracle internals

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Additive approximate robust-score maximizer | `<LEAN_PROOF_SOURCE>/Scheduleurm/RobustPolicy.lean` | `ApproxRobustScoreMaximizer` |
| Exact robust-score maximization implies zero-loss approximate maximization | `<LEAN_PROOF_SOURCE>/Scheduleurm/RobustPolicy.lean` | `robustScoreMaximizer_is_approx` |
| Queue-scaled approximate robust-score maximizer | `<LEAN_PROOF_SOURCE>/Scheduleurm/PenaltyGrowth.lean` | `QueueScaledApproxRobustScoreMaximizer` |
| Approximate oracle support bound with queue-scaled loss | `<LEAN_PROOF_SOURCE>/Scheduleurm/PenaltyGrowth.lean` | `robust_candidate_policy_approx_full_support_scaled_penalty_approx_oracle` |
| Approximate oracle Lyapunov drift | `<LEAN_PROOF_SOURCE>/Scheduleurm/PenaltyGrowth.lean` | `robust_candidate_policy_lyapunov_drift_scaled_penalty_approx_oracle` |

## Downward capacity internals

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Downward-closed capacity region with slack | `<LEAN_PROOF_SOURCE>/Scheduleurm/CapacityRegion.lean` | `InDownwardCapacityWithSlack` |
| Existing slack definition equals downward-closed capacity slack | `<LEAN_PROOF_SOURCE>/Scheduleurm/CapacityRegion.lean` | `inCapacityWithSlack_iff_downwardCapacityWithSlack` |
| Coordinatewise smaller loads remain feasible | `<LEAN_PROOF_SOURCE>/Scheduleurm/CapacityRegion.lean` | `downward_capacity_monotone` |

## Concrete stochastic model internals

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Finite-support expected drift equals weighted one-step drift | `<LEAN_PROOF_SOURCE>/Scheduleurm/ConcreteStochasticModel.lean` | `expectedLyapunovDrift_eq` |
| Coordinate conditional arrival means imply pressure bound | `<LEAN_PROOF_SOURCE>/Scheduleurm/ConcreteStochasticModel.lean` | `arrival_pressure_le_of_expectedArrival_le` |
| Coordinate conditional service means imply pressure bound | `<LEAN_PROOF_SOURCE>/Scheduleurm/ConcreteStochasticModel.lean` | `service_pressure_ge_of_expectedService_ge` |
| Coordinate sample bounds imply second-order drift bound | `<LEAN_PROOF_SOURCE>/Scheduleurm/ConcreteStochasticModel.lean` | `expectedSecondOrder_le_of_coord_sample_bounds` |
| Coordinate moments plus Foster drift imply positive recurrence | `<LEAN_PROOF_SOURCE>/Scheduleurm/ConcreteStochasticModel.lean` | `positive_recurrent_via_coordinate_moments'` |

## Current verification command

```text
cd <LEAN_PROOF_SOURCE>
lake build Scheduleurm
lake env lean ScheduleurmUpload.lean
rg -n "\\bsorry\\b|\\badmit\\b|\\baxiom\\b" Scheduleurm ScheduleurmUpload.lean lakefile.toml
```
