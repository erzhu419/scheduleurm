# Lean artifact map for Scheduleurm theory

This file maps the paper-facing claims in `paper/main.tex` and its Electronic
Companion (EC) to the split Lean source files and to theorem names that are
searchable in the consolidated upload file `ScheduleurmUpload.lean`.

Use this map to avoid the artifact inconsistency flagged in `gpt_revise_round2.md`: if only the upload file is sent to a reviewer, every theorem name below should be searchable directly in that file.

The consolidated upload is the complete checked formal source.  The manuscript
and EC do not print all implementation lemmas line by line.  Instead, the exact
theorem tables below provide the paper-to-Lean direction, while the module map
provides the Lean-to-paper direction.  Helper declarations that are not
independent paper claims remain visible in their mapped module and in the
consolidated source.

## Complete split-source module map

| Lean source module(s) | Human-readable manuscript layer | Role |
|---|---|---|
| `Basic.lean`, `SupportFunction.lean` | Section 3 and EC.1 | Finite action families, mixtures, support functions, and shared notation. |
| `CapacityRegion.lean`, `CapacityGeometry.lean`, `HausdorffCapacity.lean` | Section 3.2, EC.1, and the support-function discussion in EC.8 | Capacity slack, downward closure, support geometry, and the compact-convex Hausdorff specialization. |
| `CandidateApprox.lean`, `OperationalMetric.lean`, `DiagonalScaling.lean` | Propositions 1--2 and EC.2 | Finite-feature candidate construction, calibrated fabric cover, and heterogeneous service-unit lower bounds. |
| `RobustPolicy.lean`, `PenaltyGrowth.lean`, `MaxWeightDrift.lean` | Lemmas 1--2, Theorem 1, and EC.3 | Exact and approximate robust MaxWeight selection, bounded penalties, and drift loss accounting. |
| `StochasticQueueModel.lean`, `ConcreteStochasticModel.lean`, `IntegerQueue.lean` | Section 3.1, Lemma 3, Theorem 1, and EC.4 | Queue dynamics, finite-support conditional moments, and integer-state stochastic models. |
| `FosterLyapunov.lean`, `MarkovRecurrence.lean`, `QueueRecurrence.lean` | Theorem 1, Corollaries 1--2, and EC.4 | Foster telescoping, finite-sublevel recurrence, and positive recurrence. |
| `OperationalCapacity.lean` | EC.5 and the operational-boundary paragraph in Section 3 | Load-certified operational stabilizability and conservation-law necessity. |
| `FrameBasedStability.lean` | Proposition 3 and EC.6 | Variable-duration cumulative service, normalized oracle scores, frame drift, and embedded recurrence. |
| `ActionUnion.lean` | Proposition 4 and EC.8 | Monotonicity under finite registered action-family expansion. |
| `Concentration.lean`, `ConcentrationProbability.lean`, `Learning.lean`, `StructuredLearning.lean`, `Regret.lean` | EC.7 and the extension paragraph in Section 6 | Conditional active-bucket concentration, regret, union bounds, and certificate-event lifting. |
| `PiecewiseStationary.lean`, `RegimeBelief.lean`, `RegimeStability.lean` | EC.7 and the hidden-regime extension paragraph in Section 6 | Regime segmentation, belief/detection budgets, and switching-drift composition. |
| `SweetSpot.lean` | Experimental service-profile and candidate-family discussion in Sections 4--5 | Finite service-profile selection support; not an independent stability claim. |
| `MainTheorems.lean` | Theorem 1, Corollaries 1--2, and EC.1--EC.7 | Paper-facing composition theorems and calibrated/statewise wrappers. |

## Main theorem spine

| Paper claim | Split Lean source | Searchable theorem name |
|---|---|---|
| Full-action stationary-mix slack implies support-function slack | `/home/erzhu419/mine_code/proof/Scheduleurm/CapacityRegion.lean` | `capacity_slack_implies_support_slack` |
| Downward-closed capacity slack implies support-function slack | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_downward_capacity_support_slack` |
| Finite feature-cell candidate construction with containment, cardinality, and nonzero \(L\rho\) support loss | `/home/erzhu419/mine_code/proof/Scheduleurm/CandidateApprox.lean` | `finite_feature_cell_candidate_certificate` |
| Candidate-family expansion cannot worsen a certified support gap | `/home/erzhu419/mine_code/proof/Scheduleurm/ActionUnion.lean` | `support_gap_mono_under_candidate_expansion` |
| Candidate-family expansion preserves the base support-loss capacity margin \(\delta-\epsilon\) | `/home/erzhu419/mine_code/proof/Scheduleurm/ActionUnion.lean` | `candidate_capacity_slack_loss_under_expansion` |
| Fabric-cover candidate support approximation \(H^{full}\le H^{cand}+L\rho\|q\|_1\) | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_candidate_restricted_capacity_approximation` |
| Fabric-cover support approximation from a concrete projection and feature-sensitivity calibration | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_candidate_restricted_capacity_approximation_from_calibration` |
| Constructive coordinate-Hausdorff capacity-set approximation | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_candidate_restricted_capacity_coordinate_hausdorff` |
| Constructive coordinate-Hausdorff capacity-set approximation from calibration | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_candidate_restricted_capacity_coordinate_hausdorff_from_calibration` |
| Robust candidate MaxWeight drift with slack \(\delta-(\epsilon_{cand}+\epsilon_{est}+\beta)\) | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_robust_candidate_maxweight_drift` |
| Robust candidate MaxWeight drift with approximate oracle slack \(\delta-(\epsilon_{cand}+\epsilon_{est}+\beta+\alpha_1)\) | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_robust_candidate_maxweight_drift_approx_oracle` |
| Coordinate-scaled LCB support loss for heterogeneous service units | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean`, `/home/erzhu419/mine_code/proof/Scheduleurm/DiagonalScaling.lean` | `main_diagonal_scaled_lcb_support_loss` |
| L1-weighted coordinate-scaled LCB support loss | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean`, `/home/erzhu419/mine_code/proof/Scheduleurm/DiagonalScaling.lean` | `main_diagonal_scaled_lcb_support_loss_l1` |
| Concrete finite-support stochastic stability from coordinate moments | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_concrete_finite_support_stochastic_stability_from_coordinate_moments` |
| Robust candidate stability under bounded conditional second-order moment | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound` |
| Robust candidate stability under bounded conditional second-order moment with approximate oracle | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound_approx_oracle` |
| Bounded-sample fabric-cover robust candidate stability | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_concrete_fabric_cover_robust_candidate_stochastic_stability_from_bounded_samples'` |
| One-statement paper theorem combining cover, robust drift, bounded finite-support stochastic model, and recurrence | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_under_fabric_cover` |
| One-statement paper theorem from calibrated projection and feature-sensitivity certificates | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric` |
| Calibrated one-statement theorem with bounded conditional second-order moment | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound` |
| Calibrated one-statement theorem with bounded conditional second-order moment and approximate oracle | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle` |
| Statewise/dynamic feasible-family calibrated theorem with bounded second moment and approximate oracle | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle` |
| Finite-horizon cumulative-backlog bound from telescoping Foster drift | `/home/erzhu419/mine_code/proof/Scheduleurm/FosterLyapunov.lean` | `foster_telescoping_l1_bound` |
| Zero-slack operational necessity from a conservation law | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_operational_conservation_law_necessity` |
| Operational capacity sandwich: positive slack sufficiency plus zero-slack necessity | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_operational_capacity_sandwich` |

## Operational load binding internals

| Obligation | Split Lean source | Searchable theorem/name |
|---|---|---|
| A concrete finite-support arrival/service model explicitly encodes the offered load vector and induces the same transition kernel | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalCapacity.lean` | `ModelEncodesLoad` |
| Constructor tying a finite-support model's exact conditional mean arrivals to `lam` | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalCapacity.lean` | `finite_support_model_encodes_load` |
| Operational stabilizability uses a model bundled with its load certificate | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalCapacity.lean` | `LoadCertifiedNatQueueModel` |
| Load-certified operational stabilizability predicate | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalCapacity.lean` | `OperationallyStabilizesIntegerLoad` |

## Extension theorem spine

| Extension | Split Lean source | Searchable theorem name |
|---|---|---|
| Dwell/switching backlog budget for hidden regimes | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_hidden_regime_dwell_switching_drift` |
| Active-bucket deterministic regret bound depending on `active.card` | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_active_bucket_lcb_learning_regret` |
| Active-bucket high-probability lifting | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_active_bucket_lcb_learning_regret_high_probability` |
| Active-bucket finite local failure union bound | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_active_bucket_local_failure_union_bound` |
| Generic confidence/certificate event implies high-probability stability certificate | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | `main_high_probability_stability_from_certificate_event` |

## Variable-duration frame internals

These results are the paper-facing bridge for migration and other event-driven
configuration trajectories.  They use cumulative service and penalty units and
do not infer a stochastic migration model from deterministic cost rows.

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Unit-rate second-order bound lifted through a bounded frame duration | `/home/erzhu419/mine_code/proof/Scheduleurm/FrameBasedStability.lean` | `secondOrderTerm_frame_le` |
| Cumulative oracle obligation equals its duration-normalized form for positive duration | `/home/erzhu419/mine_code/proof/Scheduleurm/FrameBasedStability.lean` | `frameApproximateOracle_iff_durationNormalized` |
| Variable-duration robust MaxWeight pressure bound | `/home/erzhu419/mine_code/proof/Scheduleurm/FrameBasedStability.lean` | `frame_approximate_maxWeight_negative_drift` |
| Variable-duration Lyapunov drift with cumulative service and bounded frame penalty | `/home/erzhu419/mine_code/proof/Scheduleurm/FrameBasedStability.lean` | `frame_approximate_maxWeight_lyapunov_drift` |
| Positive minimum duration gives uniform embedded-chain drift | `/home/erzhu419/mine_code/proof/Scheduleurm/FrameBasedStability.lean` | `frame_approximate_maxWeight_lyapunov_drift_uniform` |
| Duration one recovers the ordinary slotted theorem | `/home/erzhu419/mine_code/proof/Scheduleurm/FrameBasedStability.lean` | `frame_lyapunov_drift_duration_one` |
| Finite upper duration converts frame count to elapsed physical time | `/home/erzhu419/mine_code/proof/Scheduleurm/FrameBasedStability.lean` | `elapsedFrameTime_le` |
| Uniform frame drift plus local return gives embedded-chain finite-set recurrence | `/home/erzhu419/mine_code/proof/Scheduleurm/FrameBasedStability.lean` | `frame_nat_model_positive_recurrent_via_finite_set` |

## Fabric calibration internals

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Candidate generator projection implies fabric cover radius \(\rho\) | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalMetric.lean` | `FabricCandidateProjection.covers` |
| Feature sensitivity envelope plus coefficient domination implies service Lipschitzness | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalMetric.lean` | `fabric_service_lipschitz_of_feature_sensitivity` |
| Calibrated projection plus sensitivity gives candidate support gap \(L\rho\) | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalMetric.lean` | `calibrated_fabric_cover_support_gap` |
| State/regime-indexed calibrated candidate support gap | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalMetric.lean` | `indexed_calibrated_fabric_cover_support_gap` |
| Uniform constant state/regime-indexed calibrated candidate support gap | `/home/erzhu419/mine_code/proof/Scheduleurm/OperationalMetric.lean` | `indexed_calibrated_fabric_cover_support_gap_uniform` |

## Approximate oracle internals

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Additive approximate robust-score maximizer | `/home/erzhu419/mine_code/proof/Scheduleurm/RobustPolicy.lean` | `ApproxRobustScoreMaximizer` |
| Exact robust-score maximization implies zero-loss approximate maximization | `/home/erzhu419/mine_code/proof/Scheduleurm/RobustPolicy.lean` | `robustScoreMaximizer_is_approx` |
| Queue-scaled approximate robust-score maximizer | `/home/erzhu419/mine_code/proof/Scheduleurm/PenaltyGrowth.lean` | `QueueScaledApproxRobustScoreMaximizer` |
| Approximate oracle support bound with queue-scaled loss | `/home/erzhu419/mine_code/proof/Scheduleurm/PenaltyGrowth.lean` | `robust_candidate_policy_approx_full_support_scaled_penalty_approx_oracle` |
| Approximate oracle Lyapunov drift | `/home/erzhu419/mine_code/proof/Scheduleurm/PenaltyGrowth.lean` | `robust_candidate_policy_lyapunov_drift_scaled_penalty_approx_oracle` |

## Downward capacity internals

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Downward-closed capacity region with slack | `/home/erzhu419/mine_code/proof/Scheduleurm/CapacityRegion.lean` | `InDownwardCapacityWithSlack` |
| Existing slack definition equals downward-closed capacity slack | `/home/erzhu419/mine_code/proof/Scheduleurm/CapacityRegion.lean` | `inCapacityWithSlack_iff_downwardCapacityWithSlack` |
| Coordinatewise smaller loads remain feasible | `/home/erzhu419/mine_code/proof/Scheduleurm/CapacityRegion.lean` | `downward_capacity_monotone` |

## Concrete stochastic model internals

| Obligation | Split Lean source | Searchable theorem name |
|---|---|---|
| Finite-support expected drift equals weighted one-step drift | `/home/erzhu419/mine_code/proof/Scheduleurm/ConcreteStochasticModel.lean` | `expectedLyapunovDrift_eq` |
| Coordinate conditional arrival means imply pressure bound | `/home/erzhu419/mine_code/proof/Scheduleurm/ConcreteStochasticModel.lean` | `arrival_pressure_le_of_expectedArrival_le` |
| Coordinate conditional service means imply pressure bound | `/home/erzhu419/mine_code/proof/Scheduleurm/ConcreteStochasticModel.lean` | `service_pressure_ge_of_expectedService_ge` |
| Coordinate sample bounds imply second-order drift bound | `/home/erzhu419/mine_code/proof/Scheduleurm/ConcreteStochasticModel.lean` | `expectedSecondOrder_le_of_coord_sample_bounds` |
| Coordinate moments plus Foster drift imply positive recurrence | `/home/erzhu419/mine_code/proof/Scheduleurm/ConcreteStochasticModel.lean` | `positive_recurrent_via_coordinate_moments'` |

## Current verification command

The general manuscript statement conditions on the complete pre-decision
filtration \(\mathcal F_t\).  The Lean statewise theorem is the queue-indexed
specialization in which all statewise full/candidate/service/feature/penalty
objects are functions of the integer queue snapshot.  The augmented-state
Markov and local-return bridge remains an explicit manuscript assumption; the
artifact must not be described as a complete measure-theoretic kernel proof for
arbitrary scheduler-visible state.

```text
cd /home/erzhu419/mine_code/proof
lake build Scheduleurm
lake env lean Scheduleurm/FrameBasedStability.lean
lake env lean ScheduleurmUpload.lean
rg -n "\\bsorry\\b|\\badmit\\b|\\baxiom\\b" Scheduleurm ScheduleurmUpload.lean lakefile.toml
```
