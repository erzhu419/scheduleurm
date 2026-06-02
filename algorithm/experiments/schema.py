"""Shared experiment schema constants."""

SCHEMA_VERSION = "scheduleurm-exp-v1"

DEFAULT_THEOREM_TARGET = (
    "main_statewise_calibrated_fabric_robust_candidate_stability_with_second_"
    "moment_bound_approx_oracle"
)

DEFAULT_ALGORITHM_MODULES = {
    "action_model": "algorithm.action_model:v1_global_action",
    "feature_map": "algorithm.features:v1_finite_class_regime_candidate_bucket",
    "candidate_set": "algorithm.candidates:v1_active_bucket_representatives",
    "score": "algorithm.scoring:v1_bounded_robust_score",
    "policy": "algorithm.placement:sweetspot_v1",
    "fabric_metric": "algorithm.experiments.fabric_metric:v1_cover_and_lipschitz",
    "slot_builder": "algorithm.experiments.slot_builder:v1_queue_step",
    "experiment_action_model": "algorithm.experiments.action_model:v1_statewise_family",
    "service_model": "algorithm.experiments.service_model:v1_empirical_bernstein_lcb",
    "penalty_fit": "algorithm.experiments.penalty_fit:v1_finite_max_envelope",
    "oracle_audit": "algorithm.experiments.oracle_audit:v1_queue_scaled_gap",
    "capacity_lp": "algorithm.experiments.capacity_lp:v1_downward_capacity_slack",
}
