"""Shared experiment schema constants."""

SCHEMA_VERSION = "scheduleurm-exp-v1"

DEFAULT_THEOREM_TARGET = (
    "main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_"
    "with_second_moment_bound_approx_oracle"
)

DEFAULT_ALGORITHM_MODULES = {
    "action_model": "algorithm.action_model:v1_global_action",
    "feature_map": "algorithm.features:v1_finite_class_regime_candidate_bucket",
    "candidate_set": "algorithm.candidates:v1_active_bucket_representatives",
    "score": "algorithm.scoring:v1_bounded_robust_score",
    "policy": "algorithm.placement:sweetspot_v1",
}
