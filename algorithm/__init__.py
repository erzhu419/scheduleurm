"""Optional scheduleurm algorithm policies.

The scheduler imports this package only through the small policy surface below.
Default policy is `legacy`, which deliberately leaves scheduler.py behavior
unchanged.
"""

from .action_model import GlobalAction, action_candidate_jsonl_row
from .features import class_key, finite_feature_metric, gpu_candidate_features, regime_key
from .placement import available_policies, load_placement_policy

__all__ = [
    "available_policies",
    "class_key",
    "finite_feature_metric",
    "GlobalAction",
    "gpu_candidate_features",
    "action_candidate_jsonl_row",
    "load_placement_policy",
    "regime_key",
]
