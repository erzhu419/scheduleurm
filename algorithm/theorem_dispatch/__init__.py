"""Theorem-facing live dispatch helpers.

This package keeps robust MaxWeight service certificates out of
``skill/scheduler.py``.  The legacy scheduler may call these helpers through the
existing optional policy hook, but the queue vector, service binding, admission
certificate, and robust score semantics live here.
"""

from .admission import AdmissionCertificate, task_admission_certificate
from .batch_policy import BatchPlacementResult, select_global_batch_placements
from .eta_lcb import OnlineServiceEstimator, ServiceEstimate, ServiceSample
from .policy import TheoremMaxWeightPlacementPolicy, theorem_policy_config
from .state_service import StateDependentServiceCache, StateServiceLookup

__all__ = [
    "AdmissionCertificate",
    "BatchPlacementResult",
    "OnlineServiceEstimator",
    "ServiceEstimate",
    "ServiceSample",
    "StateDependentServiceCache",
    "StateServiceLookup",
    "TheoremMaxWeightPlacementPolicy",
    "select_global_batch_placements",
    "task_admission_certificate",
    "theorem_policy_config",
]
