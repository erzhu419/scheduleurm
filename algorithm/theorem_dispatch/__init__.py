"""Theorem-facing live dispatch helpers.

This package keeps robust MaxWeight service certificates out of
``skill/scheduler.py``.  The legacy scheduler may call these helpers through the
existing optional policy hook, but the queue vector, service binding, admission
certificate, and robust score semantics live here.
"""

from .admission import AdmissionCertificate, task_admission_certificate
from .policy import TheoremMaxWeightPlacementPolicy, theorem_policy_config

__all__ = [
    "AdmissionCertificate",
    "TheoremMaxWeightPlacementPolicy",
    "task_admission_certificate",
    "theorem_policy_config",
]
