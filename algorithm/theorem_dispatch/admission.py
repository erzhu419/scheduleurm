"""Admission certificates for theorem-facing Scheduleurm populations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from simulation.service_cache import ServiceRateCache

from .service_registry import default_service_cache, infer_workload_key


@dataclass(frozen=True)
class AdmissionCertificate:
    task_id: str
    workload_key: str
    admitted: bool
    reason: str
    measured_profiles: tuple[int, ...]
    source: str

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workload_key": self.workload_key,
            "admitted": self.admitted,
            "reason": self.reason,
            "measured_profiles": list(self.measured_profiles),
            "source": self.source,
        }


def service_domain_certificate(
    workload_key: str,
    *,
    cache: ServiceRateCache | None = None,
) -> dict[str, Any]:
    cache = cache or default_service_cache()
    profiles = tuple(
        int(record.profile)
        for record in cache.profiles(str(workload_key or ""))
        if not record.capacity_boundary and float(record.aggregate_rate) > 0.0
    )
    return {
        "workload_key": str(workload_key or ""),
        "measured_profiles": list(profiles),
        "profile_count": len(profiles),
        "service_domain_certified": bool(profiles),
        "reason": "" if profiles else "no_positive_nonboundary_profile",
    }


def task_admission_certificate(
    task: Mapping[str, Any],
    *,
    cache: ServiceRateCache | None = None,
    workload_key: str | None = None,
    admission_mode: str | None = None,
) -> AdmissionCertificate:
    cache = cache or default_service_cache()
    selected = workload_key or infer_workload_key(
        task,
        cache,
        admission_mode=admission_mode,
    )
    task_id = str(task.get("id") or task.get("task_id") or "")
    if not selected:
        return AdmissionCertificate(
            task_id=task_id,
            workload_key="",
            admitted=False,
            reason="workload_key_not_inferred",
            measured_profiles=(),
            source="service_cache",
        )
    domain = service_domain_certificate(selected, cache=cache)
    profiles = tuple(int(x) for x in domain.get("measured_profiles") or ())
    return AdmissionCertificate(
        task_id=task_id,
        workload_key=selected,
        admitted=bool(domain.get("service_domain_certified")),
        reason=str(domain.get("reason") or ""),
        measured_profiles=profiles,
        source="service_cache",
    )
