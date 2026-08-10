"""Fail-closed replay orchestration for the final heterogeneous service cache.

The orchestrator consumes three frozen inputs:

* a statewise service cache containing lower-service and natural-completion
  models;
* the registered directional loaded-action ledger; and
* a migration certificate whose service rates were recomputed from that exact
  cache.

It never launches work.  Action selection sees only conservative lower service
and bounded penalties.  Completion-time evaluation is joined afterwards from
task-native natural-completion models.  Results are policy-semantics replay on
the measured Scheduleurm fabric, not execution of external scheduler stacks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from simulation.service_cache import ProfileRecord, ServiceRateCache
from simulation.sota_baselines import sota_baseline_specs


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_CACHE = ARTIFACT_ROOT / "service_cache_v2_cpu_gpu_loaded_final_20260809.json"
DEFAULT_LOADED_LEDGER = ARTIFACT_ROOT / "critical_gpu_loaded_action_ledger_20260809.json"
DEFAULT_MIGRATION_CERTIFICATE = ARTIFACT_ROOT / "unified_migration_recalibration_certificate_20260809.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "unified_hardware_or_replay_20260809.json"

SCHEMA_VERSION = 2
EXPECTED_QUADRANTS = ("q00", "q01", "q10", "q11")
EXPECTED_ARRIVAL_FAMILIES = ("static", "poisson", "bursty", "load_sweep")
EXPECTED_MIGRATION_MODES = ("without_migration", "with_migration")
EXPECTED_MIGRATION_FAMILIES = ("pure_cpu", "pure_gpu", "hybrid_rl")
EXPECTED_MIGRATION_POINTS = (0.25, 0.50, 0.75)
DEFAULT_LOADS = (0.50, 0.70, 0.85, 0.95)
DEFAULT_SEEDS = (41, 42, 43)
NOMINAL_ONLINE_LOAD = 0.85
PARETO_TOLERANCE = 0.005

WORKLOAD_CONTRACTS: dict[str, dict[str, str]] = {
    "light_control_local": {
        "quadrant": "q00",
        "resource_kind": "light_control",
    },
    "gpu_heavy_jax_matmul": {
        "quadrant": "q01",
        "resource_kind": "gpu_heavy",
    },
    "gpu_cnn_torch_resnet50": {
        "quadrant": "q01",
        "resource_kind": "gpu_cnn",
    },
    "gpu_llm_distilgpt2": {
        "quadrant": "q01",
        "resource_kind": "gpu_llm",
    },
    "cpu_heavy_local_bench": {
        "quadrant": "q10",
        "resource_kind": "cpu_heavy",
    },
    "freqduet_cpu_surrogate": {
        "quadrant": "q10",
        "resource_kind": "cpu_heavy",
    },
    "hybrid_rl_resac_ant": {
        "quadrant": "q11",
        "resource_kind": "hybrid_rl",
    },
}

OURS_POLICY = "scheduleurm_unified_robust_maxweight"
LEGACY_POLICY = "legacy_fixed_caps"
LEGACY_FIXED_PROFILES = {
    "gpu_cnn_torch_resnet50": 3,
    "gpu_heavy_jax_matmul": 3,
    "gpu_llm_distilgpt2": 3,
    "hybrid_rl_resac_ant": 5,
    "light_control_local": 1,
    "cpu_heavy_local_bench": 9,
    "freqduet_cpu_surrogate": 1,
}
ABLATION_POLICIES: tuple[tuple[str, str], ...] = (
    ("ablation_support_only", "support scorer without delay tie-break"),
    ("ablation_delay_only", "delay proxy without support-preserving guard"),
    ("ablation_no_bounded_penalty", "bounded action penalty removed"),
    ("ablation_no_loaded_ledger", "directional loaded actions removed"),
    ("ablation_no_migration", "migration actions removed"),
    ("ablation_high_profile_tiebreak", "high-profile rather than low-profile tie-break"),
)


@dataclass(frozen=True)
class LowerAction:
    action_id: str
    workload_key: str
    profile: int
    node_bucket: str
    resource_state: str
    resident_mix: str
    lower_service: float
    lower_service_vector: tuple[tuple[str, float], ...]
    penalty_units: float
    loaded_action: bool = False
    migration_action: bool = False
    target_batch_width: int = 0
    required_job_counts: tuple[tuple[str, int], ...] = ()

    def service_vector(self) -> dict[str, float]:
        return dict(self.lower_service_vector)

    def batch_width(self) -> int:
        return int(self.target_batch_width or self.profile)

    def job_requirements(self) -> dict[str, int]:
        if self.required_job_counts:
            return dict(self.required_job_counts)
        return {self.workload_key: self.batch_width()}

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "workload_key": self.workload_key,
            "profile": self.profile,
            "node_bucket": self.node_bucket,
            "resource_state": self.resource_state,
            "resident_mix": self.resident_mix,
            "lower_service": self.lower_service,
            "lower_service_vector": self.service_vector(),
            "penalty_units": self.penalty_units,
            "loaded_action": self.loaded_action,
            "migration_action": self.migration_action,
            "target_batch_width": self.batch_width(),
            "required_job_counts": self.job_requirements(),
            "selection_service_view": "lower_service",
        }


@dataclass(frozen=True)
class EvaluationAction:
    action_id: str
    completion_point_rate: float
    completion_group_units: float
    replay_job_units: float
    fixed_overhead_s: float
    completion_model_sample_count: int
    eta_source: str
    completion_model_kind: str

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "completion_point_rate": self.completion_point_rate,
            "completion_group_units": self.completion_group_units,
            "replay_job_units": self.replay_job_units,
            "fixed_overhead_s": self.fixed_overhead_s,
            "completion_model_sample_count": self.completion_model_sample_count,
            "eta_source": self.eta_source,
            "completion_model_kind": self.completion_model_kind,
            "evaluation_service_view": "natural_completion_point",
        }


@dataclass(frozen=True)
class ReplayAction:
    lower: LowerAction
    evaluation: EvaluationAction

    def snapshot(self) -> dict[str, Any]:
        return {
            "lower": self.lower.snapshot(),
            "evaluation": self.evaluation.snapshot(),
        }


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    quadrant: str
    node_bucket: str
    resource_count: int
    workload_keys: tuple[str, ...]
    actions: tuple[ReplayAction, ...]
    scenario_kind: str = "hardware_local"

    def snapshot(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "quadrant": self.quadrant,
            "node_bucket": self.node_bucket,
            "resource_count": self.resource_count,
            "workload_keys": list(self.workload_keys),
            "scenario_kind": self.scenario_kind,
            "action_ids": [action.lower.action_id for action in self.actions],
        }


@dataclass(frozen=True)
class TraceJob:
    job_id: str
    workload_key: str
    arrival_s: float
    total_units: float


@dataclass(frozen=True)
class TraceSpec:
    trace_id: str
    arrival_family: str
    arrival_process: str
    load_factor: float | None
    seed: int
    jobs: tuple[TraceJob, ...]
    workload_unit_scales: tuple[tuple[str, float], ...]

    def unit_scales(self) -> dict[str, float]:
        return dict(self.workload_unit_scales)


@dataclass
class ReplayJobState:
    job: TraceJob
    remaining_units: float
    available_at: float


@dataclass(frozen=True)
class PolicyDescriptor:
    name: str
    category: str
    semantic_key: str
    metadata: Mapping[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "semantic_key": self.semantic_key,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidatedInputs:
    cache_path: Path
    cache_sha256: str
    cache: ServiceRateCache
    cache_rows: tuple[ProfileRecord, ...]
    loaded_path: Path
    loaded_sha256: str
    loaded_actions: tuple[ReplayAction, ...]
    migration_path: Path
    migration_sha256: str
    migration_scenarios: tuple[Scenario, ...]
    migration_rows: tuple[dict[str, Any], ...]


def build_unified_hardware_or_replay(
    *,
    cache_path: Path = DEFAULT_CACHE,
    loaded_ledger_path: Path = DEFAULT_LOADED_LEDGER,
    migration_certificate_path: Path = DEFAULT_MIGRATION_CERTIFICATE,
    loads: Sequence[float] = DEFAULT_LOADS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    nominal_online_load: float = NOMINAL_ONLINE_LOAD,
    jobs_per_workload: int = 16,
) -> dict[str, Any]:
    """Validate frozen evidence and run the complete local replay matrix."""

    common = _report_header(
        cache_path=cache_path,
        loaded_ledger_path=loaded_ledger_path,
        migration_certificate_path=migration_certificate_path,
        loads=loads,
        seeds=seeds,
        nominal_online_load=nominal_online_load,
        jobs_per_workload=jobs_per_workload,
    )
    blockers: list[dict[str, Any]] = []
    missing = [
        (label, Path(path))
        for label, path in (
            ("unified_cache", cache_path),
            ("loaded_action_ledger", loaded_ledger_path),
            ("migration_certificate", migration_certificate_path),
        )
        if not Path(path).is_file()
    ]
    if missing:
        blockers.extend(
            _issue("MISSING_INPUT", source=label, detail=str(path))
            for label, path in missing
        )
        return {
            **common,
            "status": "WAIT_INPUTS",
            "pass": False,
            "blockers": blockers,
            "runs": [],
            "pareto_rows": [],
            "legacy_comparison_rows": [],
        }

    try:
        inputs = _validate_inputs(
            cache_path=Path(cache_path),
            loaded_path=Path(loaded_ledger_path),
            migration_path=Path(migration_certificate_path),
        )
    except ValueError as exc:
        return {
            **common,
            "status": "FAIL_INPUT_CONTRACT",
            "pass": False,
            "blockers": [_issue("INPUT_CONTRACT_INVALID", detail=str(exc))],
            "runs": [],
            "pareto_rows": [],
            "legacy_comparison_rows": [],
        }

    try:
        base_scenarios = _build_hardware_scenarios(inputs)
        scenarios = tuple(base_scenarios) + tuple(inputs.migration_scenarios)
        policies = _policy_descriptors()
        trace_protocols = _trace_protocols(
            loads=loads,
            seeds=seeds,
            nominal_online_load=nominal_online_load,
        )
        runs = _run_matrix(
            scenarios=scenarios,
            policies=policies,
            trace_protocols=trace_protocols,
            jobs_per_workload=jobs_per_workload,
        )
        pareto_rows = _pareto_rows(runs)
        legacy_rows = _legacy_comparison_rows(runs)
        checks = _coverage_checks(
            scenarios=scenarios,
            policies=policies,
            trace_protocols=trace_protocols,
            runs=runs,
            inputs=inputs,
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return {
            **common,
            "status": "FAIL_REPLAY",
            "pass": False,
            "blockers": [_issue("REPLAY_FAILED_CLOSED", detail=f"{type(exc).__name__}: {exc}")],
            "runs": [],
            "pareto_rows": [],
            "legacy_comparison_rows": [],
            "input_manifest": _input_manifest(inputs),
        }

    passed = all(bool(value) for value in checks.values())
    diagnostics = _performance_diagnostics(pareto_rows, legacy_rows)
    return {
        **common,
        "status": "PASS" if passed else "FAIL_COVERAGE",
        "pass": passed,
        "blockers": [] if passed else [
            _issue("COVERAGE_CHECK_FAILED", detail=name)
            for name, value in checks.items()
            if not value
        ],
        "input_manifest": _input_manifest(inputs),
        "scenario_count": len(scenarios),
        "scenarios": [scenario.snapshot() for scenario in scenarios],
        "policy_count": len(policies),
        "policies": [policy.snapshot() for policy in policies],
        "trace_protocol_count": len(trace_protocols),
        "trace_protocols": [dict(row) for row in trace_protocols],
        "run_count": len(runs),
        "runs": runs,
        "pareto_rows": pareto_rows,
        "legacy_comparison_rows": legacy_rows,
        "coverage_checks": checks,
        "performance_diagnostics": diagnostics,
        "claim_boundary": _claim_boundary(),
    }


def _report_header(
    *,
    cache_path: Path,
    loaded_ledger_path: Path,
    migration_certificate_path: Path,
    loads: Sequence[float],
    seeds: Sequence[int],
    nominal_online_load: float,
    jobs_per_workload: int,
) -> dict[str, Any]:
    normalized_loads = tuple(float(value) for value in loads)
    normalized_seeds = tuple(int(value) for value in seeds)
    if not normalized_loads or any(not 0.0 < value < 1.0 for value in normalized_loads):
        raise ValueError("loads must be a nonempty sequence in (0, 1)")
    if not normalized_seeds:
        raise ValueError("seeds must be nonempty")
    if not 0.0 < float(nominal_online_load) < 1.0:
        raise ValueError("nominal_online_load must lie in (0, 1)")
    if int(jobs_per_workload) <= 0:
        raise ValueError("jobs_per_workload must be positive")
    return {
        "gate": "unified_hardware_or_replay",
        "schema_version": SCHEMA_VERSION,
        "status": "WAIT_INPUTS",
        "pass": False,
        "inputs": {
            "unified_cache": str(cache_path),
            "loaded_action_ledger": str(loaded_ledger_path),
            "migration_certificate": str(migration_certificate_path),
        },
        "loads": list(normalized_loads),
        "seeds": list(normalized_seeds),
        "nominal_online_load": float(nominal_online_load),
        "jobs_per_workload": int(jobs_per_workload),
        "required_quadrants": list(EXPECTED_QUADRANTS),
        "required_arrival_families": list(EXPECTED_ARRIVAL_FAMILIES),
        "required_migration_modes": list(EXPECTED_MIGRATION_MODES),
        "selection_service_view": "lower_service",
        "evaluation_service_view": "natural_completion_point",
        "comparison_kind": "same-cache_policy-semantics",
        "full_stack_external_binary_comparison": False,
        "audited_predecessor_boundaries": {
            "critical_gpu_statewise_sota_replay_gate": "static/Poisson GPU scopes only",
            "critical_q00_q10_phase_sota_replay_gate": "static/Poisson q00/q10 only",
            "sota_quadrant_pareto_gate": "aggregates historical measured-cache policy semantics",
            "online_arrival_experiments": (
                "recomputed here by the shared-lane joint event simulator"
            ),
            "legacy_migration_artifacts": "physical costs are reusable only after rates are rebound to this cache",
        },
    }


def _validate_inputs(*, cache_path: Path, loaded_path: Path, migration_path: Path) -> ValidatedInputs:
    cache_bytes, cache_payload = _read_json(cache_path, "unified cache")
    loaded_bytes, loaded_payload = _read_json(loaded_path, "loaded action ledger")
    migration_bytes, migration_payload = _read_json(migration_path, "migration certificate")
    cache_hash = hashlib.sha256(cache_bytes).hexdigest()
    cache, cache_rows, exact = _validate_cache(cache_payload)
    loaded_actions = _validate_loaded_ledger(loaded_payload, exact=exact)
    migration_scenarios, migration_rows = _validate_migration_certificate(
        migration_payload,
        cache_sha256=cache_hash,
        exact=exact,
    )
    return ValidatedInputs(
        cache_path=cache_path.resolve(),
        cache_sha256=cache_hash,
        cache=cache,
        cache_rows=cache_rows,
        loaded_path=loaded_path.resolve(),
        loaded_sha256=hashlib.sha256(loaded_bytes).hexdigest(),
        loaded_actions=loaded_actions,
        migration_path=migration_path.resolve(),
        migration_sha256=hashlib.sha256(migration_bytes).hexdigest(),
        migration_scenarios=migration_scenarios,
        migration_rows=migration_rows,
    )


def _read_json(path: Path, label: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return raw, payload


def _validate_cache(
    payload: Mapping[str, Any],
) -> tuple[ServiceRateCache, tuple[ProfileRecord, ...], dict[tuple[Any, ...], ProfileRecord]]:
    raw_rows = payload.get("records")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("unified cache must contain a nonempty records list")
    rows: list[ProfileRecord] = []
    exact: dict[tuple[Any, ...], ProfileRecord] = {}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"cache record {index} must be an object")
        record = ProfileRecord.from_snapshot(raw)
        key = _record_key(record)
        if key in exact:
            raise ValueError(f"duplicate exact cache tuple: {key!r}")
        exact[key] = record
        rows.append(record)
        if record.workload_key in WORKLOAD_CONTRACTS and record.completion_model_ready:
            _validate_natural_completion_record(record, label=f"cache record {index}")
    cache = ServiceRateCache.from_snapshot(dict(payload))
    covered = {
        _quadrant_for_workload(record.workload_key)
        for record in rows
        if record.workload_key in WORKLOAD_CONTRACTS
        and record.completion_model_ready
        and not record.capacity_boundary
    }
    missing = sorted(set(EXPECTED_QUADRANTS) - covered)
    if missing:
        raise ValueError(f"unified cache lacks natural-completion rows for {missing!r}")
    return cache, tuple(rows), exact


def _validate_loaded_ledger(
    payload: Mapping[str, Any],
    *,
    exact: Mapping[tuple[Any, ...], ProfileRecord],
) -> tuple[ReplayAction, ...]:
    if payload.get("schema_version") != 1:
        raise ValueError("loaded ledger schema_version must be 1")
    if payload.get("ledger") != "critical_gpu_loaded_action_ledger":
        raise ValueError("loaded ledger identity mismatch")
    rows = payload.get("actions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("loaded ledger actions must be nonempty")
    if int(payload.get("action_count") or -1) != len(rows):
        raise ValueError("loaded ledger action_count mismatch")
    actions: list[ReplayAction] = []
    seen: set[str] = set()
    target_quadrants: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"loaded action {index} must be an object")
        action_id = _required_text(row, "action_id", f"loaded action {index}")
        if action_id in seen:
            raise ValueError(f"duplicate loaded action_id {action_id!r}")
        seen.add(action_id)
        if row.get("action_kind") != "directional_gpu_colocation_trajectory":
            raise ValueError(f"{action_id}: unsupported loaded action kind")
        if row.get("resource_state") != "mixed_colocation":
            raise ValueError(f"{action_id}: loaded resource_state mismatch")
        if row.get("target_completion_holdout_covered") is not True:
            raise ValueError(f"{action_id}: target holdout is not covered")
        if row.get("resident_service_holdout_covered") is not True:
            raise ValueError(f"{action_id}: resident holdout is not covered")
        target_key = _required_text(row, "target_workload_key", action_id)
        resident_key = _required_text(row, "resident_workload_key", action_id)
        if target_key not in WORKLOAD_CONTRACTS or resident_key not in WORKLOAD_CONTRACTS:
            raise ValueError(f"{action_id}: unregistered workload coordinate")
        vector = _positive_service_vector(row.get("lower_service_vector"), action_id)
        if set(vector) != {target_key, resident_key}:
            raise ValueError(f"{action_id}: loaded vector coordinates mismatch")
        profile = _positive_int(row.get("profile"), f"{action_id}.profile")
        if profile != 2:
            raise ValueError(f"{action_id}: directional loaded replay requires profile 2")
        node_bucket = _required_text(row, "node_bucket", action_id)
        resident_mix = _required_text(row, "resident_mix", action_id)
        common = {
            "node_bucket": node_bucket,
            "resource_state": "mixed_colocation",
            "resident_mix": resident_mix,
            "profile": profile,
            "allocation_workers": 1,
            "colocation_count": profile,
        }
        target_record = _exact_record(
            exact,
            workload_key=target_key,
            workload_env=_required_text(row, "target_workload_env", action_id),
            **common,
        )
        resident_record = _exact_record(
            exact,
            workload_key=resident_key,
            workload_env=_required_text(row, "resident_workload_env", action_id),
            **common,
        )
        _validate_natural_completion_record(target_record, label=f"{action_id}.target")
        _validate_lower_record(resident_record, label=f"{action_id}.resident")
        _assert_close(target_record.aggregate_rate, vector[target_key], f"{action_id}.target lower service")
        _assert_close(resident_record.aggregate_rate, vector[resident_key], f"{action_id}.resident lower service")
        upper = _positive_float(
            row.get("simultaneous_upper_target_completion_s"),
            f"{action_id}.simultaneous_upper_target_completion_s",
        )
        point_rate = _completion_point_rate(target_record)
        point_time = _completion_group_units(target_record) / point_rate
        if point_time > upper * (1.0 + 1e-9):
            raise ValueError(f"{action_id}: natural point completion exceeds certified upper bound")
        penalty = _nonnegative_float(
            row.get("bounded_action_penalty_units"),
            f"{action_id}.bounded_action_penalty_units",
        )
        actions.append(
            ReplayAction(
                lower=LowerAction(
                    action_id=action_id,
                    workload_key=target_key,
                    profile=profile,
                    node_bucket=node_bucket,
                    resource_state="mixed_colocation",
                    resident_mix=resident_mix,
                    lower_service=vector[target_key],
                    lower_service_vector=tuple(sorted(vector.items())),
                    penalty_units=penalty,
                    loaded_action=True,
                    target_batch_width=1,
                    required_job_counts=tuple(sorted(((resident_key, 1), (target_key, 1)))),
                ),
                evaluation=EvaluationAction(
                    action_id=action_id,
                    completion_point_rate=_completion_point_rate(target_record),
                    completion_group_units=_completion_group_units(target_record),
                    replay_job_units=_completion_group_units(target_record),
                    fixed_overhead_s=0.0,
                    completion_model_sample_count=int(target_record.completion_model_sample_count),
                    eta_source=target_record.eta_source,
                    completion_model_kind=(
                        "semi_markov_target_natural_completion_with_resident_lcb_accounting"
                    ),
                ),
            )
        )
        target_quadrants.add(_quadrant_for_workload(target_key))
    if not {"q01", "q11"}.issubset(target_quadrants):
        raise ValueError("loaded ledger must contain target actions for q01 and q11")
    return tuple(sorted(actions, key=lambda action: action.lower.action_id))


def _validate_migration_certificate(
    payload: Mapping[str, Any],
    *,
    cache_sha256: str,
    exact: Mapping[tuple[Any, ...], ProfileRecord],
) -> tuple[tuple[Scenario, ...], tuple[dict[str, Any], ...]]:
    if payload.get("schema_version") != 1:
        raise ValueError("migration certificate schema_version must be 1")
    if payload.get("pass") is not True or str(payload.get("status") or "") not in {
        "PASS",
        "MIGRATION_RECALIBRATION_PASS",
        "UNIFIED_MIGRATION_CERTIFICATE_PASS",
    }:
        raise ValueError("migration certificate is not passing")
    if "migration" not in str(payload.get("gate") or "").lower():
        raise ValueError("migration certificate gate identity mismatch")
    bound_hash = str(payload.get("service_cache_sha256") or payload.get("cache_sha256") or "")
    if bound_hash != cache_sha256:
        raise ValueError("migration certificate is not bound to the supplied cache hash")
    if payload.get("rates_recomputed_from_final_cache") is not True:
        raise ValueError("migration rates were not recomputed from the final cache")
    if payload.get("no_touch_safety_ready") is not True:
        raise ValueError("migration no-touch safety certificate is missing")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("migration certificate rows must be nonempty")

    scenarios: list[Scenario] = []
    audited_rows: list[dict[str, Any]] = []
    observed: dict[str, set[float]] = {family: set() for family in EXPECTED_MIGRATION_FAMILIES}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"migration row {index} must be an object")
        action_id = _required_text(row, "action_id", f"migration row {index}")
        if action_id in seen:
            raise ValueError(f"duplicate migration action_id {action_id!r}")
        seen.add(action_id)
        family = _required_text(row, "family", action_id)
        if family not in observed:
            raise ValueError(f"{action_id}: unsupported migration family {family!r}")
        point = _finite_float(row.get("progress_fraction"), f"{action_id}.progress_fraction")
        if not any(math.isclose(point, value, abs_tol=1e-12) for value in EXPECTED_MIGRATION_POINTS):
            raise ValueError(f"{action_id}: unregistered migration point {point}")
        observed[family].add(point)
        for flag in (
            "measurement_valid",
            "controlled_benchmark",
            "checkpoint_verified",
            "resume_verified",
            "rates_recomputed_from_cache",
        ):
            if row.get(flag) is not True:
                raise ValueError(f"{action_id}: {flag} is not true")
        if row.get("ordinary_running_tasks_touched") not in (False, None):
            raise ValueError(f"{action_id}: ordinary running task was touched")
        source_record = _record_from_service_key(exact, row.get("source_service_key"), f"{action_id}.source")
        target_record = _record_from_service_key(exact, row.get("destination_service_key"), f"{action_id}.destination")
        _validate_natural_completion_record(source_record, label=f"{action_id}.source")
        _validate_natural_completion_record(target_record, label=f"{action_id}.destination")
        workload_key = _required_text(row, "workload_key", action_id)
        if workload_key != source_record.workload_key or workload_key != target_record.workload_key:
            raise ValueError(f"{action_id}: migration workload/cache mismatch")
        if source_record.unit != target_record.unit:
            raise ValueError(f"{action_id}: source and destination service units differ")
        remaining_unit = _required_text(row, "remaining_work_unit", action_id)
        if remaining_unit != source_record.unit:
            raise ValueError(f"{action_id}: remaining-work unit/cache unit mismatch")
        expected_quadrant = {
            "pure_cpu": "q10",
            "pure_gpu": "q01",
            "hybrid_rl": "q11",
        }[family]
        if _quadrant_for_workload(workload_key) != expected_quadrant:
            raise ValueError(f"{action_id}: migration family/quadrant mismatch")
        current_rate = _positive_float(row.get("current_lower_service"), f"{action_id}.current_lower_service")
        target_rate = _positive_float(row.get("target_lower_service"), f"{action_id}.target_lower_service")
        _assert_close(
            current_rate,
            _per_task_lower_service(source_record, label=f"{action_id}.source"),
            f"{action_id}.source rate",
        )
        _assert_close(
            target_rate,
            _per_task_lower_service(target_record, label=f"{action_id}.destination"),
            f"{action_id}.target rate",
        )
        remaining = _positive_float(row.get("remaining_work"), f"{action_id}.remaining_work")
        cost = row.get("migration_cost")
        if not isinstance(cost, dict):
            raise ValueError(f"{action_id}: migration_cost must be an object")
        time_fields = (
            "checkpoint_flush_s",
            "sync_s",
            "environment_staging_s",
            "resume_warmup_s",
            "lost_work_s",
        )
        measured_time = sum(
            _nonnegative_float(cost.get(field), f"{action_id}.{field}")
            for field in time_fields
        )
        risk = _nonnegative_float(cost.get("risk_penalty_units"), f"{action_id}.risk_penalty_units")
        total_time = _nonnegative_float(cost.get("total_time_s"), f"{action_id}.total_time_s")
        penalty = _nonnegative_float(cost.get("total_penalty_units"), f"{action_id}.total_penalty_units")
        _assert_close(total_time, measured_time, f"{action_id}.measured migration time")
        _assert_close(penalty, total_time + risk, f"{action_id}.migration penalty")
        keep_lower_completion = remaining / current_rate
        migrate_lower_completion = penalty + remaining / target_rate
        beneficial = keep_lower_completion > migrate_lower_completion
        if row.get("beneficial_by_threshold") is not beneficial:
            raise ValueError(f"{action_id}: beneficial threshold was not recomputed")
        theorem_ready = bool(beneficial)
        if row.get("theorem_ready") is not theorem_ready:
            raise ValueError(f"{action_id}: theorem_ready mismatch")
        effective_migration_rate = remaining / migrate_lower_completion
        reported_effective = row.get("effective_migration_lower_service")
        if reported_effective is not None:
            _assert_close(
                reported_effective,
                effective_migration_rate,
                f"{action_id}.effective migration lower service",
            )

        keep_id = f"keep:{action_id}"
        keep = ReplayAction(
            lower=LowerAction(
                action_id=keep_id,
                workload_key=workload_key,
                profile=1,
                node_bucket=source_record.node_bucket,
                resource_state=source_record.resource_state,
                resident_mix=source_record.resident_mix,
                lower_service=current_rate,
                lower_service_vector=((workload_key, current_rate),),
                penalty_units=0.0,
                target_batch_width=1,
                required_job_counts=((workload_key, 1),),
            ),
            evaluation=EvaluationAction(
                action_id=keep_id,
                completion_point_rate=_per_task_completion_point_rate(source_record),
                completion_group_units=float(source_record.total_units),
                replay_job_units=remaining,
                fixed_overhead_s=0.0,
                completion_model_sample_count=int(source_record.completion_model_sample_count),
                eta_source=source_record.eta_source,
                completion_model_kind="natural_completion_keep",
            ),
        )
        migrate = ReplayAction(
            lower=LowerAction(
                action_id=action_id,
                workload_key=workload_key,
                profile=1,
                node_bucket=target_record.node_bucket,
                resource_state=target_record.resource_state,
                resident_mix=target_record.resident_mix,
                lower_service=effective_migration_rate,
                lower_service_vector=((workload_key, effective_migration_rate),),
                penalty_units=0.0,
                migration_action=True,
                target_batch_width=1,
                required_job_counts=((workload_key, 1),),
            ),
            evaluation=EvaluationAction(
                action_id=action_id,
                completion_point_rate=_per_task_completion_point_rate(target_record),
                completion_group_units=float(target_record.total_units),
                replay_job_units=remaining,
                fixed_overhead_s=total_time,
                completion_model_sample_count=int(target_record.completion_model_sample_count),
                eta_source=target_record.eta_source,
                completion_model_kind="natural_completion_after_measured_migration",
            ),
        )
        quadrant = _quadrant_for_workload(workload_key)
        scenarios.append(
            Scenario(
                scenario_id=f"migration:{action_id}",
                quadrant=quadrant,
                node_bucket=f"{source_record.node_bucket}->{target_record.node_bucket}",
                resource_count=1,
                workload_keys=(workload_key,),
                actions=((keep, migrate) if theorem_ready else (keep,)),
                scenario_kind="controlled_migration_counterfactual",
            )
        )
        audited_rows.append(
            {
                "action_id": action_id,
                "family": family,
                "progress_fraction": point,
                "workload_key": workload_key,
                "source_cache_key": list(_record_key(source_record)),
                "destination_cache_key": list(_record_key(target_record)),
                "current_lower_service": current_rate,
                "target_lower_service": target_rate,
                "effective_migration_lower_service": effective_migration_rate,
                "remaining_work": remaining,
                "migration_time_s": total_time,
                "risk_adjusted_migration_delay_s": penalty,
                "beneficial_by_threshold": beneficial,
                "natural_keep_time_s": remaining / _per_task_completion_point_rate(source_record),
                "natural_migrate_time_s": total_time + remaining / _per_task_completion_point_rate(target_record),
            }
        )
    for family, points in observed.items():
        if not all(any(math.isclose(point, expected, abs_tol=1e-12) for point in points) for expected in EXPECTED_MIGRATION_POINTS):
            raise ValueError(f"migration family {family!r} lacks 25/50/75 percent coverage")
    return tuple(sorted(scenarios, key=lambda row: row.scenario_id)), tuple(audited_rows)


def _build_hardware_scenarios(inputs: ValidatedInputs) -> tuple[Scenario, ...]:
    loaded_by_id = {action.lower.action_id: action for action in inputs.loaded_actions}
    loaded_coordinate_keys = {
        (
            action.lower.workload_key,
            action.lower.node_bucket,
            action.lower.resource_state,
            action.lower.resident_mix,
            action.lower.profile,
        )
        for action in inputs.loaded_actions
    }
    grouped: dict[tuple[str, str], list[ReplayAction]] = {}
    for record in inputs.cache_rows:
        if record.workload_key not in WORKLOAD_CONTRACTS or record.capacity_boundary:
            continue
        if not record.completion_model_ready:
            continue
        if record.resource_state == "mixed_colocation" and record.resident_mix:
            coordinate = (
                record.workload_key,
                record.node_bucket,
                record.resource_state,
                record.resident_mix,
                int(record.colocation_count),
            )
            if coordinate not in loaded_coordinate_keys:
                raise ValueError(f"unregistered completion-ready loaded coordinate {coordinate!r}")
            continue
        _validate_natural_completion_record(record, label="hardware scenario")
        quadrant = _quadrant_for_workload(record.workload_key)
        action_id = _record_action_id(record)
        action = ReplayAction(
            lower=LowerAction(
                action_id=action_id,
                workload_key=record.workload_key,
                profile=int(record.colocation_count),
                node_bucket=record.node_bucket,
                resource_state=record.resource_state,
                resident_mix=record.resident_mix,
                lower_service=float(record.aggregate_rate),
                lower_service_vector=((record.workload_key, float(record.aggregate_rate)),),
                penalty_units=0.0,
            ),
            evaluation=_evaluation_from_record(action_id, record, kind="task_native_natural_completion"),
        )
        grouped.setdefault((quadrant, record.node_bucket), []).append(action)
    for action in loaded_by_id.values():
        quadrant = _quadrant_for_workload(action.lower.workload_key)
        grouped.setdefault((quadrant, action.lower.node_bucket), []).append(action)

    scenarios = []
    for (quadrant, node_bucket), actions in sorted(grouped.items()):
        unique = {action.lower.action_id: action for action in actions}
        workloads = tuple(sorted({action.lower.workload_key for action in unique.values()}))
        if not workloads:
            continue
        scenarios.append(
            Scenario(
                scenario_id=f"{quadrant}:{_safe_id(node_bucket)}",
                quadrant=quadrant,
                node_bucket=node_bucket,
                resource_count=_resource_count(node_bucket, tuple(unique.values())),
                workload_keys=workloads,
                actions=tuple(sorted(unique.values(), key=lambda action: action.lower.action_id)),
            )
        )
    covered = {scenario.quadrant for scenario in scenarios}
    missing = sorted(set(EXPECTED_QUADRANTS) - covered)
    if missing:
        raise ValueError(f"hardware scenario construction lacks {missing!r}")
    return tuple(scenarios)


def _policy_descriptors() -> tuple[PolicyDescriptor, ...]:
    rows = [
        PolicyDescriptor(
            name=OURS_POLICY,
            category="ours",
            semantic_key="robust_maxweight",
            metadata={
                "objective": "lower-service MaxWeight with bounded penalty and support-preserving delay tie-break",
            },
        ),
        PolicyDescriptor(
            name=LEGACY_POLICY,
            category="legacy",
            semantic_key="legacy_fixed_caps",
            metadata={
                "objective": "fixed historical co-location caps without loaded-ledger or migration actions",
                "fixed_profiles": dict(LEGACY_FIXED_PROFILES),
            },
        ),
    ]
    for spec in sota_baseline_specs():
        rows.append(
            PolicyDescriptor(
                name=spec.policy.name,
                category="sota_style",
                semantic_key=spec.name,
                metadata=spec.snapshot(),
            )
        )
    rows.extend(
        PolicyDescriptor(
            name=name,
            category="ablation",
            semantic_key=name,
            metadata={"removed_or_changed_component": note},
        )
        for name, note in ABLATION_POLICIES
    )
    names = [row.name for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("policy names are not unique")
    return tuple(rows)


def _trace_protocols(
    *, loads: Sequence[float], seeds: Sequence[int], nominal_online_load: float
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for seed in (int(value) for value in seeds):
        rows.extend(
            (
                {"arrival_family": "static", "arrival_process": "static", "load_factor": None, "seed": seed},
                {"arrival_family": "poisson", "arrival_process": "poisson", "load_factor": float(nominal_online_load), "seed": seed},
                {"arrival_family": "bursty", "arrival_process": "bursty", "load_factor": float(nominal_online_load), "seed": seed},
            )
        )
        for load in (float(value) for value in loads):
            for process in ("poisson", "bursty"):
                rows.append(
                    {
                        "arrival_family": "load_sweep",
                        "arrival_process": process,
                        "load_factor": load,
                        "seed": seed,
                    }
                )
    return tuple(rows)


def _run_matrix(
    *,
    scenarios: Sequence[Scenario],
    policies: Sequence[PolicyDescriptor],
    trace_protocols: Sequence[Mapping[str, Any]],
    jobs_per_workload: int,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        protocols = (
            trace_protocols
            if scenario.scenario_kind == "hardware_local"
            else tuple(row for row in trace_protocols if row["arrival_family"] == "static")
        )
        for protocol in protocols:
            trace = _build_trace(
                scenario,
                protocol,
                jobs_per_workload=(1 if scenario.scenario_kind.startswith("controlled_migration") else jobs_per_workload),
            )
            for migration_mode in EXPECTED_MIGRATION_MODES:
                for policy in policies:
                    result = _run_policy(
                        scenario=scenario,
                        trace=trace,
                        policy=policy,
                        migration_mode=migration_mode,
                    )
                    runs.append(result)
    return runs


def _build_trace(
    scenario: Scenario,
    protocol: Mapping[str, Any],
    *,
    jobs_per_workload: int,
) -> TraceSpec:
    family = str(protocol["arrival_family"])
    process = str(protocol["arrival_process"])
    load = protocol.get("load_factor")
    seed = int(protocol["seed"])
    rng = random.Random(f"{scenario.scenario_id}|{family}|{process}|{load}|{seed}")
    jobs: list[TraceJob] = []
    unit_scales: dict[str, float] = {}
    for workload_key in scenario.workload_keys:
        actions = [action for action in scenario.actions if action.lower.workload_key == workload_key]
        if not actions:
            raise ValueError(f"{scenario.scenario_id}: no actions for {workload_key}")
        action_units = [_canonical_units(action) for action in actions]
        canonical_units = mean(action_units)
        if any(
            not math.isclose(value, canonical_units, rel_tol=1e-6, abs_tol=1e-9)
            for value in action_units
        ):
            raise ValueError(
                f"{scenario.scenario_id}/{workload_key}: action job-unit scales disagree"
            )
        unit_scales[workload_key] = canonical_units
        dedicated = [
            action
            for action in actions
            if not action.lower.loaded_action and not action.lower.migration_action
        ]
        capacity_actions = dedicated or actions
        lower_capacity = (
            max(action.lower.lower_service for action in capacity_actions)
            * scenario.resource_count
        )
        arrivals = _arrival_times(
            count=int(jobs_per_workload),
            process=process,
            load_factor=(float(load) if load is not None else None),
            lower_capacity=lower_capacity,
            canonical_units=canonical_units,
            rng=rng,
        )
        for index, arrival in enumerate(arrivals):
            variation = math.exp(rng.gauss(-0.5 * 0.05 * 0.05, 0.05))
            jobs.append(
                TraceJob(
                    job_id=f"{scenario.scenario_id}:{family}:{process}:{seed}:{workload_key}:{index:04d}",
                    workload_key=workload_key,
                    arrival_s=arrival,
                    total_units=max(1e-12, canonical_units * variation),
                )
            )
    jobs.sort(key=lambda job: (job.arrival_s, job.job_id))
    load_text = "na" if load is None else f"{float(load):.3f}"
    return TraceSpec(
        trace_id=f"{scenario.scenario_id}:{family}:{process}:load={load_text}:seed={seed}",
        arrival_family=family,
        arrival_process=process,
        load_factor=(float(load) if load is not None else None),
        seed=seed,
        jobs=tuple(jobs),
        workload_unit_scales=tuple(sorted(unit_scales.items())),
    )


def _arrival_times(
    *,
    count: int,
    process: str,
    load_factor: float | None,
    lower_capacity: float,
    canonical_units: float,
    rng: random.Random,
) -> list[float]:
    if count <= 0:
        return []
    if process == "static":
        return [0.0] * count
    if process not in {"poisson", "bursty"} or load_factor is None:
        raise ValueError(f"invalid arrival protocol {process!r}/{load_factor!r}")
    jobs_per_second = max(1e-12, lower_capacity / max(canonical_units, 1e-12))
    arrival_rate = max(1e-12, min(0.995, float(load_factor)) * jobs_per_second)
    mean_interarrival = 1.0 / arrival_rate
    now = 0.0
    arrivals: list[float] = []
    if process == "poisson":
        for _ in range(count):
            arrivals.append(now)
            now += rng.expovariate(1.0 / mean_interarrival)
        return arrivals
    burst = max(2, min(8, int(round(math.sqrt(count)))))
    while len(arrivals) < count:
        arrivals.extend([now] * min(burst, count - len(arrivals)))
        now += rng.expovariate(1.0 / (mean_interarrival * burst))
    return arrivals


def _run_policy(
    *, scenario: Scenario, trace: TraceSpec, policy: PolicyDescriptor, migration_mode: str
) -> dict[str, Any]:
    if migration_mode not in EXPECTED_MIGRATION_MODES:
        raise ValueError(f"unsupported migration mode {migration_mode!r}")
    allowed = _allowed_actions(
        scenario.actions,
        policy=policy,
        migration_mode=migration_mode,
    )
    if not allowed:
        raise ValueError(f"{scenario.scenario_id}/{policy.name}: no allowed actions")
    completions, selected_counts, selection_audit, queue_metrics = _simulate_scenario(
        trace=trace,
        actions=allowed,
        policy=policy,
        quadrant=scenario.quadrant,
        resource_count=scenario.resource_count,
    )
    flows = [completions[job.job_id] - job.arrival_s for job in trace.jobs]
    makespan = max(completions.values(), default=0.0)
    system_population = _system_population_metrics(trace.jobs, completions)
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_kind": scenario.scenario_kind,
        "quadrant": scenario.quadrant,
        "node_bucket": scenario.node_bucket,
        "trace_id": trace.trace_id,
        "arrival_family": trace.arrival_family,
        "arrival_process": trace.arrival_process,
        "load_factor": trace.load_factor,
        "seed": trace.seed,
        "migration_mode": migration_mode,
        "policy": policy.name,
        "policy_category": policy.category,
        "policy_semantic_key": policy.semantic_key,
        "job_count": len(trace.jobs),
        "completed_jobs": len(completions),
        "makespan_s": makespan,
        "mean_flow_s": mean(flows) if flows else 0.0,
        "p90_flow_s": _quantile(flows, 0.90),
        "mean_queue_backlog_jobs": queue_metrics["mean_queue_backlog_jobs"],
        "max_queue_backlog_jobs": queue_metrics["max_queue_backlog_jobs"],
        "mean_unfinished_jobs": system_population["mean_unfinished_jobs"],
        "max_unfinished_jobs": system_population["max_unfinished_jobs"],
        "selected_action_counts": dict(sorted(selected_counts.items())),
        "selection_audit": selection_audit,
        "resource_simulation": "shared_lane_joint_event_v2",
        "workload_unit_scales": trace.unit_scales(),
        "selection_service_view": "lower_service",
        "evaluation_service_view": "natural_completion_point",
    }


def _allowed_actions(
    actions: Sequence[ReplayAction], *, policy: PolicyDescriptor, migration_mode: str
) -> tuple[ReplayAction, ...]:
    out = []
    for action in actions:
        if migration_mode == "without_migration" and action.lower.migration_action:
            continue
        if policy.semantic_key == "legacy_fixed_caps" and (
            action.lower.migration_action or action.lower.loaded_action
        ):
            continue
        if policy.semantic_key == "ablation_no_migration" and action.lower.migration_action:
            continue
        if policy.semantic_key == "ablation_no_loaded_ledger" and action.lower.loaded_action:
            continue
        out.append(action)
    return tuple(out)


def _simulate_scenario(
    *,
    trace: TraceSpec,
    actions: Sequence[ReplayAction],
    policy: PolicyDescriptor,
    quadrant: str,
    resource_count: int,
) -> tuple[
    dict[str, float],
    dict[str, int],
    list[dict[str, Any]],
    dict[str, float | int],
]:
    states = {
        job.job_id: ReplayJobState(
            job=job,
            remaining_units=float(job.total_units),
            available_at=float(job.arrival_s),
        )
        for job in trace.jobs
    }
    available = [0.0 for _ in range(max(1, int(resource_count)))]
    completions: dict[str, float] = {}
    counts: dict[str, int] = {}
    audits: list[dict[str, Any]] = []
    queue_events: list[tuple[float, int]] = [
        (float(job.arrival_s), 1) for job in trace.jobs
    ]
    unit_scales = trace.unit_scales()
    while states:
        lane = min(range(len(available)), key=lambda index: (available[index], index))
        now = float(available[lane])
        ready = [
            state
            for state in states.values()
            if state.job.arrival_s <= now + 1e-12
            and state.available_at <= now + 1e-12
        ]
        if not ready:
            next_time = min(
                max(state.job.arrival_s, state.available_at)
                for state in states.values()
            )
            if next_time <= now + 1e-12:
                raise ValueError("joint replay cannot advance to a ready job")
            available[lane] = next_time
            continue

        ready_counts = _state_counts(ready)
        feasible = [
            action
            for action in actions
            if _requirements_satisfied(action.lower.job_requirements(), ready_counts)
        ]
        if not feasible:
            future = [
                max(state.job.arrival_s, state.available_at)
                for state in states.values()
                if max(state.job.arrival_s, state.available_at) > now + 1e-12
            ]
            if future:
                available[lane] = min(future)
                continue
            raise ValueError(
                f"no feasible joint action for ready counts {ready_counts!r}"
            )

        backlog_work = _state_work(ready)
        normalized_backlog = {
            key: value / _positive_float(unit_scales.get(key), f"unit scale {key}")
            for key, value in backlog_work.items()
        }
        remaining_counts = _state_counts(ready)
        oldest_ready_workload = min(
            ready,
            key=lambda state: (state.job.arrival_s, state.job.job_id),
        ).job.workload_key
        selected_lower = _select_lower_action(
            policy=policy,
            actions=tuple(action.lower for action in feasible),
            quadrant=quadrant,
            remaining_count=len(ready),
            backlog_vector=normalized_backlog,
            workload_unit_scales=unit_scales,
            remaining_counts=remaining_counts,
            oldest_ready_workload=oldest_ready_workload,
        )
        action = next(
            action for action in feasible
            if action.lower.action_id == selected_lower.action_id
        )
        target_width = action.lower.batch_width()
        target_pool = [
            state for state in ready
            if state.job.workload_key == action.lower.workload_key
        ]
        target_pool.sort(key=lambda state: _job_order_key(state, policy))
        targets = target_pool[:target_width]
        if len(targets) != target_width:
            raise ValueError(f"{action.lower.action_id}: target width is infeasible")

        used = {state.job.job_id for state in targets}
        residents: dict[str, list[ReplayJobState]] = {}
        for workload_key, required in action.lower.job_requirements().items():
            remaining_required = required - (
                target_width if workload_key == action.lower.workload_key else 0
            )
            if remaining_required <= 0:
                continue
            pool = [
                state for state in ready
                if state.job.workload_key == workload_key
                and state.job.job_id not in used
            ]
            pool.sort(key=lambda state: _job_order_key(state, policy))
            selected = pool[:remaining_required]
            if len(selected) != remaining_required:
                raise ValueError(f"{action.lower.action_id}: resident requirement is infeasible")
            residents[workload_key] = selected
            used.update(state.job.job_id for state in selected)
        queue_events.append((now, -len(used)))

        per_task_rate = (
            action.evaluation.completion_point_rate / float(target_width)
        )
        if not math.isfinite(per_task_rate) or per_task_rate <= 0.0:
            raise ValueError(f"{action.lower.action_id}: invalid completion point rate")
        overhead = float(action.evaluation.fixed_overhead_s)
        batch_end = now
        for state in targets:
            finish = now + overhead + state.remaining_units / per_task_rate
            completions[state.job.job_id] = finish
            batch_end = max(batch_end, finish)
            del states[state.job.job_id]

        frame_duration = batch_end - now
        vector = action.lower.service_vector()
        for workload_key, resident_states in residents.items():
            aggregate_rate = _positive_float(
                vector.get(workload_key),
                f"{action.lower.action_id}:{workload_key} resident rate",
            )
            per_resident_rate = aggregate_rate / len(resident_states)
            for state in resident_states:
                completion_offset = state.remaining_units / per_resident_rate
                if completion_offset <= frame_duration + 1e-12:
                    completions[state.job.job_id] = now + completion_offset
                    del states[state.job.job_id]
                else:
                    state.remaining_units -= per_resident_rate * frame_duration
                    state.available_at = batch_end
                    queue_events.append((batch_end, 1))

        available[lane] = batch_end
        counts[action.lower.action_id] = counts.get(action.lower.action_id, 0) + 1
        audits.append(
            {
                "action_id": action.lower.action_id,
                "target_workload_key": action.lower.workload_key,
                "selection_service_view": "lower_service",
                "evaluation_service_view": "natural_completion_point",
                "loaded_action": action.lower.loaded_action,
                "migration_action": action.lower.migration_action,
                "target_batch_width": target_width,
                "required_job_counts": action.lower.job_requirements(),
                "resident_job_counts": {
                    key: len(value) for key, value in residents.items()
                },
                "resource_lane": lane,
                "frame_start_s": now,
                "frame_end_s": batch_end,
                "remaining_count_before": len(ready),
                "ready_count_before": len(ready),
                "normalized_backlog_vector": normalized_backlog,
            }
        )
    horizon = max(completions.values(), default=0.0)
    return completions, counts, audits, _queue_metrics(queue_events, horizon=horizon)


def _state_counts(states: Sequence[ReplayJobState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in states:
        key = state.job.workload_key
        counts[key] = counts.get(key, 0) + 1
    return counts


def _state_work(states: Sequence[ReplayJobState]) -> dict[str, float]:
    work: dict[str, float] = {}
    for state in states:
        key = state.job.workload_key
        work[key] = work.get(key, 0.0) + float(state.remaining_units)
    return work


def _lane_frames_do_not_overlap(audits: Sequence[Mapping[str, Any]]) -> bool:
    by_lane: dict[int, list[tuple[float, float]]] = {}
    for row in audits:
        lane = int(row.get("resource_lane") or 0)
        start = _nonnegative_float(row.get("frame_start_s"), "frame start")
        end = _nonnegative_float(row.get("frame_end_s"), "frame end")
        if end + 1e-12 < start:
            return False
        by_lane.setdefault(lane, []).append((start, end))
    return all(
        all(
            current[0] + 1e-12 >= previous[1]
            for previous, current in zip(rows, rows[1:])
        )
        for rows in (sorted(values) for values in by_lane.values())
    )


def _requirements_satisfied(
    requirements: Mapping[str, int], ready_counts: Mapping[str, int]
) -> bool:
    return all(int(ready_counts.get(key, 0)) >= int(count) for key, count in requirements.items())


def _job_order_key(
    state: ReplayJobState, policy: PolicyDescriptor
) -> tuple[Any, ...]:
    if policy.semantic_key in {"delay_oracle", "ablation_delay_only"}:
        return (state.remaining_units, state.job.arrival_s, state.job.job_id)
    return (state.job.arrival_s, state.job.job_id)


def _system_population_metrics(
    jobs: Sequence[TraceJob], completions: Mapping[str, float]
) -> dict[str, float | int]:
    events: dict[float, list[int]] = {}
    for job in jobs:
        completion = _finite_float(completions[job.job_id], "completion time")
        if completion + 1e-12 < job.arrival_s:
            raise ValueError(f"{job.job_id}: completion precedes arrival")
        events.setdefault(float(job.arrival_s), [0, 0])[0] += 1
        events.setdefault(completion, [0, 0])[1] += 1
    if not events:
        return {"mean_unfinished_jobs": 0.0, "max_unfinished_jobs": 0}
    count = 0
    maximum = 0
    area = 0.0
    previous = min(events)
    for timestamp in sorted(events):
        area += count * (timestamp - previous)
        arrivals, departures = events[timestamp]
        count += arrivals
        maximum = max(maximum, count)
        count -= departures
        if count < 0:
            raise ValueError("backlog event accounting became negative")
        previous = timestamp
    horizon = max(events)
    return {
        "mean_unfinished_jobs": area / horizon if horizon > 0.0 else 0.0,
        "max_unfinished_jobs": maximum,
    }


def _queue_metrics(
    events: Sequence[tuple[float, int]], *, horizon: float
) -> dict[str, float | int]:
    grouped: dict[float, list[int]] = {}
    for timestamp, delta in events:
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("queue event time must be finite and nonnegative")
        grouped.setdefault(float(timestamp), []).append(int(delta))
    if not grouped:
        return {"mean_queue_backlog_jobs": 0.0, "max_queue_backlog_jobs": 0}
    count = 0
    maximum = 0
    area = 0.0
    previous = min(grouped)
    for timestamp in sorted(grouped):
        area += count * (timestamp - previous)
        deltas = grouped[timestamp]
        count += sum(delta for delta in deltas if delta > 0)
        maximum = max(maximum, count)
        count += sum(delta for delta in deltas if delta < 0)
        if count < 0:
            raise ValueError("queue event accounting became negative")
        previous = timestamp
    if count != 0:
        raise ValueError(f"queue event accounting did not close: {count}")
    return {
        "mean_queue_backlog_jobs": area / horizon if horizon > 0.0 else 0.0,
        "max_queue_backlog_jobs": maximum,
    }


def _select_lower_action(
    *,
    policy: PolicyDescriptor,
    actions: Sequence[LowerAction],
    quadrant: str,
    remaining_count: int,
    backlog_vector: Mapping[str, float],
    workload_unit_scales: Mapping[str, float],
    remaining_counts: Mapping[str, int],
    oldest_ready_workload: str,
) -> LowerAction:
    if not actions:
        raise ValueError("lower-service selector received no actions")
    feasible = list(actions)

    def normalized_rate(workload_key: str, rate: float) -> float:
        scale = _positive_float(
            workload_unit_scales.get(workload_key),
            f"unit scale {workload_key}",
        )
        return float(rate) / scale

    def support(action: LowerAction, *, penalty: bool = True) -> float:
        value = sum(
            max(0.0, float(backlog_vector.get(key, 0.0)))
            * normalized_rate(key, rate)
            for key, rate in action.lower_service_vector
        )
        return value - (action.penalty_units if penalty else 0.0)

    def target_support(action: LowerAction, *, penalty: bool = True) -> float:
        value = float(
            backlog_vector.get(action.workload_key, remaining_count)
        ) * normalized_rate(action.workload_key, action.lower_service)
        return value - (action.penalty_units if penalty else 0.0)

    def lower_flow(action: LowerAction) -> float:
        return float(action.batch_width()) / max(
            normalized_rate(action.workload_key, action.lower_service),
            1e-12,
        )

    def low_profile_key(action: LowerAction) -> tuple[Any, ...]:
        return (action.profile, action.penalty_units, action.action_id)

    semantic = policy.semantic_key
    if semantic == "legacy_fixed_caps":
        workload_key = oldest_ready_workload
        workload_actions = [
            action for action in feasible if action.workload_key == workload_key
        ]
        if not workload_actions:
            workload_actions = feasible
        fixed_profile = int(LEGACY_FIXED_PROFILES.get(workload_key, 1))
        return min(
            workload_actions,
            key=lambda action: (
                abs(action.profile - fixed_profile),
                action.profile > fixed_profile,
                -action.profile,
                action.action_id,
            ),
        )
    if semantic in {"throughput_table_goodput", "finish_time_fairness", "ablation_support_only"}:
        return max(feasible, key=lambda action: (target_support(action), -lower_flow(action), -action.profile, action.action_id))
    if semantic in {"delay_oracle", "ablation_delay_only"}:
        return min(feasible, key=lambda action: (lower_flow(action), -target_support(action), low_profile_key(action)))
    if semantic == "interference_guard":
        unshared = [action for action in feasible if not action.loaded_action]
        pool = unshared or feasible
        guarded = _support_guard(pool, target_support)
        return min(guarded, key=lambda action: (lower_flow(action), low_profile_key(action)))
    if semantic == "quadrant_composite":
        if quadrant in {"q00", "q10"}:
            return min(feasible, key=lambda action: (lower_flow(action), -target_support(action), low_profile_key(action)))
        if quadrant == "q11":
            guarded = _support_guard(feasible, support)
            return min(guarded, key=lambda action: (lower_flow(action), low_profile_key(action)))
        return max(feasible, key=lambda action: (target_support(action), -action.profile, action.action_id))
    if semantic == "resource_adaptive_goodput":
        high_backlog = [
            action
            for action in feasible
            if int(remaining_counts.get(action.workload_key, 0)) >= 8
        ]
        if high_backlog:
            return max(
                high_backlog,
                key=lambda action: (
                    target_support(action),
                    -lower_flow(action),
                    -action.profile,
                    action.action_id,
                ),
            )
        return min(
            feasible,
            key=lambda action: (
                lower_flow(action),
                -target_support(action),
                low_profile_key(action),
            ),
        )
    if semantic == "packing_guard":
        conservative = [action for action in feasible if action.profile <= 2]
        pool = conservative or feasible
        return max(
            pool,
            key=lambda action: (
                normalized_rate(action.workload_key, action.lower_service)
                / action.batch_width(),
                target_support(action),
                -action.profile,
                action.action_id,
            ),
        )
    if semantic == "ablation_no_bounded_penalty":
        return max(feasible, key=lambda action: (support(action, penalty=False), -lower_flow(action), -action.profile, action.action_id))
    if semantic == "ablation_high_profile_tiebreak":
        guarded = _support_guard(feasible, support)
        return min(guarded, key=lambda action: (lower_flow(action), -action.profile, action.action_id))
    if semantic in {"ablation_no_loaded_ledger", "ablation_no_migration", "robust_maxweight"}:
        guarded = _support_guard(feasible, support)
        return min(guarded, key=lambda action: (lower_flow(action), low_profile_key(action)))
    raise ValueError(f"unsupported policy semantics {semantic!r}")


def _pareto_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(
            (str(row["scenario_id"]), str(row["trace_id"]), str(row["migration_mode"])),
            [],
        ).append(row)
    out = []
    for (scenario_id, trace_id, migration_mode), rows in sorted(grouped.items()):
        ours = _one(rows, lambda row: row["policy_category"] == "ours", "ours")
        sota = [row for row in rows if row["policy_category"] == "sota_style"]
        if not sota:
            raise ValueError(f"{scenario_id}/{trace_id}: no SOTA-style rows")
        comparisons = []
        for baseline in sota:
            comparisons.append(
                {
                    "baseline_policy": baseline["policy"],
                    "ours_to_baseline_makespan_ratio": _ratio(ours["makespan_s"], baseline["makespan_s"]),
                    "ours_to_baseline_mean_flow_ratio": _ratio(ours["mean_flow_s"], baseline["mean_flow_s"]),
                    "ours_strictly_pareto_dominates": _dominates(ours, baseline, tolerance=0.0),
                    "ours_tolerance_pareto_dominates": _dominates(ours, baseline, tolerance=PARETO_TOLERANCE),
                    "baseline_strictly_pareto_dominates": _dominates(baseline, ours, tolerance=0.0),
                }
            )
        out.append(
            {
                "scenario_id": scenario_id,
                "trace_id": trace_id,
                "quadrant": ours["quadrant"],
                "arrival_family": ours["arrival_family"],
                "migration_mode": migration_mode,
                "ours_policy": ours["policy"],
                "comparisons": comparisons,
                "ours_not_pareto_dominated_by_any_sota_style_policy": not any(
                    row["baseline_strictly_pareto_dominates"] for row in comparisons
                ),
                "ours_strictly_dominates_every_sota_style_policy": all(
                    row["ours_strictly_pareto_dominates"] for row in comparisons
                ),
                "ours_tolerance_dominates_every_sota_style_policy": all(
                    row["ours_tolerance_pareto_dominates"] for row in comparisons
                ),
            }
        )
    return out


def _legacy_comparison_rows(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(
            (str(row["scenario_id"]), str(row["trace_id"]), str(row["migration_mode"])),
            [],
        ).append(row)
    out = []
    for (scenario_id, trace_id, migration_mode), rows in sorted(grouped.items()):
        ours = _one(rows, lambda row: row["policy_category"] == "ours", "ours")
        legacy = _one(rows, lambda row: row["policy_category"] == "legacy", "legacy")
        out.append(
            {
                "scenario_id": scenario_id,
                "trace_id": trace_id,
                "quadrant": ours["quadrant"],
                "arrival_family": ours["arrival_family"],
                "migration_mode": migration_mode,
                "ours_policy": ours["policy"],
                "legacy_policy": legacy["policy"],
                "legacy_to_ours_makespan_ratio": _ratio(
                    legacy["makespan_s"], ours["makespan_s"]
                ),
                "legacy_to_ours_mean_flow_ratio": _ratio(
                    legacy["mean_flow_s"], ours["mean_flow_s"]
                ),
                "ours_strictly_pareto_dominates_legacy": _dominates(
                    ours, legacy, tolerance=0.0
                ),
                "ours_tolerance_pareto_dominates_legacy": _dominates(
                    ours, legacy, tolerance=PARETO_TOLERANCE
                ),
                "legacy_strictly_pareto_dominates_ours": _dominates(
                    legacy, ours, tolerance=0.0
                ),
            }
        )
    return out


def _coverage_checks(
    *,
    scenarios: Sequence[Scenario],
    policies: Sequence[PolicyDescriptor],
    trace_protocols: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    inputs: ValidatedInputs,
) -> dict[str, bool]:
    hardware = [scenario for scenario in scenarios if scenario.scenario_kind == "hardware_local"]
    expected_policy_names = {policy.name for policy in policies}
    sota_names = {policy.name for policy in policies if policy.category == "sota_style"}
    ablation_names = {policy.name for policy in policies if policy.category == "ablation"}
    legacy_names = {policy.name for policy in policies if policy.category == "legacy"}
    quadrants = {scenario.quadrant for scenario in hardware}
    per_quadrant_arrivals = {
        quadrant: {
            str(row["arrival_family"])
            for row in runs
            if row["quadrant"] == quadrant and row["scenario_kind"] == "hardware_local"
        }
        for quadrant in EXPECTED_QUADRANTS
    }
    matrix_keys = {
        (
            str(row["scenario_id"]),
            str(row["trace_id"]),
            str(row["migration_mode"]),
            str(row["policy"]),
        )
        for row in runs
    }
    expected_keys = set()
    for scenario in scenarios:
        protocols = (
            trace_protocols
            if scenario.scenario_kind == "hardware_local"
            else tuple(row for row in trace_protocols if row["arrival_family"] == "static")
        )
        for protocol in protocols:
            load = protocol.get("load_factor")
            load_text = "na" if load is None else f"{float(load):.3f}"
            trace_id = (
                f"{scenario.scenario_id}:{protocol['arrival_family']}:"
                f"{protocol['arrival_process']}:load={load_text}:seed={int(protocol['seed'])}"
            )
            for mode in EXPECTED_MIGRATION_MODES:
                for policy in expected_policy_names:
                    expected_keys.add((scenario.scenario_id, trace_id, mode, policy))
    return {
        "all_four_quadrants_covered": quadrants == set(EXPECTED_QUADRANTS),
        "all_arrival_families_per_quadrant": all(
            per_quadrant_arrivals[quadrant] == set(EXPECTED_ARRIVAL_FAMILIES)
            for quadrant in EXPECTED_QUADRANTS
        ),
        "all_sota_style_policies_registered": sota_names == {
            spec.policy.name for spec in sota_baseline_specs()
        },
        "all_ablation_policies_registered": ablation_names == {name for name, _ in ABLATION_POLICIES},
        "legacy_fixed_caps_registered": legacy_names == {LEGACY_POLICY},
        "complete_policy_arrival_migration_matrix": matrix_keys == expected_keys,
        "all_jobs_naturally_completed": all(int(row["completed_jobs"]) == int(row["job_count"]) for row in runs),
        "all_selection_bound_to_lower_service": all(
            row["selection_service_view"] == "lower_service"
            and all(audit["selection_service_view"] == "lower_service" for audit in row["selection_audit"])
            for row in runs
        ),
        "all_evaluation_bound_to_natural_completion": all(
            row["evaluation_service_view"] == "natural_completion_point"
            and all(audit["evaluation_service_view"] == "natural_completion_point" for audit in row["selection_audit"])
            for row in runs
        ),
        "all_runs_share_physical_lanes_across_workloads": all(
            row.get("resource_simulation") == "shared_lane_joint_event_v2"
            and _lane_frames_do_not_overlap(row.get("selection_audit") or [])
            for row in runs
        ),
        "all_backlog_metrics_finite": all(
            math.isfinite(float(row.get("mean_queue_backlog_jobs") or 0.0))
            and 0.0 <= float(row.get("mean_queue_backlog_jobs") or 0.0)
            <= float(row.get("max_queue_backlog_jobs") or 0.0) + 1e-12
            and math.isfinite(float(row.get("mean_unfinished_jobs") or 0.0))
            and 0.0 <= float(row.get("mean_unfinished_jobs") or 0.0)
            <= float(row.get("max_unfinished_jobs") or 0.0) + 1e-12
            for row in runs
        ),
        "all_loaded_actions_require_resident_and_target": all(
            action.lower.batch_width() == 1
            and sum(action.lower.job_requirements().values()) == 2
            and set(action.lower.job_requirements()) == set(action.lower.service_vector())
            for action in inputs.loaded_actions
        ),
        "loaded_action_ledger_consumed": bool(inputs.loaded_actions)
        and all(any(action.lower.action_id == loaded.lower.action_id for scenario in hardware for action in scenario.actions) for loaded in inputs.loaded_actions),
        "migration_certificate_consumed": bool(inputs.migration_rows)
        and len(inputs.migration_scenarios) == len(inputs.migration_rows),
        "both_migration_modes_present": {str(row["migration_mode"]) for row in runs} == set(EXPECTED_MIGRATION_MODES),
        "legacy_excludes_loaded_and_migration_actions": all(
            not audit["loaded_action"] and not audit["migration_action"]
            for row in runs
            if row["policy_category"] == "legacy"
            for audit in row["selection_audit"]
        ),
        "policy_semantics_only": True,
    }


def _performance_diagnostics(
    pareto_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_quadrant = {}
    for quadrant in EXPECTED_QUADRANTS:
        rows = [row for row in pareto_rows if row["quadrant"] == quadrant]
        by_quadrant[quadrant] = {
            "comparison_count": len(rows),
            "ours_not_dominated_in_every_comparison": bool(rows) and all(
                row["ours_not_pareto_dominated_by_any_sota_style_policy"] for row in rows
            ),
            "ours_strictly_dominates_every_policy_in_every_comparison": bool(rows) and all(
                row["ours_strictly_dominates_every_sota_style_policy"] for row in rows
            ),
            "ours_tolerance_dominates_every_policy_in_every_comparison": bool(rows) and all(
                row["ours_tolerance_dominates_every_sota_style_policy"] for row in rows
            ),
        }
    makespan_ratios = [float(row["legacy_to_ours_makespan_ratio"]) for row in legacy_rows]
    mean_flow_ratios = [float(row["legacy_to_ours_mean_flow_ratio"]) for row in legacy_rows]
    legacy = {
        "comparison_count": len(legacy_rows),
        "legacy_to_ours_makespan_geomean_ratio": _geomean(makespan_ratios),
        "legacy_to_ours_mean_flow_geomean_ratio": _geomean(mean_flow_ratios),
        "legacy_to_ours_worst_makespan_ratio": min(makespan_ratios, default=0.0),
        "legacy_to_ours_worst_mean_flow_ratio": min(mean_flow_ratios, default=0.0),
        "ours_strictly_dominates_legacy_in_every_comparison": bool(legacy_rows) and all(
            row["ours_strictly_pareto_dominates_legacy"] for row in legacy_rows
        ),
        "ours_tolerance_dominates_legacy_in_every_comparison": bool(legacy_rows) and all(
            row["ours_tolerance_pareto_dominates_legacy"] for row in legacy_rows
        ),
        "legacy_strictly_dominates_ours_count": sum(
            bool(row["legacy_strictly_pareto_dominates_ours"]) for row in legacy_rows
        ),
    }
    return {
        "by_quadrant": by_quadrant,
        "legacy": legacy,
        "strong_superiority_claim_ready": all(
            row["ours_strictly_dominates_every_policy_in_every_comparison"]
            for row in by_quadrant.values()
        ),
        "diagnostic_only_not_a_gate_condition": True,
    }


def _input_manifest(inputs: ValidatedInputs) -> dict[str, Any]:
    return {
        "unified_cache": {
            "path": str(inputs.cache_path),
            "sha256": inputs.cache_sha256,
            "record_count": len(inputs.cache_rows),
        },
        "loaded_action_ledger": {
            "path": str(inputs.loaded_path),
            "sha256": inputs.loaded_sha256,
            "validated_action_count": len(inputs.loaded_actions),
            "cache_binding": "exact coordinate and lower-service equality",
        },
        "migration_certificate": {
            "path": str(inputs.migration_path),
            "sha256": inputs.migration_sha256,
            "validated_row_count": len(inputs.migration_rows),
            "cache_sha256": inputs.cache_sha256,
        },
    }


def _record_from_service_key(
    exact: Mapping[tuple[Any, ...], ProfileRecord], value: Any, label: str
) -> ProfileRecord:
    if not isinstance(value, dict):
        raise ValueError(f"{label}_service_key must be an object")
    return _exact_record(
        exact,
        workload_key=_required_text(value, "workload_key", label),
        workload_env=_required_text(value, "workload_env", label),
        node_bucket=_required_text(value, "node_bucket", label),
        resource_state=_required_text(value, "resource_state", label),
        resident_mix=str(value.get("resident_mix") or ""),
        profile=_positive_int(value.get("profile") or value.get("colocation_count"), f"{label}.profile"),
        allocation_workers=_positive_int(value.get("allocation_workers") or 1, f"{label}.allocation_workers"),
        colocation_count=_positive_int(value.get("colocation_count") or value.get("profile"), f"{label}.colocation_count"),
    )


def _exact_record(
    exact: Mapping[tuple[Any, ...], ProfileRecord],
    *,
    workload_key: str,
    workload_env: str,
    node_bucket: str,
    resource_state: str,
    resident_mix: str,
    profile: int,
    allocation_workers: int,
    colocation_count: int,
) -> ProfileRecord:
    if int(profile) != int(colocation_count):
        raise ValueError("profile and colocation_count disagree")
    key = (
        str(workload_key),
        str(workload_env).strip().lower(),
        str(node_bucket).strip().lower(),
        str(resource_state).strip().lower(),
        int(allocation_workers),
        int(colocation_count),
        str(resident_mix).strip().lower(),
    )
    record = exact.get(key)
    if record is None:
        raise ValueError(f"missing exact cache tuple {key!r}")
    return record


def _record_key(record: ProfileRecord) -> tuple[Any, ...]:
    return (
        record.workload_key,
        record.workload_env.strip().lower(),
        record.node_bucket.strip().lower(),
        record.resource_state.strip().lower(),
        int(record.allocation_workers),
        int(record.colocation_count),
        record.resident_mix.strip().lower(),
    )


def _record_action_id(record: ProfileRecord) -> str:
    return (
        f"profile:{record.workload_key}:{_safe_id(record.node_bucket)}:"
        f"{_safe_id(record.resource_state)}:{_safe_id(record.resident_mix or 'none')}:"
        f"w{int(record.allocation_workers)}:p{int(record.colocation_count)}"
    )


def _evaluation_from_record(action_id: str, record: ProfileRecord, *, kind: str) -> EvaluationAction:
    group_units = _completion_group_units(record)
    return EvaluationAction(
        action_id=action_id,
        completion_point_rate=_completion_point_rate(record),
        completion_group_units=group_units,
        replay_job_units=group_units / float(max(1, record.colocation_count)),
        fixed_overhead_s=0.0,
        completion_model_sample_count=int(record.completion_model_sample_count),
        eta_source=record.eta_source,
        completion_model_kind=kind,
    )


def _validate_lower_record(record: ProfileRecord, *, label: str) -> None:
    if record.capacity_boundary:
        raise ValueError(f"{label}: capacity boundary is not an action")
    if not record.stable_rate_ready:
        raise ValueError(f"{label}: stable lower service is not ready")
    _positive_float(record.aggregate_rate, f"{label}.aggregate_rate")
    eta = str(record.eta_source or "").lower()
    if "hist" in eta or "history" in eta:
        raise ValueError(f"{label}: history ETA is forbidden")
    if not any(token in eta for token in ("tqdm", "progress")):
        raise ValueError(f"{label}: task-native progress ETA is required")


def _validate_natural_completion_record(record: ProfileRecord, *, label: str) -> None:
    _validate_lower_record(record, label=label)
    if not record.completion_model_ready:
        raise ValueError(f"{label}: natural completion model is not ready")
    if int(record.completion_model_sample_count) <= 0:
        raise ValueError(f"{label}: natural completion sample count is zero")
    _positive_float(record.completion_total_wall_s, f"{label}.completion_total_wall_s")
    _positive_float(_completion_group_units(record), f"{label}.completion_group_total_units")
    _positive_float(_completion_point_rate(record), f"{label}.completion_point_rate")


def _completion_group_units(record: ProfileRecord) -> float:
    explicit = float(record.completion_group_total_units)
    return explicit if explicit > 0.0 else float(record.total_units)


def _completion_point_rate(record: ProfileRecord) -> float:
    return _completion_group_units(record) / float(record.completion_total_wall_s)


def _per_task_completion_point_rate(record: ProfileRecord) -> float:
    return _completion_point_rate(record) / float(max(1, record.colocation_count))


def _per_task_lower_service(record: ProfileRecord, *, label: str) -> float:
    rates = tuple(float(value) for value in record.per_task_rates)
    if len(rates) != int(record.colocation_count):
        raise ValueError(f"{label}: per-task lower-service count disagrees with colocation")
    if not rates or any(not math.isfinite(value) or value <= 0.0 for value in rates):
        raise ValueError(f"{label}: per-task lower service must be finite and positive")
    _assert_close(sum(rates), record.aggregate_rate, f"{label}.aggregate lower service")
    return min(rates)


def _canonical_units(action: ReplayAction) -> float:
    if action.evaluation.completion_point_rate <= 0.0:
        raise ValueError(f"{action.lower.action_id}: completion rate is nonpositive")
    return _positive_float(action.evaluation.replay_job_units, "replay_job_units")


def _support_guard(
    actions: Sequence[LowerAction], score
) -> list[LowerAction]:
    best = max(float(score(action)) for action in actions)
    threshold = best - 0.02 * max(1.0, abs(best))
    guarded = [action for action in actions if float(score(action)) >= threshold]
    return guarded or list(actions)


def _resource_count(node_bucket: str, actions: Sequence[ReplayAction]) -> int:
    text = str(node_bucket).lower()
    if "quad" in text or "4x" in text:
        return 4
    if "dual" in text or "2x" in text:
        return 2
    if any(WORKLOAD_CONTRACTS[action.lower.workload_key]["quadrant"] in {"q01", "q11"} for action in actions):
        return 1
    return 1


def _quadrant_for_workload(workload_key: str) -> str:
    contract = WORKLOAD_CONTRACTS.get(str(workload_key))
    if contract is None:
        raise ValueError(f"unregistered workload {workload_key!r}")
    return contract["quadrant"]


def _positive_service_vector(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label}: lower_service_vector must be nonempty")
    return {
        str(key): _positive_float(rate, f"{label}.lower_service_vector[{key!r}]")
        for key, rate in value.items()
    }


def _required_text(row: Mapping[str, Any], field: str, label: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{label}: {field} is required")
    return value


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _nonnegative_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _assert_close(left: Any, right: Any, label: str) -> None:
    a = _finite_float(left, label)
    b = _finite_float(right, label)
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {a!r} != {b!r}")


def _one(rows: Sequence[Mapping[str, Any]], predicate, label: str) -> Mapping[str, Any]:
    selected = [row for row in rows if predicate(row)]
    if len(selected) != 1:
        raise ValueError(f"expected one {label} row, found {len(selected)}")
    return selected[0]


def _ratio(numerator: Any, denominator: Any) -> float:
    den = _positive_float(denominator, "ratio denominator")
    return _nonnegative_float(numerator, "ratio numerator") / den


def _geomean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    positive = [_positive_float(value, "geomean value") for value in values]
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any], *, tolerance: float) -> bool:
    upper = 1.0 + max(0.0, float(tolerance))
    left_ms = float(left["makespan_s"])
    left_flow = float(left["mean_flow_s"])
    right_ms = float(right["makespan_s"])
    right_flow = float(right["mean_flow_s"])
    return bool(
        left_ms <= right_ms * upper
        and left_flow <= right_flow * upper
        and (left_ms < right_ms * (1.0 - 1e-12) or left_flow < right_flow * (1.0 - 1e-12))
    )


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_") or "none"


def _issue(code: str, **fields: Any) -> dict[str, Any]:
    return {"code": str(code), **fields}


def _claim_boundary() -> str:
    return (
        "PASS certifies a complete replay matrix over the four registered "
        "Scheduleurm quadrants and the exact supplied measured-action universe. "
        "Every action is selected using conservative lower service; JCT, mean "
        "flow, and makespan are evaluated only after selection using task-native "
        "natural-completion models. Loaded trajectories use natural completion "
        "for the target and certified resident lower-service accounting over the "
        "same semi-Markov action duration. Migration costs are physically measured "
        "and their rates must be rebound to the exact cache hash. A migration's "
        "risk-adjusted delay is folded into its duration-normalized effective lower "
        "service before selection; raw seconds are never subtracted from a service "
        "score. The legacy row implements fixed historical co-location caps on "
        "the same measured cache and is not a fresh execution of the default "
        "legacy scheduler. SOTA rows are "
        "policy-semantics implementations on the same cache, not direct full-stack "
        "executions of Gavel, Pollux, Sia, IADeep, Salus, or any other external "
        "binary. PASS is an evidence/coverage result; superiority remains a "
        "separate computed diagnostic and is never a gate assumption. Hardware "
        "scenarios use a shared-lane joint event simulator: workload classes on "
        "one node compete for the same GPU or CPU lanes, and a directional loaded "
        "action consumes one resident and one target queue coordinate. Queue and "
        "service coordinates are divided by fixed task-native canonical-work "
        "scales before MaxWeight scoring."
    )


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(report), indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--loaded-ledger", type=Path, default=DEFAULT_LOADED_LEDGER)
    parser.add_argument("--migration-certificate", type=Path, default=DEFAULT_MIGRATION_CERTIFICATE)
    parser.add_argument("--loads", default=",".join(str(value) for value in DEFAULT_LOADS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--nominal-online-load", type=float, default=NOMINAL_ONLINE_LOAD)
    parser.add_argument("--jobs-per-workload", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_unified_hardware_or_replay(
        cache_path=args.cache,
        loaded_ledger_path=args.loaded_ledger,
        migration_certificate_path=args.migration_certificate,
        loads=tuple(float(value.strip()) for value in args.loads.split(",") if value.strip()),
        seeds=tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip()),
        nominal_online_load=args.nominal_online_load,
        jobs_per_workload=args.jobs_per_workload,
    )
    write_report(report, args.output)
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, indent=2, sort_keys=True))
    return 0 if report["pass"] else (3 if str(report["status"]).startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
