"""Register and inspect the 95% joint GPU collection protocol.

This module is deliberately incapable of launching a benchmark.  It builds the
twenty registered joint work items, checks their statistical and measurement
contracts, and reads optional *local* scheduler snapshots to report whether the
target node is currently idle.  A separate launcher must consume a registered
manifest later.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments import critical_gpu_completion_campaign as empty_suite
from algorithm.experiments import critical_gpu_loaded_completion_campaign as loaded_suite
from algorithm.experiments.critical_gpu_loaded_stochastic_lcb_gate import (
    _expected_scenarios as current_loaded_scenarios,
)
from algorithm.experiments.critical_gpu_stochastic_lcb_gate import (
    _expected_cells as current_empty_cells,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMPTY_GATE = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "critical_gpu_stochastic_lcb_gate_v8_jtl110gpu2_20260803.json"
)
DEFAULT_LOADED_GATE = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "critical_gpu_loaded_stochastic_lcb_gate_jtl110gpu2_20260809.json"
)

PROTOCOL = "critical_gpu_joint95_natural_completion_v1"
COLLECTION_PREFIX = "critical_gpu_joint95_v1"
NODE = "jtl110gpu2"
MIS_COVERAGE_ALPHA = 0.05
CALIBRATION_WAVES = tuple(range(14, 33))
HOLDOUT_WAVE = 33
JOINT_WAVES = CALIBRATION_WAVES + (HOLDOUT_WAVE,)
EMPTY_SELECTION = (
    ("gpu_cnn_torch_resnet50", 1),
    ("gpu_cnn_torch_resnet50", 3),
    ("gpu_heavy_jax_matmul", 1),
    ("gpu_heavy_jax_matmul", 3),
    ("gpu_llm_distilgpt2", 1),
    ("gpu_llm_distilgpt2", 3),
    ("gpu_llm_distilgpt2", 8),
)
LOADED_SELECTION = ("cnn_after_llm", "llm_after_cnn")
SCORE_CONTRACT = {
    "cell_score": "actual_time / frozen_predicted_time - 1",
    "joint_wave_score": "maximum cell_score over exactly nine registered functionals",
    "margin": "max(0, nineteenth order statistic of 19 joint-wave scores)",
    "empty_mean_task_jct_role": "collected_suite_diagnostic_not_scored",
    "loaded_resident_overlap_service_role": "collected_suite_diagnostic_not_scored",
}

# These findings come from source diffs against the manifests embedded in the
# August campaigns.  Unknown future changes fail the compatibility audit until
# somebody classifies their effect on the registered measurements.
EMPTY_KNOWN_CHANGES: dict[str, dict[str, Any]] = {
    "algorithm/experiments/critical_gpu_completion_campaign.py": {
        "expected_old_sha256": "2efcc8ed8582db7cd42bcda45048b5a260643970c2599cdde0a99a9335092149",
        "expected_current_sha256": "bd755d5b3809eff6e6827663464a77e7a7815c335433f8d85ff1abe576fa4175",
        "classification": "selected_payload_preserving",
        "finding": (
            "The selected jtl110gpu2 workload/profile definitions, natural-"
            "completion estimands, and launch arguments are unchanged. Current "
            "changes add an injectable code-identity callback, strengthen the "
            "environment preflight, and add a jtl311linux-only profile override."
        ),
        "predictor_reuse_compatible": True,
    },
    "algorithm/experiments/progress_wrapper.py": {
        "expected_old_sha256": "c8edb95425e18241f9b5ec33fa99c50093e2e8b8e21927b4a65817020910ade0",
        "expected_current_sha256": "dbd26ed0e16b2a549cf060411cd6fee2ea402c8aae311e031db56a6796268193",
        "classification": "readiness_parser_changed_estimand_preserved",
        "finding": (
            "Current code rejects progress counters whose declared total differs "
            "from the registered total and ignores lower-ranked duplicate terminal "
            "bars. This can change readiness diagnostics and phase decomposition, "
            "but the aligned registered functional remains natural-exit drain "
            "time. The new calibration and holdout must therefore use one "
            "frozen current bundle; old calibration residuals cannot be reused."
        ),
        "predictor_reuse_compatible": True,
    },
    "algorithm/experiments/remote_workload_selected_profile_probe.py": {
        "expected_old_sha256": "bba8c23beea9c5d2ae354cde4d37280a68c81e90c09697d30e4274f30c089cde",
        "expected_current_sha256": "ab299c432130ed135184272ec1ab9bc8c397d6824db8ab4e5aad8e29eb2884ed",
        "classification": "transport_only",
        "finding": (
            "Remote output retrieval now uses bounded chunks and verifies the "
            "reconstructed bytes. Workload commands and measured clocks are not "
            "changed."
        ),
        "predictor_reuse_compatible": True,
    },
    "algorithm/experiments/templates/torch_llm_distilgpt2.cmd.tpl": {
        "expected_old_sha256": "bef2f412fce12dc8a0b636317f7a720aa5da45aab75d38ff342c1d282c4f2c1f",
        "expected_current_sha256": "4852b935470552a570c04d05d81ba306780f767b96f9aee54b06c2dc97d2dfca",
        "classification": "runtime_environment_changed_estimand_preserved",
        "finding": (
            "The current LLM command exports the registered node PYTHONPATH. This "
            "is not an exact execution replica of the August empty campaign and "
            "may shift duration, but it keeps the same model, batch, sequence, "
            "precision, step count, checkpoint, and natural-exit drain estimand. "
            "Fresh current-bundle calibration absorbs this shift."
        ),
        "predictor_reuse_compatible": True,
    },
}

LOADED_KNOWN_CHANGES: dict[str, dict[str, Any]] = {
    "algorithm/experiments/critical_gpu_completion_campaign.py": {
        "expected_old_sha256": "ed21922f769e2095583973709f1b76e07be6af1eb958b77f0a60a7532ffee3a1",
        "expected_current_sha256": "bd755d5b3809eff6e6827663464a77e7a7815c335433f8d85ff1abe576fa4175",
        "classification": "unselected_node_registry_only",
        "finding": (
            "Relative to the August loaded bundle, the only changed shared source "
            "adds a CNN capacity override for jtl311linux. Neither selected "
            "jtl110gpu2 loaded scenario, its resident/target command, nor the "
            "target natural-completion functional changes."
        ),
        "predictor_reuse_compatible": True,
    }
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _source_manifest(gate: Mapping[str, Any]) -> dict[str, Any]:
    sources = list(gate.get("source_artifacts") or [])
    if not sources:
        raise ValueError("gate has no source artifacts")
    path = Path(str(sources[0].get("path") or ""))
    if not path.is_file():
        path = REPO_ROOT / "md" / "experiment_artifacts" / path.name
    payload = _read_json(path)
    manifest = payload.get("measurement_code_manifest") or {}
    if not isinstance(manifest, dict) or not manifest.get("sha256"):
        raise ValueError(f"source artifact lacks a code manifest: {path}")
    return dict(manifest)


def _manifest_index(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["path"]): str(row["sha256"])
        for row in manifest.get("files") or []
        if isinstance(row, Mapping) and row.get("path") and row.get("sha256")
    }


def _contract_index(
    rows: Iterable[Mapping[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    return {str(row[key]): dict(row) for row in rows}


def _selected_empty_contract(node: str) -> list[dict[str, Any]]:
    wanted = set(EMPTY_SELECTION)
    rows = [
        dict(row)
        for row in current_empty_cells(node).values()
        if (str(row["workload_key"]), int(row["profile"])) in wanted
    ]
    if {(str(row["workload_key"]), int(row["profile"])) for row in rows} != wanted:
        raise RuntimeError("current empty suite does not expose the seven aligned cells")
    return sorted(rows, key=lambda row: str(row["service_cell_id"]))


def _selected_loaded_contract(node: str) -> list[dict[str, Any]]:
    current = current_loaded_scenarios(node)
    missing = set(LOADED_SELECTION) - set(current)
    if missing:
        raise RuntimeError(f"current loaded suite lacks scenarios {sorted(missing)!r}")
    return [dict(current[key]) for key in sorted(LOADED_SELECTION)]


def _compare_contracts(
    *,
    old_rows: Iterable[Mapping[str, Any]],
    current_rows: Iterable[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    old = _contract_index(old_rows, key)
    current = _contract_index(current_rows, key)
    differing = sorted(
        identity
        for identity in set(old) & set(current)
        if old[identity] != current[identity]
    )
    return {
        "match": old == current,
        "old_ids": sorted(old),
        "current_ids": sorted(current),
        "missing_current_ids": sorted(set(old) - set(current)),
        "unexpected_current_ids": sorted(set(current) - set(old)),
        "differing_ids": differing,
    }


def _code_change_audit(
    *,
    old_manifest: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
    known_changes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    old = _manifest_index(old_manifest)
    current = _manifest_index(current_manifest)
    changed = sorted(
        path for path in set(old) | set(current) if old.get(path) != current.get(path)
    )
    findings = []
    unknown = []
    for path in changed:
        finding = known_changes.get(path)
        known_pair = bool(
            finding is not None
            and old.get(path) == finding.get("expected_old_sha256")
            and current.get(path) == finding.get("expected_current_sha256")
        )
        if not known_pair:
            unknown.append(path)
            findings.append(
                {
                    "path": path,
                    "classification": "unreviewed",
                    "old_sha256": old.get(path),
                    "current_sha256": current.get(path),
                    "finding": (
                        "The measurement effect of this exact old/current source "
                        "pair is not audited."
                    ),
                    "predictor_reuse_compatible": False,
                }
            )
        else:
            findings.append(
                {
                    "path": path,
                    "old_sha256": old.get(path),
                    "current_sha256": current.get(path),
                    **dict(finding),
                }
            )
    return {
        "exact_code_match": str(old_manifest.get("sha256"))
        == str(current_manifest.get("sha256")),
        "old_sha256": old_manifest.get("sha256"),
        "current_sha256": current_manifest.get("sha256"),
        "changed_paths": changed,
        "unknown_changed_paths": unknown,
        "findings": findings,
        "all_changes_reviewed_for_predictor_reuse": bool(
            not unknown
            and all(bool(row["predictor_reuse_compatible"]) for row in findings)
        ),
    }


def build_compatibility_audit(
    *,
    node: str = NODE,
    empty_gate_path: Path = DEFAULT_EMPTY_GATE,
    loaded_gate_path: Path = DEFAULT_LOADED_GATE,
) -> dict[str, Any]:
    """Audit old training inputs against the current aligned measurement bundle."""

    empty_gate = _read_json(empty_gate_path)
    loaded_gate = _read_json(loaded_gate_path)
    empty_contract = _selected_empty_contract(node)
    loaded_contract = _selected_loaded_contract(node)
    old_empty = [
        row
        for row in empty_gate.get("expected_cells") or []
        if (str(row.get("workload_key")), int(row.get("profile") or 0))
        in set(EMPTY_SELECTION)
    ]
    old_loaded = [
        row
        for row in loaded_gate.get("expected_scenarios") or []
        if str(row.get("scenario_id")) in set(LOADED_SELECTION)
    ]
    empty_contract_audit = _compare_contracts(
        old_rows=old_empty,
        current_rows=empty_contract,
        key="service_cell_id",
    )
    loaded_contract_audit = _compare_contracts(
        old_rows=old_loaded,
        current_rows=loaded_contract,
        key="scenario_id",
    )
    empty_code = _code_change_audit(
        old_manifest=_source_manifest(empty_gate),
        current_manifest=empty_suite.measurement_code_manifest(),
        known_changes=EMPTY_KNOWN_CHANGES,
    )
    loaded_code = _code_change_audit(
        old_manifest=_source_manifest(loaded_gate),
        current_manifest=loaded_suite.measurement_code_manifest(),
        known_changes=LOADED_KNOWN_CHANGES,
    )
    predictor_reuse_ready = bool(
        empty_contract_audit["match"]
        and loaded_contract_audit["match"]
        and empty_code["all_changes_reviewed_for_predictor_reuse"]
        and loaded_code["all_changes_reviewed_for_predictor_reuse"]
    )
    return {
        "status": (
            "COMPATIBLE_FOR_FROZEN_PREDICTOR_ONLY"
            if predictor_reuse_ready
            else "INCOMPATIBLE_PENDING_SOURCE_AUDIT"
        ),
        "training_predictor_reuse_ready": predictor_reuse_ready,
        "old_calibration_or_holdout_units_eligible": 0,
        "post_hoc_august_wave_pairing_forbidden": True,
        "basis": (
            "The August measurements are used only to freeze the deterministic "
            "empty and loaded point predictors. Split-conformal validity is "
            "provided by nineteen fresh exchangeable joint calibration visits "
            "and one independent holdout measured with one frozen current bundle."
        ),
        "empty_contract": empty_contract_audit,
        "loaded_contract": loaded_contract_audit,
        "empty_code": empty_code,
        "loaded_code": loaded_code,
    }


def _joint_role(wave: int) -> str:
    if int(wave) in CALIBRATION_WAVES:
        return "calibration"
    if int(wave) == HOLDOUT_WAVE:
        return "holdout"
    raise ValueError(f"unregistered joint wave {wave}")


def _work_items(
    *,
    node: str,
    empty_contract: Sequence[Mapping[str, Any]],
    loaded_contract: Sequence[Mapping[str, Any]],
    empty_code_sha256: str,
    loaded_code_sha256: str,
) -> list[dict[str, Any]]:
    empty_ids = sorted(str(row["service_cell_id"]) for row in empty_contract)
    loaded_ids = sorted(str(row["scenario_id"]) for row in loaded_contract)
    scored_ids = [
        *(f"empty:{identity}:drain_completion" for identity in empty_ids),
        *(f"loaded:{identity}:target_completion_time" for identity in loaded_ids),
    ]
    items = []
    for order, wave in enumerate(JOINT_WAVES, start=1):
        role = _joint_role(wave)
        items.append(
            {
                "visit_order": order,
                "joint_wave": wave,
                "joint_unit_id": f"{COLLECTION_PREFIX}_{node}_r{wave:02d}",
                "split_role": role,
                "node": node,
                "suite_order": ["empty", "loaded"],
                "observation_count": len(empty_ids) + len(loaded_ids),
                "scored_functional_ids": scored_ids,
                "empty_suite": {
                    "measurement_protocol": empty_suite.PROTOCOL,
                    "measurement_code_sha256": empty_code_sha256,
                    "service_cell_ids": empty_ids,
                    "seed_base": 830_000 + wave * 10_000,
                    "run_sequentially": True,
                    "empty_preflight_required_before_each_cell": True,
                },
                "loaded_suite": {
                    "measurement_protocol": loaded_suite.PROTOCOL,
                    "measurement_code_sha256": loaded_code_sha256,
                    "scenario_ids": loaded_ids,
                    "seed_base": 920_000 + wave * 100,
                    "run_sequentially": True,
                    "full_resident_target_overlap_required": True,
                },
                "completion_rule": (
                    "Admit exactly one joint unit only after all seven empty cells "
                    "and both loaded scenarios pass under this joint identity. "
                    "Retries retain the same identity and never add a unit."
                ),
                "release_condition": (
                    "nineteen calibration units complete and the margin is frozen"
                    if role == "holdout"
                    else "registered manifest and node-idle preflight"
                ),
                "holdout_lock": {
                    "locked_until_release": role == "holdout",
                    "required_complete_calibration_units": (
                        len(CALIBRATION_WAVES) if role == "holdout" else 0
                    ),
                    "calibration_margin_must_be_frozen": role == "holdout",
                },
            }
        )
    return items


def _scored_functionals(
    *,
    empty_contract: Sequence[Mapping[str, Any]],
    loaded_contract: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        *(
            {
                "functional_id": (
                    f"empty:{row['service_cell_id']}:drain_completion"
                ),
                "source": "empty",
                "identity": row["service_cell_id"],
                "functional": "drain_completion",
                "observation_field": "drain_completion_s",
                "frozen_predictor": "frozen_training_phase_model_drain_prediction_s",
                "direction": "upper_time",
            }
            for row in empty_contract
        ),
        *(
            {
                "functional_id": (
                    f"loaded:{row['scenario_id']}:target_completion_time"
                ),
                "source": "loaded",
                "identity": row["scenario_id"],
                "functional": "target_completion_time",
                "observation_field": "target_canonical_completion_s",
                "frozen_predictor": "frozen_training_phase_model_target_prediction_s",
                "direction": "upper_time",
            }
            for row in loaded_contract
        ),
    ]


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    coverage = manifest.get("coverage_contract") or {}
    items = list(manifest.get("joint_work_items") or [])
    empty_contract = list(manifest.get("empty_contract") or [])
    loaded_contract = list(manifest.get("loaded_contract") or [])
    scored_functionals = list(manifest.get("scored_functionals") or [])
    require(manifest.get("protocol") == PROTOCOL, "protocol mismatch")
    require(manifest.get("node") == NODE, "node mismatch")
    require(manifest.get("launch_capability") is False, "manifest runner cannot launch")
    require(
        manifest.get("network_access_performed") is False,
        "registration cannot include network activity",
    )
    require(
        manifest.get("measurements_executed") is False,
        "registration cannot claim executed measurements",
    )
    require(float(coverage.get("miscoverage_alpha") or 0.0) == 0.05, "alpha must be 0.05")
    require(int(coverage.get("calibration_unit_count") or 0) == 19, "need 19 calibration units")
    require(int(coverage.get("holdout_unit_count") or 0) == 1, "need one holdout unit")
    require(int(coverage.get("finite_sample_rank") or 0) == 19, "finite-sample rank must be 19")
    require(len(empty_contract) == 7, "empty contract must contain seven cells")
    require(len(loaded_contract) == 2, "loaded contract must contain two scenarios")
    require(len(scored_functionals) == 9, "joint score must contain nine functionals")
    require(
        empty_contract == _selected_empty_contract(NODE),
        "empty measurement contract differs from the current aligned suite",
    )
    require(
        loaded_contract == _selected_loaded_contract(NODE),
        "loaded measurement contract differs from the current aligned suite",
    )
    require(len(items) == 20, "manifest must contain twenty joint work items")
    require([int(row.get("joint_wave") or -1) for row in items] == list(JOINT_WAVES), "joint waves must be 14..33")
    require(
        [row.get("split_role") for row in items]
        == ["calibration"] * 19 + ["holdout"],
        "split roles must be nineteen calibration then one holdout",
    )
    ids = [str(row.get("joint_unit_id") or "") for row in items]
    require(len(set(ids)) == len(ids) and all(ids), "joint identities must be unique")
    expected_empty = sorted(str(row.get("service_cell_id")) for row in empty_contract)
    expected_loaded = sorted(str(row.get("scenario_id")) for row in loaded_contract)
    expected_scored = sorted(
        [
            *(f"empty:{identity}:drain_completion" for identity in expected_empty),
            *(f"loaded:{identity}:target_completion_time" for identity in expected_loaded),
        ]
    )
    require(
        sorted(str(row.get("functional_id") or "") for row in scored_functionals)
        == expected_scored,
        "scored functional roster differs from the aligned nine-action envelope",
    )
    require(
        all(row.get("direction") == "upper_time" for row in scored_functionals),
        "every registered functional must be an upper-time functional",
    )
    require(
        scored_functionals
        == _scored_functionals(
            empty_contract=empty_contract,
            loaded_contract=loaded_contract,
        ),
        "scored functional definitions differ",
    )
    require(manifest.get("score_contract") == SCORE_CONTRACT, "score contract differs")
    bundles = manifest.get("measurement_bundles") or {}
    empty_code_sha256 = str(
        ((bundles.get("empty") or {}).get("manifest") or {}).get("sha256") or ""
    )
    loaded_code_sha256 = str(
        ((bundles.get("loaded") or {}).get("manifest") or {}).get("sha256") or ""
    )
    require(bool(empty_code_sha256), "empty measurement bundle identity is missing")
    require(bool(loaded_code_sha256), "loaded measurement bundle identity is missing")
    require(
        (bundles.get("empty") or {}).get("protocol") == empty_suite.PROTOCOL
        and (bundles.get("empty") or {}).get("manifest")
        == empty_suite.measurement_code_manifest(),
        "empty measurement bundle differs from current registered sources",
    )
    require(
        (bundles.get("loaded") or {}).get("protocol") == loaded_suite.PROTOCOL
        and (bundles.get("loaded") or {}).get("manifest")
        == loaded_suite.measurement_code_manifest(),
        "loaded measurement bundle differs from current registered sources",
    )
    if len(empty_contract) == 7 and len(loaded_contract) == 2:
        expected_items = _work_items(
            node=NODE,
            empty_contract=empty_contract,
            loaded_contract=loaded_contract,
            empty_code_sha256=empty_code_sha256,
            loaded_code_sha256=loaded_code_sha256,
        )
        require(
            items == expected_items,
            "joint work items differ from the registered identities, seeds, bundles, order, or release rules",
        )
    for row in items:
        require(int(row.get("observation_count") or 0) == 9, "every visit must contain nine observations")
        require(
            row.get("suite_order") == ["empty", "loaded"],
            f"wave {row.get('joint_wave')} suite order differs",
        )
        require(
            sorted((row.get("empty_suite") or {}).get("service_cell_ids") or [])
            == expected_empty,
            f"wave {row.get('joint_wave')} empty cells differ",
        )
        require(
            sorted((row.get("loaded_suite") or {}).get("scenario_ids") or [])
            == expected_loaded,
            f"wave {row.get('joint_wave')} loaded scenarios differ",
        )
        require(
            sorted(row.get("scored_functional_ids") or []) == expected_scored,
            f"wave {row.get('joint_wave')} scored functionals differ",
        )
    if items:
        holdout_lock = items[-1].get("holdout_lock") or {}
        require(
            holdout_lock.get("locked_until_release") is True
            and int(holdout_lock.get("required_complete_calibration_units") or 0)
            == 19
            and holdout_lock.get("calibration_margin_must_be_frozen") is True,
            "holdout must remain locked until 19 calibration units and margin freeze",
        )
    require(sum(int(row.get("observation_count") or 0) for row in items) == 180, "total observations must be 180")
    history = manifest.get("historical_training_inputs") or {}
    require(
        int(history.get("eligible_joint_calibration_units", -1)) == 0,
        "August waves cannot be joint calibration units",
    )
    require(bool(history.get("post_hoc_pairing_forbidden")), "post-hoc pairing must be forbidden")
    compatibility = manifest.get("compatibility_audit") or {}
    require(
        bool(compatibility.get("training_predictor_reuse_ready")),
        "historical predictor compatibility audit is not ready",
    )
    require(
        int(compatibility.get("old_calibration_or_holdout_units_eligible", -1)) == 0,
        "compatibility audit must exclude every historical calibration/holdout unit",
    )
    require(not (manifest.get("observations") or []), "registration must not contain fabricated observations")
    require(
        int(manifest.get("completed_joint_units", -1)) == 0,
        "registration cannot contain completed joint units",
    )
    return {
        "pass": not errors,
        "status": "PASS" if not errors else "FAIL_VALIDATION",
        "error_count": len(errors),
        "errors": errors,
        "joint_unit_count": len(items),
        "observation_count": sum(int(row.get("observation_count") or 0) for row in items),
    }


def local_readiness_snapshot(
    *,
    node: str = NODE,
    node_status_path: Path | None = None,
    queue_status_path: Path | None = None,
) -> dict[str, Any]:
    """Read local JSON snapshots only; no status command or remote probe is run."""

    blockers: list[dict[str, Any]] = []
    node_snapshot: dict[str, Any] | None = None
    status_ts: Any = None
    if node_status_path is None:
        blockers.append({"code": "LOCAL_NODE_STATUS_NOT_SUPPLIED"})
    else:
        payload = _read_json(node_status_path)
        status_ts = payload.get("ts")
        states = payload.get("states") or payload
        raw = states.get(node) if isinstance(states, Mapping) else None
        if not isinstance(raw, Mapping):
            blockers.append({"code": "TARGET_NODE_MISSING_FROM_LOCAL_STATUS", "node": node})
        else:
            node_snapshot = dict(raw)
            required_gpus = set(int(value) for value in empty_suite.NODE_SPECS[node].gpus)
            gpu_rows = {
                int(row.get("idx", row.get("index", -1))): dict(row)
                for row in raw.get("gpus") or []
                if int(row.get("idx", row.get("index", -1))) >= 0
            }
            selected = [gpu_rows[index] for index in sorted(required_gpus) if index in gpu_rows]
            idle = bool(
                raw.get("alive") is True
                and set(gpu_rows) >= required_gpus
                and all(
                    float(row.get("used_mb") or 0.0) <= empty_suite.GPU_IDLE_MEMORY_LIMIT_MB
                    and float(row.get("util_pct") or 0.0) <= empty_suite.GPU_IDLE_UTIL_LIMIT_PCT
                    for row in selected
                )
            )
            if not idle:
                blockers.append(
                    {
                        "code": "TARGET_GPUS_NOT_EMPTY",
                        "node": node,
                        "idle_memory_limit_mb": empty_suite.GPU_IDLE_MEMORY_LIMIT_MB,
                        "idle_util_limit_pct": empty_suite.GPU_IDLE_UTIL_LIMIT_PCT,
                        "selected_gpus": selected,
                    }
                )

    active_tasks = []
    if queue_status_path is not None:
        queue = _read_json(queue_status_path)
        tasks: Any = queue.get("tasks") or []
        if isinstance(tasks, Mapping):
            tasks = list(tasks.values())
        for task in tasks if isinstance(tasks, list) else []:
            if not isinstance(task, Mapping):
                continue
            state = str(task.get("status") or task.get("state") or "").lower()
            assigned_node = str(task.get("node") or task.get("assigned_node") or "")
            if state not in {"running", "launching"} or assigned_node != node:
                continue
            active_tasks.append(
                {
                    "id": task.get("id") or task.get("task_id"),
                    "status": state,
                    "project": task.get("project"),
                    "gpu_idx": task.get("gpu_idx", task.get("assigned_gpu")),
                    "description": task.get("description"),
                }
            )
        if active_tasks:
            blockers.append(
                {
                    "code": "ACTIVE_SCHEDULER_TASKS_ON_TARGET_NODE",
                    "count": len(active_tasks),
                    "tasks": active_tasks,
                }
            )
    return {
        "status": "READY" if not blockers else "WAIT",
        "ready": not blockers,
        "local_only": True,
        "remote_probe_run": False,
        "node_status_path": str(node_status_path) if node_status_path else None,
        "node_status_ts": status_ts,
        "queue_status_path": str(queue_status_path) if queue_status_path else None,
        "node_snapshot": node_snapshot,
        "active_task_count": len(active_tasks),
        "active_tasks": active_tasks,
        "blockers": blockers,
    }


def build_collection_manifest(
    *,
    node: str = NODE,
    empty_gate_path: Path = DEFAULT_EMPTY_GATE,
    loaded_gate_path: Path = DEFAULT_LOADED_GATE,
    node_status_path: Path | None = None,
    queue_status_path: Path | None = None,
) -> dict[str, Any]:
    if node != NODE:
        raise ValueError(f"this registered protocol is hardware-local to {NODE}")
    empty_gate = _read_json(empty_gate_path)
    loaded_gate = _read_json(loaded_gate_path)
    empty_contract = _selected_empty_contract(node)
    loaded_contract = _selected_loaded_contract(node)
    empty_manifest = empty_suite.measurement_code_manifest()
    loaded_manifest = loaded_suite.measurement_code_manifest()
    compatibility = build_compatibility_audit(
        node=node,
        empty_gate_path=empty_gate_path,
        loaded_gate_path=loaded_gate_path,
    )
    rank = math.ceil((len(CALIBRATION_WAVES) + 1) * (1.0 - MIS_COVERAGE_ALPHA))
    scored_functionals = _scored_functionals(
        empty_contract=empty_contract,
        loaded_contract=loaded_contract,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "gate": "critical_gpu_joint95_collection_runner",
        "status": "READY_TO_REGISTER",
        "registered": False,
        "protocol": PROTOCOL,
        "collection_prefix": COLLECTION_PREFIX,
        "node": node,
        "node_bucket": empty_suite.NODE_SPECS[node].node_bucket,
        "hardware_class": empty_suite.NODE_SPECS[node].hardware_class,
        "launch_capability": False,
        "network_access_performed": False,
        "measurements_executed": False,
        "coverage_contract": {
            "method": "paired_joint_wave_max_split_conformal",
            "miscoverage_alpha": MIS_COVERAGE_ALPHA,
            "nominal_simultaneous_coverage": 1.0 - MIS_COVERAGE_ALPHA,
            "calibration_unit_count": len(CALIBRATION_WAVES),
            "holdout_unit_count": 1,
            "finite_sample_rank_rule": "ceil((n_calibration + 1) * (1 - alpha))",
            "finite_sample_rank": rank,
            "calibration_joint_waves": list(CALIBRATION_WAVES),
            "holdout_joint_wave": HOLDOUT_WAVE,
        },
        "historical_training_inputs": {
            "empty_gate": str(empty_gate_path),
            "loaded_gate": str(loaded_gate_path),
            "empty_training_waves": list(empty_gate.get("training_waves") or []),
            "loaded_training_waves": list(loaded_gate.get("training_waves") or []),
            "role": "frozen_point_predictors_only",
            "eligible_joint_calibration_units": 0,
            "post_hoc_pairing_forbidden": True,
        },
        "measurement_bundles": {
            "empty": {
                "protocol": empty_suite.PROTOCOL,
                "manifest": empty_manifest,
            },
            "loaded": {
                "protocol": loaded_suite.PROTOCOL,
                "manifest": loaded_manifest,
            },
            "freeze_rule": (
                "Every calibration and holdout visit must use these same two "
                "registered current manifests. A code change invalidates the "
                "remaining split; it does not admit mixed versions."
            ),
        },
        "compatibility_audit": compatibility,
        "empty_contract": empty_contract,
        "loaded_contract": loaded_contract,
        "scored_functionals": scored_functionals,
        "score_contract": dict(SCORE_CONTRACT),
        "joint_work_items": _work_items(
            node=node,
            empty_contract=empty_contract,
            loaded_contract=loaded_contract,
            empty_code_sha256=str(empty_manifest["sha256"]),
            loaded_code_sha256=str(loaded_manifest["sha256"]),
        ),
        "counting_rule": (
            "One visit contributes one joint unit only when all seven aligned "
            "empty functionals and both aligned loaded scenarios pass. Failed "
            "attempts and retries retain that visit identity and add no unit."
        ),
        "observations": [],
        "completed_joint_units": 0,
        "readiness": local_readiness_snapshot(
            node=node,
            node_status_path=node_status_path,
            queue_status_path=queue_status_path,
        ),
        "claim_boundary": (
            "This manifest registers collection work only. It contains no new "
            "measurement, conformal margin, coverage result, capacity certificate, "
            "or paper claim."
        ),
    }
    validation = validate_manifest(report)
    report["validation"] = validation
    if not validation["pass"] or not compatibility["training_predictor_reuse_ready"]:
        report["status"] = "FAIL_PROTOCOL_VALIDATION"
    return report


def write_registration(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    registered = dict(report)
    validation = validate_manifest(registered)
    if not validation["pass"]:
        raise ValueError(f"refusing invalid registration: {validation['errors']!r}")
    registered["registered"] = True
    registered["status"] = (
        "REGISTERED_READY_FOR_SEPARATE_LAUNCH"
        if (registered.get("readiness") or {}).get("ready") is True
        else "REGISTERED_WAIT_READINESS"
    )
    registered["validation"] = validate_manifest(registered)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(registered, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return registered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register or validate the local-only joint95 collection manifest."
    )
    parser.add_argument("--node", default=NODE, choices=(NODE,))
    parser.add_argument("--empty-gate", type=Path, default=DEFAULT_EMPTY_GATE)
    parser.add_argument("--loaded-gate", type=Path, default=DEFAULT_LOADED_GATE)
    parser.add_argument("--node-status", type=Path)
    parser.add_argument("--queue-status", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.validate is not None:
        report = _read_json(args.validate)
        validation = validate_manifest(report)
        readiness = local_readiness_snapshot(
            node=args.node,
            node_status_path=args.node_status,
            queue_status_path=args.queue_status,
        )
        print(
            json.dumps(
                {"validation": validation, "readiness": readiness},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if validation["pass"] else 2
    report = build_collection_manifest(
        node=args.node,
        empty_gate_path=args.empty_gate,
        loaded_gate_path=args.loaded_gate,
        node_status_path=args.node_status,
        queue_status_path=args.queue_status,
    )
    if args.output is not None:
        report = write_registration(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "registered": report["registered"],
                "validation": report["validation"],
                "readiness": report["readiness"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["validation"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
