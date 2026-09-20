"""Audit requirement-dominance prevalence across local catalogue snapshots.

The screen reads the complete-frame catalogues already stored in local
artifacts.  It separates repeated snapshots and duration variants from a
genuinely different action catalogue.  No selector, trajectory, random draw,
or server input is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from algorithm.canonical_count_space_bnb_requirement_dominance_frame_candidates import (
    prepare_canonical_count_space_bnb_requirement_dominance_catalogue,
)
from algorithm.requirement_dominance_frame_candidates import (
    reduce_requirement_dominated_options,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_requirement_dominance_prevalence_screen_"
    "development_20260920.json"
)
EXPERIMENT = "aligned_hardware_requirement_dominance_prevalence_screen_v1"


@dataclass(frozen=True)
class CatalogueSource:
    snapshot_id: str
    source_role: str
    path: str
    model_path: tuple[str, ...]
    representation: str = "frame_model"


SOURCES = (
    CatalogueSource(
        "frame_v1",
        "frame_experiment_model",
        "aligned_hardware_frame_experiment_20260909.json",
        ("model",),
    ),
    CatalogueSource(
        "frame_v2",
        "frame_experiment_model",
        "aligned_hardware_frame_experiment_v2_20260909.json",
        ("model",),
    ),
    CatalogueSource(
        "frame_v3",
        "frame_experiment_model",
        "aligned_hardware_frame_experiment_v3_20260909.json",
        ("model",),
    ),
    CatalogueSource(
        "frame_v4",
        "frame_experiment_model",
        "aligned_hardware_frame_experiment_v4_20260909.json",
        ("model",),
    ),
    CatalogueSource(
        "joint95_v1_point",
        "capacity_protocol_point_reference",
        "aligned_hardware_joint95_capacity_closure_20260913_protocol.json",
        ("point_reference_model",),
    ),
    CatalogueSource(
        "joint95_v2_point",
        "capacity_protocol_point_reference",
        "aligned_hardware_joint95_v2_capacity_closure_20260913_protocol.json",
        ("point_reference_model",),
    ),
    CatalogueSource(
        "joint95_v3_point",
        "capacity_protocol_point_reference",
        "aligned_hardware_joint95_v3_capacity_closure_20260913_protocol.json",
        ("point_reference_model",),
    ),
    CatalogueSource(
        "joint95_v4_point",
        "capacity_protocol_point_reference",
        "aligned_hardware_joint95_v4_capacity_closure_20260913_protocol.json",
        ("point_reference_model",),
    ),
    CatalogueSource(
        "joint95_v5_point",
        "capacity_protocol_point_reference",
        "aligned_hardware_joint95_v5_capacity_closure_20260919_protocol.json",
        ("point_reference_model",),
    ),
    CatalogueSource(
        "count_space_scaling_point",
        "scaling_screen_point_reference",
        "aligned_hardware_count_space_bnb_scaling_screen_development_20260913.json",
        ("protocol", "model"),
    ),
    CatalogueSource(
        "lcb_shadow",
        "lcb_complete_frame_shadow_model",
        "aligned_hardware_lcb_frame_envelope_shadow_development_20260912.json",
        ("protocol", "shadow_model"),
    ),
    CatalogueSource(
        "joint95_v1_zero_margin",
        "capacity_protocol_zero_margin_envelopes",
        "aligned_hardware_joint95_capacity_closure_20260913_protocol.json",
        ("zero_margin_action_envelopes",),
        "zero_margin_action_envelopes",
    ),
    CatalogueSource(
        "joint95_v2_zero_margin",
        "capacity_protocol_zero_margin_envelopes",
        "aligned_hardware_joint95_v2_capacity_closure_20260913_protocol.json",
        ("zero_margin_action_envelopes",),
        "zero_margin_action_envelopes",
    ),
    CatalogueSource(
        "joint95_v3_zero_margin",
        "capacity_protocol_zero_margin_envelopes",
        "aligned_hardware_joint95_v3_capacity_closure_20260913_protocol.json",
        ("zero_margin_action_envelopes",),
        "zero_margin_action_envelopes",
    ),
    CatalogueSource(
        "joint95_v4_zero_margin",
        "capacity_protocol_zero_margin_envelopes",
        "aligned_hardware_joint95_v4_capacity_closure_20260913_protocol.json",
        ("zero_margin_action_envelopes",),
        "zero_margin_action_envelopes",
    ),
    CatalogueSource(
        "joint95_v5_zero_margin",
        "capacity_protocol_zero_margin_envelopes",
        "aligned_hardware_joint95_v5_capacity_closure_20260919_protocol.json",
        ("zero_margin_action_envelopes",),
        "zero_margin_action_envelopes",
    ),
    CatalogueSource(
        "joint95_v5_physical_envelope",
        "capacity_result_physical_envelope",
        "aligned_hardware_joint95_v5_capacity_closure_20260919.json",
        ("physical_complete_frame_envelope", "model"),
    ),
)


DECISION_RULE = {
    "INVALID_EXECUTION": "one or more read-only inventory checks fail",
    "MULTI_CATALOGUE_SCALING_ADVANCE": (
        "at least two distinct ordered action catalogue topologies each have "
        "a nonempty quotient with a valid full-catalogue canonical lift"
    ),
    "SINGLE_CATALOGUE_TOPOLOGY_STOP": (
        "the frozen roster contains fewer than two distinct ordered action "
        "catalogue topologies"
    ),
    "CROSS_CATALOGUE_DOMINANCE_NOT_OBSERVED_STOP": (
        "multiple catalogue topologies exist but fewer than two have a "
        "nonempty canonically liftable quotient"
    ),
}

CLAIM_BOUNDARY = (
    "A deterministic development inventory of local complete-frame catalogue "
    "snapshots. Equality of ordered action_id and requirement vectors defines "
    "one catalogue topology; exact duration equality defines variants within "
    "that topology. Repeated lineage snapshots do not count as independent "
    "catalogues. The screen does not establish selector scaling, wall time, "
    "trajectory performance, external-system prevalence, or a paper claim."
)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _protocol_path(output: Path) -> Path:
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(registered_at_utc: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "role": "outcome-neutral local catalogue prevalence development screen",
        "registered": True,
        "registered_at_utc": registered_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "catalogue_roster": [
            {
                **asdict(source),
                "model_path": list(source.model_path),
            }
            for source in SOURCES
        ],
        "catalogue_contract": _catalogue_contract(),
        "catalogue_topology_equivalence": (
            "exact equality of node, scenario_id, lanes, ordered workloads, and "
            "ordered (action_id, requirements); duration and model_scope are excluded"
        ),
        "duration_variant_equivalence": (
            "exact equality of topology fields plus ordered "
            "(action_id, requirements, duration)"
        ),
        "dominance_rule": (
            "same exact requirement vector and strictly smaller "
            "max(1, duration)"
        ),
        "advance_gate": DECISION_RULE["MULTI_CATALOGUE_SCALING_ADVANCE"],
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": (
            "server pulls, RNG, arrivals, trajectories, selector calls, "
            "wall-time claims, paper edits, and dropping roster members after "
            "observing their quotient"
        ),
    }


def _read_model(source: CatalogueSource) -> dict[str, Any]:
    root_payload: Any = json.loads((ARTIFACTS / source.path).read_text())
    payload = root_payload
    for key in source.model_path:
        if not isinstance(payload, dict) or key not in payload:
            raise ValueError(
                f"{source.snapshot_id} lacks model path {source.model_path}"
            )
        payload = payload[key]
    if source.representation == "frame_model":
        if not isinstance(payload, dict):
            raise ValueError(f"{source.snapshot_id} model is not an object")
        metadata = payload
        raw_options = payload.get("options")
        duration_field = "duration"
    elif source.representation == "zero_margin_action_envelopes":
        if not isinstance(root_payload, dict):
            raise ValueError(f"{source.snapshot_id} protocol is not an object")
        metadata = root_payload.get("point_reference_model")
        if not isinstance(metadata, dict) or not isinstance(payload, list):
            raise ValueError(
                f"{source.snapshot_id} lacks zero-margin envelope metadata"
            )
        raw_options = payload
        duration_field = "upper_complete_frame_duration_s"
    else:
        raise ValueError(
            f"{source.snapshot_id} has unsupported representation "
            f"{source.representation}"
        )
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError(f"{source.snapshot_id} has no nonempty option catalogue")

    options: list[dict[str, Any]] = []
    dimension: int | None = None
    seen_action_ids: set[str] = set()
    for index, raw in enumerate(raw_options):
        if not isinstance(raw, dict):
            raise ValueError(f"{source.snapshot_id} option {index} is not an object")
        action_id = raw.get("action_id")
        requirements = raw.get("requirements")
        duration = raw.get(duration_field)
        if not isinstance(action_id, str) or not action_id:
            raise ValueError(f"{source.snapshot_id} option {index} lacks action_id")
        if action_id in seen_action_ids:
            raise ValueError(f"{source.snapshot_id} repeats action_id {action_id}")
        seen_action_ids.add(action_id)
        if not isinstance(requirements, list):
            raise ValueError(f"{source.snapshot_id} option {index} lacks requirements")
        if dimension is None:
            dimension = len(requirements)
        if len(requirements) != dimension or not requirements:
            raise ValueError(f"{source.snapshot_id} has inconsistent requirements")
        if any(int(value) != value or value < 0 for value in requirements):
            raise ValueError(f"{source.snapshot_id} has invalid requirements")
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"{source.snapshot_id} has invalid duration")
        options.append(
            {
                "index": index,
                "action_id": action_id,
                "requirements": [int(value) for value in requirements],
                "duration": duration,
            }
        )
    return {
        "node": metadata.get("node"),
        "scenario_id": metadata.get("scenario_id"),
        "lanes": metadata.get("lanes"),
        "workloads": list(metadata.get("workloads") or ()),
        "model_scope": (
            metadata.get("model_scope")
            if source.representation == "frame_model"
            else "registered_zero_margin_complete_frame_upper_envelope"
        ),
        "options": options,
    }


def _catalogue_contract() -> list[dict[str, Any]]:
    """Freeze the normalized selector-relevant contents without a digest."""

    return [
        {
            "snapshot_id": source.snapshot_id,
            "model": _read_model(source),
        }
        for source in SOURCES
    ]


def _topology_signature(model: dict[str, Any]) -> tuple[Any, ...]:
    return (
        model["node"],
        model["scenario_id"],
        model["lanes"],
        tuple(model["workloads"]),
        tuple(
            (option["action_id"], tuple(option["requirements"]))
            for option in model["options"]
        ),
    )


def _duration_signature(model: dict[str, Any]) -> tuple[Any, ...]:
    return (
        model["node"],
        model["scenario_id"],
        model["lanes"],
        tuple(model["workloads"]),
        tuple(
            (
                option["action_id"],
                tuple(option["requirements"]),
                option["duration"],
            )
            for option in model["options"]
        ),
    )


def _equal_group_index(groups: Sequence[tuple[Any, ...]], signature: tuple[Any, ...]) -> int:
    for index, existing in enumerate(groups):
        if signature == existing:
            return index
    return len(groups)


def _analyse_duration_variant(
    group_id: str,
    signature: tuple[Any, ...],
) -> dict[str, Any]:
    node, scenario_id, lanes, workloads, option_signature = signature
    options = tuple(
        SimpleNamespace(action_id=action_id, requirements=requirements, duration=duration)
        for action_id, requirements, duration in option_signature
    )
    reduction = reduce_requirement_dominated_options(options, idle_duration=1.0)
    strict_pairs = []
    for dominated_index, dominated in enumerate(options):
        for dominator_index, dominator in enumerate(options):
            if (
                dominated_index != dominator_index
                and dominated.requirements == dominator.requirements
                and max(1.0, dominator.duration)
                < max(1.0, dominated.duration)
            ):
                strict_pairs.append(
                    {
                        "dominated_index": dominated_index,
                        "dominated_action_id": dominated.action_id,
                        "dominated_effective_duration": max(1.0, dominated.duration),
                        "dominator_index": dominator_index,
                        "dominator_action_id": dominator.action_id,
                        "dominator_effective_duration": max(1.0, dominator.duration),
                    }
                )

    lift_error = None
    try:
        prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
            options, idle_duration=1.0
        )
        lift_valid = True
        lift_pairs = [
            list(pair)
            for pair in prepared.certificate_lift_proof.removed_to_dominator_original_indices
        ]
        metadata = asdict(prepared.reduction)
    except ValueError as error:
        lift_valid = False
        lift_pairs = []
        lift_error = str(error)
        metadata = {
            "original_option_count": reduction.original_option_count,
            "retained_option_count": reduction.retained_option_count,
            "removed_option_count": reduction.removed_option_count,
            "retained_original_indices": list(reduction.retained_original_indices),
            "removed_original_indices": list(reduction.removed_original_indices),
            "idle_duration": 1.0,
            "reduction_option_probes": reduction.reduction_option_probes,
            "reduction_requirement_coordinate_reads": (
                reduction.reduction_requirement_coordinate_reads
            ),
            "reduction_duration_comparisons": (
                reduction.reduction_duration_comparisons
            ),
        }

    return {
        "duration_variant_group_id": group_id,
        "node": node,
        "scenario_id": scenario_id,
        "lanes": lanes,
        "workloads": list(workloads),
        "option_count": reduction.original_option_count,
        "retained_option_count": reduction.retained_option_count,
        "removed_option_count": reduction.removed_option_count,
        "retained_original_indices": list(reduction.retained_original_indices),
        "removed_original_indices": list(reduction.removed_original_indices),
        "strict_requirement_duration_pair_count": len(strict_pairs),
        "strict_requirement_duration_pairs": strict_pairs,
        "nonempty_quotient": reduction.removed_option_count > 0,
        "canonical_lift_valid": lift_valid,
        "canonical_lift_pairs": lift_pairs,
        "canonical_lift_error": lift_error,
        "preprocessing_work": _json(metadata),
        "catalogue": [
            {
                "index": index,
                "action_id": action_id,
                "requirements": list(requirements),
                "duration": duration,
            }
            for index, (action_id, requirements, duration) in enumerate(
                option_signature
            )
        ],
    }


def run_screen(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("experiment") != EXPERIMENT or not protocol.get("registered"):
        raise ValueError("the prevalence screen requires its registered protocol")
    expected_protocol = build_protocol(str(protocol.get("registered_at_utc")))
    if protocol != expected_protocol:
        raise ValueError(
            "registered protocol constants or normalized catalogue contents changed"
        )

    records = []
    topology_signatures: list[tuple[Any, ...]] = []
    duration_signatures: list[tuple[Any, ...]] = []
    topology_members: list[list[str]] = []
    duration_members: list[list[str]] = []
    topology_duration_groups: list[set[int]] = []

    for source in SOURCES:
        model = _read_model(source)
        topology = _topology_signature(model)
        duration = _duration_signature(model)
        topology_index = _equal_group_index(topology_signatures, topology)
        if topology_index == len(topology_signatures):
            topology_signatures.append(topology)
            topology_members.append([])
            topology_duration_groups.append(set())
        duration_index = _equal_group_index(duration_signatures, duration)
        if duration_index == len(duration_signatures):
            duration_signatures.append(duration)
            duration_members.append([])
        topology_members[topology_index].append(source.snapshot_id)
        duration_members[duration_index].append(source.snapshot_id)
        topology_duration_groups[topology_index].add(duration_index)
        records.append(
            {
                "snapshot_id": source.snapshot_id,
                "source_role": source.source_role,
                "path": source.path,
                "model_path": list(source.model_path),
                "representation": source.representation,
                "node": model["node"],
                "scenario_id": model["scenario_id"],
                "lanes": model["lanes"],
                "workloads": model["workloads"],
                "model_scope": model["model_scope"],
                "option_count": len(model["options"]),
                "catalogue_topology_group_id": f"T{topology_index + 1}",
                "duration_variant_group_id": f"D{duration_index + 1}",
            }
        )

    analyses = [
        _analyse_duration_variant(f"D{index + 1}", signature)
        for index, signature in enumerate(duration_signatures)
    ]
    qualified_duration_indices = {
        index
        for index, analysis in enumerate(analyses)
        if analysis["nonempty_quotient"] and analysis["canonical_lift_valid"]
    }
    topology_groups = []
    qualified_topology_count = 0
    for index, signature in enumerate(topology_signatures):
        node, scenario_id, lanes, workloads, option_signature = signature
        duration_indices = topology_duration_groups[index]
        has_quotient = bool(duration_indices & qualified_duration_indices)
        qualified_topology_count += int(has_quotient)
        topology_groups.append(
            {
                "catalogue_topology_group_id": f"T{index + 1}",
                "snapshot_ids": topology_members[index],
                "duration_variant_group_ids": [
                    f"D{duration_index + 1}"
                    for duration_index in sorted(duration_indices)
                ],
                "node": node,
                "scenario_id": scenario_id,
                "lanes": lanes,
                "workloads": list(workloads),
                "option_count": len(option_signature),
                "ordered_actions": [
                    {
                        "index": option_index,
                        "action_id": action_id,
                        "requirements": list(requirements),
                    }
                    for option_index, (action_id, requirements) in enumerate(
                        option_signature
                    )
                ],
                "has_nonempty_canonically_liftable_quotient": has_quotient,
            }
        )

    duration_groups = [
        {
            **analysis,
            "snapshot_ids": duration_members[index],
        }
        for index, analysis in enumerate(analyses)
    ]
    checks = {
        "registered_roster_loaded_exactly_once": len(records) == len(SOURCES),
        "snapshot_ids_unique": len({row["snapshot_id"] for row in records})
        == len(records),
        "every_snapshot_assigned_to_one_topology": all(
            row["catalogue_topology_group_id"] for row in records
        ),
        "every_snapshot_assigned_to_one_duration_variant": all(
            row["duration_variant_group_id"] for row in records
        ),
        "every_duration_variant_analysed": len(duration_groups)
        == len(duration_signatures),
        "all_roster_outcomes_reported": sum(
            len(group["snapshot_ids"]) for group in duration_groups
        )
        == len(SOURCES),
        "preprocessing_reported_separately": all(
            "preprocessing_work" in group for group in duration_groups
        ),
        "no_selector_or_trajectory_execution": True,
        "no_server_pull": True,
    }
    if not all(checks.values()):
        decision = "INVALID_EXECUTION"
    elif len(topology_groups) < 2:
        decision = "SINGLE_CATALOGUE_TOPOLOGY_STOP"
    elif qualified_topology_count < 2:
        decision = "CROSS_CATALOGUE_DOMINANCE_NOT_OBSERVED_STOP"
    else:
        decision = "MULTI_CATALOGUE_SCALING_ADVANCE"

    advance = decision == "MULTI_CATALOGUE_SCALING_ADVANCE"
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "PASS" if decision != "INVALID_EXECUTION" else "FAIL",
        "scientific_decision": decision,
        "protocol": _json(protocol),
        "checks": checks,
        "snapshot_count": len(records),
        "catalogue_topology_count": len(topology_groups),
        "duration_variant_count": len(duration_groups),
        "qualified_catalogue_topology_count": qualified_topology_count,
        "snapshots": records,
        "catalogue_topology_groups": topology_groups,
        "duration_variant_groups": duration_groups,
        "work_accounting": {
            "server_pulls": 0,
            "rng_draws": 0,
            "selector_calls": 0,
            "trajectory_runs": 0,
            "catalogue_snapshots_read": len(records),
            "duration_variants_reduced": len(duration_groups),
        },
        "development_advance": advance,
        "scaling_launched": False,
        "next_step": (
            "Register the frozen cross-catalogue scaling screen."
            if advance
            else "Do not launch cross-catalogue scaling from repeated versions of "
            "this roster; retain the quotient as a single-action-catalogue "
            "optimization until a genuinely different supported physical "
            "catalogue is available."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_claim_authorized": False,
        "limitations": [
            "The roster contains only the local q01 gpu_b complete-frame lineage.",
            "Duration variants derived from one ordered action catalogue are not independent catalogues.",
            "No scaling or runtime conclusion is produced when the topology gate stops.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    output = args.output
    protocol_path = args.protocol or _protocol_path(output)
    if args.register_only:
        _write_new(protocol_path, build_protocol())
        print(protocol_path)
        return 0
    protocol = json.loads(protocol_path.read_text())
    result = run_screen(protocol)
    _write_new(output, result)
    print(output)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
