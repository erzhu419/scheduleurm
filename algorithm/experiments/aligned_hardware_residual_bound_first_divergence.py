"""Registered first-search-divergence diagnosis for residual-bound regressions.

This post-outcome diagnostic uses only the four support-regressed states retained
by the preceding fixed-state shadow.  It compares the old and residual-capacity
frontiers through their first different paid attempt and independently reruns
the two frozen K26 selectors to reproduce the registered checkpoints.  It does
not generate arrivals, simulate trajectories, or evaluate a hybrid selector.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.residual_capacity_frame_candidates import (
    select_residual_capacity_frame,
)
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.experiments.residual_bound_first_divergence_trace import (
    trace_first_bound_divergence,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_residual_bound_first_divergence_development_20260910.json"
)
SOURCE_RESULT = ARTIFACTS / (
    "aligned_hardware_residual_capacity_bound_shadow_development_20260910.json"
)
SOURCE_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_residual_capacity_bound_shadow_development_20260910_protocol.json"
)
SOURCE_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_residual_capacity_bound_shadow_development_20260910_fixed_sources.json"
)
SOURCE_STATE_ROSTER = ARTIFACTS / (
    "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"
)
TARGET_STATE_INDICES = (424, 425, 437, 438)
BUDGET = 26
TARGET_COUNT = 4
SCORE_TOLERANCE = 1e-12
NEW_SOURCES = (
    "algorithm/experiments/residual_bound_first_divergence_trace.py",
    "algorithm/experiments/aligned_hardware_residual_bound_first_divergence.py",
)


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _source_texts():
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    changed = [
        name for name, text in sources.items()
        if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"residual-bound shadow source changed: {changed}")
    return {
        **sources,
        **{name: (ROOT / name).read_text() for name in NEW_SOURCES},
    }


def _target_projection(report, roster):
    regressed = sorted(
        (
            row for row in report["state_changes"]
            if row["support_outcome"] == "regressed"
        ),
        key=lambda row: row["state_index"],
    )
    if tuple(row["state_index"] for row in regressed) != TARGET_STATE_INDICES:
        raise ValueError("diagnosis requires all four registered support regressions")
    targets = []
    for row in regressed:
        state_index = row["state_index"]
        state = roster["states"][state_index]
        if (
            row["legacy_finite_budget_support_regret"] != 0.0
            or row["variant_finite_budget_support_regret"] <= 0.0
            or not row["selected_physical_signature_changed_from_legacy_k26"]
            or not row["k26_bound_strictly_activated"]
            or not row["trace_matches_independent_k26"]
            or tuple(state["queue"]) not in (
                (3, 1, 4), (3, 1, 5), (3, 2, 4), (3, 2, 5)
            )
        ):
            raise ValueError("registered regression projection is inconsistent")
        targets.append({
            "state_index": state_index,
            "queue": state["queue"],
            "legacy_k26": state["legacy_k26"],
            "residual_k26": row["variant_k26"],
            "support_reference_score": state["support_reference_score"],
            "support_score_delta_over_legacy_k26": row[
                "support_score_delta_over_legacy_k26"
            ],
            "legacy_first_exact_score_attempt": state[
                "legacy_first_exact_score_attempt"
            ],
            "residual_first_exact_score_attempt": row["first_exact_score_attempt"],
            "primary_frame_visits": sum(
                visit[3]
                for visit in state["population_visits"]
                if visit[2] == BUDGET
            ),
        })
    return targets


def _validate_source():
    report = json.loads(SOURCE_RESULT.read_text())
    protocol = json.loads(SOURCE_PROTOCOL.read_text())
    roster = json.loads(SOURCE_STATE_ROSTER.read_text())
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    if (
        report["experiment"] != "residual_capacity_binary_price_bound_shadow_v1"
        or report["status"] != "PASS"
        or not all(report["checks"].values())
        or report["shadow_advance"]
        or report["certificate_only"]
        or report["protocol"] != protocol
        or report["state_evaluation_count"] != roster["state_count"]
        or roster["state_count"] != len(roster["states"])
        or any((ROOT / name).read_text() != text for name, text in sources.items())
    ):
        raise ValueError("first-divergence diagnosis requires the frozen shadow result")
    model = load_model()
    if _json(model.snapshot()) != protocol["model"]:
        raise ValueError("first-divergence diagnosis model changed")
    targets = _target_projection(report, roster)
    return report, protocol, roster, model, targets


def _selection_projection(selection, fields):
    selected = _json(asdict(selection))
    return {field: selected[field] for field in fields}


def _same_projection(actual, expected):
    return all(actual[key] == expected[key] for key in actual)


def _attempted_candidate_signature(model, node, queue):
    if node is None:
        return None
    prefix = node["prefix"]
    option_index = prefix["next_option_index"]
    indices = tuple(prefix["indices"]) + (option_index,)
    service = tuple(prefix["service"])
    if option_index is not None:
        service = tuple(
            old + count
            for old, count in zip(
                service, model.options[option_index].requirements
            )
        )
    if any(count > available for count, available in zip(service, queue)):
        return {"feasible": False, "indices": indices}
    duration = max(
        float(prefix["duration"]),
        1.0 if option_index is None else model.options[option_index].duration,
    )
    completion_offsets = []
    for coordinate in range(len(queue)):
        offsets = []
        for index in indices:
            if index is not None:
                offsets.extend(model.options[index].completion_offsets[coordinate])
        completion_offsets.append(tuple(sorted(float(value) for value in offsets)))
    return {
        "feasible": True,
        "indices": indices,
        "service": service,
        "duration": duration,
        "completion_offsets": tuple(completion_offsets),
    }


def _classification_evidence_valid(trace):
    kind = trace["classification"]
    divergence = trace["divergence"]
    if kind == "heap_reprioritization":
        if (
            divergence is None
            or not divergence["legacy_top_present_in_residual_frontier"]
            or not divergence["residual_top_present_in_legacy_frontier"]
            or not divergence["frontier_membership_equal"]
            or trace["residual_only_prune_count_before_divergence"] != 0
        ):
            return False
        old = divergence["legacy_top"]
        new = divergence["residual_top"]
        old_key_legacy = (-old["old_upper_bound"], old["legacy_serial"])
        new_key_legacy = (-new["old_upper_bound"], new["legacy_serial"])
        old_key_residual = (-old["new_upper_bound"], old["residual_serial"])
        new_key_residual = (-new["new_upper_bound"], new["residual_serial"])
        return (
            old["new_upper_bound"] > divergence["incumbent_score"]
            and new["new_upper_bound"] > divergence["incumbent_score"]
            and old_key_legacy < new_key_legacy
            and new_key_residual < old_key_residual
        )
    if kind == "early_pruning":
        pruned = trace["first_residual_only_prune"]
        return (
            pruned is not None
            and pruned["legacy_retained"]
            and not pruned["residual_retained"]
            and pruned["new_upper_bound"] <= pruned["push_incumbent_score"]
            < pruned["old_upper_bound"]
        )
    return False


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("residual first-divergence artifacts already exist")
    report, source_protocol, _, model, targets = _validate_source()
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "residual_capacity_bound_first_divergence_v1",
        "role": (
            "post-outcome deterministic mechanism diagnosis on all four "
            "source-derived support regressions"
        ),
        "registration_context": (
            "the source shadow outcomes are already observed; registration freezes "
            "the diagnostic execution and does not make it confirmatory"
        ),
        "source_result": SOURCE_RESULT.name,
        "reused_registered_state_roster": SOURCE_STATE_ROSTER.name,
        "target_selection": (
            "all and only source state_changes with support_outcome=regressed; "
            "must equal state indices 424,425,437,438"
        ),
        "target_count": TARGET_COUNT,
        "targets": targets,
        "budget": BUDGET,
        "model": model.snapshot(),
        "source_rule": source_protocol["rule"],
        "first_divergence_definition": (
            "after the ten paid seeds, the first paid attempt whose popped logical "
            "prefix or ordered choice differs; a bound-value change alone is not a "
            "search divergence"
        ),
        "logical_prefix": (
            "indices, service, numerator, duration and next_index; lane-order "
            "permutations remain distinct search prefixes"
        ),
        "classification": {
            "early_pruning": (
                "the legacy top is absent from the residual frontier because its "
                "new bound was at most the common incumbent while its old bound "
                "was larger"
            ),
            "heap_reprioritization": (
                "both competing tops are live in both pre-pop frontiers, but the "
                "old and new heap keys select different prefixes"
            ),
        },
        "validation": (
            "independently rerun the frozen legacy and residual K26 selectors and "
            "match their registered source checkpoints exactly"
        ),
        "completion_rule": (
            "status PASS means all four targets were classified with valid frontier "
            "evidence and both K26 checkpoints reproduced; it is not an algorithm "
            "advancement decision"
        ),
        "stopping_rule": (
            "run the fixed paired K26 diagnosis once on the four targets and stop "
            "after reporting the first divergence in each; do not add states, "
            "budgets, masks, hybrid selectors or trajectory outcomes"
        ),
        "prohibited_in_this_stage": (
            "arrival generation, trajectory simulation, RNG, sidecar or hybrid "
            "selection, new experiments, tuning and paper edits"
        ),
        "source_status": {
            "status": report["status"],
            "shadow_advance": report["shadow_advance"],
            "certificate_only": report["certificate_only"],
        },
    }
    sources = _source_texts()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("residual first-divergence result already exists")
    protocol = json.loads(protocol_path.read_text())
    frozen_sources = json.loads(sources_path.read_text())
    report, _, _, model, targets = _validate_source()
    if (
        protocol.get("experiment") != "residual_capacity_bound_first_divergence_v1"
        or protocol.get("target_count") != TARGET_COUNT
        or protocol.get("budget") != BUDGET
        or protocol.get("targets") != targets
        or _json(model.snapshot()) != protocol.get("model")
        or any(
            (ROOT / name).read_text() != text
            for name, text in frozen_sources.items()
        )
    ):
        raise ValueError("registered residual first-divergence inputs changed")

    legacy_fields = tuple(targets[0]["legacy_k26"])
    residual_fields = tuple(targets[0]["residual_k26"])
    cases = []
    for target in targets:
        queue = tuple(target["queue"])
        paired = trace_first_bound_divergence(
            model.options, queue, model.lanes, budget=BUDGET
        )
        legacy = select_budgeted_frame(
            model.options, queue, model.lanes, BUDGET
        )
        residual = select_residual_capacity_frame(
            model.options, queue, model.lanes, BUDGET
        )
        legacy_projection = _selection_projection(legacy, legacy_fields)
        residual_projection = _selection_projection(residual, residual_fields)
        divergence = paired["divergence"]
        legacy_candidate = _attempted_candidate_signature(
            model, None if divergence is None else divergence["legacy_top"], queue
        )
        residual_candidate = _attempted_candidate_signature(
            model, None if divergence is None else divergence["residual_top"], queue
        )
        cases.append({
            "state_index": target["state_index"],
            "queue": target["queue"],
            "primary_frame_visits": target["primary_frame_visits"],
            "source_support_score_delta": target[
                "support_score_delta_over_legacy_k26"
            ],
            "legacy_first_exact_score_attempt": target[
                "legacy_first_exact_score_attempt"
            ],
            "residual_first_exact_score_attempt": target[
                "residual_first_exact_score_attempt"
            ],
            "legacy_k26": legacy_projection,
            "residual_k26": residual_projection,
            "legacy_k26_reproduced": _same_projection(
                legacy_projection, target["legacy_k26"]
            ),
            "residual_k26_reproduced": _same_projection(
                residual_projection, target["residual_k26"]
            ),
            "paired_trace": paired,
            "classification_evidence_valid": _classification_evidence_valid(paired),
            "first_divergence_candidate_physical_signature_equal": (
                legacy_candidate == residual_candidate
            ),
            "first_divergence_legacy_candidate": legacy_candidate,
            "first_divergence_residual_candidate": residual_candidate,
        })

    work = {
        "paired_trace": {
            key: sum(case["paired_trace"]["work"][key] for case in cases)
            for key in cases[0]["paired_trace"]["work"]
        },
        "legacy_k26_selector_calls": len(cases),
        "residual_k26_selector_calls": len(cases),
        "arrival_tapes": 0,
        "trajectory_frames": 0,
        "rng_initializations": 0,
        "random_draws": 0,
    }
    paired_work = work["paired_trace"]
    masks = 1 << len(model.workloads)
    positions = len(model.options) + 1
    thresholds = len({
        1.0,
        *(max(1.0, float(option.duration)) for option in model.options),
    })
    work_identity = (
        paired_work["bound_pair_calls"] == paired_work["generated_prefixes"]
        and paired_work["bound_option_probes"]
        == TARGET_COUNT * thresholds * positions
        and paired_work["dual_table_option_probes"]
        == TARGET_COUNT * thresholds * masks * positions
        and paired_work["dual_mask_evaluations"]
        == paired_work["bound_threshold_evaluations"] * masks
        and paired_work["dual_mask_sums"]
        == paired_work["bound_pair_calls"] * masks
        and paired_work["dual_coordinate_products"]
        == len(model.workloads) * (
            TARGET_COUNT * masks * positions + paired_work["bound_pair_calls"]
        )
    )
    checks = {
        "all_four_registered_regression_states_executed_once": (
            len(cases) == TARGET_COUNT
            and tuple(case["state_index"] for case in cases) == TARGET_STATE_INDICES
        ),
        "ten_seed_attempts_and_v4_order_match_on_every_state": all(
            case["paired_trace"]["seed_attempt_count"] == len(model.options) + 1
            and _json(case["paired_trace"]["ordered_option_indices"])
            == case["residual_k26"]["ordered_option_indices"]
            for case in cases
        ),
        "first_paid_divergence_exists_with_common_prefix_history": all(
            case["paired_trace"]["first_divergence_attempt"] is not None
            and [
                event["attempt"]
                for event in case["paired_trace"]["common_paid_attempts"]
            ]
            == list(range(
                case["paired_trace"]["seed_attempt_count"] + 1,
                case["paired_trace"]["first_divergence_attempt"],
            ))
            for case in cases
        ),
        "first_divergence_classification_is_exclusive_and_evidenced": all(
            case["paired_trace"]["classification"]
            in ("early_pruning", "heap_reprioritization")
            and case["classification_evidence_valid"]
            for case in cases
        ),
        "residual_bound_nonlooser_on_every_compared_prefix": all(
            case["paired_trace"]["all_new_bounds_nonlooser"] for case in cases
        ),
        "legacy_k26_checkpoints_reproduce_registered_source": all(
            case["legacy_k26_reproduced"] for case in cases
        ),
        "residual_k26_checkpoints_reproduce_registered_source": all(
            case["residual_k26_reproduced"] for case in cases
        ),
        "registered_support_regressions_reproduce": all(
            math.isclose(
                case["residual_k26"]["score"] - case["legacy_k26"]["score"],
                case["source_support_score_delta"],
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            )
            and case["source_support_score_delta"] < 0.0
            for case in cases
        ),
        "paired_bound_work_identities_close": work_identity,
        "no_arrivals_trajectories_or_randomness_used": (
            work["arrival_tapes"] == work["trajectory_frames"]
            == work["rng_initializations"] == work["random_draws"] == 0
        ),
        "registered_sources_unchanged": all(
            (ROOT / name).read_text() == text
            for name, text in frozen_sources.items()
        ),
        "source_shadow_remains_valid_and_closed": (
            report["status"] == "PASS"
            and all(report["checks"].values())
            and not report["shadow_advance"]
        ),
    }
    execution_passed = all(checks.values())
    classifications = {
        kind: sum(
            case["paired_trace"]["classification"] == kind for case in cases
        )
        for kind in ("heap_reprioritization", "early_pruning")
    }
    pure_priority_first = (
        execution_passed
        and classifications["heap_reprioritization"] == TARGET_COUNT
        and all(
            case["paired_trace"]["residual_only_prune_count_before_divergence"]
            == 0
            for case in cases
        )
    )
    mechanism = (
        "heap_reprioritization_is_the_first_search_change_on_all_four_regressions"
        if pure_priority_first
        else "mixed_or_invalid_first_divergence_mechanism"
    )
    next_step = (
        "separately preregister a fixed-state sidecar design that retains the "
        "legacy heap priority and applies the tighter bound only to safe pruning "
        "and certification"
        if pure_priority_first
        else "stop before designing a sidecar and inspect the mixed or invalid evidence"
    )
    result = {
        "experiment": "residual_capacity_bound_first_divergence_v1",
        "status": "PASS" if execution_passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "target_count": len(cases),
        "classification_counts": classifications,
        "mechanism": mechanism,
        "pure_priority_first": pure_priority_first,
        "cases": cases,
        "work_accounting": work,
        "limitations": (
            "The four states were selected because the source shadow regressed. "
            "The trace identifies the first causal search-order change in those "
            "states; it does not validate a sidecar on the other 4,999 states or "
            "on full trajectories."
        ),
        "next_step": next_step,
    }
    if not execution_passed:
        raise ValueError("residual first-divergence execution checks failed")
    _write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.register:
        protocol = register(args.output)
        print(json.dumps({
            "status": "REGISTERED",
            "experiment": protocol["experiment"],
            "target_count": protocol["target_count"],
        }))
    else:
        result = run_registered(args.output)
        print(json.dumps({
            "status": result["status"],
            "mechanism": result["mechanism"],
        }))


if __name__ == "__main__":
    main()
