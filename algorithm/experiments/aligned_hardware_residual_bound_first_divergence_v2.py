"""Corrected reporting for the residual-bound first-divergence diagnosis.

Version 1 correctly classified the paired frontiers but compared dictionaries
that still contained lane-ordered option indices when reporting whether the two
first-divergence candidates were physically identical.  This version preserves
the registered targets and search analysis while comparing only service,
duration, and per-class completion offsets for physical equivalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import algorithm.experiments.aligned_hardware_residual_bound_first_divergence as v1
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.residual_capacity_frame_candidates import (
    select_residual_capacity_frame,
)
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.residual_bound_first_divergence_trace import (
    trace_first_bound_divergence,
)


ROOT = v1.ROOT
ARTIFACTS = v1.ARTIFACTS
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_residual_bound_first_divergence_v2_development_20260910.json"
)
SOURCE_V1_RESULT = ARTIFACTS / (
    "aligned_hardware_residual_bound_first_divergence_development_20260910.json"
)
SOURCE_V1_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_residual_bound_first_divergence_development_20260910_protocol.json"
)
SOURCE_V1_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_residual_bound_first_divergence_development_20260910_fixed_sources.json"
)
EXPERIMENT = "residual_capacity_bound_first_divergence_v2"
NEW_SOURCE = (
    "algorithm/experiments/aligned_hardware_residual_bound_first_divergence_v2.py"
)
CORRECTION = (
    "physical candidate equivalence excludes lane-ordered option indices "
    "and compares only feasibility, service, duration and sorted per-class "
    "completion offsets"
)


def _paths(output):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _validate_v1_bundle():
    result = json.loads(SOURCE_V1_RESULT.read_text())
    protocol = json.loads(SOURCE_V1_PROTOCOL.read_text())
    sources = json.loads(SOURCE_V1_FIXED_SOURCES.read_text())
    if (
        result["experiment"] != "residual_capacity_bound_first_divergence_v1"
        or result["status"] != "PASS"
        or not all(result["checks"].values())
        or result["protocol"] != protocol
        or any((ROOT / name).read_text() != text for name, text in sources.items())
    ):
        raise ValueError("corrected diagnosis requires the preserved v1 bundle")
    return result, protocol, sources


def _source_texts():
    _, _, sources = _validate_v1_bundle()
    return {**sources, NEW_SOURCE: (ROOT / NEW_SOURCE).read_text()}


def _attempted_candidate(model, node, queue):
    if node is None:
        return None
    prefix = node["prefix"]
    option_index = prefix["next_option_index"]
    search_indices = tuple(prefix["indices"]) + (option_index,)
    service = tuple(prefix["service"])
    if option_index is not None:
        service = tuple(
            old + count
            for old, count in zip(
                service, model.options[option_index].requirements
            )
        )
    if any(count > available for count, available in zip(service, queue)):
        return {
            "search_indices": search_indices,
            "feasible": False,
            "physical_signature": None,
        }
    duration = max(
        float(prefix["duration"]),
        1.0 if option_index is None else model.options[option_index].duration,
    )
    completion_offsets = []
    for coordinate in range(len(queue)):
        offsets = []
        for index in search_indices:
            if index is not None:
                offsets.extend(model.options[index].completion_offsets[coordinate])
        completion_offsets.append(tuple(sorted(float(value) for value in offsets)))
    return {
        "search_indices": search_indices,
        "feasible": True,
        "physical_signature": {
            "service": service,
            "duration": duration,
            "completion_offsets": tuple(completion_offsets),
        },
    }


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("corrected residual first-divergence artifacts already exist")
    _, v1_protocol, _ = _validate_v1_bundle()
    report, _, _, model, targets = v1._validate_source()
    if targets != v1_protocol["targets"]:
        raise ValueError("corrected diagnosis targets changed from v1")
    protocol = json.loads(json.dumps(v1_protocol))
    protocol.update({
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "role": (
            "corrected deterministic reporting of the same four registered "
            "post-outcome first-divergence cases"
        ),
        "supersedes_result": SOURCE_V1_RESULT.name,
        "correction": CORRECTION,
        "completion_rule": (
            "status PASS means all v1 frontier and K26 checks reproduce and the "
            "corrected physical-equivalence field is internally consistent; it is "
            "not an algorithm advancement decision"
        ),
        "source_status": {
            "status": report["status"],
            "shadow_advance": report["shadow_advance"],
            "certificate_only": report["certificate_only"],
        },
    })
    sources = _source_texts()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("corrected residual first-divergence result already exists")
    protocol = json.loads(protocol_path.read_text())
    frozen_sources = json.loads(sources_path.read_text())
    v1_result, _, _ = _validate_v1_bundle()
    report, _, _, model, targets = v1._validate_source()
    if (
        protocol.get("experiment") != EXPERIMENT
        or protocol.get("supersedes_result") != SOURCE_V1_RESULT.name
        or protocol.get("correction") != CORRECTION
        or protocol.get("target_count") != v1.TARGET_COUNT
        or protocol.get("budget") != v1.BUDGET
        or protocol.get("targets") != targets
        or v1._json(model.snapshot()) != protocol.get("model")
        or any(
            (ROOT / name).read_text() != text
            for name, text in frozen_sources.items()
        )
    ):
        raise ValueError("registered corrected first-divergence inputs changed")

    legacy_fields = tuple(targets[0]["legacy_k26"])
    residual_fields = tuple(targets[0]["residual_k26"])
    cases = []
    for target in targets:
        queue = tuple(target["queue"])
        paired = trace_first_bound_divergence(
            model.options, queue, model.lanes, budget=v1.BUDGET
        )
        legacy = select_budgeted_frame(
            model.options, queue, model.lanes, v1.BUDGET
        )
        residual = select_residual_capacity_frame(
            model.options, queue, model.lanes, v1.BUDGET
        )
        legacy_projection = v1._selection_projection(legacy, legacy_fields)
        residual_projection = v1._selection_projection(residual, residual_fields)
        divergence = paired["divergence"]
        legacy_candidate = _attempted_candidate(
            model, None if divergence is None else divergence["legacy_top"], queue
        )
        residual_candidate = _attempted_candidate(
            model, None if divergence is None else divergence["residual_top"], queue
        )
        physical_equal = (
            legacy_candidate is not None
            and residual_candidate is not None
            and legacy_candidate["feasible"]
            and residual_candidate["feasible"]
            and legacy_candidate["physical_signature"]
            == residual_candidate["physical_signature"]
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
            "legacy_k26_reproduced": v1._same_projection(
                legacy_projection, target["legacy_k26"]
            ),
            "residual_k26_reproduced": v1._same_projection(
                residual_projection, target["residual_k26"]
            ),
            "paired_trace": paired,
            "classification_evidence_valid": v1._classification_evidence_valid(
                paired
            ),
            "first_divergence_candidates_are_lane_permutations": (
                physical_equal
                and legacy_candidate["search_indices"]
                != residual_candidate["search_indices"]
            ),
            "first_divergence_candidate_physical_signature_equal": physical_equal,
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
        == v1.TARGET_COUNT * thresholds * positions
        and paired_work["dual_table_option_probes"]
        == v1.TARGET_COUNT * thresholds * masks * positions
        and paired_work["dual_mask_evaluations"]
        == paired_work["bound_threshold_evaluations"] * masks
        and paired_work["dual_mask_sums"]
        == paired_work["bound_pair_calls"] * masks
        and paired_work["dual_coordinate_products"]
        == len(model.workloads) * (
            v1.TARGET_COUNT * masks * positions
            + paired_work["bound_pair_calls"]
        )
    )
    checks = {
        "all_four_registered_regression_states_executed_once": (
            len(cases) == v1.TARGET_COUNT
            and tuple(case["state_index"] for case in cases)
            == v1.TARGET_STATE_INDICES
        ),
        "ten_seed_attempts_and_v4_order_match_on_every_state": all(
            case["paired_trace"]["seed_attempt_count"] == len(model.options) + 1
            and v1._json(case["paired_trace"]["ordered_option_indices"])
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
        "first_search_divergence_candidates_are_physical_lane_permutations": all(
            case["first_divergence_candidates_are_lane_permutations"]
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
                abs_tol=v1.SCORE_TOLERANCE,
            )
            and case["source_support_score_delta"] < 0.0
            for case in cases
        ),
        "paired_bound_work_identities_close": work_identity,
        "no_arrivals_trajectories_or_randomness_used": (
            work["arrival_tapes"] == work["trajectory_frames"]
            == work["rng_initializations"] == work["random_draws"] == 0
        ),
        "registered_v1_and_v2_sources_unchanged": all(
            (ROOT / name).read_text() == text
            for name, text in frozen_sources.items()
        ),
        "source_shadow_remains_valid_and_closed": (
            report["status"] == "PASS"
            and all(report["checks"].values())
            and not report["shadow_advance"]
        ),
        "superseded_v1_result_is_preserved": (
            v1_result["status"] == "PASS"
            and v1_result["mechanism"]
            == "heap_reprioritization_is_the_first_search_change_on_all_four_regressions"
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
        and classifications["heap_reprioritization"] == v1.TARGET_COUNT
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
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "supersedes_result": SOURCE_V1_RESULT.name,
        "correction": protocol["correction"],
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
            "The trace identifies their first causal search-order change; it does "
            "not validate a sidecar on the other 4,999 states or full trajectories."
        ),
        "next_step": next_step,
    }
    if not execution_passed:
        raise ValueError("corrected residual first-divergence checks failed")
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
            "supersedes_result": protocol["supersedes_result"],
        }))
    else:
        result = run_registered(args.output)
        print(json.dumps({
            "status": result["status"],
            "mechanism": result["mechanism"],
            "supersedes_result": result["supersedes_result"],
        }))


if __name__ == "__main__":
    main()
