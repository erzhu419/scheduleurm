"""Global queue-space extension of the frozen aligned-frame support curve.

This runner proves that every queue outside a small integer tail box closes at
the mandatory homogeneous seeds.  The already-frozen complete finite-box curve
therefore contains every queue that can have a positive returned search gap.
No selector, oracle, arrival process, or trajectory is run here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
import json
import math
from pathlib import Path
import time

from algorithm.experiments.aligned_hardware_frame_model import load_model


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
SOURCE_FRONTIER = ARTIFACTS / (
    "aligned_hardware_budget_support_frontier_development_20260912.json"
)
SOURCE_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_budget_support_frontier_development_20260912_protocol.json"
)
SOURCE_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_budget_support_frontier_development_20260912_fixed_sources.json"
)
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_global_support_extension_development_20260912.json"
)

EXPERIMENT = "aligned_hardware_global_support_extension_v1"
SOURCE_EXPERIMENT = "aligned_hardware_budget_support_frontier_v1"
CHECKPOINT_BUDGETS = tuple(range(10, 121))
PURE_OPTION_INDICES = (3, 5, 8)
EXPECTED_REPLICATION_THRESHOLDS = (6, 6, 16)
EXPECTED_TAIL_MAXIMUM = (40, 18, 15)
EXPECTED_TAIL_STATE_COUNT = 41 * 19 * 16
EXPECTED_KERNEL_STATE_COUNT = 4471
GLOBAL_MARGIN_BUDGET = 73
GLOBAL_EXACT_BUDGET = 76
TREE_EXACT_BUDGET = 120
ACTUAL_EXACT_BUDGET = 51
SCORE_TOLERANCE = 1e-10
COMPUTE_GAIN_NEGLIGIBLE_FRACTION = Fraction(1, 1000)
THIS_SOURCE = "algorithm/experiments/aligned_hardware_global_support_extension.py"

RULE = {
    "pure_rate_envelope": (
        "for every q in Z_+^3, max local support is attained by options 3, 5, "
        "or 8; exact Fraction comparisons prove every other local rate lies "
        "below their coordinatewise max envelope"
    ),
    "root_upper_bound": (
        "for two lanes, every root threshold upper bound is at most twice the "
        "maximum local support; if a maximizing pure option fits two copies, "
        "its mandatory homogeneous seed meets that bound and closes the search"
    ),
    "analytic_tail": (
        "all queues that can fail the replicated-maximizer condition lie in "
        "the inclusive integer box [0,40]x[0,18]x[0,15]"
    ),
    "global_extension": (
        "the frozen source box contains the analytic tail box; every queue "
        "outside the tail has zero returned gap already at K10, so every source "
        "K10..K120 maximum is the corresponding all-queue maximum"
    ),
    "quantity_separation": (
        "returned upper-minus-selected gap is the primary runtime certificate; "
        "oracle-minus-selected actual regret remains a separate offline result"
    ),
}

ADVANCE_GATE = {
    "all_execution_checks": True,
    "global_k73_returned_gap_below_all_capacity_margins": True,
    "global_k76_returned_certificate_exact": True,
    "k120_finite_tree_budget_exact": True,
    "actual_regret_and_returned_gap_reported_separately": True,
}

CLAIM_BOUNDARY = (
    "Fixed-catalogue, two-lane, deterministic cached-phase support guarantees "
    "for every supported nonnegative integer queue. The result does not extend "
    "to another catalogue, lane count, physical service distribution, or a "
    "continuous-valued queue API."
)


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output=DEFAULT_OUTPUT):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _write_new(path, payload, *, compact=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
        stream.write("\n")


def _fraction(value: float) -> Fraction:
    """Use the exact binary value consumed by the frozen float implementation."""
    return Fraction.from_float(float(value))


def _fraction_row(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _fraction_from_row(row) -> Fraction:
    return Fraction(int(row["numerator"]), int(row["denominator"]))


def _current_fixed_sources():
    frozen = json.loads(SOURCE_FIXED_SOURCES.read_text())
    changed = [
        name for name, text in frozen.items() if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen budget-frontier source changed: {changed}")
    return {**frozen, THIS_SOURCE: (ROOT / THIS_SOURCE).read_text()}


def _load_source():
    source = json.loads(SOURCE_FRONTIER.read_text())
    protocol = json.loads(SOURCE_PROTOCOL.read_text())
    if source.get("protocol") != protocol:
        raise ValueError("source frontier result does not embed its registered protocol")
    if (
        source.get("experiment") != SOURCE_EXPERIMENT
        or source.get("status") != "PASS"
        or not source.get("checks")
        or not all(source["checks"].values())
        or protocol.get("checkpoint_budgets") != list(CHECKPOINT_BUDGETS)
        or protocol.get("minimum_budget") != CHECKPOINT_BUDGETS[0]
        or protocol.get("exact_budget") != TREE_EXACT_BUDGET
    ):
        raise ValueError("global extension requires the frozen passing frontier")
    rows = source.get("budget_summaries", ())
    if [row.get("budget") for row in rows] != list(CHECKPOINT_BUDGETS):
        raise ValueError("source frontier does not contain every K10..K120 row")
    domain = source.get("domain", {})
    if (
        domain.get("minimum") != [0, 0, 0]
        or domain.get("maximum") != [40, 25, 24]
        or domain.get("state_count") != 26650
    ):
        raise ValueError("source frontier box changed")
    return source, protocol


def _budget_contract(row):
    fields = (
        "budget",
        "domain_state_count",
        "nonempty_state_count",
        "exact_score_state_count",
        "exact_certificate_state_count",
        "positive_gap_state_count",
        "max_certified_normalized_gap",
        "max_actual_normalized_regret_diagnostic",
        "max_frontier_normalized_slack_diagnostic",
        "alpha_witness",
        "actual_regret_witness",
        "frontier_slack_witness",
        "mean_normalized_certified_gap",
        "mean_normalized_actual_regret_diagnostic",
        "exact_score_domain_fraction",
        "exact_certificate_domain_fraction",
        "selector_work",
        "capacity_bridge",
    )
    return {field: row[field] for field in fields}


def _source_contract(source, source_protocol):
    return {
        "source_experiment": source["experiment"],
        "source_status": source["status"],
        "source_checks": source["checks"],
        "source_domain": source["domain"],
        "source_model": source_protocol["model"],
        "source_capacity_certificates": source_protocol["capacity_certificates"],
        "source_budget_summaries": [
            _budget_contract(row) for row in source["budget_summaries"]
        ],
        "source_actual_continuous_trace_work": source[
            "actual_continuous_trace_work"
        ],
        "source_scientific_decision": source["scientific_decision"],
    }


def _rate(option, coordinate):
    duration = _fraction(max(1.0, option.duration))
    return Fraction(option.requirements[coordinate], 1) / duration


def _pure_dominance_certificate(model):
    if model.lanes != 2 or len(model.workloads) != 3:
        raise ValueError("global extension is frozen to two lanes and three queues")
    if len(model.options) != 9:
        raise ValueError("global extension is frozen to the nine-option catalogue")

    rates = tuple(
        tuple(_rate(option, coordinate) for coordinate in range(3))
        for option in model.options
    )
    pure = []
    for coordinate in range(3):
        eligible = [
            index
            for index, option in enumerate(model.options)
            if option.requirements[coordinate] > 0
            and sum(value > 0 for value in option.requirements) == 1
        ]
        pure.append(max(eligible, key=lambda index: rates[index][coordinate]))
    pure = tuple(pure)
    pure_rates = tuple(rates[index][coordinate] for coordinate, index in enumerate(pure))

    dominance = []
    for index, option_rates in enumerate(rates):
        coefficients = tuple(
            option_rates[coordinate] / pure_rates[coordinate]
            for coordinate in range(3)
        )
        total = sum(coefficients, Fraction())
        dominance.append({
            "option_index": index,
            "coordinate_coefficients": [_fraction_row(value) for value in coefficients],
            "coefficient_sum": _fraction_row(total),
            "dominated_by_pure_maximum": total <= 1,
        })

    thresholds = tuple(
        model.lanes * model.options[index].requirements[coordinate]
        for coordinate, index in enumerate(pure)
    )
    underfilled = tuple(value - 1 for value in thresholds)
    underfilled_supports = tuple(
        pure_rates[coordinate] * underfilled[coordinate]
        for coordinate in range(3)
    )
    maximum_underfilled_support = max(underfilled_supports)
    tail_maximum = tuple(
        maximum_underfilled_support // rate for rate in pure_rates
    )
    tail_maximum = tuple(int(value) for value in tail_maximum)
    tail_shape = tuple(value + 1 for value in tail_maximum)
    outside_cutoff_supports = tuple(
        pure_rates[coordinate] * (tail_maximum[coordinate] + 1)
        for coordinate in range(3)
    )
    maximum_pure_rate = max(pure_rates)
    return {
        "lane_count": model.lanes,
        "option_count": len(model.options),
        "pure_option_indices": list(pure),
        "pure_rate_fractions": [_fraction_row(value) for value in pure_rates],
        "all_option_dominance": dominance,
        "all_options_dominated": all(
            row["dominated_by_pure_maximum"] for row in dominance
        ),
        "replication_thresholds": list(thresholds),
        "maximum_underfilled_counts": list(underfilled),
        "underfilled_support_fractions": [
            _fraction_row(value) for value in underfilled_supports
        ],
        "maximum_underfilled_support": _fraction_row(maximum_underfilled_support),
        "first_outside_cutoff_support_fractions": [
            _fraction_row(value) for value in outside_cutoff_supports
        ],
        "every_first_outside_cutoff_strictly_exceeds_underfilled_support": all(
            value > maximum_underfilled_support for value in outside_cutoff_supports
        ),
        "tail_maximum": list(tail_maximum),
        "tail_shape": list(tail_shape),
        "tail_state_count": math.prod(tail_shape),
        "root_upper_formula": (
            "U_root(q)<=max_T 2*max_{o:t_o<=T}(q dot d_o)/T"
            "<=2*max_o(q dot d_o/t_o)"
        ),
        "replicated_seed_formula": (
            "if pure maximizer i has q_i>=2*d_i then its seed score equals "
            "2*max_o(q dot d_o/t_o), hence U_root<=incumbent and gap=0"
        ),
        "coarse_all_k_actual_regret_bound": {
            "minimum_budget": len(model.options) + 1,
            "seed_score_fraction_of_oracle_at_least": _fraction_row(
                Fraction(1, model.lanes)
            ),
            "normalized_actual_regret_at_most": _fraction_row(
                (model.lanes - 1) * maximum_pure_rate
            ),
            "formula": (
                "S_K>=S_seed>=S_star/R, so (S_star-S_K)/L1(q)"
                "<=(R-1)*max_i,o d_o_i/t_o"
            ),
        },
    }


def _kernel_queues(certificate):
    """Queues in the analytic tail without a replicable pure maximizer."""
    rates = tuple(
        _fraction_from_row(row) for row in certificate["pure_rate_fractions"]
    )
    thresholds = tuple(certificate["replication_thresholds"])
    maximum = tuple(certificate["tail_maximum"])
    for queue in product(*(range(value + 1) for value in maximum)):
        supports = tuple(Fraction(queue[i], 1) * rates[i] for i in range(3))
        best = max(supports)
        if not any(
            supports[i] == best and queue[i] >= thresholds[i] for i in range(3)
        ):
            yield queue


def _tail_lemma_checks(certificate, source):
    source_maximum = tuple(source["domain"]["maximum"])
    tail_maximum = tuple(certificate["tail_maximum"])
    kernel = tuple(_kernel_queues(certificate))
    return {
        "pure_options_are_exactly_3_5_8": (
            tuple(certificate["pure_option_indices"]) == PURE_OPTION_INDICES
        ),
        "all_local_options_are_fraction_dominated_by_pure_envelope": (
            certificate["all_options_dominated"]
        ),
        "replication_thresholds_are_exactly_6_6_16": (
            tuple(certificate["replication_thresholds"])
            == EXPECTED_REPLICATION_THRESHOLDS
        ),
        "analytic_tail_is_exactly_40_18_15": (
            tail_maximum == EXPECTED_TAIL_MAXIMUM
            and certificate["tail_state_count"] == EXPECTED_TAIL_STATE_COUNT
        ),
        "source_box_contains_complete_analytic_tail": all(
            source_value >= tail_value
            for source_value, tail_value in zip(
                source_maximum, tail_maximum, strict=True
            )
        ),
        "kernel_count_matches_exact_fraction_enumeration": (
            len(kernel) == EXPECTED_KERNEL_STATE_COUNT
            and kernel[0] == (0, 0, 0)
            and all(
                0 <= queue[i] <= tail_maximum[i]
                for queue in kernel for i in range(3)
            )
        ),
        "outside_tail_has_replicable_pure_maximizer": (
            certificate[
                "every_first_outside_cutoff_strictly_exceeds_underfilled_support"
            ]
        ),
        "replicable_pure_maximizer_closes_root_upper_at_k10": (
            certificate["lane_count"] == 2
            and certificate["option_count"] + 1 == CHECKPOINT_BUDGETS[0]
            and certificate["all_options_dominated"]
        ),
    }, kernel


def _global_curve(source):
    return [
        {
            "budget": row["budget"],
            "global_max_returned_normalized_gap": row[
                "max_certified_normalized_gap"
            ],
            "global_max_actual_normalized_regret_offline": row[
                "max_actual_normalized_regret_diagnostic"
            ],
            "returned_gap_witness": row["alpha_witness"],
            "actual_regret_witness": row["actual_regret_witness"],
            "source_exact_score_state_count": row["exact_score_state_count"],
            "source_exact_certificate_state_count": row[
                "exact_certificate_state_count"
            ],
            "source_selector_work": row["selector_work"],
        }
        for row in source["budget_summaries"]
    ]


def _row(curve, budget):
    return next(row for row in curve if row["budget"] == budget)


def _capacity_rows(source, alpha):
    certificates = source["protocol"]["capacity_certificates"]
    return [
        {
            "load": certificate["load_factor"],
            "effective_slack_eta": certificate["effective_slack_eta"],
            "global_returned_gap_alpha": alpha,
            "global_residual_margin": certificate["effective_slack_eta"] - alpha,
            "global_residual_margin_positive": (
                certificate["effective_slack_eta"] > alpha
            ),
            "generic_global_negative_drift_backlog_threshold": (
                certificate["drift_constant"]
                / (certificate["tau_min"] * (certificate["effective_slack_eta"] - alpha))
                if certificate["effective_slack_eta"] > alpha
                else None
            ),
            "stronger_k_independent_homogeneous_seed_threshold": certificate[
                "negative_drift_backlog_threshold"
            ],
        }
        for certificate in certificates
    ]


def _evaluate(source, dominance):
    tail_checks, kernel = _tail_lemma_checks(dominance, source)
    curve = _global_curve(source)
    returned = [row["global_max_returned_normalized_gap"] for row in curve]
    actual = [row["global_max_actual_normalized_regret_offline"] for row in curve]
    k51 = _row(curve, ACTUAL_EXACT_BUDGET)
    k73 = _row(curve, GLOBAL_MARGIN_BUDGET)
    k76 = _row(curve, GLOBAL_EXACT_BUDGET)
    k120 = _row(curve, TREE_EXACT_BUDGET)
    capacity = _capacity_rows(
        source, k73["global_max_returned_normalized_gap"]
    )
    source_certificates = source["protocol"]["capacity_certificates"]
    availability_losses = {
        certificate["availability_loss_C"] for certificate in source_certificates
    }

    option_count = dominance["option_count"]
    lanes = dominance["lane_count"]
    width = option_count + 1
    tree_budget = width + sum(width ** depth for depth in range(1, lanes + 1))
    work73 = k73["source_selector_work"]["candidate_attempts"]
    work120 = k120["source_selector_work"]["candidate_attempts"]
    saved = work120 - work73
    saving_fraction = Fraction(saved, work120)

    checks = {
        **tail_checks,
        "source_curve_is_nonincreasing": all(
            right <= left + SCORE_TOLERANCE
            for left, right in zip(returned[:-1], returned[1:], strict=True)
        ),
        "source_actual_regret_is_nonincreasing": all(
            right <= left + SCORE_TOLERANCE
            for left, right in zip(actual[:-1], actual[1:], strict=True)
        ),
        "finite_source_curve_equals_global_curve_by_tail_lemma": True,
        "k51_actual_support_regret_is_globally_zero_offline": (
            k51["global_max_actual_normalized_regret_offline"]
            <= SCORE_TOLERANCE
        ),
        "k73_actual_regret_and_returned_gap_are_not_conflated": (
            k73["global_max_actual_normalized_regret_offline"]
            <= SCORE_TOLERANCE
            and k73["global_max_returned_normalized_gap"] > SCORE_TOLERANCE
        ),
        "k73_global_returned_gap_below_all_capacity_margins": all(
            row["global_residual_margin_positive"] for row in capacity
        ),
        "k73_global_witness_is_the_frozen_4_2_5_state": (
            k73["returned_gap_witness"]["queue"] == [4, 2, 5]
            and math.isclose(
                k73["global_max_returned_normalized_gap"],
                0.009883424929280758,
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            )
        ),
        "k76_returned_certificate_is_globally_exact": (
            k76["global_max_returned_normalized_gap"] <= SCORE_TOLERANCE
        ),
        "k120_tree_count_is_exact_and_returned_gap_is_zero": (
            tree_budget == TREE_EXACT_BUDGET
            and k120["global_max_returned_normalized_gap"] <= SCORE_TOLERANCE
        ),
        "k73_paid_attempt_saving_is_positive_but_negligible": (
            saved == 6
            and saving_fraction > 0
            and saving_fraction < COMPUTE_GAIN_NEGLIGIBLE_FRACTION
        ),
        "homogeneous_seed_stability_threshold_is_stronger_at_every_load": all(
            row["stronger_k_independent_homogeneous_seed_threshold"]
            < row["generic_global_negative_drift_backlog_threshold"]
            for row in capacity
        ),
        "homogeneous_seed_projection_certificate_is_global_and_k_independent": (
            len(availability_losses) == 1
            and all(
                certificate["capacity_projection_passed"]
                and certificate["conditional_model_drift_certified"]
                for certificate in source_certificates
            )
        ),
        "proof_adds_no_selector_or_oracle_work": True,
        "proof_adds_no_arrivals_trajectories_drains_or_rng": True,
    }
    return {
        "checks": checks,
        "analytic_tail": {
            "maximum": dominance["tail_maximum"],
            "shape": dominance["tail_shape"],
            "state_count_including_zero": dominance["tail_state_count"],
            "kernel_state_count_including_zero": len(kernel),
            "source_box_maximum": source["domain"]["maximum"],
            "source_box_contains_tail": tail_checks[
                "source_box_contains_complete_analytic_tail"
            ],
        },
        "global_budget_curve": curve,
        "primary_k73": k73,
        "offline_actual_exact_k51": k51,
        "returned_certificate_exact_k76": k76,
        "finite_tree_exact_k120": {
            **k120,
            "option_plus_idle_width": width,
            "lane_count": lanes,
            "complete_tree_attempt_bound": tree_budget,
        },
        "capacity_bridge_at_k73": capacity,
        "homogeneous_seed_global_baseline": {
            **dominance["coarse_all_k_actual_regret_bound"],
            "mask_projection_additive_support_loss": next(iter(availability_losses)),
            "pointwise_normalized_additive_bound": (
                "actual_regret/L1(q) <= availability_loss_C/L1(q) for nonzero q"
            ),
            "applies_to_every_budget_at_least_10": True,
            "negative_drift_backlog_thresholds": [
                certificate["negative_drift_backlog_threshold"]
                for certificate in source_certificates
            ],
            "comparison": (
                "this K-independent additive certificate gives a smaller drift "
                "threshold than subtracting the global K73 returned-gap alpha"
            ),
        },
        "compute_comparison": {
            "k73_paid_candidate_attempts": work73,
            "k120_paid_candidate_attempts": work120,
            "paid_attempts_saved": saved,
            "paid_attempt_saving_fraction": float(saving_fraction),
            "paid_attempt_saving_percent": 100.0 * float(saving_fraction),
            "scientific_interpretation": "negligible",
        },
        "proof_work": {
            "fraction_dominance_rows": len(dominance["all_option_dominance"]),
            "tail_integer_states_classified": dominance["tail_state_count"],
            "kernel_states_retained": len(kernel),
            "source_budget_rows_reused": len(curve),
            "selector_calls": 0,
            "oracle_action_checks": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "rng_or_random_draws": 0,
        },
    }


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("global support extension artifacts already exist")
    source, source_protocol = _load_source()
    fixed_sources = _current_fixed_sources()
    model = load_model()
    if _json(model.snapshot()) != source_protocol["model"]:
        raise ValueError("current aligned model differs from the frozen frontier model")
    dominance = _pure_dominance_certificate(model)
    checks, _kernel = _tail_lemma_checks(dominance, source)
    if not all(checks.values()):
        raise ValueError("pure-dominance tail lemma does not match the frozen design")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "role": "analytic all-integer-queue extension of frozen support curve",
        "source_frontier": SOURCE_FRONTIER.name,
        "source_protocol": SOURCE_PROTOCOL.name,
        "checkpoint_budgets": list(CHECKPOINT_BUDGETS),
        "pure_option_indices": list(PURE_OPTION_INDICES),
        "expected_replication_thresholds": list(EXPECTED_REPLICATION_THRESHOLDS),
        "expected_tail_maximum": list(EXPECTED_TAIL_MAXIMUM),
        "expected_tail_state_count": EXPECTED_TAIL_STATE_COUNT,
        "expected_kernel_state_count": EXPECTED_KERNEL_STATE_COUNT,
        "global_margin_budget": GLOBAL_MARGIN_BUDGET,
        "global_exact_budget": GLOBAL_EXACT_BUDGET,
        "tree_exact_budget": TREE_EXACT_BUDGET,
        "actual_exact_budget": ACTUAL_EXACT_BUDGET,
        "score_tolerance": SCORE_TOLERANCE,
        "compute_gain_negligible_fraction": _fraction_row(
            COMPUTE_GAIN_NEGLIGIBLE_FRACTION
        ),
        "rule": RULE,
        "advance_gate": ADVANCE_GATE,
        "claim_boundary": CLAIM_BOUNDARY,
        "model": _json(model.snapshot()),
        "pure_dominance_certificate": dominance,
        "source_contract": _source_contract(source, source_protocol),
        "timing_is_scientific_gate": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("global support extension result already exists")
    protocol = json.loads(protocol_path.read_text())
    registered_sources = json.loads(fixed_sources_path.read_text())
    if registered_sources != _current_fixed_sources():
        raise ValueError("registered global-extension sources changed")
    source, source_protocol = _load_source()
    model = load_model()
    expected_constants = {
        "experiment": EXPERIMENT,
        "source_frontier": SOURCE_FRONTIER.name,
        "source_protocol": SOURCE_PROTOCOL.name,
        "checkpoint_budgets": list(CHECKPOINT_BUDGETS),
        "pure_option_indices": list(PURE_OPTION_INDICES),
        "expected_replication_thresholds": list(EXPECTED_REPLICATION_THRESHOLDS),
        "expected_tail_maximum": list(EXPECTED_TAIL_MAXIMUM),
        "expected_tail_state_count": EXPECTED_TAIL_STATE_COUNT,
        "expected_kernel_state_count": EXPECTED_KERNEL_STATE_COUNT,
        "global_margin_budget": GLOBAL_MARGIN_BUDGET,
        "global_exact_budget": GLOBAL_EXACT_BUDGET,
        "tree_exact_budget": TREE_EXACT_BUDGET,
        "actual_exact_budget": ACTUAL_EXACT_BUDGET,
        "score_tolerance": SCORE_TOLERANCE,
        "compute_gain_negligible_fraction": _fraction_row(
            COMPUTE_GAIN_NEGLIGIBLE_FRACTION
        ),
        "rule": RULE,
        "advance_gate": ADVANCE_GATE,
        "claim_boundary": CLAIM_BOUNDARY,
        "timing_is_scientific_gate": False,
    }
    if any(protocol.get(key) != value for key, value in expected_constants.items()):
        raise ValueError("registered global support extension protocol changed")
    if protocol.get("model") != _json(model.snapshot()):
        raise ValueError("registered global-extension model changed")
    if protocol.get("source_contract") != _source_contract(source, source_protocol):
        raise ValueError("frozen source frontier contract changed")

    started = time.perf_counter()
    dominance = _pure_dominance_certificate(model)
    if protocol.get("pure_dominance_certificate") != dominance:
        raise ValueError("registered pure-dominance certificate changed")
    evaluated = _evaluate(source, dominance)
    proof_wall_s = time.perf_counter() - started
    checks = {
        **evaluated.pop("checks"),
        "registered_sources_unchanged": True,
        "registered_protocol_and_source_contract_unchanged": True,
    }
    passed = all(checks.values())
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "theorem": {
            "queue_domain": "all q in Z_+^3",
            "pure_support_envelope": (
                "R(q)=max_i q_i*r_i with pure option indices (3,5,8)"
            ),
            "root_bound": "U_root(q)<=2*R(q)",
            "tail_implication": (
                "q outside [0,40]x[0,18]x[0,15] implies a maximizing pure "
                "option fits two lanes, so U_root equals the seed incumbent and "
                "the returned gap is zero at K10"
            ),
            "global_curve_implication": (
                "the frozen source box contains the complete analytic tail, so "
                "its K10..K120 maxima equal the all-queue maxima"
            ),
        },
        "pure_dominance_certificate": dominance,
        **evaluated,
        "scientific_decision": (
            "GLOBAL_K73_RETURNED_GAP_CERTIFIED_K76_EXACT_"
            "COMPUTE_GAIN_NEGLIGIBLE_CLOSE"
            if passed else "INVALID_EXECUTION"
        ),
        "development_advance": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The all-queue proof is catalogue- and two-lane-specific.",
            "Actual regret uses the frozen offline 100-action oracle; returned gap is the runtime certificate.",
            "K73 saves only six paid attempts over the early-closing K120 search on the frozen source box.",
            "The existing homogeneous-seed drift certificate is stronger and K-independent.",
            "No new physical-service confidence, trajectory, or mean-flow result is produced.",
        ],
        "next_step": (
            "audit computation-versus-returned-gap scaling as the lane Cartesian "
            "action family grows; keep this fixed-catalogue result closed"
        ),
        "timing": {
            "proof_wall_seconds": proof_wall_s,
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--run", "--run-registered", dest="run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.register == args.run:
        parser.error("choose exactly one of --register or --run")
    payload = register(args.output) if args.register else run_registered(args.output)
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", "REGISTERED"),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
