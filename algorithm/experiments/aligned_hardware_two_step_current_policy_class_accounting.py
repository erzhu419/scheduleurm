"""Joint class accounting on all 22 frozen old-current continuation labels."""
from collections import Counter
import json
from math import isclose
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
RANKING = ARTIFACTS / "aligned_hardware_two_step_current_policy_root_ranking_20261001.json"
CROSS = ARTIFACTS / "aligned_hardware_two_step_bg_incumbent_continuation_cross_20261001.json"
PROTOCOL = ARTIFACTS / "aligned_hardware_two_step_bg_incumbent_continuation_cross_20261001_protocol.json"
OUTPUT = ARTIFACTS / "aligned_hardware_two_step_current_policy_class_accounting_20261001.json"
GROUPS = ("kept_beneficial_A", "kept_harmful_A", "rejected_harmful_A", "rejected_beneficial_A")


def indexed(records):
    result = {(row["tape_id"], row["frame_index"]): row for row in records}
    if len(result) != len(records):
        raise ValueError("duplicate frozen anchor")
    return result


def distribution(values):
    return {"negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "positive": sum(value > 0 for value in values),
            "minimum": min(values), "maximum": max(values),
            "unweighted_anchor_mean_descriptive": mean(values)}


def summarize(records):
    classes = []
    for coordinate in range(len(records[0]["classes"])):
        cells = [row["classes"][coordinate] for row in records]
        classes.append({"coordinate": coordinate, "workload": cells[0]["workload"],
            "T_class_mean_delta_s": distribution([cell["T_class_mean_delta_s"] for cell in cells]),
            "T_weighted_mean_contribution_s": distribution([cell["T_weighted_mean_contribution_s"] for cell in cells]),
            "class_900_unfinished_delta": distribution([cell["class_900_unfinished_delta"] for cell in cells]),
            "same_state_mean_improved_900_tail_increased": sum(
                cell["mean_improved_900_tail_increased"] for cell in cells)})
    return {"anchors": len(records), "classes": classes,
        "mixed_class_T_mean_signs": sum(
            any(cell["T_class_mean_delta_s"] < 0 for cell in row["classes"])
            and any(cell["T_class_mean_delta_s"] > 0 for cell in row["classes"])
            for row in records),
        "anchors_with_any_class_mean_improved_900_tail_increased": sum(
            any(cell["mean_improved_900_tail_increased"] for cell in row["classes"])
            for row in records)}


def main():
    ranking = json.loads(RANKING.read_text())
    cross = json.loads(CROSS.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    ranked, crossed = indexed(ranking["records"]), indexed(cross["records"])
    if (not ranking["execution_checks_passed"] or not cross["execution_checks_passed"]
            or len(ranked) != 22 or set(ranked) != set(crossed)):
        raise ValueError("all22 matching execution-valid frozen anchors required")
    records, reconstruction_errors = [], []
    for key, prior in ranked.items():
        source = crossed[key]
        if (not prior["pair_available"] or prior["native_choice_kind"] not in ("A", "BG")
                or any(prior[name] != source[name] for name in
                       ("factual_option_index", "replacement_option_index"))):
            raise ValueError("frozen native A/BG pair differs: " + str(key))
        snapshots = source["outcomes"]["snapshots"]
        end, horizon = snapshots["observation_end"], snapshots["arrival_horizon"]
        a, b = end["original_current"], end["BG_current"]
        label = source["outcomes"]["root_effects"]["current"]["delta_T_mean"]
        if (label == 0 or not isclose(label, prior["cached_current_labels"]["delta_T_mean"],
                                      rel_tol=1e-10, abs_tol=1e-9)
                or a["arrivals"] != b["arrivals"]):
            raise ValueError("old-current cohort mean label differs: " + str(key))
        group = ("kept_" if prior["native_choice_kind"] == "A" else "rejected_")
        group += "beneficial_A" if label < 0 else "harmful_A"
        classes = []
        labels = source["outcomes"]["root_effects"]["current"]["per_coordinate_deltas"]
        for coordinate, (ca, cb) in enumerate(zip(a["per_coordinate"], b["per_coordinate"], strict=True)):
            if ca["arrivals"] != cb["arrivals"]:
                raise ValueError("class cohort counts differ: " + str(key))
            class_mean = ca["restricted_mean_flow_s"] - cb["restricted_mean_flow_s"]
            tail = (horizon["original_current"]["per_coordinate"][coordinate]["unfinished_jobs"]
                    - horizon["BG_current"]["per_coordinate"][coordinate]["unfinished_jobs"])
            classes.append({"coordinate": coordinate, "workload": labels[coordinate]["workload"],
                "class_cohort_N": ca["arrivals"], "cohort_fraction": ca["arrivals"] / a["arrivals"],
                "T_class_mean_delta_s": class_mean,
                "T_weighted_mean_contribution_s": ca["arrivals"] * class_mean / a["arrivals"],
                "class_900_unfinished_delta": tail,
                "mean_improved_900_tail_increased": class_mean < 0 and tail > 0})
        reconstructed = sum(cell["T_weighted_mean_contribution_s"] for cell in classes)
        if (sum(cell["class_cohort_N"] for cell in classes) != a["arrivals"]
                or not isclose(reconstructed, label, rel_tol=1e-10, abs_tol=1e-9)):
            raise ValueError("class contributions do not reconstruct the root mean: " + str(key))
        reconstruction_errors.append(abs(reconstructed - label))
        records.append({"tape_id": key[0], "frame_index": key[1], "group": group,
            "A_option_index": prior["factual_option_index"], "BG_option_index": prior["replacement_option_index"],
            "new_native_choice_kind": prior["native_choice_kind"], "cohort_N": a["arrivals"],
            "old_current_T_mean_A_minus_BG_s": label, "reconstructed_T_mean_delta_s": reconstructed,
            "classes": classes})
    counts = Counter(row["group"] for row in records)
    if any(counts[group] != ranking["summary"][group] for group in GROUPS):
        raise ValueError("native ranking group membership differs")
    result = {"decision": "TWO_STEP_CURRENT_POLICY_CLASS_ACCOUNTING_COMPLETE",
        "execution_checks_passed": True, "performance_gain_established": False,
        "new_service_trajectory_executions": 0, "new_native_proposals": 0,
        "source_inputs": [str(path.relative_to(ROOT)) for path in (RANKING, CROSS, PROTOCOL)],
        "protocol_metadata": {key: protocol[key] for key in
            ("birth_cohort", "arrival_horizon", "observation_end", "cutoff_semantics")},
        "source_registered_at_utc": {"ranking": ranking["registered_at_utc"],
                                     "cross": cross["registered_at_utc"]},
        "scope": "All22 existing fixed A/BG pairs under OLD-current continuation. Each class contribution is class cohort N / full cohort N times its T restricted-mean delta. Compare these with same-state class900 unfinished deltas. Anchor distribution means are unweighted descriptive summaries, not additive counterfactual benefits or new-policy performance. No new scores, policy weights, service gates, fits, proposals or trajectories.",
        "accounting_checks": {"joined_anchors": len(records), "class_cohort_counts_match": True,
            "native_mean_reconstruction_passed": True,
            "maximum_absolute_reconstruction_error_s": max(reconstruction_errors),
            "identity_tolerance": {"relative": 1e-10, "absolute_s": 1e-9}},
        "summary": summarize(records),
        "groups": {group: summarize([row for row in records if row["group"] == group])
                   for group in GROUPS}, "records": records}
    with OUTPUT.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)),
        "group_counts": counts, "summary": result["summary"],
        "accounting_checks": result["accounting_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
