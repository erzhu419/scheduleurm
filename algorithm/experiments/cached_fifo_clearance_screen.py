"""Rescore the completed three-state native terminal caches without simulation."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

from algorithm.experiments.fifo_clearance_externality import clearance_externality

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_cached_fifo_clearance_20261003"
NATIVE = "aligned_hardware_native_terminal_flow_20261003"


def read(path):
    return json.loads(Path(path).read_text())


def exact(value):
    return Fraction(*value) if isinstance(value, list) else Fraction.from_float(float(value))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preference(delta):
    return "parent" if delta < 0 else "native" if delta > 0 else "tie"


def jsonable(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def analyze(directory):
    design = read(directory / (STEM + "_design.json"))
    protocol = read(directory / (NATIVE + "_protocol.json"))
    summary = read(directory / (NATIVE + "_summary.json"))
    require(protocol["registered_at_utc"] == summary["registered_at_utc"]
        == design["source_native_epoch"] and protocol["source_horizon_registered_at_utc"]
        == design["source_horizon_epoch"] and summary["execution_checks_passed"]
        and summary["completed_root_sample_units"] == 48 and protocol["planning_samples"] == 8
        and protocol["planning_event_cap"] == 128 and protocol["horizon_tau_max_multiple"] == 2,
        "completed matching native48 and original H/8/cap128 required")
    require([target["tape_id"] for target in protocol["targets"]] == design["tapes"],
        "retain all three original states without outcome selection")
    records = []
    for target, reference in zip(protocol["targets"], summary["records"], strict=True):
        require(target["tape_id"] == reference["tape_id"], "native summary target differs")
        samples = []
        for sample in range(8):
            arms = {}
            environments, rosters = [], []
            for slot, role in enumerate(("parent", "native")):
                index = 16 * target["index"] + 8 * slot + sample
                row = read(directory / (NATIVE + f"_shard_{index:02d}.json"))
                old = read(protocol["source_shards"][index])
                assignment = protocol["assignments"][index]
                require(row["registered_at_utc"] == protocol["registered_at_utc"]
                    and old["registered_at_utc"] == protocol["source_horizon_registered_at_utc"]
                    and row["assignment"] == old["assignment"] == assignment
                    and assignment["root"] == target[role + "_root"]
                    and row["execution_checks_passed"] and old["execution_checks_passed"],
                    "fixed H/native root-sample identity differs")
                result, source = row["result"], old["result"]
                flow, hflow = result["flow"], source["flow"]
                require(flow["complete"] and not flow["overflow"] and hflow["complete"]
                    and not hflow["overflow"] and result["environment"] == source["environment"]
                    and flow["horizon"] == hflow["horizon"] and all(flow[key] == hflow[key]
                        for key in ("common_area", "waiting_area", "inflight_area")),
                    "unchanged completed H population and pre-H costs required")
                cost = exact(flow["cost"])
                require(cost == exact(flow["common_area"]) + exact(flow["terminal_cost"]),
                    "original full population cost no longer reconstructs")
                externality = clearance_externality(flow["population"]["jobs"],
                    hflow["terminal_state"]["waiting_ids"], result["environment"]["rates"], flow["horizon"])
                arms[role] = dict(root=assignment["root"], original_cost=cost,
                    augmented_cost=cost + externality["externality"], **externality)
                environments.append(result["environment"])
                rosters.append(result["scored_population"])
            require(environments[0] == environments[1] and rosters[0] == rosters[1],
                "paired roots must share the original H population and anchor rates")
            samples.append(dict(sample_index=sample, arms=arms, **{name + "_delta":
                arms["parent"][name] - arms["native"][name]
                for name in ("original_cost", "externality", "augmented_cost")}))
        means = {name: sum((s[name + "_delta"] for s in samples), Fraction(0)) / 8
            for name in ("original_cost", "externality", "augmented_cost")}
        require(means["original_cost"] == exact(reference["comparison"]
            ["exact_parent_minus_native_mean"]["cost"]), "cached original eight-path cost changed")
        original, augmented = (preference(means[key]) for key in ("original_cost", "augmented_cost"))
        # Actual labels are read only after both root scores are fixed.
        context = reference["actual_context"]
        effects = context["cohort_effect"]
        change = int(augmented == "parent") - int(original == "parent")
        sensitive = [(sum((s["augmented_cost_delta"] for s in samples), Fraction(0))
            - removed["augmented_cost_delta"]) / 7 for removed in samples]
        records.append(dict(tape_id=target["tape_id"], seed=target["seed"],
            parent_root=target["parent_root"], native_root=target["native_root"],
            original_risk_roots=target["original_risk_roots"], mean_parent_minus_native=means,
            original_preference=original, augmented_preference=augmented,
            sample_preferences={role: sum(preference(s["augmented_cost_delta"]) == role for s in samples)
                for role in ("parent", "native", "tie")},
            leave_one_out_mean_range=[min(sensitive), max(sensitive)],
            actual_H_complete_delta=context["H_population_complete_cost_delta"],
            actual_cohort_effect=effects,
            changed_choice_conditional_effect={key: change * value for key, value in effects.items()},
            per_coordinate_externality_delta=[sum((s["arms"]["parent"]["per_coordinate"][c]["externality"]
                - s["arms"]["native"]["per_coordinate"][c]["externality"] for s in samples), Fraction(0)) / 8
                for c in range(len(samples[0]["arms"]["parent"]["per_coordinate"]))], samples=samples))
    changed = [r for r in records if r["original_preference"] != r["augmented_preference"]]
    nonneutral = [r for r in records if r["actual_cohort_effect"]["delta_T_mean"] != 0]
    aligned = all(r["augmented_preference"] == preference(r["actual_cohort_effect"]["delta_T_mean"])
        for r in nonneutral)
    improves = any(r["changed_choice_conditional_effect"]["delta_T_mean"] < 0 for r in changed)
    no_added_harm = all(all(value <= 0 for value in r["changed_choice_conditional_effect"].values())
        for r in changed)
    proceed = bool(changed) and aligned and improves and no_added_harm
    return jsonable(dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
        design_registered_at_utc=design["registered_at_utc"],
        source_native_registered_at_utc=protocol["registered_at_utc"],
        source_horizon_registered_at_utc=protocol["source_horizon_registered_at_utc"],
        decision="FIFO_CLEARANCE_SCREEN_SUPPORTS_RULE_IMPLEMENTATION" if proceed else "FIFO_CLEARANCE_SCREEN_HOLD",
        cached_native_units_read=48, cached_H_units_read=48, new_simulation_executions=0,
        new_planner_calls=0, performance_gain_established=False, changed_state_rankings=len(changed),
        nonneutral_cohort_mean_states=len(nonneutral), all_nonneutral_mean_directions_aligned=aligned,
        full24_authorized_by_screen=False, records=records,
        interpretation="Development diagnostic on the unchanged three selected states, two roots and eight paths. E prices later arrivals using endogenous class FIFO clearance clocks; it is a proxy, not a proven expected cost or lower bound. Actual conditional effects never enter J. A favorable screen warrants implementing a rule before the original full24 gates; it is not policy gain or independent confirmation. Busy-only congestion and later service costs are omitted."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    result = analyze(args.artifact_dir)
    output = args.artifact_dir / (STEM + "_summary.json")
    with output.open("x") as target:
        json.dump(result, target, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        target.write("\n")
    print(json.dumps({key: result[key] for key in ("decision", "changed_state_rankings",
        "all_nonneutral_mean_directions_aligned", "new_planner_calls")}), flush=True)


if __name__ == "__main__":
    main()
