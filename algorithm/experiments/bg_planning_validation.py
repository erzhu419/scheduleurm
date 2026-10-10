"""Validate unit462's saved trajectory with its independently proved float tie.

The frozen BG policy, checker, trajectory and original missing PASS row remain
unchanged. All frozen checker predicates are retained; only the lost strictness
of one serialized S1 comparison can be discharged by its exact certificate.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
from math import ceil
from pathlib import Path
from time import perf_counter, process_time

from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_background_competition_continuous import EXTRA_WORK, background_work
from algorithm.experiments.aligned_hardware_bg_planning_streams import ROOT, STEM, births, load_protocol, path, read, require, save
from algorithm.experiments.aligned_hardware_performance_necessity import load_models


def _fraction(record):
    return Fraction(record["numerator"], record["denominator"])


def exact_certificate_matches(certificate, frame):
    """Require strict rational S1 improvement and nonincreasing rational S2."""
    info = frame["rollout"]
    baseline = _fraction(certificate["baseline_S1_sum"])
    actual = _fraction(certificate["actual_S1_sum"])
    baseline_squared = _fraction(certificate["baseline_S2_sum"])
    actual_squared = _fraction(certificate["actual_S2_sum"])
    rounded = {
        "baseline_predicted_completion_s": float(baseline / info["sample_count"]),
        "chosen_predicted_completion_s": float(actual / info["sample_count"]),
        "baseline_predicted_squared_flow_s2": float(baseline_squared / info["sample_count"]),
        "chosen_predicted_squared_flow_s2": float(actual_squared / info["sample_count"]),
    }
    return (certificate["start"] == frame["start"] and info["changed"]
        and certificate["actual_root"] == frame["option_indices"][0]
        and certificate["baseline_root"] != certificate["actual_root"]
        and actual < baseline and actual_squared <= baseline_squared
        and baseline - actual == _fraction(certificate["strict_S1_improvement_sum"])
        and baseline_squared - actual_squared == _fraction(certificate["S2_nonincrease_sum"])
        and all(info[key] == value == certificate["rounded_scores"][key] for key, value in rounded.items())
        and certificate["exact_S1_strict_improvement"] is True
        and certificate["exact_S2_nonincrease"] is True
        and certificate["actual_exact_rule_root_matched"] is True
        and certificate["target_work_metadata_matched"] is True)


def corrected_rollout_checks(result, model, protocol, planner, certificates):
    # Copied frozen check_rollout predicates; the final float-strict predicate
    # alone admits a matching independently recomputed rational certificate.
    m, r, cap, samples = len(model.options), model.lanes, protocol["planning_event_cap"], protocol["planning_samples"]
    for frame_index, frame in enumerate(result["frames"]):
        info, now, counts = frame["rollout"], frame["start"], tuple(frame["observed_arrival_counts"])
        n, ages = sum(frame["queue"]), frame["waiting_ages"]
        if (tuple(map(len, ages)) != tuple(frame["queue"]) or any(
                any(age < 0 for age in group) or list(group) != sorted(group, reverse=True) for group in ages)
                or info["queue_limit"] != r * planner.batch_width
                or info["eligible"] != (0 < n <= info["queue_limit"])):
            return False
        if (tuple(info["observed_arrival_counts"]) != counts or info["rate_exposure_s"] != now
                or tuple(info["estimated_arrival_rates"]) != tuple(value / now if now else 0. for value in counts)
                or info["rate_estimation_coordinate_reads"] != len(counts)
                or info["rate_estimation_divisions"] != (len(counts) if now else 0)):
            return False
        k = info["candidate_count"]
        if (not 0 <= k <= m or info["rollout_option_probes"] != m * info["rollout_calls"]
                or info["planning_events"] != info["tagged_planning_events"] + info["background_planning_events"]
                or info["admitted_job_reads"] != info["flow_evaluations"] + info["background_admissions"]
                or info["tag_flag_reads"] != info["admitted_job_reads"]
                or info["background_seed_initializations"] > info["sample_count"]):
            return False
        rescheduled = info["background_planning_events"] - int(info["background_rollout_overflow"])
        positive_rates = sum(value > 0 for value in info["estimated_arrival_rates"])
        if (info["background_planning_rate_reads"] != len(counts) * info["background_seed_initializations"] + rescheduled
                or info["background_planning_random_draws"] != positive_rates * info["background_seed_initializations"] + rescheduled):
            return False
        calls_bound = samples * k * (n + cap - 1 + r * (ceil(planner.maximum_duration) + 1))
        if (info["rollout_calls"] > calls_bound or info["admitted_job_reads"] > samples * k * (n + cap)
                or info["flow_evaluations"] > k * (samples * n + info["tagged_planning_events"])):
            return False
        if not info["eligible"]:
            if info["root_option_probes"] != m or any(info[key] for key in EXTRA_WORK) or info["changed"] or k:
                return False
            continue
        if info["window_s"] != planner.maximum_duration or not 1 <= info["sample_count"] <= samples:
            return False
        if (info["planning_events"] > info["sample_count"] * (cap + 1)
                or info["background_planning_random_draws"] > info["planning_random_draws"]
                or info["background_planning_rate_reads"] > info["planning_rate_reads"]):
            return False
        if info["planning_overflow"]:
            if (info["changed"] or info["baseline_predicted_completion_s"] is not None
                    or info["chosen_predicted_completion_s"] is not None
                    or info["risk_feasible_candidates"] or info["risk_filtered_candidates"]):
                return False
            if info["background_rollout_overflow"]:
                if info["root_option_probes"] != 2 * m or not k or info["sample_count"] != samples:
                    return False
            elif info["root_option_probes"] != m or info["rollout_calls"] or k:
                return False
            continue
        if (info["background_rollout_overflow"] or info["root_option_probes"] != 2 * m
                or info["sample_count"] != samples or not 1 <= info["risk_feasible_candidates"] <= k
                or info["flow_evaluations"] != k * (samples * n + info["tagged_planning_events"])
                or info["risk_filtered_candidates"] + info["risk_feasible_candidates"] != k
                or info["chosen_predicted_completion_s"] > info["baseline_predicted_completion_s"]
                or info["chosen_predicted_squared_flow_s2"] > info["baseline_predicted_squared_flow_s2"]):
            return False
        if info["changed"] and not info["chosen_predicted_completion_s"] < info["baseline_predicted_completion_s"]:
            if frame_index not in certificates or not exact_certificate_matches(certificates[frame_index], frame):
                return False
    return True


def _work_matches_saved_frames(result):
    frames, work = result["frames"], result["rollout_work"]
    for field in ("root_option_probes", "candidate_count", "rollout_calls", "rollout_option_probes"):
        if work[field] != sum(frame["rollout"][field] for frame in frames):
            return False
    return (work["eligible_decisions"] == sum(frame["rollout"]["eligible"] for frame in frames)
        and work["changed_decisions"] == sum(frame["rollout"]["changed"] for frame in frames)
        and work["maximum_rollout_calls"] == max(frame["rollout"]["rollout_calls"] for frame in frames)
        and work["age_context_job_reads"] == sum(sum(frame["queue"]) for frame in frames)
        and work["observed_arrival_counter_updates"] == sum(frames[-1]["observed_arrival_counts"])
        and work["observed_arrival_context_coordinate_reads"] == sum(len(frame["queue"]) for frame in frames))


def validate_saved_unit(directory):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    unit = protocol["units"][462]
    require(not path(directory, "_unit_462.json").exists(), "original failed-check PASS row must remain absent")
    diagnostic_file = path(directory, "_runner_unit462_exact_fraction_proof.json")
    diagnostic = read(diagnostic_file)
    trajectory_file = path(directory, "_unit_462_trajectory.json.gz")
    saved = read(trajectory_file)
    require(saved["registered_at_utc"] == diagnostic["registered_at_utc"] == protocol["registered_at_utc"]
        and saved["unit"] == diagnostic["unit"] == unit and diagnostic["unit_index"] == 462
        and diagnostic["trajectory_file"] == trajectory_file.name,
        "exact proof and saved trajectory must match the complete registered unit and epoch")
    require(diagnostic["exact_proof_passed"] is True and diagnostic["read_exit_code"] == 0
        and diagnostic["new_simulations"] == diagnostic["policy_calls"] == 0
        and diagnostic["rng_reconstruction_only"] is True and diagnostic["diagnostic_source_literal"],
        "authentic independently reconstructed exact proof required")
    fixture = read(Path(directory) / unit["fixture_file"])
    tape = next(tape for tape in protocol["tapes"] if tape["tape_id"] == unit["tape_id"])
    model_record = next(model for model in protocol["models"] if model["node"] == unit["node"])
    require(fixture["registered_at_utc"] == protocol["source_execution_epoch"]
        and fixture["tape"] == tape and fixture["model"] == model_record, "source fixture differs")
    model, = load_models([model_record])
    result = saved["result"]
    frames = result["frames"]
    require(births(result) == [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])],
        "saved actual births differ from registered literal arrivals")
    certificates = {item["frame_index"]: item for item in diagnostic["certificates"]}
    targets = {index for index, frame in enumerate(frames) if frame["rollout"]["changed"]
        and not frame["rollout"]["chosen_predicted_completion_s"] < frame["rollout"]["baseline_predicted_completion_s"]}
    require(targets == set(certificates) and len(certificates) == diagnostic["exact_target_frames_verified"] == 1
        and targets == {16}, "every actual lost-strictness frame must have its exact certificate")
    require(diagnostic["prefix_frames_verified"] == len(frames)
        and diagnostic["eligible_frames_verified"] == sum(frame["rollout"]["eligible"] for frame in frames)
        and diagnostic["tagged_events_verified"] == sum(frame["rollout"]["tagged_planning_events"] for frame in frames)
        and diagnostic["prefix_draws_verified"] == sum(frame["rollout"]["planning_random_draws"]
            - frame["rollout"]["background_planning_random_draws"] for frame in frames)
        and diagnostic["prefix_rate_reads_verified"] == sum(frame["rollout"]["planning_rate_reads"]
            - frame["rollout"]["background_planning_rate_reads"] for frame in frames),
        "independent reconstructed prefix counters differ from saved frames")
    planner = BackgroundCompetitionContinuousLaneRollout(model.options, unit["planning_seed"], 8, 128)
    require(corrected_rollout_checks(result, model, protocol, planner, certificates),
        "unchanged causal/budget/selection predicates or exact certificate failed")
    require(result["decision_count"] == result["frame_count"] == len(frames)
        and all(result["invariants"].values()) and _work_matches_saved_frames(result),
        "saved engine invariants or aggregate work differ from frame records")
    work = background_work(result, planner, 900)
    work["controller_constructors"] = 1
    source_file = str(Path(__file__).resolve().relative_to(ROOT))
    validation_utc = datetime.now(timezone.utc).isoformat()
    proof_file = path(directory, "_validation_unit_462.json")
    proof = dict(diagnostic, original_checker_passed=False, corrected_checker_passed=True,
        exact_strict_improvement_proved=True, frozen_sources_unchanged=True,
        all_other_frozen_checks_certified=True, literal_births_and_saved_work_valid=True,
        validation_registered_at_utc=validation_utc, validator_source_file=source_file,
        validator_source_literal=(ROOT / source_file).read_text(),
        validation_work=dict(saved_frames_read=len(frames), rational_certificates_consumed=len(certificates),
            new_simulations=0, new_policy_calls=0, new_scoring_calls=0,
            wall_s=perf_counter() - started, cpu_s=process_time() - cpu))
    save(proof_file, proof)
    analysis_validation = dict(original_frozen_checker_passed=False, exact_checker_passed=True,
        exact_proof_file=proof_file.name, analysis_registered_at_utc=validation_utc,
        validator_source_file=source_file, actual_lost_float_strictness_frames=[16],
        mathematical_change="None; consume the independently proved strict Fraction comparison lost in float metadata.",
        new_simulations=0, original_PASS_row_created=False)
    analysis_row = dict(**unit, registered_at_utc=protocol["registered_at_utc"],
        execution_checks_passed=True, trajectory_file=trajectory_file.name, work=work,
        computation=None, computation_unavailable_reason=(
            "Original trajectory was saved before checker failure; original runtime timing was not persisted."),
        analysis_validation=analysis_validation)
    save(path(directory, "_analysis_unit_462.json"), analysis_row)
    print('{"unit_index":462,"exact_checker_passed":true,"all_other_frozen_checks_certified":true,"new_simulations":0,"original_PASS_row_created":false}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "md/experiment_artifacts")
    parser.add_argument("--validate", action="store_true", required=True)
    args = parser.parse_args()
    validate_saved_unit(args.artifact_dir)


if __name__ == "__main__":
    main()
