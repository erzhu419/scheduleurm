"""Extract measured phase repeats for the unchanged BG native action roster.

Dedicated samples retain each accepted wave's mean-child phase. Loaded samples
retain target phase and resident overlap rate from the same measured row, then
combine them with independently measured empty resident-p1 phases. Their
Cartesian product is an explicit composite-model assumption, not additional
physical repetitions. Only gate-listed artifacts are read; no filename search
or measured-performance filtering selects a wave.
"""
import argparse
from copy import deepcopy
import json
from math import fsum, isclose, isfinite
from pathlib import Path

from algorithm.experiments.aligned_hardware_frame_model import DEFAULT_CACHE, DEFAULT_LOADED_LEDGER, WORKLOADS
from algorithm.experiments.aligned_hardware_physical_catalogue_models import load_catalogues
from algorithm.experiments.unified_hardware_or_replay import _record_action_id, _validate_cache


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_BG_empirical_service_samples_20261004.json"
PHASE_FIELDS = ("startup_overhead_s", "completion_unit_s", "terminal_overhead_s")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def phase(model):
    values = {key: float(model[key]) for key in PHASE_FIELDS}
    require(model["completion_model_ready"] is True
        and all(isfinite(value) and value >= 0 for value in values.values())
        and values["completion_unit_s"] > 0, "accepted natural phase model required")
    return values


def empty_phase(payload, certificate):
    require(payload["pass"] is True and payload["node"] == certificate["node"], "empty source gate/node mismatch")
    rows = [row for row in payload["rows"] if all(row[key] == certificate[key]
        for key in ("workload_key", "workload_env", "profile"))]
    require(len(rows) == 1 and rows[0]["ready"] is True and not rows[0]["capacity_boundary"],
        "exact admitted empty action must be ready")
    row = rows[0]
    require(row["node_bucket"] == certificate["node_bucket"], "empty hardware bucket mismatch")
    children = row["summary"]["rows"]
    require(len(children) == certificate["total_task_count"], "empty child count differs from accepted action")
    phases = []
    for child in children:
        model = child["completion_model"]
        require(child["completion_model_ready"] is True and child["returncode"] == 0
            and float(model["total_units"]) == float(certificate["total_units_per_task"]),
            "empty child native size/natural completion mismatch")
        item = phase(model)
        require(isclose(item["startup_overhead_s"] + model["total_units"] * item["completion_unit_s"]
            + item["terminal_overhead_s"], model["total_wall_s"], rel_tol=1e-9, abs_tol=1e-9),
            "empty phase does not reconstruct measured natural completion")
        phases.append(item)
    return {**{key: fsum(item[key] for item in phases) / len(phases) for key in PHASE_FIELDS},
        "total_units_per_task": float(certificate["total_units_per_task"]), "child_count": len(children)}


def loaded_phase(payload, certificate):
    require(payload["pass"] is True and payload["node"] == certificate["node"], "loaded source gate/node mismatch")
    rows = [row for row in payload["rows"] if row["scenario_id"] == certificate["scenario_id"]]
    require(len(rows) == 1 and rows[0]["ready"] is True and not rows[0]["capacity_boundary"],
        "exact admitted loaded action must be ready")
    row = rows[0]
    require(all(row[key] == certificate[key] for key in
        ("target_workload_key", "resident_workload_key", "target_workload_env", "resident_workload_env", "resident_mix"))
        and row["post_colocation_count"] == certificate["profile"], "loaded resident/target pair mismatch")
    target, resident = row["target_completion_model"], row["resident_completion_model"]
    require(float(target["total_units"]) == certificate["target_measurement_total_units"]
        and float(resident["total_units"]) == certificate["resident_measurement_total_units"],
        "loaded measured native sizes differ")
    target_phase, resident_phase = phase(target), phase(resident)
    overlap = row["resident_overlap_service"]
    rate = float(overlap["conservative_rate_units_per_s"])
    require(overlap["ready"] is True and isfinite(rate) and rate > 0
        and overlap["unit"] == certificate["resident_service_unit"], "paired resident overlap service invalid")
    return dict(target_phase=target_phase, resident_phase=resident_phase,
        resident_overlap_service_units_per_s=rate,
        resident_overlap_service_source_field="resident_overlap_service.conservative_rate_units_per_s",
        target_measured_total_units=float(target["total_units"]),
        resident_measured_total_units=float(resident["total_units"]))


def source_components(gate_path, certificate, *, loaded=False):
    gate = read(gate_path)
    require(gate["pass"] is True and gate["status"] == "PASS" and gate["certificate_ready"] is True,
        "passing final source gate required")
    key_fields = ("scenario_id",) if loaded else ("workload_key", "workload_env", "profile")
    require(any(all(row[key] == certificate[key] for key in key_fields) for row in gate["certificate"]["rows"]),
        "action not admitted by final gate")
    artifacts = gate["source_artifacts"]
    # The v9 node007 gate replaces only transport-truncated DistilGPT2 p10.
    corrected = (not loaded and certificate["node"] == "node007"
                 and certificate["workload_key"] == "gpu_llm_distilgpt2" and certificate["profile"] == 10)
    if corrected:
        require(gate["correction_scope"] == "pre_registered_measurement_transport_integrity",
            "registered p10 transport correction required")
        artifacts = gate["correction_artifacts"]
    require(len(artifacts) == 13 and {item["wave"] for item in artifacts} == set(range(1, 14)),
        "all thirteen gate-listed measured repetitions required")
    result = []
    for artifact in sorted(artifacts, key=lambda item: item["wave"]):
        payload = read(artifact["path"])
        wave = artifact["wave"]
        require(payload["wave"] == wave and payload["split_role"] ==
            ("training" if wave <= 3 else "calibration" if wave <= 12 else "holdout"),
            "source wave/split identity differs")
        values = (loaded_phase if loaded else empty_phase)(payload, certificate)
        result.append(dict(wave=wave, split_role=payload["split_role"], source_path=artifact["path"],
            gate_path=str(gate_path), transport_corrected=corrected, **values))
    return result


def dedicated_sample(option, native_units, component):
    duration = component["startup_overhead_s"] + native_units * component["completion_unit_s"] + component["terminal_overhead_s"]
    offsets = [[duration] * count for count in option.requirements]
    return dict(empty_wave=component["wave"], duration=duration, effective_duration=max(1., duration),
        completion_offsets=offsets)


def composite_sample(option, target, resident, native_units, empty, loaded):
    target_phase = loaded["target_phase"]
    target_frame = target_phase["startup_overhead_s"] + native_units[target] * target_phase["completion_unit_s"] + target_phase["terminal_overhead_s"]
    remaining = max(0., native_units[resident] - loaded["resident_overlap_service_units_per_s"] * target_frame)
    drain = remaining * empty["completion_unit_s"]
    target_finish = empty["startup_overhead_s"] + target_frame
    duration = target_finish + drain + empty["terminal_overhead_s"]
    offsets = [[target_finish] if index == target else [duration] if index == resident else []
               for index in range(len(native_units))]
    require(tuple(map(len, offsets)) == option.requirements, "composite offsets must preserve the native action")
    return dict(empty_wave=empty["wave"], loaded_wave=loaded["wave"], duration=duration,
        effective_duration=max(1., duration), completion_offsets=offsets)


def summarize_action(option, training, holdout, components):
    require(training and all(tuple(map(len, sample["completion_offsets"])) == option.requirements
        and all(0 <= offset <= sample["effective_duration"] for group in sample["completion_offsets"] for offset in group)
        for sample in (*training, *holdout)), "native sample requirements/offsets mismatch")
    mean_duration = fsum(sample["effective_duration"] for sample in training) / len(training)
    mean_raw_duration = fsum(sample["duration"] for sample in training) / len(training)
    offsets = [[fsum(sample["completion_offsets"][coordinate][slot] for sample in training) / len(training)
        for slot in range(count)] for coordinate, count in enumerate(option.requirements)]
    return dict(action_id=option.action_id, loaded=option.loaded, requirements=option.requirements,
        source_action_ids=option.source_action_ids, catalogue_duration=option.duration,
        catalogue_completion_offsets=option.completion_offsets, components=components,
        training=training, holdout=holdout, training_sample_count=len(training),
        training_mean_duration=mean_duration, training_mean_raw_duration=mean_raw_duration,
        training_mean_completion_offsets=offsets,
        training_mean_duration_over_catalogue=mean_duration / max(1., option.duration))


def build_bank(cache_path=DEFAULT_CACHE, loaded_ledger_path=DEFAULT_LOADED_LEDGER):
    models = load_catalogues(cache_path, loaded_ledger_path=loaded_ledger_path)
    _, rows, _ = _validate_cache(read(cache_path))
    records = {_record_action_id(record): record for record in rows}
    ledger = {row["action_id"]: row for row in read(loaded_ledger_path)["actions"]}
    catalogues, mean_models, gates = [], [], {}
    for model in models:
        empty_components, actions = {}, []
        for option in model.options:
            if option.loaded:
                continue
            record = records[option.action_id]
            gate_path = Path(record.source.split(";", 1)[0])
            gate = gates.setdefault(str(gate_path), read(gate_path))
            certificate = next(row for row in gate["certificate"]["rows"] if all(row[key] == getattr(record, key)
                for key in ("workload_key", "workload_env", "profile")))
            coordinate = WORKLOADS.index(record.workload_key)
            require(certificate["total_units_per_task"] == model.scales[coordinate], "BG native job size changed")
            components = source_components(gate_path, certificate)
            empty_components[option.action_id] = components
            samples = [dedicated_sample(option, model.scales[coordinate], component) for component in components]
            actions.append(summarize_action(option, samples[:12], samples[12:], dict(empty=components)))
        for option in model.options:
            if not option.loaded:
                continue
            source = ledger[option.action_id]
            gate_path = Path(source["source"])
            gate = gates.setdefault(str(gate_path), read(gate_path))
            scenario = option.action_id.rsplit(":", 1)[1]
            certificate = next(row for row in gate["certificate"]["rows"] if row["scenario_id"] == scenario)
            loaded_components = source_components(gate_path, certificate, loaded=True)
            resident_components = empty_components[option.source_action_ids[1]]
            target, resident = (WORKLOADS.index(source[f"{role}_workload_key"]) for role in ("target", "resident"))
            require(certificate["target_canonical_total_units"] == model.scales[target]
                and certificate["resident_canonical_total_units"] == model.scales[resident], "loaded BG native sizes changed")
            training = [composite_sample(option, target, resident, model.scales, empty, loaded)
                for empty in resident_components[:12] for loaded in loaded_components[:12]]
            holdout = [composite_sample(option, target, resident, model.scales, resident_components[12], loaded_components[12])]
            actions.append(summarize_action(option, training, holdout,
                dict(resident_empty=resident_components, paired_loaded=loaded_components)))
        actions.sort(key=lambda action: action["action_id"])
        mean_model = deepcopy(model.snapshot())
        mean_model["model_scope"] = "empirical_pre_holdout_phase_mean_complete_job_renewal"
        by_id = {action["action_id"]: action for action in actions}
        for option in mean_model["options"]:
            action = by_id[option["action_id"]]
            option["duration"] = action["training_mean_duration"]
            option["completion_offsets"] = action["training_mean_completion_offsets"]
            option["phases"] = ()
        mean_models.append(mean_model)
        catalogues.append(dict(node=model.node, lanes=model.lanes, workloads=model.workloads,
            native_job_sizes=model.scales, actions=actions, excluded_actions=model.excluded_actions))
    samples = [sample for model in catalogues for action in model["actions"] for sample in (*action["training"], *action["holdout"])]
    source_gates = [dict(path=path, node=gate["node"], resource_state=gate["resource_state"],
        training_waves=gate["training_waves"], calibration_waves=gate["calibration_waves"], holdout_wave=gate["holdout_wave"],
        training_capacity_excluded_cells=gate.get("training_capacity_excluded_cells", []),
        correction_scope=gate.get("correction_scope"), claim_boundary=gate["claim_boundary"]) for path, gate in gates.items()]
    return dict(experiment="aligned_hardware_BG_empirical_service_samples_20261004",
        source_cache=str(cache_path), source_loaded_ledger=str(loaded_ledger_path),
        catalogues=catalogues, mean_model_snapshots=mean_models, source_gates=source_gates,
        coverage=dict(nodes=len(catalogues), actions=sum(len(model["actions"]) for model in catalogues),
            dedicated_actions=sum(not action["loaded"] for model in catalogues for action in model["actions"]),
            loaded_actions=sum(action["loaded"] for model in catalogues for action in model["actions"]),
            training_samples=sum(action["training_sample_count"] for model in catalogues for action in model["actions"]),
            minimum_raw_duration=min(sample["duration"] for sample in samples),
            duration_floor_active=any(sample["duration"] < 1 for sample in samples)),
        composition="Dedicated: accepted wave mean-child phase at the frozen native job size, all batch offsets at completion. Loaded: same-row target phase and conservative resident-overlap rate estimate, independently combined with empty resident-p1 initialization/drain/finalization. This is a phase-derived joint composite model;12x12 derived support points use12 empty plus12 loaded rounds, not144 measured complete actions. Component phases are never independently shuffled.",
        split="Gate-accepted waves1..12 form the pre-holdout empirical distribution; source training/calibration labels are retained. Wave13 components and derived comparison are separate and never enter training means.",
        mean_catalogue="Duration is E[max(1,D)] and each completion offset is its slotwise empirical mean. Mean service can differ from the legacy median/LCB catalogue; no normalization to legacy means or artificial jitter.",
        scope="Measured phase-model variability for the current native action roster. Canonical loaded full actions remain compositions; samples do not establish production stationarity, thermal-drift control, cross-campaign dependence or physical P90 guarantees. Source wave13 was already used for the historical gate holdout, not a new physical confirmation. No simulations, server downloads or new measurements.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve existing sample bank")
    args.output.write_text(json.dumps(build_bank(), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    print(json.dumps(dict(sample_bank=str(args.output), simulations=0)), flush=True)


if __name__ == "__main__":
    main()
