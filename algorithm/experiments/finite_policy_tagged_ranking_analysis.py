"""Decompose completed fixed tagged rankings without executing any policy."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

STEM = "aligned_hardware_finite_policy_tagged_ranking_20261003"
CROSS = "aligned_hardware_persistent_action_controller_cross_20261002"
FIRST = "aligned_hardware_persistent_first_divergence_20261002"
LABELS = ("CNN", "matmul", "LLM")


def read(path):
    return json.loads(path.read_text())


def exact(value):
    return Fraction(*value)


def encode(value):
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def extract_units(directory):
    """Read server artifacts and return only small paired-sample statistics."""
    protocol = read(directory / (STEM + "_protocol.json"))
    rows = []
    for assignment in protocol["assignments"]:
        row = read(directory / (STEM + f"_shard_{assignment['index']:02d}.json"))
        assert row["registered_at_utc"] == protocol["registered_at_utc"]
        assert row["assignment"] == assignment and row["execution_checks_passed"]
        result = row["result"]
        rows.append(dict(assignment=assignment, window_s=result["environment"]["window_s"],
            tagged_population=len(result["tagged_population"]),
            scores={method: {**{key: result[method][key] for key in ("cost", "squared", "complete", "overflow")},
                "sources": result[method]["population"]["sources"]} for method in ("approx", "free")}))
    return dict(registered_at_utc=protocol["registered_at_utc"], rows=rows)


def component(record, method, field, source=None, coordinate=None):
    values = []
    for role in ("parent", "native"):
        population = record["root_scores"][role]["scores"][method]["population_totals"]
        if coordinate is not None:
            population = population["per_coordinate"][coordinate]
        if source is not None:
            population = population["sources"][source]
        values.append(exact(population[field]))
    return (values[0] - values[1]) / 8


def decomposition(record, field):
    def values(source=None, coordinate=None):
        a, f = (component(record, method, field, source, coordinate) for method in ("approx", "free"))
        return dict(approx=a, free=f, shift=f - a)
    total = values()
    sources = {source: values(source) for source in ("visible", "prefix")}
    coordinates = [dict(coordinate=i, label=label, **values(coordinate=i),
        sources={source: values(source, i) for source in ("visible", "prefix")})
        for i, label in enumerate(LABELS)]
    # Detect attribution/normalization mistakes before interpreting a component.
    for method in ("approx", "free", "shift"):
        assert sum(source[method] for source in sources.values()) == total[method]
        assert sum(coordinate[method] for coordinate in coordinates) == total[method]
        for coordinate in coordinates:
            assert sum(source[method] for source in coordinate["sources"].values()) == coordinate[method]
    return dict(total=total, sources=sources, coordinates=coordinates)


def sample_diagnostics(record, unit_rows):
    pairs = {(r["assignment"]["role"], r["assignment"]["sample_index"]): r for r in unit_rows}
    assert len(pairs) == 16
    result = {}
    for method in ("approx", "free"):
        deltas = []
        for sample in range(8):
            p, n = (pairs[role, sample]["scores"][method] for role in ("parent", "native"))
            assert all(value["complete"] and not value["overflow"] for value in (p, n))
            deltas.append(exact(p["cost"]) - exact(n["cost"]))
        assert sum(deltas) / 8 == component(record, method, "cost")
        leave_one_out = [(sum(deltas) - delta) / 7 for delta in deltas]
        result[method] = dict(parent_minus_native_S1_by_sample=deltas,
            parent_preferred_samples=sum(delta < 0 for delta in deltas),
            native_preferred_samples=sum(delta > 0 for delta in deltas),
            tied_samples=sum(delta == 0 for delta in deltas),
            leave_one_out_mean_range=[min(leave_one_out), max(leave_one_out)],
            pooled_per_tagged_job_delta=sum(deltas) / sum(pairs["parent", i]["tagged_population"] for i in range(8)))
    result["strict_sample_reversals"] = sum(
        a * f < 0 for a, f in zip(result["approx"]["parent_minus_native_S1_by_sample"],
            result["free"]["parent_minus_native_S1_by_sample"], strict=True))
    return result


def analyze(directory):
    summary = read(directory / (STEM + "_summary.json"))
    cross = read(directory / (CROSS + "_summary.json"))
    first = read(directory / (FIRST + "_summary.json"))
    compact = read(directory / (STEM + "_compact_samples.json"))
    assert summary["execution_checks_passed"] and cross["execution_checks_passed"]
    assert compact["registered_at_utc"] == summary["registered_at_utc"]
    actual = {r["tape_id"]: r for r in cross["records"]}
    anchors = {r["tape_id"]: r["first_divergence"] for r in first["records"]}
    records = []
    for record in summary["records"]:
        target = actual[record["tape_id"]]
        assert all(record[key] == target[key] for key in ("seed", "frame_index", "parent_root", "native_root"))
        units = [r for r in compact["rows"] if r["assignment"]["tape_id"] == record["tape_id"]]
        window, = {r["window_s"] for r in units}
        anchor = anchors[record["tape_id"]]
        assert anchor["frame_index"] == record["frame_index"]
        s1, s2 = decomposition(record, "cost"), decomposition(record, "squared")
        for method in ("approx", "free"):
            assert s1["total"][method] == exact(record["comparisons"][method]["exact_parent_minus_native_mean"]["S1"])
            assert s2["total"][method] == exact(record["comparisons"][method]["exact_parent_minus_native_mean"]["S2"])
        result = dict(tape_id=record["tape_id"], seed=record["seed"], frame_index=record["frame_index"],
            parent_root=record["parent_root"], native_root=record["native_root"],
            anchor_start=anchor["start"], planning_window_s=window,
            tagged_birth_upper_bound=anchor["start"] + window,
            scored_births_before_actual_cohort=anchor["start"] + window < 300,
            S1=s1, S2=s2, samples=sample_diagnostics(record, units),
            prefix_share_of_S1_shift=s1["sources"]["prefix"]["shift"] / s1["total"]["shift"],
            finite_actual_birth_cohort={**{key: target["root_effects"]["finite"][key] for key in
                ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished", "delta_1200_unfinished", "per_coordinate_deltas")},
                "population": target["cohort_population"]})
        records.append(result)
        print(json.dumps(dict(seed=record["seed"], birth_upper_bound=result["tagged_birth_upper_bound"],
            source_S1={source: {key: float(value) for key, value in delta.items()}
                for source, delta in s1["sources"].items()},
            sample_preferences={method: {key: value for key, value in result["samples"][method].items()
                if key.endswith("samples")} for method in ("approx", "free")},
            leave_one_out={method: list(map(float, result["samples"][method]["leave_one_out_mean_range"]))
                for method in ("approx", "free")}), ensure_ascii=False))
    output = dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_registered_at_utc=summary["registered_at_utc"], arithmetic_checks_passed=True,
        source_files=[STEM + "_summary.json", STEM + "_compact_samples.json", CROSS + "_summary.json", FIRST + "_summary.json"],
        new_policy_executions=0, full24_decision=summary["full24_decision"], performance_gain_established=False,
        interpretation="Exact fixed8 score attribution; sums are per planning path, not per job. Sample signs and leave-one-out ranges describe only these reused8 paths and are not independent confirmation. Visible/prefix splits retain competition from both populations; they are not arrival-removal interventions. Real[300,900] outcomes describe different births, so no forecast-error subtraction or fitted discount is justified.",
        records=records)
    destination = directory / (STEM + "_analysis.json")
    destination.write_text(json.dumps(encode(output), ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("md/experiment_artifacts"))
    parser.add_argument("--extract-units", action="store_true")
    args = parser.parse_args()
    if args.extract_units:
        print(json.dumps(extract_units(args.artifact_dir), separators=(",", ":")))
    else:
        analyze(args.artifact_dir)


if __name__ == "__main__":
    main()
