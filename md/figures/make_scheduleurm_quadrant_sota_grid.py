#!/usr/bin/env python3
"""Draw a four-quadrant comparison on jointly completed replay traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from algorithm.experiments.unified_hardware_or_replay import _common_complete_runs
from algorithm.experiments.unified_hardware_paper_results import SOTA_POLICY_LABELS


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "unified_hardware_or_replay_20260809.json"
)
OUT = REPO_ROOT / "md" / "figures" / "scheduleurm_quadrant_sota_grid"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


QUADRANTS = {
    "q00": ("q00_light_control",),
    "q01": (
        "q01_gpu_bound_compute",
        "q01_gpu_bound_cnn_resnet50",
        "q01_gpu_bound_llm_inference",
        "q01_gpu_model_portfolio",
    ),
    "q10": ("q10_cpu_host_bound",),
    "q11": ("q11_cpu_gpu_coupled",),
}

QUADRANT_TITLES = {
    "q00": "q00 low CPU / low GPU",
    "q01": "q01 low CPU / high GPU",
    "q10": "q10 high CPU / low GPU",
    "q11": "q11 high CPU / high GPU",
}

POLICY_COLORS = {
    "throughput_table_goodput": "#4C78A8",
    "delay_oracle": "#F58518",
    "interference_guard": "#54A24B",
    "finish_time_fairness": "#B279A2",
    "resource_adaptive_goodput": "#72B7B2",
    "packing_guard": "#E45756",
    "quadrant_composite": "#8C6D31",
}

POLICY_MARKERS = {
    "throughput_table_goodput": "o",
    "delay_oracle": "s",
    "interference_guard": "^",
    "finish_time_fairness": "D",
    "resource_adaptive_goodput": "P",
    "packing_guard": "v",
    "quadrant_composite": "X",
}

POLICY_ID_TO_BASELINE = {
    "sota_gavel_pollux_sia_table_goodput": "throughput_table_goodput",
    "sota_srpt_gittins_mean_flow_oracle": "delay_oracle",
    "sota_iadeep_salus_interference_guard": "interference_guard",
    "sota_gavel_finish_time_fairness": "finish_time_fairness",
    "sota_sia_pollux_resource_adaptive": "resource_adaptive_goodput",
    "sota_salus_iadeep_packing_guard": "packing_guard",
    "sota_quadrant_composite": "quadrant_composite",
}
POLICY_LABELS = {
    baseline: SOTA_POLICY_LABELS[policy_id]
    for policy_id, baseline in POLICY_ID_TO_BASELINE.items()
}


def load_report(path: Path = ARTIFACT) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def quadrant_points(report: dict) -> dict[str, dict[str, dict[str, float]]]:
    if report.get("gate") == "unified_hardware_or_replay" or report.get("pareto_rows"):
        return unified_quadrant_points(report)
    return legacy_quadrant_points(report)


def unified_quadrant_points(report: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate baseline/Scheduleurm ratios from the final replay artifact.

    The replay stores ``ours / baseline``.  The figure contract is the inverse:
    each SOTA-style point is ``baseline / ours`` and Scheduleurm is fixed at
    ``(1, 1)``.  Only hardware-local scenarios and the without-migration view
    enter the four-quadrant figure. Every policy is evaluated on the same
    traces, conditional on completion by all registered policies.
    """

    hardware_ids = {
        str(row.get("scenario_id") or "")
        for row in report.get("scenarios") or []
        if row.get("scenario_kind") == "hardware_local"
    }
    complete_trace_keys = {
        (row["scenario_id"], row["trace_id"], row["migration_mode"])
        for row in _common_complete_runs(report["runs"])
    }
    by_quad: dict[str, dict[str, list[tuple[float, float]]]] = {
        key: defaultdict(list) for key in QUADRANTS
    }
    for row in report.get("pareto_rows") or []:
        if str(row.get("scenario_id") or "") not in hardware_ids:
            continue
        if row.get("migration_mode") != "without_migration":
            continue
        if (row["scenario_id"], row["trace_id"], row["migration_mode"]) not in complete_trace_keys:
            continue
        quadrant = str(row.get("quadrant") or "")
        if quadrant not in by_quad:
            continue
        for comparison in row.get("comparisons") or []:
            policy_id = str(comparison.get("baseline_policy") or "")
            policy = POLICY_ID_TO_BASELINE.get(policy_id)
            if not policy:
                continue
            ours_to_baseline_flow = _positive(
                comparison.get("ours_to_baseline_mean_flow_ratio")
            )
            ours_to_baseline_makespan = _positive(
                comparison.get("ours_to_baseline_makespan_ratio")
            )
            if ours_to_baseline_flow <= 0.0 or ours_to_baseline_makespan <= 0.0:
                continue
            by_quad[quadrant][policy].append(
                (1.0 / ours_to_baseline_flow, 1.0 / ours_to_baseline_makespan)
            )
    return _aggregate_points(by_quad)


def figure_population(report: dict) -> dict:
    """Record the shared completion condition and the traces it excludes."""
    hardware = {
        str(row["scenario_id"]): str(row["quadrant"])
        for row in report["scenarios"]
        if row.get("scenario_kind") == "hardware_local"
    }
    registered = [
        row for row in report["runs"]
        if row["scenario_id"] in hardware
        and row["migration_mode"] == "without_migration"
    ]
    registered_keys = {(row["scenario_id"], row["trace_id"]) for row in registered}
    complete_keys = {
        (row["scenario_id"], row["trace_id"])
        for row in _common_complete_runs(registered)
    }
    excluded_keys = registered_keys - complete_keys
    return {
        "aggregation_population": "all_registered_policies_complete_on_same_trace",
        "registered_hardware_trace_count": len(registered_keys),
        "included_complete_trace_count": len(complete_keys),
        "excluded_incomplete_trace_count": len(excluded_keys),
        "excluded_incomplete_trace_count_by_quadrant": {
            quadrant: sum(hardware[scenario_id] == quadrant for scenario_id, _ in excluded_keys)
            for quadrant in QUADRANTS
        },
        "registered_hardware_scenario_ids": sorted(hardware),
        "included_hardware_scenario_ids": sorted({scenario_id for scenario_id, _ in complete_keys}),
    }


def legacy_quadrant_points(report: dict) -> dict[str, dict[str, dict[str, float]]]:
    by_quad: dict[str, dict[str, list[tuple[float, float]]]] = {
        key: defaultdict(list) for key in QUADRANTS
    }
    taskset_to_quad = {
        taskset: quadrant
        for quadrant, tasksets in QUADRANTS.items()
        for taskset in tasksets
    }
    for row in report.get("policy_matrix") or []:
        if row.get("policy_family") != "sota_style":
            continue
        quadrant = taskset_to_quad.get(str(row.get("taskset") or ""))
        if not quadrant:
            continue
        policy = str(row.get("baseline_name") or row.get("policy") or "")
        x = _positive(row.get("candidate_vs_policy_mean_flow"))
        y = _positive(row.get("candidate_vs_policy_makespan"))
        if x > 0.0 and y > 0.0:
            by_quad[quadrant][policy].append((x, y))
    return _aggregate_points(by_quad)


def _aggregate_points(
    by_quad: dict[str, dict[str, list[tuple[float, float]]]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for quadrant, policy_rows in by_quad.items():
        out[quadrant] = {}
        for policy, pairs in policy_rows.items():
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            out[quadrant][policy] = {
                "x": _geomean(xs),
                "y": _geomean(ys),
                "scenario_count": len(pairs),
            }
    return out


def draw(*, artifact: Path = ARTIFACT, output: Path = OUT) -> None:
    artifact = Path(artifact).resolve()
    artifact_raw = artifact.read_bytes()
    report = json.loads(artifact_raw.decode("utf-8"))
    _validate_final_replay(report)
    points = quadrant_points(report)
    population = figure_population(report)
    best_by_quadrant = scenario_best_policies(report)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.65))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.20, wspace=0.26, hspace=0.32)
    flat = axes.ravel()
    axis_ranges = {}

    handles = {}
    for ax, quadrant in zip(flat, ("q00", "q01", "q10", "q11")):
        ax.axvline(1.0, color="#9aa4b2", lw=0.7, ls="--", zorder=0)
        ax.axhline(1.0, color="#9aa4b2", lw=0.7, ls="--", zorder=0)
        if points.get(quadrant):
            ax.scatter(1.0, 1.0, marker="*", s=88, color="#C62828", edgecolor="white", linewidth=0.55, zorder=5)
        else:
            ax.text(0.5, 0.5, "No common completed traces", transform=ax.transAxes, ha="center")
        best_policy = best_by_quadrant.get(quadrant) or _best_policy(points.get(quadrant, {}))
        for policy, point in sorted(points.get(quadrant, {}).items()):
            label = POLICY_LABELS.get(policy, policy)
            color = POLICY_COLORS.get(policy, "#666666")
            marker = POLICY_MARKERS.get(policy, "o")
            h = ax.scatter(
                point["x"],
                point["y"],
                s=34,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.96,
                zorder=3,
            )
            handles[label] = h
        ax.set_title(QUADRANT_TITLES[quadrant], loc="left", fontweight="bold", pad=3)
        x_limits = _padded_limits([1.0] + [point["x"] for point in points.get(quadrant, {}).values()])
        y_limits = _padded_limits([1.0] + [point["y"] for point in points.get(quadrant, {}).values()])
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
        ax.ticklabel_format(useOffset=False)
        axis_ranges[quadrant] = {"x": x_limits, "y": y_limits}
        ax.grid(True, color="#e7ebf0", lw=0.45)
        if best_policy:
            ax.text(
                0.03,
                0.94,
                "nearest rule\n" + POLICY_LABELS.get(best_policy, best_policy),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.6,
                color="#1f2937",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cfd6df", linewidth=0.5),
            )
        if quadrant == "q00":
            near_points = {
                policy: point for policy, point in points.get(quadrant, {}).items()
                if abs(point["x"] - 1.0) <= 0.1 and abs(point["y"] - 1.0) <= 0.1
            }
            inset = ax.inset_axes([0.14, 0.28, 0.44, 0.43])
            inset.axvline(1.0, color="#9aa4b2", lw=0.6, ls="--")
            inset.axhline(1.0, color="#9aa4b2", lw=0.6, ls="--")
            inset.scatter(1.0, 1.0, marker="*", s=54, color="#C62828", edgecolor="white", linewidth=0.4, zorder=5)
            for policy, point in sorted(near_points.items()):
                inset.scatter(
                    point["x"], point["y"], s=22,
                    color=POLICY_COLORS[policy], marker=POLICY_MARKERS[policy],
                    edgecolor="white", linewidth=0.3, zorder=3,
                )
            inset.set_xlim(*_padded_limits([1.0] + [point["x"] for point in near_points.values()]))
            inset.set_ylim(*_padded_limits([1.0] + [point["y"] for point in near_points.values()]))
            inset.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            inset.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            inset.ticklabel_format(useOffset=False)
            inset.tick_params(labelsize=4.8, pad=1, length=2)
            inset.set_title("Near (1, 1)", fontsize=5.5, loc="left", pad=2)
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("#9aa4b2")
                spine.set_linewidth(0.5)
            axis_ranges[quadrant]["inset_x"] = inset.get_xlim()
            axis_ranges[quadrant]["inset_y"] = inset.get_ylim()

    axes[1, 0].set_xlabel("Mean-flow cost (baseline / Scheduleurm)")
    axes[1, 1].set_xlabel("Mean-flow cost (baseline / Scheduleurm)")
    axes[0, 0].set_ylabel("Makespan cost (baseline / Scheduleurm)")
    axes[1, 0].set_ylabel("Makespan cost (baseline / Scheduleurm)")
    scheduleurm_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor="#C62828",
        markeredgecolor="white",
        markersize=9,
        label="Scheduleurm",
    )
    legend_labels = ["Scheduleurm"] + list(handles)
    fig.legend(
        [scheduleurm_handle] + [handles[label] for label in legend_labels[1:]],
        legend_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=6.2,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.text(
        0.02,
        0.965,
        "Measured-cache policy comparison",
        ha="left",
        va="top",
        fontsize=7.7,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.925,
        f"Common completed traces: {population['included_complete_trace_count']}; "
        f"incomplete traces excluded: {population['excluded_incomplete_trace_count']}\n"
        "Geometric-mean cost; lower is better. Axes vary by panel; q00 inset expands the near-unity group.",
        ha="left",
        va="top",
        fontsize=6.6,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = Path(f"{output}.svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    fig.savefig(f"{output}.pdf", bbox_inches="tight")
    fig.savefig(f"{output}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{output}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    data_path = output.with_name(output.name + "_data.json")
    data_path.write_text(
        json.dumps(
            {
                "source_artifact": str(artifact),
                "source_artifact_sha256": hashlib.sha256(artifact_raw).hexdigest(),
                "source_gate": report["gate"],
                "source_schema_version": report["schema_version"],
                "source_run_count": int(report["run_count"]),
                "source_policy_count": int(report["policy_count"]),
                "normalization": "baseline_policy_cost_divided_by_scheduleurm_cost",
                "included_scenario_kind": "hardware_local",
                "included_migration_mode": "without_migration",
                "aggregation": "geometric_mean_across_common_complete_traces",
                **population,
                "points": points,
                "axis_ranges": axis_ranges,
                "best_sota_by_quadrant": best_by_quadrant,
                "nearest_sota_definition": "minimum maximum aggregate cost ratio; ties broken by sum, nonduplicate label, then policy id",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_final_replay(report: dict) -> None:
    expected = {
        "gate": "unified_hardware_or_replay",
        "schema_version": 3,
        "status": "PASS",
        "pass": True,
        "comparison_kind": "same-cache_policy-semantics",
        "full_stack_external_binary_comparison": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"figure source {key} expected {value!r}, got {report.get(key)!r}"
            )
    if not report.get("pareto_rows") or not report.get("scenarios"):
        raise ValueError("figure source lacks scenarios or Pareto rows")
    if int(report.get("run_count") or 0) != len(report.get("runs") or []):
        raise ValueError("figure source run_count does not match run matrix")


def _best_policy(points: dict[str, dict[str, float]]) -> str:
    if not points:
        return ""
    return min(points, key=lambda p: (
        max(points[p]["x"], points[p]["y"]),
        points[p]["x"] + points[p]["y"],
        "(duplicate)" in POLICY_LABELS.get(p, p),
        p,
    ))


def _padded_limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    margin = max(0.001, 0.12 * (high - low))
    return low - margin, high + margin


def scenario_best_policies(report: dict) -> dict[str, str]:
    if report.get("gate") == "unified_hardware_or_replay" or report.get("pareto_rows"):
        points = unified_quadrant_points(report)
        return {
            quadrant: _best_policy(policy_points)
            for quadrant, policy_points in points.items()
            if policy_points
        }

    from collections import Counter

    taskset_to_quad = {
        taskset: quadrant
        for quadrant, tasksets in QUADRANTS.items()
        for taskset in tasksets
    }
    counts = {quadrant: Counter() for quadrant in QUADRANTS}
    for row in report.get("scenarios") or []:
        quadrant = taskset_to_quad.get(str(row.get("taskset") or ""))
        if not quadrant:
            continue
        for field in ("best_sota_makespan_policy", "best_sota_mean_flow_policy"):
            baseline = POLICY_ID_TO_BASELINE.get(str(row.get(field) or ""))
            if baseline:
                counts[quadrant][baseline] += 1
    out: dict[str, str] = {}
    for quadrant, counter in counts.items():
        if counter:
            out[quadrant] = counter.most_common(1)[0][0]
    return out


def _geomean(values: list[float]) -> float:
    clean = [max(1e-12, float(v)) for v in values if float(v) > 0.0]
    if not clean:
        return 0.0
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def _positive(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out > 0.0 and math.isfinite(out) else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--output", type=Path, default=OUT)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    draw(artifact=args.artifact, output=args.output)
