#!/usr/bin/env python3
"""Draw a four-quadrant SOTA-style policy-semantics comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

POLICY_LABELS = {
    "throughput_table_goodput": "Gavel/Pollux/Sia goodput",
    "delay_oracle": "SRPT/Gittins delay",
    "interference_guard": "IADeep/Salus guard",
    "finish_time_fairness": "Gavel finish-time",
    "resource_adaptive_goodput": "Sia/Pollux adaptive",
    "packing_guard": "Salus/IADeep packing",
    "quadrant_composite": "best quadrant composite",
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
    enter the four-quadrant figure; controlled migration counterfactuals are
    reported separately.
    """

    hardware_ids = {
        str(row.get("scenario_id") or "")
        for row in report.get("scenarios") or []
        if row.get("scenario_kind") == "hardware_local"
    }
    by_quad: dict[str, dict[str, list[tuple[float, float]]]] = {
        key: defaultdict(list) for key in QUADRANTS
    }
    for row in report.get("pareto_rows") or []:
        if str(row.get("scenario_id") or "") not in hardware_ids:
            continue
        if row.get("migration_mode") != "without_migration":
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
    report = load_report(artifact)
    points = quadrant_points(report)
    best_by_quadrant = scenario_best_policies(report)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.20, wspace=0.12, hspace=0.25)
    flat = axes.ravel()
    all_values = [1.0]
    for quadrant in QUADRANTS:
        for point in points.get(quadrant, {}).values():
            all_values.extend([point["x"], point["y"]])
    lo = min(0.995, min(all_values) - 0.006)
    hi = max(1.025, max(all_values) + 0.010)

    handles = {}
    for ax, quadrant in zip(flat, ("q00", "q01", "q10", "q11")):
        ax.axvline(1.0, color="#9aa4b2", lw=0.7, ls="--", zorder=0)
        ax.axhline(1.0, color="#9aa4b2", lw=0.7, ls="--", zorder=0)
        ax.scatter(1.0, 1.0, marker="*", s=88, color="#C62828", edgecolor="white", linewidth=0.55, zorder=5)
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
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, color="#e7ebf0", lw=0.45)
        if best_policy:
            ax.text(
                0.97,
                0.94,
                "best SOTA\n" + POLICY_LABELS.get(best_policy, best_policy),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.6,
                color="#1f2937",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cfd6df", linewidth=0.5),
            )

    axes[1, 0].set_xlabel("Mean-flow cost (SOTA / Scheduleurm)")
    axes[1, 1].set_xlabel("Mean-flow cost (SOTA / Scheduleurm)")
    axes[0, 0].set_ylabel("Makespan cost (SOTA / Scheduleurm)")
    axes[1, 0].set_ylabel("Makespan cost (SOTA / Scheduleurm)")
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
        "Four-quadrant measured-cache policy-semantics comparison (lower is better; Scheduleurm = 1)",
        ha="left",
        va="top",
        fontsize=7.7,
        fontweight="bold",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output}.svg", bbox_inches="tight")
    fig.savefig(f"{output}.pdf", bbox_inches="tight")
    fig.savefig(f"{output}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{output}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    data_path = output.with_name(output.name + "_data.json")
    data_path.write_text(
        json.dumps(
            {
                "source_artifact": str(Path(artifact).resolve()),
                "normalization": "baseline_policy_cost_divided_by_scheduleurm_cost",
                "included_scenario_kind": "hardware_local",
                "included_migration_mode": "without_migration",
                "aggregation": "geometric_mean_across_registered_scenarios_and_traces",
                "points": points,
                "best_sota_by_quadrant": best_by_quadrant,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _best_policy(points: dict[str, dict[str, float]]) -> str:
    if not points:
        return ""
    return min(points, key=lambda p: (max(points[p]["x"], points[p]["y"]), points[p]["x"] + points[p]["y"], p))


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
