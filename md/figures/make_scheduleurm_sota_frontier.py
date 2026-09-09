#!/usr/bin/env python3
"""Draw the Scheduleurm measured-cache SOTA frontier figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "sota_candidate_union_gate_fresh_eta_full_v6_20260626.json"
)
OUT = REPO_ROOT / "md" / "figures" / "scheduleurm_sota_frontier"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


LABELS = {
    "legacy_fixed_caps": "Legacy caps",
    "scheduleurm_sota_union_online_pareto_slack": "Scheduleurm candidate",
    "scheduleurm_sota_union_adaptive_scalarized": "Action union: adaptive",
    "scheduleurm_sota_union_guarded_mean_flow": "Action union: guarded",
    "scheduleurm_sota_union_makespan": "Action union: makespan",
    "scheduleurm_sota_union_mean_flow": "Action union: mean-flow",
    "scheduleurm_sota_union_pareto_slack": "Action union: Pareto slack",
    "sota_gavel_finish_time_fairness": "Finish-time fairness",
    "sota_gavel_pollux_sia_table_goodput": "Throughput goodput",
    "sota_iadeep_salus_interference_guard": "Interference guard",
    "sota_quadrant_composite": "Quadrant composite",
    "sota_salus_iadeep_packing_guard": "Packing guard",
    "sota_sia_pollux_resource_adaptive": "Resource-adaptive",
    "sota_srpt_gittins_mean_flow_oracle": "Delay oracle",
}

SCENARIO_LABELS = {
    "q00_light_control": "q00",
    "q01_gpu_bound_compute": "q01-GPU",
    "q01_gpu_bound_cnn_resnet50": "q01-CNN",
    "q01_gpu_bound_llm_inference": "q01-LLM",
    "q01_gpu_model_portfolio": "q01-port",
    "q10_cpu_host_bound": "q10",
    "q11_cpu_gpu_coupled": "q11",
    "hybrid_research_portfolio": "mixed",
}

SCENARIO_COLORS = {
    "q00": "#5b6c84",
    "q01-GPU": "#4c78a8",
    "q01-CNN": "#5b8fd9",
    "q01-LLM": "#7aa6dc",
    "q01-port": "#93b7e3",
    "q10": "#8a6fb4",
    "q11": "#d08b5b",
    "mixed": "#5f9e73",
}


def load_report() -> dict:
    with ARTIFACT.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def aggregate_points(report: dict) -> list[dict]:
    rows = report["aggregate"]["policy_aggregate"]
    legacy = next(row for row in rows if row["policy"] == "legacy_fixed_caps")
    points = []
    for row in rows:
        points.append(
            {
                "policy": row["policy"],
                "label": LABELS.get(row["policy"], row["policy"]),
                "family": row["policy_family"],
                "makespan_speedup": legacy["sum_makespan_s"] / row["sum_makespan_s"],
                "mean_flow_speedup": legacy["job_weighted_mean_flow_s"]
                / row["job_weighted_mean_flow_s"],
            }
        )
    return points


def style_for(point: dict) -> dict:
    policy = point["policy"]
    family = point["family"]
    if policy == "scheduleurm_sota_union_online_pareto_slack":
        return {
            "marker": "*",
            "s": 170,
            "facecolor": "#c43c39",
            "edgecolor": "#7f1d1d",
            "linewidth": 0.9,
            "zorder": 6,
        }
    if family == "sota_style":
        return {
            "marker": "o",
            "s": 48,
            "facecolor": "#6f8fb9",
            "edgecolor": "#2f4f73",
            "linewidth": 0.75,
            "zorder": 4,
        }
    return {
        "marker": "D",
        "s": 42,
        "facecolor": "#e1b56b",
        "edgecolor": "#916a22",
        "linewidth": 0.75,
        "zorder": 5,
    }


def draw_aggregate_panel(ax: plt.Axes, points: list[dict]) -> None:
    ax.grid(True, color="#e8ebef", lw=0.55)

    plotted = [
        point
        for point in points
        if point["family"] == "sota_style"
        or point["policy"] == "scheduleurm_sota_union_online_pareto_slack"
    ]
    for point in plotted:
        st = style_for(point)
        ax.scatter(
            point["makespan_speedup"],
            point["mean_flow_speedup"],
            marker=st["marker"],
            s=st["s"],
            facecolor=st["facecolor"],
            edgecolor=st["edgecolor"],
            linewidth=st["linewidth"],
            zorder=st["zorder"],
        )

    candidate = next(p for p in points if p["policy"] == "scheduleurm_sota_union_online_pareto_slack")
    ax.annotate(
        "Scheduleurm\ncandidate",
        xy=(candidate["makespan_speedup"], candidate["mean_flow_speedup"]),
        xytext=(-12, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", lw=0.65, color="#7f1d1d"),
        ha="right",
        va="bottom",
        color="#7f1d1d",
        fontsize=7,
    )
    sota_points = [p for p in plotted if p["family"] == "sota_style"]
    if sota_points:
        mid_x = np.mean([p["makespan_speedup"] for p in sota_points])
        mid_y = np.mean([p["mean_flow_speedup"] for p in sota_points])
        ax.annotate(
            "SOTA-style\npolicies",
            xy=(mid_x, mid_y),
            xytext=(-8, -26),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#2f4f73"),
            ha="right",
            va="top",
            color="#2f4f73",
            fontsize=7,
        )

    ax.set_xlim(2.260, 2.294)
    ax.set_ylim(5.1500, 5.1573)
    ax.set_xlabel("Aggregate makespan speedup over legacy")
    ax.set_ylabel("Aggregate mean-flow speedup over legacy")
    ax.set_title("a  Aggregate policy frontier", loc="left", fontweight="bold")


def draw_scenario_panel(ax: plt.Axes, scenarios: list[dict]) -> None:
    ax.axhline(1.0, color="#c7ccd1", lw=0.8, ls="--")
    ax.axvline(1.0, color="#c7ccd1", lw=0.8, ls="--")
    ax.grid(True, color="#e8ebef", lw=0.55)

    offsets = {"static": (-0.00008, 0.00006), "poisson": (0.00008, -0.00006)}
    markers = {"static": "o", "poisson": "^"}
    tie_count = 0
    for scenario in scenarios:
        short = SCENARIO_LABELS[scenario["taskset"]]
        x = scenario["best_sota_makespan_s"] / scenario["candidate_makespan_s"]
        y = scenario["best_sota_mean_flow_s"] / scenario["candidate_mean_flow_s"]
        if abs(x - 1.0) < 1e-9 and abs(y - 1.0) < 1e-9:
            tie_count += 1
        dx, dy = offsets.get(scenario["arrival_mode"], (0.0, 0.0))
        ax.scatter(
            x + dx,
            y + dy,
            marker=markers.get(scenario["arrival_mode"], "o"),
            s=46,
            facecolor=SCENARIO_COLORS[short],
            edgecolor="#26313d",
            linewidth=0.55,
            alpha=0.95,
            zorder=4,
        )
        if abs(x - 1.0) > 0.0005 or abs(y - 1.0) > 0.0005:
            ax.annotate(
                short,
                xy=(x + dx, y + dy),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=5.8,
                color="#26313d",
            )

    ax.set_xlim(0.9990, 1.0100)
    ax.set_ylim(0.9990, 1.0100)
    ax.set_xlabel("Best SOTA-envelope makespan / candidate makespan")
    ax.set_ylabel("Best SOTA-envelope mean flow / candidate mean flow")
    ax.set_title("b  Scenario-level SOTA envelope comparison", loc="left", fontweight="bold")
    ax.text(
        1.0044,
        1.00955,
        "upper-right favors candidate;\npoints on dashed lines are ties",
        ha="left",
        va="top",
        fontsize=6.2,
        color="#5b6470",
    )
    if tie_count:
        ax.annotate(
            f"{tie_count} ties",
            xy=(1.0, 1.0),
            xytext=(10, 9),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", lw=0.55, color="#5b6470"),
            fontsize=6.1,
            color="#5b6470",
        )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#8aa9cf", markeredgecolor="#26313d", markersize=5.2, label="static"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#8aa9cf", markeredgecolor="#26313d", markersize=5.2, label="Poisson"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6.5, handletextpad=0.45)


def main() -> None:
    report = load_report()
    points = aggregate_points(report)
    scenarios = report["scenarios"]

    fig = plt.figure(figsize=(7.1, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.0], wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    draw_aggregate_panel(ax0, points)
    draw_scenario_panel(ax1, scenarios)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


if __name__ == "__main__":
    main()
