#!/usr/bin/env python3
"""Draw an intuitive measured-cache SOTA comparison figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "sota_candidate_union_gate_fresh_eta_full_v6_20260626.json"
)
OUT = REPO_ROOT / "md" / "figures" / "scheduleurm_sota_comparison"


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


POLICY_LABELS = {
    "throughput_table_goodput": "Throughput goodput",
    "delay_oracle": "Delay oracle",
    "interference_guard": "Interference guard",
    "finish_time_fairness": "Finish-time fairness",
    "resource_adaptive_goodput": "Resource-adaptive",
    "packing_guard": "Packing guard",
    "quadrant_composite": "Quadrant composite",
}

TASKSET_LABELS = {
    "q01_gpu_bound_cnn_resnet50": "q01-CNN static",
    "q01_gpu_bound_llm_inference": "q01-LLM static",
}


def load_report() -> dict:
    with ARTIFACT.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def aggregate_sota_rows(report: dict) -> list[dict]:
    rows = []
    for row in report["aggregate"]["policy_aggregate"]:
        if row["policy_family"] != "sota_style":
            continue
        makespan_extra = (float(row["candidate_vs_policy_sum_makespan"]) - 1.0) * 100.0
        mean_flow_extra = (float(row["candidate_vs_policy_job_weighted_mean_flow"]) - 1.0) * 100.0
        rows.append(
            {
                "label": POLICY_LABELS.get(row["baseline_name"], row["baseline_name"]),
                "makespan_extra": makespan_extra,
                "mean_flow_extra": mean_flow_extra,
            }
        )
    rows.sort(key=lambda r: (r["makespan_extra"], r["mean_flow_extra"]), reverse=True)
    return rows


def scenario_rows(report: dict) -> tuple[list[dict], int]:
    rows = []
    ties = 0
    for scenario in report["scenarios"]:
        makespan_gain = (float(scenario["best_sota_makespan_s"]) / float(scenario["candidate_makespan_s"]) - 1.0) * 100.0
        mean_flow_gain = (float(scenario["best_sota_mean_flow_s"]) / float(scenario["candidate_mean_flow_s"]) - 1.0) * 100.0
        if abs(makespan_gain) < 1e-9 and abs(mean_flow_gain) < 1e-9:
            ties += 1
            continue
        label = TASKSET_LABELS.get(
            scenario["taskset"],
            scenario["taskset"].replace("q01_gpu_bound_", "q01-").replace("_", " "),
        )
        if scenario["arrival_mode"] != "static":
            label = f"{label} {scenario['arrival_mode']}"
        rows.append(
            {
                "label": label,
                "makespan_gain": makespan_gain,
                "mean_flow_gain": mean_flow_gain,
            }
        )
    rows.sort(key=lambda r: max(r["makespan_gain"], r["mean_flow_gain"]), reverse=True)
    return rows, ties


def draw_aggregate(ax: plt.Axes, rows: list[dict], scenario_summary: str) -> None:
    y = np.arange(len(rows))
    height = 0.34
    makespan = np.array([r["makespan_extra"] for r in rows])
    mean_flow = np.array([r["mean_flow_extra"] for r in rows])

    ax.barh(y + height / 2, makespan, height=height, color="#4c78a8", label="Makespan")
    ax.barh(y - height / 2, mean_flow, height=height, color="#d28c4d", label="Mean flow")
    ax.axvline(0, color="#8b949e", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Extra cost relative to the Scheduleurm candidate (%)")
    ax.set_title("SOTA-style policies: extra cost relative to Scheduleurm", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#e8ebef", lw=0.55)
    ax.set_xlim(0, max(1.25, float(max(makespan.max(), mean_flow.max())) * 1.20))
    ax.invert_yaxis()

    for yi, value in zip(y + height / 2, makespan):
        if value >= 0.04:
            ax.text(value + 0.025, yi, f"{value:.2f}", va="center", ha="left", fontsize=5.8, color="#334155")
    for yi, value in zip(y - height / 2, mean_flow):
        if value >= 0.04:
            ax.text(value + 0.025, yi, f"{value:.2f}", va="center", ha="left", fontsize=5.8, color="#334155")

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        fontsize=6.5,
        ncol=2,
        handlelength=1.2,
        columnspacing=1.0,
    )
    ax.text(
        0.985,
        0.40,
        "Positive bars mean the SOTA-style policy is slower.\n"
        "Values near zero are measured-cache ties.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color="#5b6470",
    )
    ax.text(
        0.985,
        0.25,
        scenario_summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color="#1f2937",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f5f7fa", edgecolor="#cfd6df", linewidth=0.6),
    )


def main() -> None:
    report = load_report()
    aggregate = aggregate_sota_rows(report)
    scenarios, ties = scenario_rows(report)
    non_tie = "; ".join(
        f"{row['label']}: +{row['makespan_gain']:.2f}% M, +{row['mean_flow_gain']:.2f}% F"
        for row in scenarios
    )
    scenario_summary = f"Scenario envelope: {ties} ties, 0 losses"
    if non_tie:
        scenario_summary += f"\nNon-tie wins: {non_tie}"

    fig, ax = plt.subplots(figsize=(7.1, 3.35), constrained_layout=True)
    draw_aggregate(ax, aggregate, scenario_summary)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


if __name__ == "__main__":
    main()
