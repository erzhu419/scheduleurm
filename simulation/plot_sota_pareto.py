"""Plot Pareto comparison against SOTA-style replay baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .defaults import build_default_cache
from .sota_baselines import compare_against_sota_suite
from .tasksets import taskset_by_name


PANEL_TASKSETS = (
    ("q01_gpu_bound_compute", "q01 GPU-heavy"),
    ("q11_cpu_gpu_coupled", "q11 Hybrid RL"),
    ("hybrid_research_portfolio", "Mixed portfolio"),
)

ALL_TASKSETS = (
    "q00_light_control",
    "q01_gpu_bound_compute",
    "q10_cpu_host_bound",
    "q11_cpu_gpu_coupled",
    "hybrid_research_portfolio",
)

BASELINE_MARKERS = {
    "throughput_table_goodput": ("o", "#1f77b4"),
    "delay_oracle": ("s", "#ff7f0e"),
    "interference_guard": ("^", "#2ca02c"),
    "quadrant_composite": ("D", "#9467bd"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="md/figures")
    parser.add_argument("--trials", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = build_default_cache()
    reports = {
        name: compare_against_sota_suite(
            cache,
            taskset_by_name(name).workload_specs(),
            trials=args.trials,
            seed=args.seed,
        )
        for name in ALL_TASKSETS
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    for ax, (taskset_name, title) in zip(axes, PANEL_TASKSETS):
        _plot_panel(ax, reports[taskset_name], title)

    handles, labels = [], []
    seen = set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.08, 0.52), ncol=1, frameon=False)
    fig.suptitle("Scheduleurm candidate vs SOTA-style replay baselines", fontsize=15)

    png = out_dir / "module27_sota_pareto.png"
    svg = out_dir / "module27_sota_pareto.svg"
    data = out_dir / "module27_sota_pareto_data.json"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    data.write_text(json.dumps(_compact_reports(reports), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"png": str(png), "svg": str(svg), "data": str(data)}, indent=2))
    return 0


def _plot_panel(ax, report: dict, title: str) -> None:
    ax.set_xlim(0.94, 1.05)
    ax.set_ylim(0.94, 1.25)
    ax.add_patch(
        Rectangle(
            (0.94, 0.80),
            0.06,
            0.06,
            facecolor="#f8d7da",
            edgecolor="none",
            alpha=0.45,
            label="dominates ours",
            zorder=0,
        )
    )
    ax.axvline(1.0, color="#333333", linewidth=1.0)
    ax.axhline(1.0, color="#333333", linewidth=1.0)
    ax.scatter([1.0], [1.0], marker="*", s=240, color="#111111", label="ours", zorder=5)

    for row in report["baselines"]:
        name = row["baseline"]["name"]
        marker, color = BASELINE_MARKERS.get(name, ("o", "#777777"))
        x = float(row["candidate_vs_baseline_makespan"])
        y = float(row["candidate_vs_baseline_mean_flow"])
        ax.scatter([x], [y], marker=marker, s=90, color=color, edgecolor="white", linewidth=0.8, label=name, zorder=4)
        ax.annotate(
            _short_name(name),
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("SOTA baseline makespan / ours (lower is better)")
    ax.set_ylabel("SOTA baseline mean-flow / ours (lower is better)")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.text(
        0.945,
        0.955,
        "red region: SOTA baseline\nPareto-dominates ours",
        fontsize=8,
        color="#7a1f28",
    )


def _short_name(name: str) -> str:
    return {
        "throughput_table_goodput": "throughput",
        "delay_oracle": "delay",
        "interference_guard": "interf.",
        "quadrant_composite": "composite",
    }.get(name, name)


def _compact_reports(reports: dict[str, dict]) -> dict:
    out = {}
    for taskset_name, report in reports.items():
        out[taskset_name] = {
            "candidate_policy": report["candidate_policy"],
            "candidate_not_pareto_dominated": report["candidate_not_pareto_dominated"],
            "candidate_pareto_dominated_by": report["candidate_pareto_dominated_by"],
            "baselines": [
                {
                    "name": row["baseline"]["name"],
                    "representative_systems": row["baseline"]["representative_systems"],
                    "baseline_profiles": row["baseline_profiles"],
                    "candidate_profiles": row["candidate_profiles"],
                    "candidate_vs_baseline_makespan": row["candidate_vs_baseline_makespan"],
                    "candidate_vs_baseline_mean_flow": row["candidate_vs_baseline_mean_flow"],
                    "baseline_over_ours_makespan": float(row["candidate_vs_baseline_makespan"]),
                    "baseline_over_ours_mean_flow": float(row["candidate_vs_baseline_mean_flow"]),
                }
                for row in report["baselines"]
            ],
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
