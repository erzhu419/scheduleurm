#!/usr/bin/env python3
"""Figure 1 combo candidate: v1 quadrant panel plus tightened v2 action table.

This version preserves the existing v1/v2 outputs.  It uses the left panel from
``make_scheduleurm_fig1_nature_python.py`` because that panel states the four
workload regimes and policy-family anchors most directly.  The right panels are
a tightened theorem-evidence view aligned with the manuscript's theorem objects.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


OUT = Path(__file__).resolve().parent
BASE = OUT / "scheduleurm_fig1_nature_python_combo"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = _load_module(OUT / "make_scheduleurm_fig1_nature_python.py", "scheduleurm_fig1_v1")
V2 = _load_module(OUT / "make_scheduleurm_fig1_nature_python_v2.py", "scheduleurm_fig1_v2")
COL = V2.COL


def panel_b(ax):
    V2.clean(ax)
    ax.text(0.03, 0.965, "b", ha="left", va="top", fontsize=8.5, fontweight="bold", color=COL["ink"])
    ax.text(0.070, 0.965, "Measured-service action table", ha="left", va="top", fontsize=8.3, fontweight="bold", color=COL["ink"])

    x0, y0, w, h = 0.035, 0.19, 0.93, 0.65
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.006,rounding_size=0.020", facecolor="white", edgecolor=COL["hair"], linewidth=0.9))
    headers = ["row", "lower service", "penalty", "oracle", "status"]
    xs = [x0 + 0.085, x0 + 0.345, x0 + 0.555, x0 + 0.665, x0 + 0.855]
    for x, head in zip(xs, headers):
        ax.text(x, y0 + h - 0.065, head, ha="center", va="center", fontsize=5.25, color=COL["muted"], fontweight="bold")
    ax.plot([x0 + 0.025, x0 + w - 0.025], [y0 + h - 0.105, y0 + h - 0.105], color=COL["hair"], lw=0.8)

    rows = [
        ("declared\nq00-q11", 0.72, 0.18, "0 gap", "admit", COL["green"]),
        ("candidate\ncover", 0.64, 0.24, "check", "admit", COL["blue"]),
        ("external\npolicy row", 0.58, 0.20, "bounded", "diagnostic", COL["violet"]),
        ("unmeasured\nstate", 0.00, 0.00, "-", "probe/defer", COL["red"]),
    ]
    row_y = [y0 + h - 0.18, y0 + h - 0.305, y0 + h - 0.43, y0 + h - 0.555]
    for (name, svc, pen, gap, status, col), yy in zip(rows, row_y):
        ax.text(xs[0], yy, name, ha="center", va="center", fontsize=5.35, color=COL["ink"], linespacing=1.02)
        ax.add_patch(Rectangle((xs[1] - 0.060, yy - 0.017), 0.12, 0.034, facecolor=COL["panel"], edgecolor="none"))
        if svc > 0:
            ax.add_patch(Rectangle((xs[1] - 0.060, yy - 0.017), 0.12 * svc, 0.034, facecolor=col, edgecolor="none"))
        ax.add_patch(Rectangle((xs[2] - 0.048, yy - 0.013), 0.096, 0.026, facecolor=COL["panel"], edgecolor="none"))
        if pen > 0:
            ax.add_patch(Rectangle((xs[2] - 0.048, yy - 0.013), 0.096 * pen, 0.026, facecolor=COL["gold"], edgecolor="none"))
        ax.text(xs[3], yy, gap, ha="center", va="center", fontsize=5.15, color=COL["ink"])
        status_col = COL["green"] if status == "admit" else COL["violet"] if status == "diagnostic" else COL["red"]
        status_width = 0.106 if status == "admit" else 0.120
        V2.pill(ax, (xs[4] - status_width / 2, yy - 0.020), (status_width, 0.040), status, status_col, fs=4.25)


def panel_c(ax):
    V2.clean(ax)
    ax.text(0.03, 0.965, "c", ha="left", va="top", fontsize=8.5, fontweight="bold", color=COL["ink"])
    ax.text(0.070, 0.965, "Slack test for theorem-facing claims", ha="left", va="top", fontsize=8.4, fontweight="bold", color=COL["ink"])

    x0, y0, total, hh = 0.06, 0.70, 0.86, 0.080
    ax.text(x0, y0 + 0.13, "residual drift margin", ha="left", va="center", fontsize=5.8, color=COL["ink"], fontweight="bold")
    ax.add_patch(Rectangle((x0, y0), total, hh, facecolor="white", edgecolor=COL["hair"], linewidth=0.8))
    losses = [
        ("candidate", 0.22, COL["rose"]),
        ("estimate", 0.19, COL["gold"]),
        ("penalty", 0.17, COL["violet"]),
        ("oracle", 0.13, COL["teal"]),
    ]
    x = x0
    for name, frac, col in losses:
        ww = total * frac
        ax.add_patch(Rectangle((x, y0), ww, hh, facecolor=col, edgecolor="white", linewidth=0.8))
        ax.text(x + ww / 2, y0 + hh / 2, name, ha="center", va="center", fontsize=5.0, color="white" if name != "estimate" else COL["ink"])
        x += ww
    residual = x0 + total - x
    ax.add_patch(Rectangle((x, y0), residual, hh, facecolor=COL["green"], edgecolor="white", linewidth=0.8))
    ax.text(x + residual / 2, y0 + hh / 2, "eta > 0", ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")

    V2.round_box(ax, (0.07, 0.50), (0.35, 0.105), "robust MaxWeight\nlower-service score", fc=COL["blue_light"], ec=COL["blue"], fs=5.55)
    V2.round_box(ax, (0.60, 0.50), (0.31, 0.105), "Foster recurrence\ncertificate", fc=COL["green_light"], ec=COL["green"], fs=5.55)
    V2.arrow(ax, (0.425, 0.552), (0.595, 0.552), color=COL["muted"], lw=1.0)

    ax.text(0.06, 0.385, "claim scope", ha="left", va="center", fontsize=5.8, color=COL["ink"], fontweight="bold")
    V2.round_box(ax, (0.07, 0.235), (0.25, 0.095), "proved\nmeasured slice", fc=COL["green_light"], ec=COL["green"], fs=5.35)
    V2.round_box(ax, (0.375, 0.235), (0.25, 0.095), "checked\nlive hook", fc=COL["panel"], ec=COL["hair"], fs=5.35)
    V2.round_box(ax, (0.68, 0.235), (0.25, 0.095), "not claimed\nfuture/external", fc=COL["red_light"], ec=COL["red"], fs=5.35)
    V2.arrow(ax, (0.32, 0.282), (0.375, 0.282), color=COL["muted"], lw=0.8, ms=8)
    V2.arrow(ax, (0.625, 0.282), (0.68, 0.282), color=COL["muted"], lw=0.8, ms=8)

def main():
    fig = plt.figure(figsize=(7.35, 4.28), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1, 1], wspace=0.045, hspace=0.15)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    V1.draw_quadrants(ax_a)
    ax_a.set_xlim(0, 0.945)
    panel_b(ax_b)
    panel_c(ax_c)

    fig.text(
        0.02,
        0.984,
        "Scheduleurm certificate: four-regime actions to robust MaxWeight claims",
        ha="left",
        va="top",
        fontsize=8.1,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.subplots_adjust(left=0.030, right=0.985, top=0.905, bottom=0.055)

    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(BASE.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
