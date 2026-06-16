#!/usr/bin/env python3
"""Generate an alternative Figure 1 for the Scheduleurm OR manuscript.

Figure contract
---------------
Core conclusion:
    Scheduleurm converts heterogeneous four-quadrant workloads into measured
    finite configuration actions, then certifies robust MaxWeight stability with
    explicit slack and bounded claim gates.
Figure archetype:
    schematic-led composite.
Target output:
    OPRE/Nature-style two-column method figure; editable SVG/PDF plus raster
    preview.
Backend:
    Python / matplotlib only.
Panel map:
    a: Four workload quadrants and where external policy families sit.
    b: Full configuration actions are admitted through measured service rows.
    c: Robust MaxWeight score, slack accounting, and claim boundary.
Review risk:
    Avoid suggesting direct full-stack SOTA superiority or arbitrary future
    workload coverage; keep policy families as diagnostic action rows.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# Mandatory editable-text settings from the figure skill.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


OUT = Path(__file__).resolve().parent
BASE = OUT / "scheduleurm_fig1_nature_python"

COL = {
    "ink": "#272727",
    "muted": "#666A73",
    "line": "#AEB4BF",
    "panel": "#F7F8FA",
    "blue_dark": "#284B75",
    "blue": "#5E84B8",
    "blue_soft": "#DCE8F7",
    "teal": "#4AA6A6",
    "teal_soft": "#DDF2F1",
    "rose": "#C76D7E",
    "rose_soft": "#F4DDE2",
    "gold": "#C9942B",
    "gold_soft": "#F7E8BE",
    "green": "#4B9A67",
    "green_soft": "#DDEEDF",
    "violet": "#7565A8",
    "violet_soft": "#E8E4F4",
    "red": "#B64342",
    "red_soft": "#F6D9D7",
}


def ax_setup(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("white")


def add_panel_label(ax, label, x=0.0, y=1.03):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=COL["ink"],
    )


def box(
    ax,
    xy,
    wh,
    text,
    fc="white",
    ec=None,
    lw=0.8,
    radius=0.025,
    fontsize=6.2,
    weight="normal",
    color=None,
    align="center",
    pad=0.012,
    z=3,
):
    if ec is None:
        ec = COL["line"]
    if color is None:
        color = COL["ink"]
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=1.15,
        zorder=z + 1,
    )
    return patch


def arrow(ax, start, end, color=None, lw=1.0, ms=9, rad=0.0, z=4):
    if color is None:
        color = COL["muted"]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def chip(ax, x, y, text, fc, ec=None, w=None, h=0.055, fontsize=5.6):
    if w is None:
        w = 0.052 + 0.010 * len(text)
    return box(
        ax,
        (x, y),
        (w, h),
        text,
        fc=fc,
        ec=ec or fc,
        lw=0.6,
        radius=0.020,
        fontsize=fontsize,
        pad=0.004,
    )


def draw_quadrants(ax):
    ax_setup(ax)
    add_panel_label(ax, "a")
    ax.text(
        0.03,
        0.97,
        "Four workload regimes define the action fabric",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COL["ink"],
    )
    ax.text(
        0.03,
        0.925,
        "Candidate actions are placements + co-location profiles + drain rules.",
        ha="left",
        va="top",
        fontsize=6.0,
        color=COL["muted"],
    )

    x0, y0, w, h = 0.08, 0.16, 0.82, 0.68
    quads = [
        (x0, y0 + h / 2, w / 2, h / 2, COL["green_soft"], "q10", "CPU / data-loader\nhost-bound"),
        (x0 + w / 2, y0 + h / 2, w / 2, h / 2, COL["violet_soft"], "q11", "Hybrid RL\nCPU + GPU coupled"),
        (x0, y0, w / 2, h / 2, COL["gold_soft"], "q00", "Light control\nqueue overhead"),
        (x0 + w / 2, y0, w / 2, h / 2, COL["blue_soft"], "q01", "GPU-heavy\nCNN / LLM kernels"),
    ]
    for x, y, ww, hh, fc, q, title in quads:
        ax.add_patch(Rectangle((x, y), ww, hh, facecolor=fc, edgecolor="white", linewidth=1.2))
        ax.text(x + 0.025, y + hh - 0.060, q, ha="left", va="top", fontsize=9, fontweight="bold", color=COL["ink"])
        ax.text(x + 0.025, y + hh - 0.118, title, ha="left", va="top", fontsize=6.3, color=COL["ink"], linespacing=1.12)

    ax.plot([x0 + w / 2, x0 + w / 2], [y0, y0 + h], color="white", lw=1.4)
    ax.plot([x0, x0 + w], [y0 + h / 2, y0 + h / 2], color="white", lw=1.4)
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=COL["line"], linewidth=0.9))

    # Axis arrows and labels.
    arrow(ax, (x0, y0 - 0.055), (x0 + w, y0 - 0.055), color=COL["ink"], lw=0.9, ms=8)
    arrow(ax, (x0 - 0.050, y0), (x0 - 0.050, y0 + h), color=COL["ink"], lw=0.9, ms=8)
    ax.text(x0 + w / 2, y0 - 0.105, "GPU co-location sensitivity", ha="center", va="top", fontsize=6.1)
    ax.text(x0 - 0.078, y0 + h / 2, "host / data coupling", ha="center", va="center", rotation=90, fontsize=6.1)

    # External policy families as diagnostic anchors.
    chip(ax, x0 + w * 0.56, y0 + 0.035, "Gavel / Pollux / Sia", COL["blue"], ec=COL["blue"], w=0.27, h=0.045, fontsize=5.2)
    chip(ax, x0 + w * 0.58, y0 + h * 0.53, "packing guards", COL["violet"], ec=COL["violet"], w=0.19, h=0.045, fontsize=5.2)
    chip(ax, x0 + 0.045, y0 + h * 0.53, "CPU admission", COL["green"], ec=COL["green"], w=0.17, h=0.045, fontsize=5.2)
    chip(ax, x0 + 0.045, y0 + 0.035, "SRPT / overhead", COL["gold"], ec=COL["gold"], w=0.18, h=0.045, fontsize=5.2)
    for txt in ax.texts[-4:]:
        txt.set_color("white")


def draw_pipeline(ax):
    ax_setup(ax)
    add_panel_label(ax, "b")
    ax.text(
        0.02,
        0.97,
        "Measured finite actions replace informal sweet spots",
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=COL["ink"],
    )

    box(ax, (0.03, 0.76), (0.25, 0.12), "Full\nconfiguration space", fc=COL["panel"], ec=COL["line"], fontsize=6.2)
    box(ax, (0.37, 0.76), (0.24, 0.12), "Measured\nservice cache", fc=COL["blue_soft"], ec=COL["blue"], fontsize=6.2)
    box(ax, (0.72, 0.76), (0.23, 0.12), "Candidate\nfamily", fc=COL["rose_soft"], ec=COL["rose"], fontsize=6.2)
    arrow(ax, (0.285, 0.82), (0.365, 0.82), color=COL["muted"], lw=1.0)
    arrow(ax, (0.615, 0.82), (0.715, 0.82), color=COL["muted"], lw=1.0)

    # Dots in the measured cache.
    dots = [(0.42, 0.62), (0.47, 0.67), (0.52, 0.61), (0.57, 0.66), (0.48, 0.56), (0.55, 0.54)]
    for i, (x, y) in enumerate(dots):
        ax.scatter([x], [y], s=50 + i * 4, color=[COL["blue"], COL["teal"], COL["gold"], COL["violet"], COL["green"], COL["rose"]][i], edgecolor="white", linewidth=0.8, zorder=5)
    ax.text(
        0.50,
        0.485,
        "lower-service rows + capacity boundaries",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COL["muted"],
    )

    # Candidate cards.
    cards = [
        (0.05, 0.31, "placement", COL["green_soft"]),
        (0.27, 0.31, "profile", COL["blue_soft"]),
        (0.49, 0.31, "admission", COL["gold_soft"]),
        (0.71, 0.31, "drain rule", COL["violet_soft"]),
    ]
    for x, y, t, fc in cards:
        box(ax, (x, y), (0.17, 0.075), t, fc=fc, ec=COL["line"], fontsize=5.7, radius=0.018, pad=0.006)

    box(
        ax,
        (0.08, 0.08),
        (0.84, 0.11),
        "External policies become finite diagnostic action rows on the same service cache",
        fc="white",
        ec=COL["line"],
        fontsize=6.1,
        radius=0.020,
    )
    ax.text(0.50, 0.245, "a candidate action is a configuration trajectory", ha="center", va="center", fontsize=6.1, fontweight="bold", color=COL["ink"])


def draw_certificate(ax):
    ax_setup(ax)
    add_panel_label(ax, "c")
    ax.text(
        0.02,
        0.97,
        "The theorem checks slack, not slogan-level dominance",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COL["ink"],
    )

    score = "MaxWeight score\nqueue x lower service - penalty"
    slack = "Residual slack\neta = delta - certified losses"
    box(ax, (0.04, 0.74), (0.41, 0.13), score, fc=COL["blue_soft"], ec=COL["blue"], fontsize=5.75, radius=0.020)
    box(ax, (0.55, 0.74), (0.39, 0.13), slack, fc=COL["green_soft"], ec=COL["green"], fontsize=5.75, radius=0.020)
    arrow(ax, (0.455, 0.805), (0.545, 0.805), color=COL["muted"], lw=1.0)

    # Slack budget bar.
    ax.text(0.05, 0.62, "slack accounting", ha="left", va="center", fontsize=6.4, fontweight="bold", color=COL["ink"])
    x0, y0, total_w, h = 0.05, 0.53, 0.88, 0.07
    segments = [
        ("candidate", 0.22, COL["rose"]),
        ("estimate", 0.20, COL["gold"]),
        ("penalty", 0.18, COL["violet"]),
        ("oracle", 0.14, COL["teal"]),
        ("eta > 0", 0.26, COL["green"]),
    ]
    x = x0
    for name, frac, color in segments:
        ww = total_w * frac
        ax.add_patch(Rectangle((x, y0), ww, h, facecolor=color, edgecolor="white", linewidth=0.8))
        ax.text(x + ww / 2, y0 + h / 2, name, ha="center", va="center", fontsize=5.2, color="white" if name != "estimate" else COL["ink"])
        x += ww
    ax.add_patch(Rectangle((x0, y0), total_w, h, facecolor="none", edgecolor=COL["line"], linewidth=0.8))

    box(ax, (0.05, 0.32), (0.26, 0.11), "admit\nmeasured bucket", fc=COL["green_soft"], ec=COL["green"], fontsize=5.8)
    box(ax, (0.37, 0.32), (0.25, 0.11), "probe / defer\nunknown state", fc=COL["panel"], ec=COL["line"], fontsize=5.8)
    box(ax, (0.68, 0.32), (0.25, 0.11), "exclude\nstrong claim", fc=COL["red_soft"], ec=COL["red"], fontsize=5.8)
    arrow(ax, (0.31, 0.375), (0.365, 0.375), color=COL["muted"], lw=0.9)
    arrow(ax, (0.62, 0.375), (0.675, 0.375), color=COL["muted"], lw=0.9)

    box(
        ax,
        (0.08, 0.11),
        (0.36, 0.10),
        "proved: finite-slice\nstability certificate",
        fc=COL["green_soft"],
        ec=COL["green"],
        fontsize=5.8,
        radius=0.020,
    )
    box(
        ax,
        (0.55, 0.11),
        (0.36, 0.10),
        "not claimed: arbitrary\nfuture/full-stack SOTA",
        fc=COL["red_soft"],
        ec=COL["red"],
        fontsize=5.8,
        radius=0.020,
    )
    arrow(ax, (0.44, 0.16), (0.55, 0.16), color=COL["muted"], lw=0.9)


def main():
    fig = plt.figure(figsize=(7.35, 4.75), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1, 1], wspace=0.13, hspace=0.18)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    draw_quadrants(ax_a)
    draw_pipeline(ax_b)
    draw_certificate(ax_c)

    fig.text(
        0.015,
        0.984,
        "Scheduleurm certificate: from heterogeneous actions to bounded stability claims",
        ha="left",
        va="top",
        fontsize=7.9,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.text(
        0.985,
        0.018,
        "Schematic; external policies are diagnostic action families, not universal full-stack baselines.",
        ha="right",
        va="bottom",
        fontsize=5.7,
        color=COL["muted"],
    )

    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.075)
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(BASE.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
