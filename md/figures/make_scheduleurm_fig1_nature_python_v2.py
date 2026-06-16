#!/usr/bin/env python3
"""Alternative Figure 1 v2 for the Scheduleurm OR manuscript.

Figure contract
---------------
Core conclusion:
    Robust candidate MaxWeight can be presented as an auditable chain: four
    heterogeneous workload regimes define configuration actions; measured rows
    admit finite candidate actions; slack gates convert those rows into a
    theorem-facing stability certificate.
Figure archetype:
    asymmetric schematic-led composite with one dominant mechanism panel.
Target output:
    OPRE/Nature-style two-column Figure 1 candidate; editable SVG/PDF plus
    raster preview.
Backend:
    Python / matplotlib only.
Panel map:
    a: configuration-action fabric across four regimes.
    b: measured action ledger and external-policy row semantics.
    c: slack accounting and claim gate.
Review risk:
    The figure must not imply arbitrary future-workload coverage or direct
    full-stack SOTA superiority.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# Mandatory editable-text settings from the figure skill.
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
BASE = OUT / "scheduleurm_fig1_nature_python_v2"

COL = {
    "ink": "#24272B",
    "muted": "#6E7480",
    "hair": "#B7BEC8",
    "panel": "#F6F7F9",
    "blue": "#3E6FA8",
    "blue_light": "#DCE9F7",
    "green": "#4E9A66",
    "green_light": "#DDEEDF",
    "gold": "#C9952C",
    "gold_light": "#F6E7BD",
    "violet": "#6F62A9",
    "violet_light": "#E8E3F4",
    "rose": "#C96E80",
    "rose_light": "#F5DDE3",
    "teal": "#4BA7A8",
    "teal_light": "#DDF1F1",
    "red": "#B84643",
    "red_light": "#F7D9D7",
    "white": "#FFFFFF",
}


def clean(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("white")


def label(ax, letter, x=0.0, y=1.02):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=COL["ink"],
    )


def round_box(ax, xy, wh, text, fc, ec=None, lw=0.9, fs=6.0, weight="normal", radius=0.020, color=None, z=3):
    if ec is None:
        ec = COL["hair"]
    if color is None:
        color = COL["ink"]
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.009,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color=color,
        linespacing=1.08,
        zorder=z + 1,
    )
    return patch


def arrow(ax, p0, p1, color=None, lw=1.1, ms=10, rad=0.0, z=4):
    if color is None:
        color = COL["muted"]
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
        zorder=z,
    )
    ax.add_patch(arr)


def pill(ax, xy, wh, text, fc, ec=None, fs=5.3, color=None):
    if ec is None:
        ec = fc
    if color is None:
        color = COL["white"] if fc not in {COL["panel"], COL["blue_light"], COL["green_light"], COL["gold_light"], COL["violet_light"], COL["rose_light"], COL["teal_light"]} else COL["ink"]
    return round_box(ax, xy, wh, text, fc=fc, ec=ec, lw=0.75, fs=fs, radius=0.028, color=color)


def panel_a(ax):
    clean(ax)
    label(ax, "a", x=0.01)
    ax.text(0.06, 0.965, "Configuration-action fabric", ha="left", va="top", fontsize=9.2, fontweight="bold", color=COL["ink"])
    ax.text(0.06, 0.915, "one action = placement + profile + admission + drain rule", ha="left", va="top", fontsize=5.8, color=COL["muted"])

    # Background field and subtle axes.
    ax.add_patch(FancyBboxPatch((0.06, 0.11), 0.88, 0.74, boxstyle="round,pad=0.006,rounding_size=0.030", facecolor=COL["panel"], edgecolor=COL["hair"], linewidth=0.9))
    ax.plot([0.50, 0.50], [0.16, 0.80], color=COL["white"], lw=2.0, zorder=1)
    ax.plot([0.12, 0.88], [0.47, 0.47], color=COL["white"], lw=2.0, zorder=1)

    cards = [
        (0.14, 0.54, "q10", "CPU / data\nhost-bound", COL["green_light"], COL["green"]),
        (0.54, 0.54, "q11", "hybrid RL\ncoupled", COL["violet_light"], COL["violet"]),
        (0.14, 0.20, "q00", "light control\nsmall jobs", COL["gold_light"], COL["gold"]),
        (0.54, 0.20, "q01", "GPU-heavy\nCNN / LLM", COL["blue_light"], COL["blue"]),
    ]
    for x, y, q, desc, fc, ec in cards:
        round_box(ax, (x, y), (0.31, 0.20), "", fc=fc, ec="none", lw=0, radius=0.020, z=2)
        ax.text(x + 0.025, y + 0.155, q, ha="left", va="top", fontsize=10, fontweight="bold", color=COL["ink"], zorder=3)
        ax.text(x + 0.025, y + 0.095, desc, ha="left", va="top", fontsize=6.2, color=COL["ink"], linespacing=1.08, zorder=3)
        pill(ax, (x + 0.18, y + 0.012), (0.105, 0.038), "measured", ec, fs=4.6)

    # Central action token.
    ax.add_patch(Circle((0.50, 0.47), 0.082, facecolor=COL["ink"], edgecolor=COL["white"], linewidth=1.2, zorder=5))
    ax.text(0.50, 0.485, "action", ha="center", va="center", fontsize=7.0, color="white", fontweight="bold", zorder=6)
    ax.text(0.50, 0.448, "a", ha="center", va="center", fontsize=8.5, color="white", zorder=6)
    for p in [(0.295, 0.64), (0.705, 0.64), (0.295, 0.30), (0.705, 0.30)]:
        arrow(ax, p, (0.50, 0.47), color=COL["hair"], lw=0.9, ms=8, z=2.4)

    ax.text(0.50, 0.075, "low to high GPU co-location pressure", ha="center", va="center", fontsize=5.5, color=COL["muted"])
    ax.text(0.025, 0.47, "low to high host pressure", ha="center", va="center", rotation=90, fontsize=5.5, color=COL["muted"])


def panel_b(ax):
    clean(ax)
    label(ax, "b", x=0.0)
    ax.text(0.03, 0.95, "Measured action ledger", ha="left", va="top", fontsize=8.7, fontweight="bold", color=COL["ink"])
    ax.text(0.03, 0.895, "same service cache for our policy and diagnostic policy families", ha="left", va="top", fontsize=5.6, color=COL["muted"])

    x0, y0, w, h = 0.05, 0.17, 0.90, 0.64
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.006,rounding_size=0.020", facecolor="white", edgecolor=COL["hair"], linewidth=0.9))
    headers = ["row", "lower service", "penalty", "oracle gap", "status"]
    xs = [x0 + 0.06, x0 + 0.31, x0 + 0.53, x0 + 0.70, x0 + 0.84]
    for x, head in zip(xs, headers):
        ax.text(x, y0 + h - 0.065, head, ha="center", va="center", fontsize=5.4, color=COL["muted"], fontweight="bold")
    ax.plot([x0 + 0.025, x0 + w - 0.025], [y0 + h - 0.105, y0 + h - 0.105], color=COL["hair"], lw=0.8)

    rows = [
        ("q00-q11 bucket", 0.70, 0.18, "0", "admit", COL["green"]),
        ("candidate cover", 0.64, 0.24, "small", "admit", COL["blue"]),
        ("external policy", 0.58, 0.20, "bounded", "diagnostic", COL["violet"]),
        ("unknown state", 0.10, 0.08, "-", "probe", COL["red"]),
    ]
    row_y = [y0 + h - 0.18, y0 + h - 0.30, y0 + h - 0.42, y0 + h - 0.54]
    for (name, svc, pen, gap, status, col), yy in zip(rows, row_y):
        ax.text(xs[0], yy, name, ha="center", va="center", fontsize=5.7, color=COL["ink"])
        ax.add_patch(Rectangle((xs[1] - 0.065, yy - 0.018), 0.13, 0.036, facecolor=COL["panel"], edgecolor="none"))
        ax.add_patch(Rectangle((xs[1] - 0.065, yy - 0.018), 0.13 * svc, 0.036, facecolor=col, edgecolor="none"))
        ax.add_patch(Rectangle((xs[2] - 0.050, yy - 0.014), 0.10, 0.028, facecolor=COL["panel"], edgecolor="none"))
        ax.add_patch(Rectangle((xs[2] - 0.050, yy - 0.014), 0.10 * pen, 0.028, facecolor=COL["gold"], edgecolor="none"))
        ax.text(xs[3], yy, gap, ha="center", va="center", fontsize=5.5, color=COL["ink"])
        status_col = COL["green"] if status == "admit" else COL["violet"] if status == "diagnostic" else COL["red"]
        pill(ax, (xs[4] - 0.052, yy - 0.020), (0.104, 0.040), status, status_col, fs=4.9)

    ax.text(0.50, 0.075, "External schedulers stress the candidate family; they are not used as universal full-stack claims.", ha="center", va="center", fontsize=5.45, color=COL["muted"])


def panel_c(ax):
    clean(ax)
    label(ax, "c", x=0.0)
    ax.text(0.03, 0.95, "Slack gate to theorem-facing claims", ha="left", va="top", fontsize=8.7, fontweight="bold", color=COL["ink"])

    # Slack bar.
    x0, y0, total, hh = 0.06, 0.70, 0.86, 0.080
    ax.text(x0, y0 + 0.13, "full-region slack delta", ha="left", va="center", fontsize=5.8, color=COL["ink"], fontweight="bold")
    ax.add_patch(Rectangle((x0, y0), total, hh, facecolor=COL["green_light"], edgecolor=COL["green"], linewidth=0.8))
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
        ax.text(x + ww / 2, y0 + hh / 2, name, ha="center", va="center", fontsize=5.1, color="white" if name != "estimate" else COL["ink"])
        x += ww
    ax.text(x + (x0 + total - x) / 2, y0 + hh / 2, "eta > 0", ha="center", va="center", fontsize=5.5, color=COL["green"], fontweight="bold")

    # MaxWeight rule compact.
    round_box(ax, (0.07, 0.50), (0.35, 0.105), "robust MaxWeight\nselection", fc=COL["blue_light"], ec=COL["blue"], fs=5.8)
    round_box(ax, (0.60, 0.50), (0.31, 0.105), "Foster recurrence\ncertificate", fc=COL["green_light"], ec=COL["green"], fs=5.8)
    arrow(ax, (0.425, 0.552), (0.595, 0.552), color=COL["muted"], lw=1.0)

    # Claim boundary.
    ax.text(0.06, 0.385, "claim boundary", ha="left", va="center", fontsize=5.8, color=COL["ink"], fontweight="bold")
    round_box(ax, (0.07, 0.235), (0.25, 0.095), "proved\nmeasured slice", fc=COL["green_light"], ec=COL["green"], fs=5.4)
    round_box(ax, (0.375, 0.235), (0.25, 0.095), "trace-only\nlive hook", fc=COL["panel"], ec=COL["hair"], fs=5.4)
    round_box(ax, (0.68, 0.235), (0.25, 0.095), "not claimed\narbitrary future", fc=COL["red_light"], ec=COL["red"], fs=5.4)
    arrow(ax, (0.32, 0.282), (0.375, 0.282), color=COL["muted"], lw=0.8, ms=8)
    arrow(ax, (0.625, 0.282), (0.68, 0.282), color=COL["muted"], lw=0.8, ms=8)
    ax.text(0.50, 0.105, "The figure shows the certificate interface, not an exhaustive engineering race.", ha="center", va="center", fontsize=5.45, color=COL["muted"])


def main():
    fig = plt.figure(figsize=(7.35, 4.55), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1, 1], wspace=0.13, hspace=0.20)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    fig.text(
        0.02,
        0.984,
        "Scheduleurm: audited configuration actions for robust MaxWeight scheduling",
        ha="left",
        va="top",
        fontsize=8.1,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.subplots_adjust(left=0.030, right=0.985, top=0.905, bottom=0.070)

    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(BASE.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
