#!/usr/bin/env python3
"""Create a compact vector Figure 1 for the Scheduleurm OR manuscript."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "md" / "figures"


GREEN = "#0b7a34"
BLUE = "#1d4f9a"
RED = "#a12b2b"
BLACK = "#111111"
GRAY = "#f4f4f4"


def box(ax, x, y, w, h, fc="white", ec=BLACK, lw=1.1, r=0.018):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=BLACK, lw=1.4, ms=14):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def chip(ax, x, y, s=0.055, color=GREEN, fan=False):
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=color, lw=1.5))
    for i in range(4):
        t = y + s * (0.18 + i * 0.21)
        ax.plot([x - s * 0.10, x], [t, t], color=color, lw=1.0)
        ax.plot([x + s, x + s * 1.10], [t, t], color=color, lw=1.0)
        u = x + s * (0.18 + i * 0.21)
        ax.plot([u, u], [y - s * 0.10, y], color=color, lw=1.0)
        ax.plot([u, u], [y + s, y + s * 1.10], color=color, lw=1.0)
    if fan:
        ax.add_patch(plt.Circle((x + s / 2, y + s / 2), s * 0.22, fill=False, ec=color, lw=1.2))
        ax.plot([x + s * 0.34, x + s * 0.66], [y + s * 0.50, y + s * 0.50], color=color, lw=1.0)
        ax.plot([x + s * 0.50, x + s * 0.37], [y + s * 0.50, y + s * 0.65], color=color, lw=1.0)
        ax.plot([x + s * 0.50, x + s * 0.64], [y + s * 0.50, y + s * 0.35], color=color, lw=1.0)
    else:
        ax.add_patch(Rectangle((x + s * 0.25, y + s * 0.25), s * 0.5, s * 0.5, fill=False, ec=color, lw=1.0))


def terminal(ax, x, y, w=0.070, h=0.048):
    box(ax, x, y, w, h, fc="white", ec=BLACK, lw=1.4, r=0.006)
    ax.plot([x + w * 0.20, x + w * 0.34, x + w * 0.20], [y + h * 0.72, y + h * 0.50, y + h * 0.28], color=BLACK, lw=1.4)
    ax.plot([x + w * 0.43, x + w * 0.62], [y + h * 0.28, y + h * 0.28], color=BLACK, lw=1.4)


def mini_action_space(ax, x, y, w, h):
    for row in range(4):
        yy = y + h * (0.15 + row * 0.22)
        ax.add_patch(Rectangle((x + w * 0.05, yy), w * 0.08, h * 0.065, fill=False, ec=BLACK, lw=0.9))
        for k in range(3):
            y2 = yy + h * (0.015 + k * 0.035)
            x2 = x + w * (0.56 + 0.09 * k)
            ax.plot([x + w * 0.15, x2], [yy + h * 0.032, y2], color=BLACK, lw=0.8)
            ax.add_patch(Rectangle((x2, y2 - h * 0.025), w * 0.055, h * 0.050, fill=False, ec=BLACK, lw=0.8))
    ax.text(x + w * 0.50, y + h * 0.07, r"$\cdots$", ha="center", va="center", fontsize=14, color=BLACK)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.8, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.940,
        "Robust Candidate MaxWeight: finite actions, audited slack",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    panel_y, panel_h = 0.250, 0.565
    panel_top = panel_y + panel_h
    center_y = panel_y + panel_h / 2
    left_x, left_w = 0.055, 0.235
    mid_x, mid_w = 0.382, 0.310
    right_x, right_w = 0.757, 0.220

    # Panel 1: full space.
    box(ax, left_x, panel_y, left_w, panel_h, fc=GRAY, ec="#333333", lw=1.0)
    ax.text(
        left_x + left_w / 2,
        panel_top - 0.075,
        "Combinatorial\nfull action space",
        ha="center",
        va="center",
        fontsize=10.7,
        fontweight="bold",
    )
    mini_action_space(ax, left_x + 0.030, panel_y + 0.090, left_w - 0.070, panel_h - 0.215)

    # Candidate-cover arrow.
    arrow(ax, left_x + left_w + 0.006, center_y, mid_x - 0.010, center_y, lw=1.8, ms=18)
    ax.text(
        (left_x + left_w + mid_x) / 2,
        center_y + 0.105,
        "cover +\nprofile",
        ha="center",
        va="center",
        fontsize=7.9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )

    # Panel 2: measured candidate quadrants.
    box(ax, mid_x, panel_y, mid_w, panel_h, fc="white", ec="#333333", lw=1.0)
    ax.text(
        mid_x + mid_w / 2,
        panel_top - 0.065,
        "Measured candidate\nactions",
        ha="center",
        va="center",
        fontsize=10.7,
        fontweight="bold",
    )
    gx, gy, gw, gh = mid_x + 0.032, panel_y + 0.075, mid_w - 0.064, 0.335
    ax.add_patch(Rectangle((gx, gy), gw, gh, fill=False, ec=BLACK, lw=1.1))
    ax.plot([gx + gw / 2] * 2, [gy, gy + gh], color=BLACK, lw=0.9)
    ax.plot([gx, gx + gw], [gy + gh / 2] * 2, color=BLACK, lw=0.9)
    labels = [
        (0.25, 0.80, "q00\nLight/control"),
        (0.75, 0.80, "q01\nGPU-heavy"),
        (0.25, 0.33, "q10\nCPU-heavy"),
        (0.75, 0.33, "q11\nHybrid"),
    ]
    for xx, yy, text in labels:
        ax.text(gx + gw * xx, gy + gh * yy, text, ha="center", va="center", fontsize=8.6)
    terminal(ax, gx + gw * 0.19, gy + gh * 0.565, w=0.054, h=0.038)
    chip(ax, gx + gw * 0.70, gy + gh * 0.545, s=0.040, color=GREEN, fan=True)
    chip(ax, gx + gw * 0.19, gy + gh * 0.075, s=0.040, color=RED)
    chip(ax, gx + gw * 0.60, gy + gh * 0.075, s=0.040, color=RED)
    chip(ax, gx + gw * 0.79, gy + gh * 0.075, s=0.040, color=GREEN, fan=True)

    # Panel 3: certificate.
    arrow(ax, mid_x + mid_w + 0.006, center_y, right_x - 0.010, center_y, lw=1.8, ms=18)
    box(ax, right_x, panel_y, right_w, panel_h, fc="#fbfffb", ec=GREEN, lw=1.3)
    ax.text(
        right_x + right_w / 2,
        panel_top - 0.065,
        "Robust MaxWeight\ncertificate",
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=GREEN,
    )
    inner_x, inner_w = right_x + 0.024, right_w - 0.048
    box(ax, inner_x, panel_y + 0.315, inner_w, 0.075, fc="white", ec=GREEN, lw=1.0, r=0.010)
    ax.text(inner_x + inner_w / 2, panel_y + 0.353, r"$Q^{T}$ lower-service$(a)-p(a)$", ha="center", va="center", fontsize=8.6)
    box(ax, inner_x, panel_y + 0.205, inner_w, 0.075, fc="white", ec=GREEN, lw=1.0, r=0.010)
    ax.text(
        inner_x + inner_w / 2,
        panel_y + 0.243,
        r"$\delta > L\rho+\epsilon_{\rm est}+\beta+\alpha_{1}$",
        ha="center",
        va="center",
        fontsize=8.8,
    )
    ax.plot([inner_x + 0.010, inner_x + inner_w - 0.010], [panel_y + 0.160, panel_y + 0.160], color="#9acaa5", lw=0.8)
    ax.plot([right_x + 0.050], [panel_y + 0.075], marker="o", ms=15, color=GREEN)
    ax.text(right_x + 0.050, panel_y + 0.075, r"$\checkmark$", ha="center", va="center", fontsize=11.0, color="white", fontweight="bold")
    ax.text(
        right_x + 0.075,
        panel_y + 0.075,
        "Foster recurrence\non audited finite slice",
        ha="left",
        va="center",
        fontsize=8.1,
        color=GREEN,
        fontweight="bold",
    )

    # Bottom boundary strip.
    bottom_x, bottom_y, bottom_w, bottom_h = 0.260, 0.075, 0.480, 0.105
    box(ax, bottom_x, bottom_y, bottom_w, bottom_h, fc="#f8fbff", ec=BLUE, lw=1.0, r=0.012)
    ax.text(
        bottom_x + bottom_w / 2,
        bottom_y + bottom_h / 2,
        "External policies are diagnostic finite actions on the same service cache\n"
        "not universal full-stack superiority claims.",
        ha="center",
        va="center",
        fontsize=7.9,
        color=BLUE,
        fontweight="bold",
    )
    for x in (mid_x + mid_w * 0.25, mid_x + mid_w * 0.50, mid_x + mid_w * 0.75):
        arrow(ax, x, bottom_y + bottom_h, x + 0.025, panel_y, color=BLUE, lw=0.9, ms=9)

    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"scheduleurm_core_vector.{ext}", bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
