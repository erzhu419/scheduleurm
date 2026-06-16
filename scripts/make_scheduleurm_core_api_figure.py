#!/usr/bin/env python3
"""Overlay stable publication labels on the API-generated Figure 1 artwork."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "md" / "figures"
BASE = FIG / "scheduleurm_core_api_base.png"
PNG_OUT = FIG / "scheduleurm_core_api_labeled.png"
PDF_OUT = FIG / "scheduleurm_core_api_labeled.pdf"

BLACK = (18, 24, 31, 255)
GREEN = (10, 117, 49, 255)
BLUE = (30, 79, 154, 255)
RED = (158, 32, 32, 255)
WHITE = (255, 255, 255, 238)
PALE_GREEN = (238, 250, 241, 232)
PALE_BLUE = (238, 245, 255, 232)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded_label(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    lines: list[str],
    *,
    size: int,
    fg: tuple[int, int, int, int] = BLACK,
    fill: tuple[int, int, int, int] = WHITE,
    outline: tuple[int, int, int, int] | None = None,
    bold: bool = True,
    pad_x: int = 18,
    pad_y: int = 10,
    radius: int = 14,
) -> None:
    f = font(size, bold=bold)
    line_gap = int(size * 0.24)
    widths = [draw.textbbox((0, 0), line, font=f)[2] for line in lines]
    heights = [draw.textbbox((0, 0), line, font=f)[3] - draw.textbbox((0, 0), line, font=f)[1] for line in lines]
    text_w = max(widths)
    text_h = sum(heights) + line_gap * (len(lines) - 1)
    x0 = int(center[0] - text_w / 2 - pad_x)
    y0 = int(center[1] - text_h / 2 - pad_y)
    x1 = int(center[0] + text_w / 2 + pad_x)
    y1 = int(center[1] + text_h / 2 + pad_y)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=fill, outline=outline, width=2 if outline else 1)
    y = int(center[1] - text_h / 2)
    for line, h in zip(lines, heights):
        w = draw.textbbox((0, 0), line, font=f)[2]
        draw.text((center[0] - w / 2, y), line, font=f, fill=fg)
        y += h + line_gap


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"Missing base image: {BASE}")
    image = Image.open(BASE).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, _ = image.size

    rounded_label(draw, (285, 82), ["Full action space"], size=23, fg=BLACK, fill=WHITE, pad_x=16, pad_y=7)
    rounded_label(draw, (875, 74), ["Measured candidate set"], size=23, fg=BLACK, fill=WHITE, pad_x=16, pad_y=7)
    rounded_label(draw, (1428, 72), ["Robust certificate"], size=24, fg=GREEN, fill=PALE_GREEN, outline=GREEN, pad_x=16, pad_y=7)

    rounded_label(draw, (744, 116), ["q00", "light/control"], size=14, fg=BLACK, fill=(255, 255, 255, 218), bold=True, pad_x=8, pad_y=4, radius=10)
    rounded_label(draw, (1018, 116), ["q01", "GPU-heavy"], size=14, fg=GREEN, fill=(255, 255, 255, 218), bold=True, pad_x=8, pad_y=4, radius=10)
    rounded_label(draw, (744, 418), ["q10", "CPU-heavy"], size=15, fg=RED, fill=(255, 255, 255, 218), bold=True, pad_x=9, pad_y=5, radius=10)
    rounded_label(draw, (1018, 418), ["q11", "hybrid"], size=15, fg=BLACK, fill=(255, 255, 255, 218), bold=True, pad_x=9, pad_y=5, radius=10)

    rounded_label(
        draw,
        (1433, 382),
        ["queue-weighted", "lower-service score"],
        size=18,
        fg=BLACK,
        fill=WHITE,
        outline=GREEN,
        pad_x=16,
        pad_y=8,
    )
    rounded_label(
        draw,
        (1433, 650),
        ["positive slack", "pays all losses"],
        size=18,
        fg=GREEN,
        fill=PALE_GREEN,
        outline=GREEN,
        pad_x=16,
        pad_y=8,
    )

    rounded_label(
        draw,
        (865, 730),
        ["External schedulers enter only as diagnostic finite actions"],
        size=21,
        fg=BLUE,
        fill=PALE_BLUE,
        outline=BLUE,
        pad_x=18,
        pad_y=9,
    )

    labeled = Image.alpha_composite(image, overlay).convert("RGB")
    labeled.save(PNG_OUT, quality=95)
    labeled.save(PDF_OUT, "PDF", resolution=300.0)


if __name__ == "__main__":
    main()
