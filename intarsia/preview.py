"""Visual verification artifacts. Every stage failure here is silent and
visual, so previews are first-class outputs, not debug helpers."""

import numpy as np
from PIL import Image, ImageDraw

_ROW_H = 26
_PAD = 8


def level_map_image(levels, palette_rgb):
    """Levels (H,W) + palette (N,3) -> flat-color PIL image of the separation."""
    return Image.fromarray(palette_rgb[levels])


def legend_image(report, width, heights_mm=None):
    """Swatch rows: level, hex, coverage, optional height."""
    n = report.n_levels
    img = Image.new("RGB", (width, _ROW_H * n + 2 * _PAD), "white")
    d = ImageDraw.Draw(img)
    for lvl in range(n - 1, -1, -1):  # top of legend = highest level
        y = _PAD + (n - 1 - lvl) * _ROW_H
        color = tuple(report.palette_rgb[lvl])
        d.rectangle([_PAD, y + 3, _PAD + 40, y + _ROW_H - 3], fill=color, outline="black")
        cov = report.coverage[lvl]
        text = f"L{lvl}  {report.palette_hex()[lvl]}  {cov * 100:5.1f}%"
        if heights_mm is not None:
            text += f"  top at {heights_mm[lvl]:.2f} mm"
        if cov < 0.005:
            text += "  << suspiciously small: separation error?"
        d.text((_PAD + 50, y + 6), text, fill="black")
    return img


def side_by_side(original, levels, report, heights_mm=None, max_w=1400):
    """original | level map, legend underneath. The main stage-1 artifact."""
    lm = level_map_image(levels, report.palette_rgb)
    orig = original.convert("RGB")
    if orig.size != lm.size:
        orig = orig.resize(lm.size, Image.NEAREST)
    w, h = lm.size
    scale = min(1.0, (max_w // 2) / w)
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
        orig = orig.resize((w, h), Image.LANCZOS)
        lm = lm.resize((w, h), Image.NEAREST)
    legend = legend_image(report, 2 * w + 3 * _PAD, heights_mm)
    out = Image.new("RGB", (2 * w + 3 * _PAD, h + legend.size[1] + 2 * _PAD), "white")
    out.paste(orig, (_PAD, _PAD))
    out.paste(lm, (2 * _PAD + w, _PAD))
    out.paste(legend, (0, h + 2 * _PAD))
    return out


def hillshade_image(height_mm, px_mm):
    """Shaded relief render of the final heightmap, light from upper-left."""
    z = np.asarray(height_mm, dtype=np.float64)
    gy, gx = np.gradient(z, px_mm)
    # light direction
    lx, ly, lz = -0.5, -0.5, 0.8
    norm = np.sqrt(gx**2 + gy**2 + 1.0)
    shade = (-gx * lx - gy * ly + lz) / norm / np.sqrt(lx**2 + ly**2 + lz**2)
    shade = np.clip(shade, 0, 1)
    # mix in a bit of height so levels read even on flat tops
    hn = (z - z.min()) / max(z.ptp(), 1e-9)
    val = np.clip(0.75 * shade + 0.25 * hn, 0, 1)
    return Image.fromarray((val * 255).astype(np.uint8), "L").convert("RGB")
