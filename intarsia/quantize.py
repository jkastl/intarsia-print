"""Deterministic color quantization: image -> N discrete levels.

No RNG anywhere. Centers are seeded from the most frequent distinct colors
(subject to a minimum Lab separation) and polished with Lloyd iterations,
so the same input always produces the same levels.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .color import delta_e, srgb_to_lab

# Colors are binned to 5 bits/channel for the histogram; flat graphic art
# collapses to a handful of bins, anti-aliased edges spread thinly around them.
_BIN_SHIFT = 3

# A pixel further than this (CIE76) from every final center is "unexplained".
UNSUITABLE_DE = 15.0
# If more than this fraction of pixels are unexplained, the image is probably
# a gradient/photo rather than flat graphics.
UNSUITABLE_FRACTION = 0.10


@dataclass
class LevelReport:
    n_levels: int
    palette_rgb: np.ndarray  # (N, 3) uint8, display color per level, index = level
    coverage: np.ndarray  # (N,) float, fraction of pixels per level
    mean_residual: float  # mean deltaE of pixels to their center
    bad_fraction: float  # fraction of pixels with deltaE > UNSUITABLE_DE
    warnings: list = field(default_factory=list)

    def palette_hex(self):
        return ["#%02x%02x%02x" % tuple(c) for c in self.palette_rgb]


def _binned_histogram(rgb):
    """rgb uint8 (H,W,3) -> (unique binned colors as float rgb in [0,1], counts)."""
    q = (rgb >> _BIN_SHIFT).astype(np.uint32)
    packed = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    uniq, inverse, counts = np.unique(packed.ravel(), return_inverse=True, return_counts=True)
    # bin center back to rgb
    r = ((uniq >> 10) & 31).astype(np.float64)
    g = ((uniq >> 5) & 31).astype(np.float64)
    b = (uniq & 31).astype(np.float64)
    colors = (np.stack([r, g, b], axis=1) * (1 << _BIN_SHIFT) + (1 << (_BIN_SHIFT - 1))) / 255.0
    return colors, counts, inverse.reshape(rgb.shape[:2])


def _seed_centers(lab, counts, n, min_sep=14.0):
    """Pick n seed centers: frequency-ordered, each at least min_sep from the rest.

    If the image doesn't have n colors that far apart, relax the separation
    until it does. Fully deterministic.
    """
    order = np.argsort(-counts, kind="stable")
    sep = min_sep
    while sep > 0.5:
        chosen = []
        for i in order:
            if all(delta_e(lab[i], lab[j]) >= sep for j in chosen):
                chosen.append(i)
                if len(chosen) == n:
                    return np.array(chosen), sep
        sep /= 2.0
    # Fewer distinct colors than requested levels.
    return np.array(chosen), sep


def _lloyd(lab, counts, centers, iters=20):
    """Weighted k-means refinement on the binned-color histogram."""
    for _ in range(iters):
        d = np.linalg.norm(lab[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        new = np.empty_like(centers)
        for k in range(len(centers)):
            m = assign == k
            if not m.any():
                new[k] = centers[k]  # empty cluster: keep it where it was
                continue
            w = counts[m].astype(np.float64)
            new[k] = (lab[m] * w[:, None]).sum(axis=0) / w.sum()
        if np.allclose(new, centers, atol=1e-6):
            centers = new
            break
        centers = new
    d = np.linalg.norm(lab[:, None, :] - centers[None, :, :], axis=2)
    return centers, np.argmin(d, axis=1), d.min(axis=1)


def quantize(
    image: Image.Image,
    n_levels: int,
    order: str = "dark-low",
    assign_overrides: dict | None = None,
):
    """Quantize image to n_levels discrete levels.

    order: 'dark-low' (darker colors get lower levels) or 'light-low'.
    assign_overrides: {'#rrggbb': level_index} — forces the cluster nearest
        that color to the given level; remaining clusters fill the free
        levels in brightness order.

    Returns (levels int array (H,W) with values 0..n_levels-1, LevelReport).
    """
    rgb = np.asarray(image.convert("RGB"))
    colors, counts, bin_map = _binned_histogram(rgb)
    lab = srgb_to_lab(colors)

    seed_idx, sep = _seed_centers(lab, counts, n_levels)
    warnings = []
    if len(seed_idx) < n_levels:
        raise SystemExit(
            f"error: image has only {len(seed_idx)} distinguishable colors, "
            f"cannot make {n_levels} levels. Lower --levels."
        )
    if sep < 7.0:
        warnings.append(
            f"color separation is weak (deltaE ~{sep:.1f}); levels may be arbitrary. "
            "Consider fewer levels or a flatter source image."
        )

    centers, assign, dmin = _lloyd(lab, counts, lab[seed_idx].copy())

    total = counts.sum()
    mean_residual = float((dmin * counts).sum() / total)
    bad_fraction = float(counts[dmin > UNSUITABLE_DE].sum() / total)
    if bad_fraction > UNSUITABLE_FRACTION:
        warnings.append(
            f"{bad_fraction:.0%} of pixels don't match any of the {n_levels} level colors "
            f"(mean residual deltaE {mean_residual:.1f}). This image looks like it has "
            "gradients/shading/photographic content — output will likely be noisy. "
            "Use a flat, poster-style source image."
        )

    # Display color per cluster: weighted mean of the actual rgb bins.
    disp = np.empty((n_levels, 3), dtype=np.uint8)
    for k in range(n_levels):
        m = assign == k
        w = counts[m].astype(np.float64)
        disp[k] = np.round((colors[m] * w[:, None]).sum(axis=0) / w.sum() * 255)

    # Cluster -> level: brightness order, then apply explicit overrides.
    lightness = centers[:, 0]
    bright_order = np.argsort(lightness, kind="stable")  # dark ... light
    if order == "light-low":
        bright_order = bright_order[::-1]
    cluster_to_level = np.empty(n_levels, dtype=np.int64)
    cluster_to_level[bright_order] = np.arange(n_levels)

    if assign_overrides:
        forced = {}  # cluster -> level
        for hexcolor, lvl in assign_overrides.items():
            if not (0 <= lvl < n_levels):
                raise SystemExit(f"error: level {lvl} out of range 0..{n_levels - 1}")
            c = hexcolor.lstrip("#")
            target = srgb_to_lab(np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0)
            forced[int(np.argmin(delta_e(centers, target)))] = lvl
        if len(set(forced.values())) < len(forced):
            raise SystemExit("error: two --assign overrides map to the same level")
        free_levels = [l for l in range(n_levels) if l not in forced.values()]
        free_clusters = [c for c in bright_order if c not in forced]
        cluster_to_level[:] = -1
        for c, l in forced.items():
            cluster_to_level[c] = l
        for c, l in zip(free_clusters, free_levels):
            cluster_to_level[c] = l

    levels = cluster_to_level[assign][bin_map]

    palette = np.empty_like(disp)
    coverage = np.zeros(n_levels)
    for k in range(n_levels):
        palette[cluster_to_level[k]] = disp[k]
        coverage[cluster_to_level[k]] = counts[assign == k].sum() / total

    report = LevelReport(
        n_levels=n_levels,
        palette_rgb=palette,
        coverage=coverage,
        mean_residual=mean_residual,
        bad_fraction=bad_fraction,
        warnings=warnings,
    )
    return levels, report
