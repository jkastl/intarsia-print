"""Geometry hygiene between quantization and meshing.

Anti-aliased edges and JPEG noise leave one-pixel slivers between levels that
become real print geometry; features thinner than ~0.5 mm at final scale won't
survive desupporting. This pass removes both, then reassigns every orphaned
pixel to its nearest surviving region — no pixel may end up unassigned, or the
mesh gets pinholes.
"""

import numpy as np
from scipy import ndimage

# 4-connectivity everywhere: diagonal-only connections produce non-manifold
# knife edges in the mesh, so regions joined only at a corner are separate.
_CROSS = ndimage.generate_binary_structure(2, 1)


def _disk(radius_px):
    r = max(1, int(round(radius_px)))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def _mode_filter(levels, n_levels, passes=2):
    """3x3 majority vote; melts single-pixel anti-alias noise into neighbors."""
    for _ in range(passes):
        counts = np.stack(
            [
                ndimage.uniform_filter((levels == k).astype(np.float32), size=3, mode="nearest")
                for k in range(n_levels)
            ]
        )
        levels = counts.argmax(axis=0).astype(levels.dtype)
    return levels


def clean_levels(levels, n_levels, min_feature_px):
    """Returns (cleaned levels array, stats dict)."""
    before = levels
    levels = _mode_filter(levels, n_levels)

    unassigned = np.zeros(levels.shape, dtype=bool)

    # Drop connected components smaller than a min_feature x min_feature square.
    min_area = max(4, int(round(min_feature_px * min_feature_px)))
    for k in range(n_levels):
        lab, n = ndimage.label(levels == k, structure=_CROSS)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        small = sizes < min_area
        small[0] = False
        unassigned |= small[lab]

    # Drop parts thinner than min_feature: erosion (border counts as solid so
    # shapes touching the frame aren't thinned from outside) then dilation;
    # what the opening can't rebuild is a sliver.
    disk = _disk(min_feature_px / 2)
    for k in range(n_levels):
        mask = (levels == k) & ~unassigned
        if not mask.any():
            continue
        opened = ndimage.binary_dilation(
            ndimage.binary_erosion(mask, structure=disk, border_value=1),
            structure=disk,
        )
        unassigned |= mask & ~opened

    if unassigned.all():
        raise SystemExit("error: cleanup removed everything — image is all sub-printable noise")

    # Reassign orphans to the nearest surviving pixel's level.
    if unassigned.any():
        _, (iy, ix) = ndimage.distance_transform_edt(unassigned, return_indices=True)
        levels = levels.copy()
        levels[unassigned] = levels[iy[unassigned], ix[unassigned]]

    stats = {
        "changed_fraction": float((levels != before).mean()),
        "coverage": np.bincount(levels.ravel(), minlength=n_levels) / levels.size,
    }
    return levels, stats
