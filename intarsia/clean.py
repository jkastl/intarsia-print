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


def _break_diagonal_pinches(levels, n_levels, max_passes=16):
    """Remove corner pinches, where solid meets solid only diagonally.

    At such a corner four wall faces share one vertical edge: non-manifold,
    and it prints as a zero-thickness join that snaps off during cleanup.

    The mesh is built in height bands, so the test is per band, not per
    level. Around a 2x2 of heights a b / c d, band k is pinched when one
    diagonal is solid there and the other is not, which happens for some k
    exactly when one diagonal lies entirely above the other. Unequal levels
    pinch too — 3 0 / 0 2 is solid on one diagonal only in bands 0 and 1 —
    so comparing levels for equality is not enough.

    Each fix raises the lower pixel to its diagonal neighbour's level,
    filling the pinch rather than severing it. Levels only ever increase, so
    this terminates.
    """
    lv = levels.copy()
    fixed = 0
    for _ in range(max_passes):
        a, b = lv[:-1, :-1], lv[:-1, 1:]
        c, d = lv[1:, :-1], lv[1:, 1:]
        main_hi = np.maximum(b, c) < np.minimum(a, d)  # a,d strictly above b,c
        anti_hi = np.maximum(a, d) < np.minimum(b, c)  # b,c strictly above a,d
        site = main_hi | anti_hi
        if not site.any():
            break
        jj, ii = np.nonzero(site)
        mh = main_hi[jj, ii]
        # main high -> raise bottom-left to a; anti high -> raise top-left to b
        lv[np.where(mh, jj + 1, jj), ii] = np.where(mh, a[jj, ii], b[jj, ii])
        fixed += len(jj)
    return lv, fixed


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

    levels, pinches = _break_diagonal_pinches(levels, n_levels)

    stats = {
        "changed_fraction": float((levels != before).mean()),
        "coverage": np.bincount(levels.ravel(), minlength=n_levels) / levels.size,
        "pinches_fixed": pinches,
    }
    return levels, stats
