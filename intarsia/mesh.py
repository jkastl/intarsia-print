"""Level map -> watertight binary STL.

Geometry is a voxel-column surface: one quad per pixel on top, banded
vertical walls at level changes, flat bottom at Z=0. Every vertex lies on
the integer grid (column i, row j, z-level index), and walls are split at
every intermediate level height, so shared edges match exactly — no
T-junctions, no pinholes, no contour/nested-hole handling at all.

Coordinates: Z-up, bottom at Z=0, x = width (exact mm), y = image height,
image top row at max y so the STL matches the image seen from above.
"""

import struct

import numpy as np


def level_heights(n_levels, base_mm, step_mm):
    """Top surface height of each level. Level 0 is flush with the plate."""
    return base_mm + np.arange(n_levels) * step_mm


def build_mesh(levels, px_mm, base_mm, step_mm):
    """Returns (verts_int (V,3) int32 grid coords, tris (T,3) vertex ids,
    zvals) — grid coords are (i, j, zindex); zvals maps zindex -> mm."""
    lv = np.asarray(levels)
    H, W = lv.shape
    n = int(lv.max()) + 1
    # z index of a pixel's top = level + 1; zvals[0] = 0 (the bottom).
    zvals = np.concatenate([[0.0], level_heights(n, base_mm, step_mm)])
    zi = lv.astype(np.int64) + 1

    quads = []  # each: (i0, j0, i1, j1, zA, zB, kind)

    jj, ii = np.mgrid[0:H, 0:W]

    def emit_horiz(z_idx, flip):
        """Full-coverage horizontal quads at per-pixel z (tops or bottom)."""
        i, j, z = ii.ravel(), jj.ravel(), np.broadcast_to(z_idx, (H, W)).ravel()
        quads.append(("h", i, j, z, np.full(z.shape, flip, dtype=bool)))

    emit_horiz(zi, False)  # tops, +Z
    emit_horiz(np.zeros_like(zi), True)  # bottom, -Z

    # Interior + perimeter walls, banded at every level height.
    # For x-facing walls at column boundary b (0..W): left pixel is (j, b-1),
    # right pixel is (j, b); off the image the height index is 0.
    zx = np.zeros((H, W + 1), dtype=np.int64)
    zx_l = np.zeros((H, W + 1), dtype=np.int64)
    zx_l[:, 1:] = zi
    zx[:, :-1] = zi
    zy_t = np.zeros((H + 1, W), dtype=np.int64)  # pixel above boundary row
    zy_b = np.zeros((H + 1, W), dtype=np.int64)
    zy_t[1:, :] = zi
    zy_b[:-1, :] = zi

    for k in range(len(zvals) - 1):  # band k spans zvals[k]..zvals[k+1]
        # x-facing: wall exists where band k is covered on one side only
        lo = np.minimum(zx_l, zx)
        hi = np.maximum(zx_l, zx)
        m = (lo <= k) & (k < hi)
        if m.any():
            j, b = np.nonzero(m)
            # unflipped x-wall faces +x, which is outward when solid is on the
            # left; flip when the right column is the taller (solid) one
            flip = zx[m] > zx_l[m]
            quads.append(("x", b, j, np.full(j.shape, k), flip))
        lo = np.minimum(zy_t, zy_b)
        hi = np.maximum(zy_t, zy_b)
        m = (lo <= k) & (k < hi)
        if m.any():
            b, i = np.nonzero(m)
            # unflipped y-wall faces -y; image rows grow toward -y, so that is
            # outward when the solid is the upper pixel (smaller j, zy_t)
            flip = zy_b[m] > zy_t[m]
            quads.append(("y", i, b, np.full(i.shape, k), flip))

    # Assemble quads into triangles with integer corner vertices.
    tri_corners = []
    for kind, a, b, z, flip in quads:
        if kind == "h":
            # pixel (j=b? no: a=i, b=j) corners at z
            i, j = a, b
            c0 = np.stack([i, j + 1, z], 1)  # x0, y_low
            c1 = np.stack([i + 1, j + 1, z], 1)
            c2 = np.stack([i + 1, j, z], 1)
            c3 = np.stack([i, j, z], 1)
        elif kind == "x":
            # boundary at x=a, spans y over row j=b, z band z..z+1
            i, j = a, b
            c0 = np.stack([i, j + 1, z], 1)
            c1 = np.stack([i, j, z], 1)
            c2 = np.stack([i, j, z + 1], 1)
            c3 = np.stack([i, j + 1, z + 1], 1)
        else:  # "y": boundary at row j=b, spans x over column i=a
            i, j = a, b
            c0 = np.stack([i, j, z], 1)
            c1 = np.stack([i + 1, j, z], 1)
            c2 = np.stack([i + 1, j, z + 1], 1)
            c3 = np.stack([i, j, z + 1], 1)
        f = flip[:, None]
        # two triangles per quad: (c0,c1,c2) and (c0,c2,c3), reversed when flipped
        t1 = np.where(f[:, :, None], np.stack([c0, c2, c1], 1), np.stack([c0, c1, c2], 1))
        t2 = np.where(f[:, :, None], np.stack([c0, c3, c2], 1), np.stack([c0, c2, c3], 1))
        tri_corners.append(t1)
        tri_corners.append(t2)

    tris_int = np.concatenate(tri_corners)  # (T, 3, 3) integer grid coords
    return tris_int, zvals


def to_float_coords(tris_int, zvals, px_mm, H):
    """Integer grid coords -> mm coords (float64). Image row 0 -> max y."""
    out = np.empty(tris_int.shape, dtype=np.float64)
    out[..., 0] = tris_int[..., 0] * px_mm
    out[..., 1] = (H - tris_int[..., 1]) * px_mm
    out[..., 2] = zvals[tris_int[..., 2]]
    return out


def check_mesh(tris_int, tris_mm, levels, px_mm, base_mm, step_mm):
    """Closure, orientation (signed volume vs analytic), non-manifold count."""
    t = tris_int
    key = (
        t[..., 0].astype(np.int64) * (2**42)
        + t[..., 1].astype(np.int64) * (2**21)
        + t[..., 2].astype(np.int64)
    )  # unique vertex id per grid point
    edges = np.concatenate([key[:, [0, 1]], key[:, [1, 2]], key[:, [2, 0]]])
    # undirected counts
    und = np.sort(edges, axis=1)
    _, counts = np.unique(und, axis=0, return_counts=True)
    open_edges = int((counts % 2).sum())
    nonmanifold = int((counts > 2).sum())
    # directed matching: every (a,b) must be matched by a (b,a)
    fwd, fcnt = np.unique(edges, axis=0, return_counts=True)
    rev_order = np.lexsort((fwd[:, 0], fwd[:, 1]))
    matched = bool(
        np.array_equal(fwd[rev_order][:, ::-1], fwd) and np.array_equal(fcnt[rev_order], fcnt)
    )

    v0, v1, v2 = tris_mm[:, 0], tris_mm[:, 1], tris_mm[:, 2]
    signed_vol = float(np.einsum("ij,ij->", v0, np.cross(v1, v2)) / 6.0)
    heights = level_heights(int(levels.max()) + 1, base_mm, step_mm)
    true_vol = float((heights[levels] * px_mm * px_mm).sum())

    return {
        "watertight": open_edges == 0 and matched,
        "open_edges": open_edges,
        "nonmanifold_edges": nonmanifold,
        "signed_volume_mm3": signed_vol,
        "analytic_volume_mm3": true_vol,
        "volume_ok": abs(signed_vol - true_vol) < max(1e-6 * true_vol, 1e-6),
    }


def write_stl(path, tris_mm, name=b"intarsia-print"):
    tris = tris_mm.astype(np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    T = len(tris)
    rec = np.zeros(T, dtype=[("n", "<3f4"), ("v", "<(3,3)f4"), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(struct.pack("<80s", name))
        f.write(struct.pack("<I", T))
        rec.tofile(f)
