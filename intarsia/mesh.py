"""Level map -> watertight binary STL.

Geometry is a voxel-column surface: flat tops at each level, banded vertical
walls at level changes, flat bottom at Z=0. Every vertex lies on a doubled
integer grid (column i, row j, z-level index) so rectangle centres stay
exact, and walls are split at every intermediate level height, so shared
edges match exactly — no T-junctions, no pinholes, no contour/nested-hole
handling at all.

Flat areas are merged: a quadtree finds constant-level rectangles and each
becomes a fan from its centre that still carries every integer point on its
boundary. That keeps the vertex-exactness above while making a large flat
region cost its perimeter rather than its area.

Coordinates: Z-up, bottom at Z=0, x = width (exact mm), y = image height,
image top row at max y so the STL matches the image seen from above.
"""

import struct

import numpy as np


def uniform_heights(n_levels, base_mm, step_mm):
    """Evenly spaced level tops. Level 0 is flush with the plate."""
    return base_mm + np.arange(n_levels) * step_mm


def snap_heights(heights, layer_mm, base_mm, min_step_mm):
    """Round level tops to whole printer layers, keeping them strictly
    increasing by at least min_step_mm (also layer-rounded).

    Resin prints in discrete layers, so a height that isn't a layer multiple
    is silently rounded by the slicer anyway — doing it here means the STL,
    the previews and the reported numbers all agree with what gets printed.
    """
    heights = np.asarray(heights, dtype=np.float64)
    step_layers = max(1, int(round(min_step_mm / layer_mm)))
    out = np.round(heights / layer_mm).astype(np.int64)
    out[0] = round(base_mm / layer_mm)
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1] + step_layers)
    return out * layer_mm


def build_mesh(levels, heights_mm):
    """Returns (tris_int (T,3,3) integer grid coords, zvals) — grid coords are
    (i, j, zindex); zvals maps zindex -> mm."""
    lv = np.asarray(levels)
    H, W = lv.shape
    n = int(lv.max()) + 1
    heights = np.asarray(heights_mm, dtype=np.float64)
    if len(heights) < n:
        raise ValueError(f"got {len(heights)} heights for {n} levels")
    if np.any(np.diff(heights) <= 0) or heights[0] <= 0:
        raise ValueError("heights must be positive and strictly increasing")
    # z index of a pixel's top = level + 1; zvals[0] = 0 (the bottom).
    zvals = np.concatenate([[0.0], heights[:n]])
    zi = lv.astype(np.int64) + 1

    faces = []  # flat horizontal surfaces, already triangulated
    quads = []  # vertical walls, assembled below

    # Tops: one flat face per constant-level rectangle, not per pixel.
    for i0, j0, i1, j1, k in _quadtree_rects(lv):
        faces.append(_flat_rect(i0, j0, i1, j1, k + 1, flip=False))
    # Bottom: the whole plate is one rectangle.
    faces.append(_flat_rect(0, 0, W, H, 0, flip=True))

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

    # Assemble wall quads into triangles with integer corner vertices.
    tri_corners = list(faces)
    for kind, a, b, z, flip in quads:
        if kind == "x":
            # boundary at x=a, spans y over row j=b, z band z..z+1
            i, j = a * 2, b * 2
            c0 = np.stack([i, j + 2, z], 1)
            c1 = np.stack([i, j, z], 1)
            c2 = np.stack([i, j, z + 1], 1)
            c3 = np.stack([i, j + 2, z + 1], 1)
        else:  # "y": boundary at row j=b, spans x over column i=a
            i, j = a * 2, b * 2
            c0 = np.stack([i, j, z], 1)
            c1 = np.stack([i + 2, j, z], 1)
            c2 = np.stack([i + 2, j, z + 1], 1)
            c3 = np.stack([i, j, z + 1], 1)
        f = flip[:, None]
        # two triangles per quad: (c0,c1,c2) and (c0,c2,c3), reversed when flipped
        t1 = np.where(f[:, :, None], np.stack([c0, c2, c1], 1), np.stack([c0, c1, c2], 1))
        t2 = np.where(f[:, :, None], np.stack([c0, c3, c2], 1), np.stack([c0, c2, c3], 1))
        tri_corners.append(t1)
        tri_corners.append(t2)

    tris_int = np.concatenate(tri_corners)  # (T, 3, 3) integer grid coords
    return tris_int, zvals


def _quadtree_rects(lv):
    """Split the level map into axis-aligned rectangles of constant level.

    A quadtree rather than greedy strip meshing: a rectangle's cost here is
    its perimeter (see _flat_rect), so long thin strips — which is what
    greedy meshing degenerates to along a diagonal edge — save nothing.
    Recursive halving keeps big interior blocks square and only refines
    down to single pixels where levels actually change.
    """
    H, W = lv.shape
    out = []
    stack = [(0, 0, W, H)]
    while stack:
        i0, j0, i1, j1 = stack.pop()
        block = lv[j0:j1, i0:i1]
        first = block[0, 0]
        if (block == first).all():
            out.append((i0, j0, i1, j1, int(first)))
            continue
        im, jm = (i0 + i1) // 2, (j0 + j1) // 2
        for a, b, c, d in ((i0, j0, im, jm), (im, j0, i1, jm),
                           (i0, jm, im, j1), (im, jm, i1, j1)):
            if c > a and d > b:
                stack.append((a, b, c, d))
    return out


def _perimeter(i0, j0, i1, j1):
    """Every integer grid point on the rectangle boundary, walked in the order
    that gives a +Z normal. Doubled coordinates."""
    xs = np.arange(i0, i1 + 1) * 2
    ys = np.arange(j0, j1 + 1) * 2
    bottom = np.stack([xs, np.full(xs.shape, j1 * 2)], 1)
    right = np.stack([np.full(ys.shape, i1 * 2), ys[::-1]], 1)
    top = np.stack([xs[::-1], np.full(xs.shape, j0 * 2)], 1)
    left = np.stack([np.full(ys.shape, i0 * 2), ys], 1)
    # each run repeats the next run's first point, so drop the last of each
    return np.concatenate([bottom[:-1], right[:-1], top[:-1], left[:-1]])


def _flat_rect(i0, j0, i1, j1, z, flip):
    """A flat rectangle as a fan from its centre, with every integer boundary
    point as a vertex.

    Keeping the boundary points is what makes merging safe: a 64x64 block
    still meets its single-pixel neighbours vertex-to-vertex, so no
    T-junctions appear and the mesh stays closed under edge matching. Cost is
    therefore the perimeter, 2*(w+h) triangles, versus 2*w*h per pixel — so
    for 1-wide strips, where that trade loses, fall back to per-pixel quads.
    """
    w, h = i1 - i0, j1 - j0
    if w == 1 or h == 1:
        jj, ii = np.mgrid[j0:j1, i0:i1]
        i, j = ii.ravel() * 2, jj.ravel() * 2
        zz = np.full(i.shape, z)
        c0 = np.stack([i, j + 2, zz], 1)
        c1 = np.stack([i + 2, j + 2, zz], 1)
        c2 = np.stack([i + 2, j, zz], 1)
        c3 = np.stack([i, j, zz], 1)
        tris = np.concatenate([np.stack([c0, c1, c2], 1), np.stack([c0, c2, c3], 1)])
        return tris[:, ::-1, :] if flip else tris

    P = _perimeter(i0, j0, i1, j1)
    Q = np.roll(P, -1, axis=0)
    tri = np.empty((len(P), 3, 3), dtype=np.int64)
    tri[:, 0, :2] = (i0 + i1, j0 + j1)  # centre: exact in doubled coords
    tri[:, 1, :2] = P
    tri[:, 2, :2] = Q
    tri[:, :, 2] = z
    return tri[:, ::-1, :] if flip else tri


def to_float_coords(tris_int, zvals, px_mm, H):
    """Doubled integer grid coords -> mm coords (float64). Row 0 -> max y."""
    out = np.empty(tris_int.shape, dtype=np.float64)
    out[..., 0] = tris_int[..., 0] * (px_mm / 2)
    out[..., 1] = (2 * H - tris_int[..., 1]) * (px_mm / 2)
    out[..., 2] = zvals[tris_int[..., 2]]
    return out


def check_mesh(tris_int, tris_mm, levels, px_mm, heights_mm):
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
    true_vol = float((np.asarray(heights_mm)[levels] * px_mm * px_mm).sum())

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
