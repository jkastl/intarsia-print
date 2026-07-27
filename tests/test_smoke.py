"""Minimal smoke test: tiny image -> STL, must be watertight with exact
volume and exact physical size. Run with: python3 tests/test_smoke.py"""

import numpy as np

from intarsia.clean import clean_levels
from intarsia.mesh import build_mesh, check_mesh, to_float_coords, uniform_heights
from intarsia.quantize import quantize
from PIL import Image


def test_pipeline():
    # 3-color image with a nested hole: ring around a hole around an island
    img = np.zeros((60, 80, 3), np.uint8)
    img[:] = (240, 230, 210)
    yy, xx = np.mgrid[0:60, 0:80]
    r2 = (xx - 40) ** 2 + (yy - 30) ** 2
    img[r2 < 400] = (200, 60, 30)
    img[r2 < 200] = (240, 230, 210)
    img[r2 < 60] = (30, 60, 120)

    levels, report = quantize(Image.fromarray(img), 3)
    assert sorted(np.unique(levels)) == [0, 1, 2]
    levels, _ = clean_levels(levels, 3, 2.0)

    px_mm = 100.0 / 80
    heights = uniform_heights(3, base_mm=2.0, step_mm=0.4)
    tris_int, zvals = build_mesh(levels, heights)
    tris_mm = to_float_coords(tris_int, zvals, px_mm, 60)
    chk = check_mesh(tris_int, tris_mm, levels, px_mm, heights)
    assert chk["watertight"], chk
    assert chk["volume_ok"], chk

    flat = tris_mm.reshape(-1, 3)
    assert abs(flat[:, 0].max() - 100.0) < 1e-9  # exact width
    assert flat[:, 2].min() == 0.0  # flat bottom on Z=0
    print("smoke test OK:", len(tris_mm), "triangles, volume",
          round(chk["signed_volume_mm3"], 1), "mm3")


if __name__ == "__main__":
    test_pipeline()
