"""sRGB <-> CIELAB conversion, vectorized, no dependencies beyond numpy."""

import numpy as np

# D65 reference white
_WHITE = np.array([0.95047, 1.0, 1.08883])
_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def srgb_to_lab(rgb):
    """rgb: float array (..., 3) in [0, 1] -> Lab array (..., 3)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    lab = np.empty_like(xyz)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


def delta_e(lab1, lab2):
    """Plain CIE76 distance; adequate for separating flat graphic colors."""
    return np.linalg.norm(np.asarray(lab1) - np.asarray(lab2), axis=-1)
