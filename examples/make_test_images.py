"""Generate deterministic test images that exercise the known failure modes:
anti-aliased edges, nested holes (island inside a hole inside a shape),
similar-brightness hues, and a gradient image that should be rejected."""

import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def poster(path, aa=True):
    """Flat 5-color poster with a nested hole: ring -> hole -> island."""
    s = 4 if aa else 1  # supersample then downscale = anti-aliased edges
    W, H = 800 * s, 600 * s
    img = Image.new("RGB", (W, H), "#f2e8d5")  # cream background
    d = ImageDraw.Draw(img)
    # big teal mountain silhouette
    d.polygon([(0, H), (0, int(H * 0.55)), (int(W * 0.35), int(H * 0.18)),
               (int(W * 0.62), int(H * 0.62)), (int(W * 0.8), int(H * 0.4)),
               (W, int(H * 0.7)), (W, H)], fill="#2a6f77")
    # dark green foreground hills
    d.polygon([(0, H), (0, int(H * 0.8)), (int(W * 0.3), int(H * 0.68)),
               (int(W * 0.7), int(H * 0.85)), (W, int(H * 0.75)), (W, H)],
              fill="#1d3b2a")
    # orange sun-ring with a hole, and an island inside the hole (nested holes)
    cx, cy, r = int(W * 0.72), int(H * 0.25), int(W * 0.11)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#e07a2f")
    d.ellipse([cx - r * 0.62, cy - r * 0.62, cx + r * 0.62, cy + r * 0.62], fill="#f2e8d5")
    d.ellipse([cx - r * 0.3, cy - r * 0.3, cx + r * 0.3, cy + r * 0.3], fill="#e07a2f")
    # similar-brightness hue pair: red bird on the teal (hard separation)
    bx, by, br = int(W * 0.3), int(H * 0.42), int(W * 0.04)
    d.polygon([(bx - br * 2, by), (bx, by - br), (bx + br * 2, by), (bx, by + br)],
              fill="#b5484a")
    if s > 1:
        img = img.resize((W // s, H // s), Image.LANCZOS)
    img.save(path)


def gradient(path):
    """Photograph-ish gradient image: the tool should flag this as unsuitable."""
    W, H = 800, 600
    x = np.linspace(0, 1, W)[None, :]
    y = np.linspace(0, 1, H)[:, None]
    r = (np.sin(3 * x + y) * 0.5 + 0.5) * 255
    g = (x * y) * 255
    b = (1 - x) * 255 * np.ones_like(y)
    img = np.stack([r * np.ones_like(y), g, b], axis=-1).astype(np.uint8)
    Image.fromarray(img).save(path)


if __name__ == "__main__":
    poster(os.path.join(HERE, "poster.png"), aa=True)
    poster(os.path.join(HERE, "poster-hard.jpg"), aa=True)  # jpeg artifacts on top
    gradient(os.path.join(HERE, "gradient.png"))
    print("wrote poster.png, poster-hard.jpg, gradient.png")
