# intarsia-print

A headless Python CLI: a text prompt or a flat graphic image goes in, a
printable multi-level bas-relief STL comes out. Distinct color regions become
discrete height levels stacked on a flat backing plate — think intarsia, or a
topographic map made from a poster.

Relief is deliberately shallow: the intended output is a plaque that gets
printed, painted, and mounted on a sign, so the default is 1.6 mm of relief on
a 2 mm plate.

The image→STL pipeline is deterministic: same input, same output, no network.
Gemini is used only for the two steps that need to understand a picture —
generating the source image, and judging scene depth — and both print flags
that reproduce their result offline.

## Install

```sh
pip install -e .          # or: pip install numpy pillow scipy
```

For image generation, set `GEMINI_API_KEY` (get one at
https://aistudio.google.com/apikey).

## Usage

End to end — prompt → image → **look at the image and approve it** → STL:

```sh
intarsia run "a fox sitting under a pine tree" -n 5 --width-mm 120 -o fox.stl
```

Step by step:

From a photo — attach a reference picture and say what to keep from it:

```sh
intarsia run "only the dog's head from this photo, facing forward" \
    --from-image my-dog.jpg -n 4 --width-mm 100 -o dog.stl
```

Gemini redraws the subject as flat poster art (it does not trace the photo);
the usual approval stop lets you reject and retry until the flattening looks
right.

Step by step:

```sh
# 1. generate a flat, poster-style source image (or bring your own)
intarsia gen "a fox sitting under a pine tree" -n 5 -o fox.png

# 2. check how the colors will split into height levels
intarsia levels fox.png -n 5 --width-mm 120 -o fox-levels.png

# 3. mesh it
intarsia build fox.png -n 5 --width-mm 120 -o fox.stl
```

Every command writes a preview PNG. **Open it.** Every failure mode in this
pipeline is silent and visual — a bad separation runs without error and
produces an unprintable model.

## Key options

| Option | Meaning |
| --- | --- |
| `-n, --levels` | number of height levels (default 5) |
| `--width-mm` | physical width, exact in the STL (default 100) |
| `--base-mm` | backing plate thickness (default 2.0) |
| `--step-mm` | height difference per level, uniform mode (default 0.4) |
| `--relief-mm` | total relief with `--depth-order` (default 1.6) |
| `--layer-mm` | printer layer height; all level tops snap to a multiple (default 0.05) |
| `--min-step-mm` | floor on the gap between adjacent levels (default 0.1) |
| `--heights` | explicit absolute height per level, overriding all of the above |
| `--order` | `dark-low` (default) or `light-low` |
| `--depth-order` | order levels by real-world distance instead of brightness (see below) |
| `--assign '#rrggbb=2'` | force the region nearest that color to level 2; repeatable |
| `--min-feature` | smallest printable feature in mm; thinner detail is absorbed (default 0.5) |
| `--max-px` | working resolution cap; higher = finer detail (default 512) |

When the automatic level assignment picks badly, read the palette hex codes
from the `levels` report and pin regions with `--assign`.

## Realistic depth

By default levels follow brightness, which is arbitrary with respect to the
scene: a dark foreground and a dark sky land on the same height, and every
level is one fixed step above the last. `--depth-order` instead places levels
by how far the depicted things actually are from the viewer:

```sh
intarsia build dog.png -n 5 --width-mm 100 --depth-order -o dog.stl
```

Two things change. The **order** follows the scene — background flush with the
plate, the nearest parts (a nose, a foreground paw) protruding furthest. And
the **spacing** is proportional rather than fixed: things at nearly the same
distance sit at nearly the same height, while a distant background drops well
below the subject. A sky and a sun at almost the same distance collapse to the
`--min-step-mm` floor; a foreground that is metres nearer than the midground
gets a correspondingly larger gap:

```
scene depth (farthest -> nearest): #e47c2d@100, #f4ecd4@95, #2c6c74@55, ...
heights (scene-relative over 1.6 mm of relief, snapped to 50 um layers):
  L0 2.00, L1 2.10, L2 2.75, L3 3.15, L4 3.60 mm
```

This is the one judgement in the pipeline that needs to understand the
picture, so it asks Gemini and needs `GEMINI_API_KEY`. It prints both the
`--assign` and `--heights` flags that reproduce the result exactly, offline
and deterministically. Explicit `--assign` pins always win over the ranking,
so you can correct one region and let Gemini place the rest.

## Layer heights

Every level top is snapped to a whole multiple of `--layer-mm` (default 0.05,
the Saturn 4 Ultra's reliable layer height). The slicer would round these
anyway; doing it here means the STL, the previews and the reported numbers all
agree with what actually gets printed. `--min-step-mm` keeps adjacent levels
from collapsing into each other when the scene puts them at nearly the same
depth.

## Output

- Binary STL, watertight, Z-up, flat bottom on Z=0, level 0 flush with the
  backing plate. Exact `--width-mm` wide; height follows the image aspect.
- Self-checked before writing: edge closure, orientation, non-manifold edges,
  and signed volume against the volume computed straight from the level map.
  The tool refuses to emit a mesh that fails.
- Flat areas are merged, so cost scales with the length of the edges between
  regions rather than with pixel count: raising `--max-px` from 512 to 2048
  takes the poster example from 4.0 MB to 6.8 MB, not 16x that.
- Sized against the Elegoo Saturn 4 Ultra plate (218 × 123 mm); slicing
  happens downstream (e.g. Lychee).

## Source image requirements

Flat and graphic: large solid color regions, no gradients, no shading, no
texture, no outlines, no tiny details. The `gen` prompt template asks Gemini
for exactly this. Unsuitable images (photos, gradients) are detected and
flagged rather than silently producing garbage.
