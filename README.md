# intarsia-print

A headless Python CLI: a text prompt or a flat graphic image goes in, a
printable multi-level bas-relief STL comes out. Distinct color regions become
discrete height levels stacked on a flat backing plate — think intarsia, or a
topographic map made from a poster. See [spec.md](spec.md).

The image→STL pipeline is deterministic: same input, same output. The only
network call is the optional Gemini image generation step.

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
| `--step-mm` | height difference per level (default 1.0) |
| `--order` | `dark-low` (default) or `light-low` |
| `--depth-order` | order levels by real-world distance instead of brightness (see below) |
| `--assign '#rrggbb=2'` | force the region nearest that color to level 2; repeatable |
| `--min-feature` | smallest printable feature in mm; thinner detail is absorbed (default 0.5) |
| `--max-px` | working resolution cap; higher = finer detail, bigger STL (default 512) |

When the automatic level assignment picks badly, read the palette hex codes
from the `levels` report and pin regions with `--assign`.

## Realistic depth ordering

By default levels follow brightness, which is arbitrary with respect to the
scene: a dark foreground and a dark sky land on the same height. `--depth-order`
instead stacks levels by how far the depicted things actually are from the
viewer — background flush with the plate, the nearest parts (a nose, a
foreground paw) protruding furthest:

```sh
intarsia build dog.png -n 5 --width-mm 100 --depth-order -o dog.stl
```

This is the one judgement in the pipeline that needs to understand the
picture, so it asks Gemini to rank the palette and needs `GEMINI_API_KEY`.
It prints the equivalent `--assign` flags, so the exact same result can be
rebuilt offline and deterministically:

```
depth order (farthest -> nearest): #f4ecd4 #e47c2d #2c6c74 #b44c4c #1c3c2c
reproduce offline with: --assign '#f4ecd4=0' --assign '#e47c2d=1' ...
```

Explicit `--assign` pins always win over the ranking, so you can correct one
region and let Gemini order the rest.

## Output

- Binary STL, watertight, Z-up, flat bottom on Z=0, level 0 flush with the
  backing plate. Exact `--width-mm` wide; height follows the image aspect.
- Self-checked before writing: edge closure, orientation, signed volume vs
  analytic volume. The tool refuses to emit a mesh that fails.
- Sized against the Elegoo Saturn 4 Ultra plate (218 × 123 mm); slicing
  happens downstream (e.g. Lychee).

## Source image requirements

Flat and graphic: large solid color regions, no gradients, no shading, no
texture, no outlines, no tiny details. The `gen` prompt template asks Gemini
for exactly this. Unsuitable images (photos, gradients) are detected and
flagged rather than silently producing garbage.
