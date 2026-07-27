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
| `--xy-um` | printer XY pixel in microns (default 19); sets working resolution |
| `--max-px` | override working resolution; lower is faster but leaves visible steps |

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

## Edge smoothness

Regions are built as columns on a pixel grid, so region boundaries are
staircases rather than smooth curves. What matters is the step size relative
to the printer: at `--xy-um` (default 19, the Saturn 4 Ultra's XY pixel) a step
is one printer pixel, which the printer cannot render as a visible ledge. The
working resolution defaults to whatever achieves that, and every run reports
the step size it actually got:

```
edge resolution: 19 um steps at 90.0 mm wide — below the printer's XY pixel,
                 so edges cannot step visibly
```

**The source image is the hard ceiling.** Upsampling a small image only makes
bigger copies of the same staircase, so an 800 px source cannot produce smooth
edges at any setting — the run warns when this is the case and tells you the
resolution you need. `gen` requests 4K by default for this reason, which covers
a 90 mm piece at 19 µm.

Cost scales with resolution: the poster example at 90 mm wide is 4.0 MB at
`--max-px 512` (176 µm steps, visibly stepped) and 54 MB at the 19 µm default.
Use `--max-px 512` while iterating on levels and depth, then drop it for the
final build.

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
  happens downstream (e.g. Lychee). See [Printing](#printing) for orientation
  and plate thickness.

## Source image requirements

Flat and graphic: large solid color regions, no gradients, no shading, no
texture, no outlines, no tiny details. The `gen` prompt template asks Gemini
for exactly this. Unsuitable images (photos, gradients) are detected and
flagged rather than silently producing garbage.

## Printing

This tool stops at the STL — slicing and printing are yours. But the output is
shaped by a few assumptions worth writing down, because they interact with the
options above. None of this is verified against a real print yet; treat the
numbers as starting points, not measurements.

**Orientation: flat on the build plate, face up, no supports.** The back face
is a glue surface. Tilting the model onto supports would leave nubs across it
and ruin that, so the flat orientation is worth its costs. It picks up the
build plate's texture, which if anything helps adhesion, and first-layer
squish makes the plate run slightly thicker than `--base-mm` — measure a test
piece if the total thickness has to be exact.

**Backing plate thickness.** `--base-mm` accepts anything down to one layer,
and will happily build a plate too thin to survive handling. Nothing warns
you. Practical floors:

| Piece width | `--base-mm` |
| --- | --- |
| up to ~100 mm | 2.0 |
| 120–150 mm | 2.5–3.0 |
| beyond that | 3.0+, roughly span/50 |

Two things drive this, and neither is about print forces: a flat resin panel
cups as it post-cures, and thin panels cup worse; and a large flat area
printed directly on the plate adheres hard enough that prising a thin one off
can crack it. A thin flexible scraper and a slightly warmed plate help.

**A large flat bottom layer is demanding on printers without vat tilt.** A
120 × 90 mm cross-section parallel to the plate is close to the worst case for
straight-pull peel force. Printers with a tilting vat release (the Saturn 4
Ultra among them) peel such a layer off progressively and largely remove this
as a concern. On a conventional straight-lift printer it is a real constraint,
and the fixes are all downstream of this tool — slower lift speeds, longer
rest before lift, and a thicker plate for margin against delamination. If your
printer lifts straight up, start at the thicker end of the table above and
expect to tune lift settings rather than assuming the defaults transfer.

**Layer height.** `--layer-mm` defaults to 0.05, which the Saturn 4 Ultra
prints reliably. Set it to whatever your printer and resin actually do — level
heights snap to it, so a mismatch means the slicer quietly rounds your relief
somewhere other than where this tool reported it.
