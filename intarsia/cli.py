"""intarsia-print CLI."""

import argparse
import io
import os
import sys

import numpy as np
from PIL import Image

from .clean import clean_levels
from .color import delta_e, srgb_to_lab
from .gemini_gen import build_prompt, depth_profile, generate_image
from .mesh import (
    build_mesh,
    check_mesh,
    snap_heights,
    to_float_coords,
    uniform_heights,
    write_stl,
)
from .preview import hillshade_image, level_map_image, side_by_side
from .quantize import quantize

# Elegoo Saturn 4 Ultra
BUILD_X_MM, BUILD_Y_MM = 218.0, 123.0


def _parse_assign(items):
    """['#rrggbb=2', ...] -> {'#rrggbb': 2}"""
    out = {}
    for item in items or []:
        try:
            color, lvl = item.split("=")
            out[color] = int(lvl)
        except ValueError:
            raise SystemExit(f"error: bad --assign '{item}', expected '#rrggbb=LEVEL'")
    return out


def _print_report(report):
    for lvl in range(report.n_levels - 1, -1, -1):
        cov = report.coverage[lvl]
        flag = "  <-- suspiciously small: separation error?" if cov < 0.005 else ""
        print(f"  L{lvl}  {report.palette_hex()[lvl]}  {cov * 100:5.1f}%{flag}")
    print(f"  fit: mean residual deltaE {report.mean_residual:.1f}, "
          f"{report.bad_fraction:.1%} of pixels unexplained")
    for w in report.warnings:
        print(f"  warning: {w}", file=sys.stderr)


def _hex_lab(h):
    c = h.lstrip("#")
    return srgb_to_lab(np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0)


def _depth_overrides(args, levels, report):
    """Ask Gemini how far away each palette color is; explicit --assign pins
    win. Returns ({palette_hex: level}, relative nearness per level in 0..1)."""
    hexes = [h.lower() for h in report.palette_hex()]
    buf = io.BytesIO()
    level_map_image(levels, report.palette_rgb).save(buf, "PNG")
    prof = depth_profile(buf.getvalue(), hexes)

    pal_lab = np.stack([_hex_lab(h) for h in hexes])
    pinned = {}
    for h, lvl in _parse_assign(args.assign).items():
        nearest = hexes[int(np.argmin(delta_e(pal_lab, _hex_lab(h))))]
        pinned[nearest] = lvl
    # farthest thing gets the lowest level, i.e. sits flush with the plate
    far_first = sorted(hexes, key=lambda h: -prof[h])
    free = [l for l in range(args.levels) if l not in pinned.values()]
    mapping = dict(pinned)
    for h in far_first:
        if h not in mapping:
            mapping[h] = free.pop(0)

    d = np.array([prof[h] for h in hexes])
    span = float(d.max() - d.min())
    rel = np.zeros(args.levels)
    for h, l in mapping.items():
        # 0 = farthest (flush), 1 = nearest (full relief)
        rel[l] = (d.max() - prof[h]) / span if span > 0 else l / max(args.levels - 1, 1)

    by_level = sorted(mapping.items(), key=lambda kv: kv[1])
    print("  scene depth (farthest -> nearest): "
          + ", ".join(f"{h}@{prof[h]:.0f}" for h, _ in by_level))
    return mapping, rel


def _compute_heights(args, n_levels, rel):
    """Absolute top-surface height of each level, in mm, snapped to whole
    printer layers. rel is None for plain brightness/uniform ordering."""
    if args.heights:
        h = np.array([float(x) for x in args.heights.split(",")])
        if len(h) != n_levels:
            raise SystemExit(f"error: --heights needs {n_levels} values, got {len(h)}")
        how = "explicit --heights"
    elif rel is not None:
        # spacing follows the scene: a distant background sits far below a
        # foreground that is nearly touching the camera, not one fixed step
        h = args.base_mm + rel * args.relief_mm
        how = f"scene-relative over {args.relief_mm} mm of relief"
    else:
        h = uniform_heights(n_levels, args.base_mm, args.step_mm)
        how = f"uniform {args.step_mm} mm steps"
    if args.base_mm < args.layer_mm:
        raise SystemExit(f"error: --base-mm {args.base_mm} is thinner than one "
                         f"{args.layer_mm} mm layer; the plate needs at least one layer "
                         "(and realistically 2 mm — see README)")
    h = snap_heights(h, args.layer_mm, args.base_mm, args.min_step_mm)
    print(f"  heights ({how}, snapped to {args.layer_mm * 1000:.0f} um layers): "
          + ", ".join(f"L{k} {v:.2f}" for k, v in enumerate(h)) + " mm")
    print("  reproduce offline with: --heights " + ",".join(f"{v:g}" for v in h))
    return h


def make_levels(image, args):
    """Shared quantize -> downsample -> clean pipeline.
    Returns (levels, report, heights_mm)."""
    levels, report = quantize(
        image, args.levels, order=args.order, assign_overrides=_parse_assign(args.assign)
    )
    print(f"{args.image}: {image.size[0]}x{image.size[1]}, {args.levels} levels")
    _print_report(report)
    rel = None
    if args.depth_order:
        mapping, rel = _depth_overrides(args, levels, report)
        levels, report = quantize(
            image, args.levels, order=args.order, assign_overrides=mapping,
        )
        print("  reproduce offline with: "
              + " ".join(f"--assign '{h}={l}'"
                         for h, l in sorted(mapping.items(), key=lambda kv: kv[1])))
    if max(image.size) > args.max_px:
        scale = args.max_px / max(image.size)
        w, h = round(image.size[0] * scale), round(image.size[1] * scale)
        levels = np.asarray(
            Image.fromarray(levels.astype(np.uint8)).resize((w, h), Image.NEAREST),
            dtype=np.int64,
        )
        print(f"  working resolution {w}x{h} (--max-px {args.max_px})")
    if not args.no_clean:
        px_mm = args.width_mm / levels.shape[1]
        min_feature_px = args.min_feature / px_mm
        levels, stats = clean_levels(levels, args.levels, min_feature_px)
        report.coverage = stats["coverage"]
        print(f"  cleanup: reassigned {stats['changed_fraction']:.2%} of pixels "
              f"(min feature {args.min_feature} mm = {min_feature_px:.1f} px at {args.width_mm} mm wide)"
              + (f", unpinched {stats['pinches_fixed']} corner touches"
                 if stats["pinches_fixed"] else ""))
        for lvl in range(args.levels):
            if 0 < report.coverage[lvl] < 0.005:
                print(f"  warning: L{lvl} covers only {report.coverage[lvl]:.2%} after cleanup — "
                      "separation error, or a feature too small to matter?", file=sys.stderr)
    return levels, report, _compute_heights(args, args.levels, rel)


def cmd_levels(args):
    image = Image.open(args.image)
    levels, report, heights = make_levels(image, args)
    preview = side_by_side(image, levels, report, heights_mm=heights)
    preview.save(args.out)
    print(f"wrote {args.out} — open it and check the separation before meshing")


def cmd_build(args):
    image = Image.open(args.image)
    levels, report, heights = make_levels(image, args)

    H, W = levels.shape
    px_mm = args.width_mm / W
    depth_mm = H * px_mm
    if args.width_mm > BUILD_X_MM or depth_mm > BUILD_Y_MM:
        print(f"  warning: {args.width_mm:.0f} x {depth_mm:.1f} mm exceeds the Saturn 4 Ultra "
              f"plate ({BUILD_X_MM:.0f} x {BUILD_Y_MM:.0f} mm)", file=sys.stderr)

    tris_int, zvals = build_mesh(levels, heights)
    tris_mm = to_float_coords(tris_int, zvals, px_mm, H)
    chk = check_mesh(tris_int, tris_mm, levels, px_mm, heights)

    write_stl(args.out, tris_mm)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {len(tris_mm)} triangles, {size_mb:.1f} MB")
    print(f"  size: {args.width_mm:.2f} x {depth_mm:.2f} x {zvals[-1]:.2f} mm, flat bottom on Z=0")
    print(f"  watertight: {'yes' if chk['watertight'] else 'NO'}"
          f"  (open edges: {chk['open_edges']}, non-manifold edges: {chk['nonmanifold_edges']})")
    print(f"  volume: mesh {chk['signed_volume_mm3']:.1f} mm3, "
          f"analytic {chk['analytic_volume_mm3']:.1f} mm3 "
          f"{'(match)' if chk['volume_ok'] else '(MISMATCH — orientation bug)'}")
    if not chk["watertight"] or not chk["volume_ok"] or chk["nonmanifold_edges"]:
        raise SystemExit("error: mesh failed self-check, not safe to print"
                         + (" (try without --no-clean)" if args.no_clean else ""))

    base = os.path.splitext(args.out)[0]
    side_by_side(image, levels, report, heights_mm=heights).save(base + "-levels.png")
    hillshade_image(zvals[levels + 1], px_mm).save(base + "-relief.png")
    print(f"wrote {base}-levels.png and {base}-relief.png — look at both before slicing")


def _add_level_options(sp):
    sp.add_argument("-n", "--levels", type=int, default=5, help="number of height levels (default 5)")
    sp.add_argument("--order", choices=["dark-low", "light-low"], default="dark-low",
                    help="map darker or lighter colors to lower levels (default dark-low)")
    sp.add_argument("--assign", action="append", metavar="#RRGGBB=LEVEL",
                    help="force the level for the region nearest this color; repeatable")
    sp.add_argument("--width-mm", type=float, default=100.0,
                    help="physical width of the piece in mm (default 100)")
    sp.add_argument("--min-feature", type=float, default=0.5, metavar="MM",
                    help="smallest printable feature in mm; thinner details are absorbed (default 0.5)")
    sp.add_argument("--depth-order", action="store_true",
                    help="order levels by real-world distance from the viewer instead of "
                         "brightness — nearer parts protrude further. Asks Gemini to rank "
                         "the palette (needs GEMINI_API_KEY) and prints the equivalent "
                         "--assign flags so the result can be reproduced offline")
    sp.add_argument("--no-clean", action="store_true", help="skip the cleanup pass (debugging)")
    sp.add_argument("--max-px", type=int, default=512,
                    help="cap working resolution (long side, default 512); higher = finer detail, bigger STL")
    sp.add_argument("--base-mm", type=float, default=2.0,
                    help="backing plate thickness in mm (default 2.0); level 0 is flush with it")
    sp.add_argument("--step-mm", type=float, default=0.4,
                    help="height difference between adjacent levels in mm (default 0.4)")
    sp.add_argument("--relief-mm", type=float, default=1.6,
                    help="with --depth-order, total relief from farthest to nearest (default 1.6)")
    sp.add_argument("--layer-mm", type=float, default=0.05,
                    help="printer layer height in mm; all level tops snap to a multiple (default 0.05)")
    sp.add_argument("--min-step-mm", type=float, default=0.1,
                    help="floor on the gap between adjacent levels so they stay readable (default 0.1)")
    sp.add_argument("--heights", metavar="MM,MM,...",
                    help="explicit absolute top height per level, overriding all of the above")


def cmd_gen(args):
    prompt = build_prompt(args.prompt, args.levels, raw=args.raw_prompt,
                          with_ref=bool(args.from_image))
    print(f"prompt: {prompt}")
    generate_image(prompt, args.out, aspect=args.aspect, ref_image=args.from_image)
    print(f"wrote {args.out} — inspect it, then run: intarsia build {args.out}")


def cmd_run(args):
    prompt = build_prompt(args.prompt, args.levels, raw=args.raw_prompt,
                          with_ref=bool(args.from_image))
    print(f"prompt: {prompt}")
    generate_image(prompt, args.image_out, aspect=args.aspect, ref_image=args.from_image)
    print(f"\nwrote {args.image_out} — OPEN AND LOOK AT IT before continuing.")
    print("Check: flat solid colors, big simple shapes, no gradients or fine detail.")
    if not args.yes:
        answer = input("Proceed to STL with this image? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("stopped. Re-run `intarsia run` for a new image, or tweak the prompt.")
            return
    args.image = args.image_out
    cmd_build(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="intarsia", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    lv = sub.add_parser("levels", help="quantize an image into height levels and preview the separation")
    lv.add_argument("image")
    lv.add_argument("-o", "--out", default="levels-preview.png", help="preview PNG path")
    _add_level_options(lv)
    lv.set_defaults(func=cmd_levels)

    bd = sub.add_parser("build", help="image -> STL (quantize, clean, mesh, self-check, previews)")
    bd.add_argument("image")
    bd.add_argument("-o", "--out", default="relief.stl", help="output STL path")
    _add_level_options(bd)
    bd.set_defaults(func=cmd_build)

    gn = sub.add_parser("gen", help="generate a flat source image from a text prompt (Gemini)")
    gn.add_argument("prompt")
    gn.add_argument("-o", "--out", default="source.png", help="output image path")
    gn.add_argument("-n", "--levels", type=int, default=5,
                    help="planned number of height levels; sets the color count in the prompt")
    gn.add_argument("--raw-prompt", action="store_true",
                    help="send the prompt verbatim, without the flat-art template")
    gn.add_argument("--aspect", metavar="W:H",
                    help="image aspect ratio, e.g. 16:9 or 4:3 (default: model's choice)")
    gn.add_argument("--from-image", metavar="PHOTO",
                    help="reference photo; the prompt says what to keep from it "
                         "(e.g. 'the dog's head from this photo')")
    gn.set_defaults(func=cmd_gen)

    rn = sub.add_parser("run", help="prompt -> image -> approve -> STL, end to end")
    rn.add_argument("prompt")
    rn.add_argument("-o", "--out", default="relief.stl", help="output STL path")
    rn.add_argument("--image-out", default="source.png", help="where to save the generated image")
    rn.add_argument("--raw-prompt", action="store_true",
                    help="send the prompt verbatim, without the flat-art template")
    rn.add_argument("--yes", action="store_true",
                    help="skip the image approval question (non-interactive use)")
    rn.add_argument("--aspect", metavar="W:H",
                    help="image aspect ratio, e.g. 16:9 or 4:3 (default: model's choice)")
    rn.add_argument("--from-image", metavar="PHOTO",
                    help="reference photo; the prompt says what to keep from it "
                         "(e.g. 'the dog's head from this photo')")
    _add_level_options(rn)
    rn.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
