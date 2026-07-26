"""intarsia-print CLI."""

import argparse
import sys

from PIL import Image

from .clean import clean_levels
from .preview import side_by_side
from .quantize import quantize


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


def make_levels(image, args):
    """Shared quantize + clean pipeline; returns (levels, report)."""
    levels, report = quantize(
        image, args.levels, order=args.order, assign_overrides=_parse_assign(args.assign)
    )
    print(f"{args.image}: {image.size[0]}x{image.size[1]}, {args.levels} levels")
    _print_report(report)
    if not args.no_clean:
        px_mm = args.width_mm / image.size[0]
        min_feature_px = args.min_feature / px_mm
        levels, stats = clean_levels(levels, args.levels, min_feature_px)
        report.coverage = stats["coverage"]
        print(f"  cleanup: reassigned {stats['changed_fraction']:.2%} of pixels "
              f"(min feature {args.min_feature} mm = {min_feature_px:.1f} px at {args.width_mm} mm wide)")
        for lvl in range(args.levels):
            if 0 < report.coverage[lvl] < 0.005:
                print(f"  warning: L{lvl} covers only {report.coverage[lvl]:.2%} after cleanup — "
                      "separation error, or a feature too small to matter?", file=sys.stderr)
    return levels, report


def cmd_levels(args):
    image = Image.open(args.image)
    levels, report = make_levels(image, args)
    preview = side_by_side(image, levels, report)
    preview.save(args.out)
    print(f"wrote {args.out} — open it and check the separation before meshing")


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
    sp.add_argument("--no-clean", action="store_true", help="skip the cleanup pass (debugging)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="intarsia", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    lv = sub.add_parser("levels", help="quantize an image into height levels and preview the separation")
    lv.add_argument("image")
    lv.add_argument("-o", "--out", default="levels-preview.png", help="preview PNG path")
    _add_level_options(lv)
    lv.set_defaults(func=cmd_levels)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
