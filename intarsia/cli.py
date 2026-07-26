"""intarsia-print CLI."""

import argparse
import sys

from PIL import Image

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


def cmd_levels(args):
    image = Image.open(args.image)
    levels, report = quantize(
        image, args.levels, order=args.order, assign_overrides=_parse_assign(args.assign)
    )
    print(f"{args.image}: {image.size[0]}x{image.size[1]}, {args.levels} levels")
    _print_report(report)
    preview = side_by_side(image, levels, report)
    preview.save(args.out)
    print(f"wrote {args.out} — open it and check the separation before meshing")


def main(argv=None):
    p = argparse.ArgumentParser(prog="intarsia", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    lv = sub.add_parser("levels", help="quantize an image into height levels and preview the separation")
    lv.add_argument("image")
    lv.add_argument("-n", "--levels", type=int, default=5, help="number of height levels (default 5)")
    lv.add_argument("-o", "--out", default="levels-preview.png", help="preview PNG path")
    lv.add_argument("--order", choices=["dark-low", "light-low"], default="dark-low",
                    help="map darker or lighter colors to lower levels (default dark-low)")
    lv.add_argument("--assign", action="append", metavar="#RRGGBB=LEVEL",
                    help="force the level for the region nearest this color; repeatable")
    lv.set_defaults(func=cmd_levels)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
