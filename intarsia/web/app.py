"""Flask backend for the intarsia web UI.

Each browser session is a directory under outputs/<session-id>/ holding a
meta.json history plus the actual generated files (gen/, levels/, build/).
Every step (generate, levels, build) appends a numbered run and records
which run is "active"; re-running a step never deletes earlier runs, and
picking an old run as active is how the UI "unwinds" back to it. There is
no in-memory session state — every request re-reads meta.json — so the
dev server can restart without losing anything.
"""

import io
import json
import os
import re
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from ..cli import BUILD_X_MM, BUILD_Y_MM, make_levels
from ..gemini_gen import build_prompt, generate_image
from ..mesh import build_mesh, check_mesh, to_float_coords, write_stl
from ..preview import hillshade_image, side_by_side

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_SID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

LEVEL_PARAM_DEFAULTS = dict(
    levels=5,
    order="dark-low",
    assign=None,
    width_mm=100.0,
    min_feature=0.5,
    depth_order=True,
    no_clean=False,
    xy_um=19.0,
    max_px=None,
    base_mm=2.0,
    step_mm=0.4,
    relief_mm=1.6,
    layer_mm=0.05,
    min_step_mm=0.1,
    heights=None,
)

app = Flask(__name__, static_folder=None)


class PipelineError(Exception):
    pass


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _new_session_id():
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def _session_dir(sid, must_exist=True):
    if not _SID_RE.match(sid):
        abort(404)
    d = OUTPUT_ROOT / sid
    if must_exist and not d.is_dir():
        abort(404)
    return d


def _load_meta(sid):
    d = _session_dir(sid)
    try:
        return json.loads((d / "meta.json").read_text())
    except FileNotFoundError:
        abort(404)


def _save_meta(sid, meta):
    (OUTPUT_ROOT / sid / "meta.json").write_text(json.dumps(meta, indent=2))


def _get_run(meta, key, run_id):
    for r in meta[key]:
        if r["id"] == run_id:
            return r
    raise PipelineError(f"no {key} run {run_id}")


def _levels_namespace(meta, body, image_label):
    vals = dict(LEVEL_PARAM_DEFAULTS)
    vals["levels"] = meta.get("n_colors", 5)
    for key in vals:
        if key in body and body[key] is not None:
            vals[key] = body[key]
    vals["levels"] = int(vals["levels"])
    vals["width_mm"] = float(vals["width_mm"])
    vals["min_feature"] = float(vals["min_feature"])
    vals["xy_um"] = float(vals["xy_um"])
    vals["max_px"] = int(vals["max_px"]) if vals["max_px"] else None
    vals["base_mm"] = float(vals["base_mm"])
    vals["step_mm"] = float(vals["step_mm"])
    vals["relief_mm"] = float(vals["relief_mm"])
    vals["layer_mm"] = float(vals["layer_mm"])
    vals["min_step_mm"] = float(vals["min_step_mm"])
    vals["depth_order"] = bool(vals["depth_order"])
    vals["no_clean"] = bool(vals["no_clean"])
    if vals["assign"] and isinstance(vals["assign"], dict):
        vals["assign"] = [f"{k}={v}" for k, v in vals["assign"].items()]
    elif not vals["assign"]:
        vals["assign"] = None
    return SimpleNamespace(image=image_label, **vals)


def _report_to_dict(report, heights):
    hexes = report.palette_hex()
    return {
        "levels": [
            {
                "level": lvl,
                "hex": hexes[lvl],
                "coverage": float(report.coverage[lvl]),
                "height_mm": float(heights[lvl]),
            }
            for lvl in range(report.n_levels)
        ],
        "mean_residual": float(report.mean_residual),
        "bad_fraction": float(report.bad_fraction),
        "warnings": list(report.warnings),
    }


def _resolve_ref_image(sid, meta, ref_source):
    """ref_source is one of: falsy (no reference), "input" (the uploaded
    photo), "active" (whatever gen run is currently active, resolved to a
    concrete id now), or an explicit gen run id (int or numeric string) —
    any previous generation can be used as the reference for a refinement.

    Returns (ref_image_path_or_None, stored_ref_source) where stored_ref_source
    is what actually gets recorded in history (never the abstract "active").
    """
    if not ref_source:
        return None, None
    if ref_source == "input":
        if not meta.get("has_ref"):
            raise PipelineError("no reference photo was uploaded for this session")
        return str(OUTPUT_ROOT / sid / "input.png"), "input"
    if ref_source == "active":
        ref_source = meta.get("active_gen")
        if not ref_source:
            raise PipelineError("no generated image is currently selected to use as a reference")
    try:
        gid = int(ref_source)
    except (TypeError, ValueError):
        raise PipelineError(f"invalid reference source {ref_source!r}")
    if not any(r["id"] == gid for r in meta["gen_runs"]):
        raise PipelineError(f"no generated image #{gid} to use as a reference")
    return str(OUTPUT_ROOT / sid / "gen" / f"{gid}.png"), gid


def _run_gen(sid, meta, *, prompt_text, n_colors, aspect, image_size, raw_prompt, ref_source):
    prompt_text = (prompt_text or "").strip()
    ref_image, stored_ref = _resolve_ref_image(sid, meta, ref_source)
    if not prompt_text:
        if not ref_image:
            raise PipelineError("enter a text prompt, or select a reference image")
        prompt_text = "the main subject in the photo"

    gen_dir = OUTPUT_ROOT / sid / "gen"
    gen_dir.mkdir(exist_ok=True)
    run_id = len(meta["gen_runs"]) + 1
    out_path = gen_dir / f"{run_id}.png"

    prompt = build_prompt(prompt_text, n_colors, raw=raw_prompt, with_ref=bool(ref_image))
    print(f"prompt: {prompt}")
    print(f"reference image: {ref_image or '(none)'}")
    print(f"aspect: {aspect or '(model default)'}, size: {image_size}")
    t0 = time.monotonic()
    try:
        generate_image(prompt, str(out_path), aspect=aspect or None, ref_image=ref_image, size=image_size)
    except SystemExit as e:
        print(f"error: {e}")
        raise PipelineError(str(e)) from e
    with Image.open(out_path) as img:
        print(f"wrote {out_path.name}: {img.size[0]}x{img.size[1]} px in {time.monotonic() - t0:.1f}s")

    meta["gen_runs"].append({
        "id": run_id,
        "created": _now(),
        "prompt_text": prompt_text,
        "full_prompt": prompt,
        "n_colors": n_colors,
        "aspect": aspect,
        "image_size": image_size,
        "raw_prompt": raw_prompt,
        "ref_source": stored_ref,
    })
    meta["active_gen"] = run_id
    meta["active_levels"] = None
    meta["active_build"] = None
    meta["prompt"] = prompt_text
    meta["n_colors"] = n_colors
    meta["aspect"] = aspect
    meta["image_size"] = image_size
    meta["raw_prompt"] = raw_prompt
    return meta


def _state(sid, meta):
    base = f"/files/{sid}"
    state = dict(meta)
    state["ref_url"] = f"{base}/input.png" if meta.get("has_ref") else None
    state["gen_runs"] = [{**r, "url": f"{base}/gen/{r['id']}.png"} for r in meta["gen_runs"]]
    state["levels_runs"] = [{**r, "url": f"{base}/levels/{r['id']}.png"} for r in meta["levels_runs"]]
    state["build_runs"] = [
        {
            **r,
            "stl_url": f"{base}/build/{r['id']}.stl",
            "levels_preview_url": f"{base}/build/{r['id']}-levels.png",
            "relief_preview_url": f"{base}/build/{r['id']}-relief.png",
        }
        for r in meta["build_runs"]
    ]
    return state


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:relpath>")
def static_files(relpath):
    return send_from_directory(STATIC_DIR, relpath)


@app.get("/files/<sid>/<path:relpath>")
def serve_output(sid, relpath):
    return send_from_directory(_session_dir(sid), relpath)


@app.get("/api/sessions")
def list_sessions():
    out = []
    if OUTPUT_ROOT.is_dir():
        for d in sorted(OUTPUT_ROOT.iterdir(), reverse=True):
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            thumb = None
            if meta.get("active_gen"):
                thumb = f"/files/{meta['id']}/gen/{meta['active_gen']}.png"
            out.append({
                "id": meta["id"],
                "created": meta.get("created"),
                "prompt": meta.get("prompt", ""),
                "thumb": thumb,
            })
    return jsonify(out)


@app.post("/api/sessions")
def create_session():
    prompt_text = request.form.get("prompt", "")
    n_colors = int(request.form.get("n_colors", 5))
    aspect = request.form.get("aspect") or None
    image_size = request.form.get("image_size", "4K")
    raw_prompt = request.form.get("raw_prompt") == "true"

    sid = _new_session_id()
    sdir = _session_dir(sid, must_exist=False)
    sdir.mkdir(parents=True)

    has_ref = False
    file = request.files.get("image")
    if file and file.filename:
        try:
            img = Image.open(file.stream)
            img.load()
        except UnidentifiedImageError:
            return jsonify(error="uploaded file is not a readable image"), 400
        img.convert("RGB").save(sdir / "input.png")
        has_ref = True

    meta = {
        "id": sid,
        "created": _now(),
        "prompt": prompt_text,
        "n_colors": n_colors,
        "aspect": aspect,
        "image_size": image_size,
        "raw_prompt": raw_prompt,
        "has_ref": has_ref,
        "active_gen": None,
        "active_levels": None,
        "active_build": None,
        "gen_runs": [],
        "levels_runs": [],
        "build_runs": [],
    }

    if prompt_text.strip() or has_ref:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                meta = _run_gen(sid, meta, prompt_text=prompt_text, n_colors=n_colors, aspect=aspect,
                                 image_size=image_size, raw_prompt=raw_prompt,
                                 ref_source=("input" if has_ref else None))
            meta["gen_runs"][-1]["log"] = buf.getvalue()
        except PipelineError as e:
            _save_meta(sid, meta)
            return jsonify(error=str(e), log=buf.getvalue(), **_state(sid, meta)), 400

    _save_meta(sid, meta)
    return jsonify(_state(sid, meta)), 201


@app.post("/api/sessions/<sid>/gen")
def gen_step(sid):
    meta = _load_meta(sid)
    body = request.get_json(force=True) or {}
    default_ref = "active" if meta.get("active_gen") else ("input" if meta.get("has_ref") else None)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            meta = _run_gen(
                sid, meta,
                prompt_text=body.get("prompt", meta.get("prompt", "")),
                n_colors=int(body.get("n_colors", meta.get("n_colors", 5))),
                aspect=body.get("aspect", meta.get("aspect")),
                image_size=body.get("image_size", meta.get("image_size", "4K")),
                raw_prompt=bool(body.get("raw_prompt", meta.get("raw_prompt", False))),
                ref_source=body.get("ref_source", default_ref),
            )
        meta["gen_runs"][-1]["log"] = buf.getvalue()
    except PipelineError as e:
        return jsonify(error=str(e), log=buf.getvalue()), 400
    _save_meta(sid, meta)
    return jsonify(_state(sid, meta))


@app.delete("/api/sessions/<sid>/gen/<int:run_id>")
def delete_gen(sid, run_id):
    meta = _load_meta(sid)
    if not any(r["id"] == run_id for r in meta["gen_runs"]):
        return jsonify(error=f"no gen run {run_id}"), 404

    # Cascade: any levels/build run computed from this image is now
    # dangling (its source file is gone), so it goes too — files and all.
    dead_level_ids = {r["id"] for r in meta["levels_runs"] if r["gen_id"] == run_id}
    for lid in dead_level_ids:
        (OUTPUT_ROOT / sid / "levels" / f"{lid}.png").unlink(missing_ok=True)
    meta["levels_runs"] = [r for r in meta["levels_runs"] if r["id"] not in dead_level_ids]

    dead_build_ids = {r["id"] for r in meta["build_runs"] if r["gen_id"] == run_id}
    for bid in dead_build_ids:
        for suffix in (".stl", "-levels.png", "-relief.png"):
            (OUTPUT_ROOT / sid / "build" / f"{bid}{suffix}").unlink(missing_ok=True)
    meta["build_runs"] = [r for r in meta["build_runs"] if r["id"] not in dead_build_ids]

    (OUTPUT_ROOT / sid / "gen" / f"{run_id}.png").unlink(missing_ok=True)
    meta["gen_runs"] = [r for r in meta["gen_runs"] if r["id"] != run_id]

    if meta["active_gen"] == run_id:
        meta["active_gen"] = meta["gen_runs"][-1]["id"] if meta["gen_runs"] else None
    if meta["active_levels"] in dead_level_ids:
        meta["active_levels"] = None
    if meta["active_build"] in dead_build_ids:
        meta["active_build"] = None

    _save_meta(sid, meta)
    return jsonify(_state(sid, meta))


@app.post("/api/sessions/<sid>/levels")
def levels_step(sid):
    meta = _load_meta(sid)
    if not meta["active_gen"]:
        return jsonify(error="generate an image first"), 400
    body = request.get_json(force=True) or {}
    gen_run = _get_run(meta, "gen_runs", meta["active_gen"])
    image_path = OUTPUT_ROOT / sid / "gen" / f"{gen_run['id']}.png"
    image = Image.open(image_path)

    ns = _levels_namespace(meta, body, image_label=str(image_path))
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            levels, report, heights = make_levels(image, ns)
    except SystemExit as e:
        return jsonify(error=str(e), log=buf.getvalue()), 400

    levels_dir = OUTPUT_ROOT / sid / "levels"
    levels_dir.mkdir(exist_ok=True)
    run_id = len(meta["levels_runs"]) + 1
    side_by_side(image, levels, report, heights_mm=heights).save(levels_dir / f"{run_id}.png")

    meta["levels_runs"].append({
        "id": run_id,
        "gen_id": gen_run["id"],
        "created": _now(),
        "params": dict(vars(ns), image=None),
        "report": _report_to_dict(report, heights),
        "log": buf.getvalue(),
    })
    meta["active_levels"] = run_id
    meta["active_build"] = None
    _save_meta(sid, meta)
    return jsonify(_state(sid, meta))


@app.post("/api/sessions/<sid>/build")
def build_step(sid):
    meta = _load_meta(sid)
    if not meta["active_gen"]:
        return jsonify(error="generate an image first"), 400
    body = request.get_json(force=True) or {}
    gen_run = _get_run(meta, "gen_runs", meta["active_gen"])
    image_path = OUTPUT_ROOT / sid / "gen" / f"{gen_run['id']}.png"
    image = Image.open(image_path)

    ns = _levels_namespace(meta, body, image_label=str(image_path))
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            levels, report, heights = make_levels(image, ns)
            H, W = levels.shape
            px_mm = ns.width_mm / W
            depth_mm = H * px_mm
            tris_int, zvals = build_mesh(levels, heights)
            tris_mm = to_float_coords(tris_int, zvals, px_mm, H)
            chk = check_mesh(tris_int, tris_mm, levels, px_mm, heights)
    except SystemExit as e:
        return jsonify(error=str(e), log=buf.getvalue()), 400

    if not chk["watertight"] or not chk["volume_ok"] or chk["nonmanifold_edges"]:
        return jsonify(error="mesh failed self-check, not safe to print"
                        + ("" if not ns.no_clean else " (try without --no-clean)"),
                        log=buf.getvalue()), 400

    build_dir = OUTPUT_ROOT / sid / "build"
    build_dir.mkdir(exist_ok=True)
    run_id = len(meta["build_runs"]) + 1
    stl_path = build_dir / f"{run_id}.stl"
    write_stl(str(stl_path), tris_mm)
    side_by_side(image, levels, report, heights_mm=heights).save(build_dir / f"{run_id}-levels.png")
    hillshade_image(zvals[levels + 1], px_mm).save(build_dir / f"{run_id}-relief.png")

    meta["build_runs"].append({
        "id": run_id,
        "gen_id": gen_run["id"],
        "levels_id": meta["active_levels"],
        "created": _now(),
        "params": dict(vars(ns), image=None),
        "triangles": len(tris_mm),
        "size_mb": os.path.getsize(stl_path) / 1e6,
        "size_mm": [ns.width_mm, depth_mm, float(zvals[-1])],
        "watertight": bool(chk["watertight"]),
        "open_edges": int(chk["open_edges"]),
        "nonmanifold_edges": int(chk["nonmanifold_edges"]),
        "volume_mesh": float(chk["signed_volume_mm3"]),
        "volume_analytic": float(chk["analytic_volume_mm3"]),
        "volume_ok": bool(chk["volume_ok"]),
        "plate_warning": bool(ns.width_mm > BUILD_X_MM or depth_mm > BUILD_Y_MM),
        "log": buf.getvalue(),
    })
    meta["active_build"] = run_id
    _save_meta(sid, meta)
    return jsonify(_state(sid, meta))


@app.post("/api/sessions/<sid>/select")
def select_step(sid):
    meta = _load_meta(sid)
    body = request.get_json(force=True) or {}
    step = body.get("step")
    key = {"gen": "gen_runs", "levels": "levels_runs", "build": "build_runs"}.get(step)
    if key is None:
        return jsonify(error=f"unknown step '{step}'"), 400
    try:
        run_id = int(body.get("run_id"))
    except (TypeError, ValueError):
        return jsonify(error="run_id is required"), 400
    if not any(r["id"] == run_id for r in meta[key]):
        return jsonify(error=f"no {step} run {run_id}"), 404
    meta[f"active_{step}"] = run_id
    # Reactivating an old run restores whatever was last built on top of it
    # (if anything), rather than discarding that history — "unwind" means
    # going back to exactly the state you were reviewing, not starting over.
    if step == "gen":
        later = [r for r in meta["levels_runs"] if r["gen_id"] == run_id]
        meta["active_levels"] = later[-1]["id"] if later else None
        run_id = meta["active_levels"]
    if step in ("gen", "levels"):
        later = [r for r in meta["build_runs"] if r["gen_id"] == meta["active_gen"]
                  and r.get("levels_id") == run_id]
        meta["active_build"] = later[-1]["id"] if later else None
    _save_meta(sid, meta)
    return jsonify(_state(sid, meta))


@app.get("/api/sessions/<sid>")
def get_session(sid):
    return jsonify(_state(sid, _load_meta(sid)))


def main(host="127.0.0.1", port=5050, debug=False):
    OUTPUT_ROOT.mkdir(exist_ok=True)
    print(f"intarsia web UI: http://{host}:{port}  (outputs in {OUTPUT_ROOT})")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
