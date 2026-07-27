"""Text prompt -> flat graphic source image, via the Gemini API.

This is the only part of the tool that touches the network, and it is
quarantined to the `gen` step: the image->STL pipeline stays deterministic.
The prompt template matters more than any downstream cleverness — a source
image with gradients or fine detail cannot be rescued later, so we ask
Gemini very specifically for poster-flat art.
"""

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request

# Nano Banana Pro — much stronger prompt adherence (exact color counts, flat
# fills) than the base flash image model, which is what this pipeline needs.
MODEL = "gemini-3-pro-image-preview"
# Cheap vision model for the depth-ranking question; no image output needed.
DEPTH_MODEL = "gemini-2.5-flash"
_BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_URL = _BASE.format(model=MODEL)

# The constraints mirror the pipeline's failure modes one by one:
# gradients -> banding; texture/small details -> sub-printable slivers;
# similar-brightness adjacent hues -> weak separation; outlines -> 1px walls.
TEMPLATE = (
    "A flat vector-style poster illustration of: {subject}. "
    "Use exactly {colors} solid matte colors total, including the background. "
    "Large, simple, bold shapes only — like a woodcut intarsia or paper-cut "
    "collage. Absolutely no gradients, no shading, no shadows, no texture, "
    "no patterns, no outlines or strokes, no tiny details, no text. Every "
    "region is one uniform flat color, and adjacent regions use clearly "
    "different colors with distinct brightness. Clean sharp edges."
)

# When a reference photo is attached, the subject line says what to keep from
# it ("only the dog's head, facing forward"); the style constraints are the
# same because the pipeline's failure modes are the same.
REF_TEMPLATE = (
    "Using the attached photo only as a reference for the subject, redraw "
    "{subject} as a flat vector-style poster illustration. Do not reproduce "
    "the photo — recreate the subject in this style: exactly {colors} solid "
    "matte colors total, including a plain background. Large, simple, bold "
    "shapes only — like a woodcut intarsia or paper-cut collage. Absolutely "
    "no gradients, no shading, no shadows, no texture, no patterns, no "
    "outlines or strokes, no tiny details, no text. Every region is one "
    "uniform flat color, and adjacent regions use clearly different colors "
    "with distinct brightness. Clean sharp edges."
)


def build_prompt(subject, n_colors, raw=False, with_ref=False):
    if raw:
        return subject
    tpl = REF_TEMPLATE if with_ref else TEMPLATE
    return tpl.format(subject=subject, colors=n_colors)


def _require_key(api_key):
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "error: set GEMINI_API_KEY (https://aistudio.google.com/apikey), "
            "or skip generation and pass your own image to `build`."
        )
    return api_key


def _image_part(path, flag="--from-image"):
    mime = mimetypes.guess_type(path)[0]
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise SystemExit(f"error: {flag} must be png/jpg/webp, got {path}")
    with open(path, "rb") as f:
        return _bytes_part(f.read(), mime)


def _bytes_part(data, mime="image/png"):
    return {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}}


def _post(model, parts, config, api_key):
    body = json.dumps(
        {"contents": [{"parts": parts}], "generationConfig": config}
    ).encode()
    req = urllib.request.Request(
        _BASE.format(model=model),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise SystemExit(f"error: Gemini API returned {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"error: could not reach Gemini API: {e.reason}")


def generate_image(prompt, out_path, api_key=None, aspect=None, ref_image=None):
    """Calls Gemini, writes the first returned image to out_path (PNG/etc).

    ref_image: optional path to a photo sent along with the prompt, for
    "flatten this photo" / "just the dog's head from this picture" requests.
    """
    api_key = _require_key(api_key)
    parts = [{"text": prompt}]
    if ref_image:
        parts.append(_image_part(ref_image))
    # Nano Banana Pro is a thinking model: it returns TEXT parts alongside the
    # IMAGE part, so both modalities must be requested.
    config = {"responseModalities": ["TEXT", "IMAGE"]}
    if aspect:
        config["imageConfig"] = {"aspectRatio": aspect}
    data = _post(MODEL, parts, config, api_key)

    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(blob["data"]))
                return out_path
    raise SystemExit(f"error: Gemini returned no image. Response: {json.dumps(data)[:500]}")


DEPTH_PROMPT = (
    "The attached image is a flat poster-style illustration that will become "
    "a layered bas-relief carving: each color becomes one physical height "
    "layer. Its {n} colors are, as hex codes: {hexes}. "
    "For each color, estimate how far from the viewer the thing it depicts "
    "is, on a 0 to 100 scale: 0 is nearest to the camera (a nose, a "
    "foreground paw), 100 is the farthest background (sky, distant scenery). "
    "Judge the depicted scene, not the colors themselves. Use the full range, "
    "and make the gaps proportional to the real depth gaps — two things at "
    "nearly the same distance should get nearly the same number, and a "
    "distant background should sit far from the subject. "
    "Reply with ONLY a JSON object mapping each of the {n} hex codes to its "
    "number, and no other text."
)


def depth_profile(level_map_png, palette_hex, api_key=None):
    """Asks Gemini how far away each palette color is. Returns {hex: distance}
    with distance in 0..100, 0 = nearest the viewer.

    level_map_png: PNG bytes of the *quantized* level map, so the colors the
    model sees are exactly the hex codes it is being asked about.
    """
    api_key = _require_key(api_key)
    lower = [h.lower() for h in palette_hex]
    if len(set(lower)) != len(lower):
        raise SystemExit(
            "error: two levels share a display color; depth ranking would be "
            "ambiguous. Pin levels manually with --assign instead."
        )
    prompt = DEPTH_PROMPT.format(n=len(lower), hexes=", ".join(lower))
    parts = [{"text": prompt}, _bytes_part(level_map_png)]
    data = _post(DEPTH_MODEL, parts, {"temperature": 0}, api_key)

    text = "".join(
        part.get("text", "")
        for cand in data.get("candidates", [])
        for part in cand.get("content", {}).get("parts", [])
    )
    m = re.search(r"\{.*\}", text, re.DOTALL)
    profile = None
    if m:
        try:
            raw = json.loads(m.group(0))
            profile = {str(k).lower(): float(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            profile = None
    if profile is None or sorted(profile) != sorted(lower):
        raise SystemExit(
            f"error: could not get a usable depth profile from Gemini "
            f"(reply: {text[:300]!r}). Pin levels manually with --assign."
        )
    return profile
