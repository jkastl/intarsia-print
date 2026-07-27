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
import urllib.error
import urllib.request

# Nano Banana Pro — much stronger prompt adherence (exact color counts, flat
# fills) than the base flash image model, which is what this pipeline needs.
MODEL = "gemini-3-pro-image-preview"
_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

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


def generate_image(prompt, out_path, api_key=None, aspect=None, ref_image=None):
    """Calls Gemini, writes the first returned image to out_path (PNG/etc).

    ref_image: optional path to a photo sent along with the prompt, for
    "flatten this photo" / "just the dog's head from this picture" requests.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "error: set GEMINI_API_KEY (https://aistudio.google.com/apikey), "
            "or skip generation and pass your own image to `build`."
        )
    parts = [{"text": prompt}]
    if ref_image:
        mime = mimetypes.guess_type(ref_image)[0]
        if mime not in ("image/png", "image/jpeg", "image/webp"):
            raise SystemExit(f"error: --from-image must be png/jpg/webp, got {ref_image}")
        with open(ref_image, "rb") as f:
            parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(f.read()).decode()}})
    # Nano Banana Pro is a thinking model: it returns TEXT parts alongside the
    # IMAGE part, so both modalities must be requested.
    config = {"responseModalities": ["TEXT", "IMAGE"]}
    if aspect:
        config["imageConfig"] = {"aspectRatio": aspect}
    body = json.dumps(
        {
            "contents": [{"parts": parts}],
            "generationConfig": config,
        }
    ).encode()
    req = urllib.request.Request(
        _URL,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise SystemExit(f"error: Gemini API returned {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"error: could not reach Gemini API: {e.reason}")

    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(blob["data"]))
                return out_path
    raise SystemExit(f"error: Gemini returned no image. Response: {json.dumps(data)[:500]}")
