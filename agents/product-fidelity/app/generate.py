"""Generation plugin for the product-fidelity eval loop.

This is the ONE function the evaluation_wrapper needs us to provide. It:
  1. Builds a text-to-image prompt from the ground-truth description
     (plus any failing-verdict emphasis on retries and optional creative
     direction from the user),
  2. Calls `gemini-3.1-flash-image` (Nano Banana 2, region
     `global`) with the reference images as multimodal context,
  3. Uploads the generated PNG to GCS so Gecko can read it,
  4. Returns the `gs://` URI of the candidate.

Signature matches evaluation_wrapper.pipeline.GenerateFn.
"""

import io
import logging
import os
import uuid

from google import genai
from google.genai import types
from google.cloud import storage

logger = logging.getLogger(__name__)

# Nano Banana 2 — image in+out, served on the global endpoint.
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image")
IMAGE_LOCATION = os.environ.get("GENERATION_LOCATION", "global")


def _project() -> str:
    project = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("Missing PROJECT_ID / GOOGLE_CLOUD_PROJECT env var.")
    return project


def _candidate_bucket() -> str:
    bucket = os.environ.get("CANDIDATE_BUCKET") or os.environ.get("BUCKET_NAME")
    if not bucket:
        raise ValueError("Missing CANDIDATE_BUCKET / BUCKET_NAME env var.")
    return bucket.replace("gs://", "").split("/", 1)[0]


def _mime_for(uri: str) -> str:
    ext = uri.lower().rsplit(".", 1)[-1]
    return "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"


# Recontextualization framing (ported from product-fidelity-eval): the model is
# fed the reference image and asked to place the SAME product into a natural,
# contextually appropriate setting (not a white packshot), preserving every
# product detail. This is what lets an optional creative direction (scene, model,
# lighting) actually steer the output instead of being cancelled out by a
# "clean neutral background / no people" instruction.
_RECONTEXTUALIZATION_PROMPT = (
    "Using the provided reference image(s), generate a new photorealistic image "
    "of the SAME product in a contextually appropriate setting. The new image "
    "should NOT have a plain white background — contextualize it based on the "
    "product itself (e.g. a bag in a natural ad/professional photo setting; a "
    "dress worn in a natural model photo setting). If there is a person in the "
    "reference image, create a variation of the person with the product without "
    "copying the exact same pose and environment. Keep the product exactly as it "
    "is — do not alter its design, colors, patterns, logos, or any visual details."
)


def _build_prompt(description: str, failing_verdicts: list[str], user_prompt: str) -> str:
    parts = [
        _RECONTEXTUALIZATION_PROMPT,
        f"\nPRODUCT DESCRIPTION (the product must remain faithful to this):\n{description}",
    ]
    if user_prompt:
        parts.append(
            "\nAdditionally, follow this creative direction from the user for the "
            f"scene/setting/composition:\n{user_prompt}"
        )
    if failing_verdicts:
        bullets = "\n".join(f"- {v}" for v in failing_verdicts)
        parts.append(
            "\nA previous attempt failed to reproduce these product attributes — "
            f"pay special attention to them:\n{bullets}"
        )
    return "\n".join(parts)


def generate_candidate_image(
    reference_uris: list[str],
    description: str,
    attempt: int,
    failing_verdicts: list[str],
    **kwargs,
) -> str:
    """Generate a candidate product image and return its gs:// URI.

    Conforms to evaluation_wrapper.pipeline.GenerateFn. `kwargs` carries
    `sku_id` and `user_prompt` supplied by the pipeline.
    """
    sku_id = kwargs.get("sku_id", "sku")
    user_prompt = kwargs.get("user_prompt", "")
    # Per-run image model override (from the UI dropdown); falls back to the env default.
    model = kwargs.get("image_model") or IMAGE_MODEL

    client = genai.Client(vertexai=True, project=_project(), location=IMAGE_LOCATION)

    contents: list = []
    for uri in reference_uris:
        contents.append(types.Part.from_uri(file_uri=uri, mime_type=_mime_for(uri)))
    contents.append(_build_prompt(description, failing_verdicts, user_prompt))

    logger.info(
        "generate | sku=%s attempt=%d refs=%d model=%s",
        sku_id, attempt, len(reference_uris), model,
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    image_bytes = None
    out_mime = "image/png"
    for part in response.candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            image_bytes = inline.data
            out_mime = inline.mime_type or "image/png"
            break
    if not image_bytes:
        raise RuntimeError(
            f"{model} returned no image for sku={sku_id} attempt={attempt}."
        )

    ext = "png" if "png" in out_mime else out_mime.split("/")[-1]
    blob_name = f"candidates/{sku_id}/attempt_{attempt}_{uuid.uuid4().hex[:8]}.{ext}"
    bucket = storage.Client(project=_project()).bucket(_candidate_bucket())
    blob = bucket.blob(blob_name)
    blob.upload_from_file(io.BytesIO(image_bytes), content_type=out_mime)

    gs_uri = f"gs://{_candidate_bucket()}/{blob_name}"
    logger.info("generate | sku=%s attempt=%d candidate=%s", sku_id, attempt, gs_uri)
    return gs_uri
