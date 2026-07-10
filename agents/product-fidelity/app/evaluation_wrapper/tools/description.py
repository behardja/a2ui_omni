"""Standalone description generation and refinement — no ToolContext, no global config.

Synced from: product_fidelity_agent/tools/gemini.py
Last synced:  2026-02-28

When updating product_fidelity_agent/tools/gemini.py, re-sync core logic here.
The only differences should be:
  - No ToolContext parameter or state writes
  - project_id / location / model passed explicitly
  - image_uris is a list[str], not a comma-separated string
  - Returns plain strings, not dicts
"""

import os

from google import genai
from google.genai import types

_DEFAULT_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_prompt(filename: str, prompts_dir: str | None = None) -> str:
    d = prompts_dir or _DEFAULT_PROMPTS_DIR
    with open(os.path.join(d, filename)) as f:
        return f.read()


def _make_client(project_id: str, location: str) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        http_options=types.HttpOptions(
            timeout=60 * 1000,
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=2.0,
                jitter=0.3,
                max_delay=30.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )


def generate_description(
    image_uris: list[str],
    project_id: str,
    location: str,
    model: str,
    prompts_dir: str | None = None,
) -> str:
    """Generate a ground-truth product description from reference images.

    Args:
        image_uris: List of GCS URIs for product reference images.
        project_id: GCP project ID.
        location: GCP region.
        model: Gemini model ID for description generation.
        prompts_dir: Optional path to directory containing prompt templates.
            Defaults to evaluation_wrapper/prompts/.

    Returns:
        The generated description text.
    """
    client = _make_client(project_id, location)

    system_instruction = _load_prompt("description_system.txt", prompts_dir)
    user_prompt = _load_prompt("description_user.txt", prompts_dir)

    content_parts = []
    for uri in image_uris:
        ext = uri.lower().rsplit(".", 1)[-1]
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        content_parts.append(types.Part.from_uri(file_uri=uri, mime_type=mime))
    content_parts.append(user_prompt)

    response = client.models.generate_content(
        model=model,
        contents=content_parts,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=1,
        ),
    )

    return response.text


def refine_description(
    description: str,
    failing_verdicts: list[str],
    project_id: str,
    location: str,
    model: str,
) -> str:
    """Refine a product description to emphasize attributes that failed evaluation.

    Always derives from the original description (not previous refinements)
    to prevent drift across iterations.

    Args:
        description: The original ground-truth description to refine.
        failing_verdicts: List of failing rubric verdict descriptions.
        project_id: GCP project ID.
        location: GCP region.
        model: Gemini model ID for refinement.

    Returns:
        The refined description text.
    """
    client = _make_client(project_id, location)

    failing_text = "\n".join(f"- {v}" for v in failing_verdicts)

    refinement_prompt = f"""You are refining a product description for text-to-image generation.

The original description was used to generate an image, but the following attributes
were NOT faithfully reproduced in the generated image:

FAILING ATTRIBUTES:
{failing_text}

ORIGINAL DESCRIPTION:
{description}

Your task: Rewrite the description to MORE STRONGLY EMPHASIZE the failing attributes.
- Keep ALL original details intact
- Add stronger, more explicit language for the failing attributes
- Add spatial/visual cues that help image generation models render these attributes correctly
- Do NOT remove any attributes — reinforce them
- Do NOT add new attributes that were not in the original
- Output only the refined description paragraph. 750 words max."""

    response = client.models.generate_content(
        model=model,
        contents=refinement_prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )

    return response.text
