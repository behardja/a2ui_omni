"""A2UI Product-Fidelity agent.

The LLM orchestrates: browse/ingest reference images from GCS, run the Gecko
fidelity-eval loop, and render everything as A2UI JSON. Tools return raw JSON;
the model (grounded by the A2UI schema + few-shot examples) turns it into UI
wrapped in <a2ui-json> tags, which the executor extracts into A2UI DataParts.
"""

import os
import logging

from google.adk.agents import Agent
from google.adk.tools import FunctionTool, load_artifacts
from google.genai import types as genai_types
from a2ui.schema.manager import A2uiSchemaManager
from a2ui.basic_catalog.provider import BasicCatalog
from a2ui.schema.common_modifiers import remove_strict_validation
from a2ui.schema.constants import VERSION_0_8

logger = logging.getLogger(__name__)

try:
    from .tools import (
        list_gcs_images,
        ingest_uploaded_image_tool,
        run_fidelity_eval,
        get_eval_defaults,
    )
except ImportError:
    from tools import (
        list_gcs_images,
        ingest_uploaded_image_tool,
        run_fidelity_eval,
        get_eval_defaults,
    )


ROLE_DESCRIPTION = (
    "You are a product catalog fidelity assistant. You help users select a "
    "product reference image from Google Cloud Storage (or an uploaded image), "
    "run an automated fidelity-evaluation loop that generates a candidate asset "
    "and scores how faithfully it matches the reference, and present the results "
    "as rich interactive A2UI cards and charts."
)

WORKFLOW_DESCRIPTION = """
Choose the workflow based on the user's intent:

A. BROWSE GCS IMAGES
   1. When the user asks to browse/list images in a GCS bucket or prefix, call
      `list_gcs_images` with their gs:// prefix.
   2. Render the returned images as a selectable grid: one Image per item plus a
      Button whose action name is "select_reference" and whose context carries
      the image's `gs_uri` (key "referenceUri") and `name`.

B. EVALUATE A REFERENCE
   1. If the user uploaded an image, first call `ingest_uploaded_image_tool` to
      store it and get its gs_uri.
   2. If the user selected/pasted a gs:// URI, use that directly.
   3. Call `run_fidelity_eval` with reference_uris=[the gs_uri] (and optional
      user_prompt, threshold, max_retries, image_model when specified — pass any
      `image_model` string from the request through verbatim).
   4. Parse the returned JSON and render the results UI (see UI rules): the
      reference image, the best candidate image, a score chart across attempts,
      a pass/fail status, and the passing/failing rubric verdicts.

C. ADJUST SETTINGS BEFORE EVALUATING
   1. When the user wants to tune the run (or asks for settings/options), first
      call `get_eval_defaults`, then render the "Evaluation settings" panel
      (see UI rules) pre-filled with those defaults and the chosen referenceUri.
   2. When the user clicks the "run_eval" action, read referenceUri, threshold,
      maxRetries, and userPrompt from the action context and call
      `run_fidelity_eval` with them, then render the results UI (workflow B.4).

Keep prose to at most ONE short sentence (e.g. "Here are the images." or
"Evaluation complete."), then emit the A2UI UI block. Do NOT restate scores,
verdicts, or a written evaluation report in prose — ALL results belong in the
A2UI widgets (images, score list, verdict lists), not the text.
When a user clicks a "select_reference" action, treat the provided referenceUri
as the reference and proceed with workflow B (or C if they asked to adjust settings).
"""

UI_DESCRIPTION = """
Emit A2UI **v0.8**: a BARE JSON ARRAY of messages (no wrapper object, NO
`version` field), in order `beginRendering` → `surfaceUpdate` → `dataModelUpdate`.
- `beginRendering`: `{"surfaceId": "...", "catalogId": "https://a2ui.org/specification/v0_8/standard_catalog_definition.json", "root": "root", "styles": {"primaryColor": "#135bec"}}`
  — ALWAYS include `catalogId` set to exactly that URL (the renderer needs it to
  load the component catalog), and the `styles` so widgets match the brand color.
- `surfaceUpdate`: `{"surfaceId": "...", "components": [...]}`. Components are
  NESTED objects: `{"id": "x", "component": {"<Type>": { ...props }}}` — e.g.
  `{"id":"root","component":{"Column":{"children":{"explicitList":["a","b"]}}}}`.
  The root component (id "root") MUST be first; parents before children.
- Children: multi-child → `"children": {"explicitList": ["id1","id2"]}`;
  single-child (Card, Button) → `"child": "id"`; repeated list items →
  `"children": {"template": {"componentId": "item-id", "dataBinding": "/arrayPath"}}`.
- Every value is an OBJECT: literal → `{"literalString": "..."}` (or
  `literalNumber`/`literalBoolean`); data-bound → `{"path": "/field"}` (inside a
  template, paths resolve relative to each array element).
- `dataModelUpdate`: `{"surfaceId": "...", "path": "/", "contents": [...]}` where
  each entry is `{"key": "...", "valueString"|"valueNumber"|"valueBoolean": ...}`
  and arrays/objects use `"valueMap": [ {"key": "...", ...}, ... ]` (an adjacency
  list; array items are entries like `{"key":"item1","valueMap":[...]}`).
Follow the provided few-shot examples closely.

- GCS image browser: a `Column` → `List` whose children template is
  `{"componentId":"image-card","dataBinding":"/images"}`; each item is a `Card`/`Row`
  with an `Image` (`"url":{"path":"/url"}`, `"fit":"cover"`, `"usageHint":"smallFeature"`),
  a `Text` name, and a `Button` (child Text reads **"Generate and Evaluate"**) with
  `"action":{"name":"select_reference","context":[{"key":"referenceUri","value":{"path":"/gs_uri"}},{"key":"name","value":{"path":"/name"}}]}`.
  Put the images (name, gs_uri, url) in `dataModelUpdate` under key "images".
- Fidelity results — title it **"Fidelity Report"** (Text usageHint "h2"), then:
  * A status `Text` (usageHint "h3") like "✅ PASS · Score 0.82 · 3 attempts" (use ❌
    and "FAIL" when not passed; ALWAYS include the number of attempts).
  * A `Row` of two `Card`s: reference `Image` (left) and best-candidate `Image`
    (right, the highest-scoring attempt's candidate_url), each `Image` with
    `"fit":"contain"` and `"usageHint":"largeFeature"` (prominent), plus a `Text` caption.
  * A `Tabs` component grouping the details:
    `"tabItems":[{"title":{"literalString":"Passing (N)"},"child":"passing-list"},{"title":{"literalString":"Failing (M)"},"child":"failing-list"},{"title":{"literalString":"Scores"},"child":"scores-list"}]`.
    - passing-list / failing-list: `List` template bound to `/passing` / `/failing`;
      each item is a `Row` of two `Text`s — a mark ("✅" for passing, "❌" for
      failing) then the verdict bound to `{"path":"/text"}`.
    - scores-list: `List` template bound to `/attempts`; each item a `Row` of two
      `Text`s — the attempt label and its score (format the score as text, e.g.
      "0.82"). (Do NOT use any chart component.)
- Evaluation settings panel (a `Card` → `Column`):
  * `TextField` `"label":{"literalString":"Reference gs:// URI"}`, `"text":{"path":"/referenceUri"}`.
  * `Slider` `"label":{"literalString":"Passing threshold (0-1)"}`, `"value":{"path":"/threshold"}`, `"minValue":0`, `"maxValue":1`.
  * `Slider` `"label":{"literalString":"Max attempts"}`, `"value":{"path":"/maxRetries"}`, `"minValue":1`, `"maxValue":5`.
  * `TextField` `"label":{"literalString":"Creative direction (optional)"}`, `"text":{"path":"/userPrompt"}`, `"textFieldType":"longText"`.
  * `Button` `"action":{"name":"run_eval","context":[{"key":"referenceUri","value":{"path":"/referenceUri"}},{"key":"threshold","value":{"path":"/threshold"}},{"key":"maxRetries","value":{"path":"/maxRetries"}},{"key":"userPrompt","value":{"path":"/userPrompt"}}]}`.
  Pre-fill `dataModelUpdate` with `get_eval_defaults` values + the referenceUri.
- Image components MUST use the signed `url` fields returned by the tools (not
  gs:// URIs) so they can be displayed.
- Do NOT put markdown syntax (##, **, -, etc.) inside `Text` values; write plain
  words only and use the `usageHint` property ("h2","h3","h4","caption","body")
  for size/emphasis. (e.g. text "Select a reference image" with usageHint "h2" —
  never "## Select a reference image".)
- Size `Image` with `usageHint`: grid/browser thumbnails use "smallFeature";
  reference vs candidate result images use "largeFeature".
- The fidelity report (scores, pass/fail, verdicts) MUST be rendered as the A2UI
  widgets above — never written out as prose text.
- ALL UI MUST be wrapped in `<a2ui-json>` and `</a2ui-json>` tags. DO NOT output
  raw JSON without these tags.
"""


def create_agent() -> Agent:
    # The orchestrator model (gemini-3.5-flash) is served ONLY on the global
    # endpoint — it 404s on regional ones. ADK's genai client reads
    # GOOGLE_CLOUD_LOCATION at call time, and Agent Engine's runtime overrides
    # that env var to the engine's region (us-central1), so we must force it
    # here, in-process, where the platform can't override it. Gecko eval stays
    # regional via the separate LOCATION var.
    os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("MODEL_LOCATION", "global")

    schema_manager = A2uiSchemaManager(
        version=VERSION_0_8,
        catalogs=[
            BasicCatalog.get_config(
                version=VERSION_0_8,
                examples_path=os.path.join(os.path.dirname(__file__), "examples/0.8"),
            )
        ],
        schema_modifiers=[remove_strict_validation],
    )

    instruction = schema_manager.generate_system_prompt(
        role_description=ROLE_DESCRIPTION,
        workflow_description=WORKFLOW_DESCRIPTION,
        ui_description=UI_DESCRIPTION,
        include_schema=True,
        include_examples=True,
        validate_examples=False,
    )

    return Agent(
        name="ProductFidelityAgent",
        model=os.environ.get("GOOGLE_GENAI_MODEL", "gemini-3.5-flash"),
        description=(
            "Product catalog fidelity agent: browse GCS references, run the "
            "Gecko eval loop, render results as A2UI."
        ),
        # Pass the instruction as a callable (InstructionProvider) so ADK skips
        # {var} state-injection — the embedded A2UI schema contains literal
        # braces like `{expression}` that would otherwise raise KeyError.
        instruction=lambda _ctx: instruction,
        # A2UI grids/results embed long signed URLs verbatim; give the model
        # ample output budget so it never truncates mid-<a2ui-json> block.
        generate_content_config=genai_types.GenerateContentConfig(
            max_output_tokens=int(os.environ.get("MAX_OUTPUT_TOKENS", "16384")),
        ),
        tools=[
            load_artifacts,
            FunctionTool(list_gcs_images),
            FunctionTool(ingest_uploaded_image_tool),
            FunctionTool(get_eval_defaults),
            FunctionTool(run_fidelity_eval),
        ],
    )


_root_agent = None


def get_agent() -> Agent:
    global _root_agent
    if _root_agent is None:
        _root_agent = create_agent()
    return _root_agent


root_agent = get_agent()
