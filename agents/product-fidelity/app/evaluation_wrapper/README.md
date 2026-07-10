# Evaluation Wrapper

Standalone product fidelity evaluation library. Scores generated media (images or videos) against original product reference images using [Gecko](https://cloud.google.com/vertex-ai/generative-ai/docs/evaluation/gecko-metrics) rubric-based evaluation, with iterative refinement on failure.

**You provide the generation function. The wrapper handles everything else:** ground-truth description generation, Gecko scoring, threshold checking, description refinement, and retry orchestration.

---

## Step 1: Configure Your Settings

Edit `evaluation_wrapper/settings.py` with your GCP project details:

```python
# evaluation_wrapper/settings.py
PROJECT_ID = "my-project"
LOCATION = "us-central1"
BUCKET_NAME = "my-bucket"
PASSING_THRESHOLD = 0.7
MAX_RETRIES = 3
DESCRIPTION_MODEL = "gemini-3-pro-preview"
MEDIA_TYPE = "image"  # or "video"
```

Environment variables (`PROJECT_ID`, `LOCATION`, `BUCKET_NAME`) override these values if set.

---

## Step 2: Write Your Generation Function

This is the only code you need to write. The wrapper calls your function at each attempt, passing everything your model needs to produce a candidate. Your function generates media and returns a GCS URI.

### Function Signature

```python
# The protocol is defined in evaluation_wrapper/pipeline.py (GenerateFn)
def my_generate(
    reference_uris: list[str],
    description: str,
    attempt: int,
    failing_verdicts: list[str],
    **kwargs,
) -> str:
```

### Parameters Your Function Receives

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `reference_uris` | `list[str]` | `["gs://bucket/front.png", "gs://bucket/side.png"]` | GCS URIs of the original product photos. These are the same URIs you pass to `pipeline.run()`. Your generation model can use these as visual references. |
| `description` | `str` | `"A red leather handbag with gold buckle..."` | Text description of the product. On **attempt 1**, this is an auto-generated ground-truth description from the reference images. On **retries**, this is a refined version that more strongly emphasizes the attributes that failed. Use this as your generation prompt or supplementary guidance. |
| `attempt` | `int` | `1`, `2`, `3` | Current attempt number (1-indexed). Use for logging, naming output files, or adjusting generation parameters on retries (e.g., increasing guidance scale). |
| `failing_verdicts` | `list[str]` | `["The logo 'COACH' is missing", "Handle is brown, not black"]` | Attribute descriptions that failed Gecko evaluation on the previous attempt. **Empty list on attempt 1.** Use these to adjust your generation strategy on retries. |
| `**kwargs` | | | Additional context: `sku_id` (str) and `user_prompt` (str, optional creative direction). |

### What Your Function Must Return

A **GCS URI** (`gs://...`) pointing to the generated candidate media.

The file must be in a GCS bucket readable by your GCP project, because the Gecko evaluation API reads it from GCS to score it.

- Images: `gs://my-bucket/output/candidate.png`
- Videos: `gs://my-bucket/output/candidate.mp4`

### What Your Function Must Do

1. **Generate media** — call your generation model (Imagen, Veo, DALL-E, Stable Diffusion, a custom pipeline, etc.)
2. **Upload to GCS** — save the generated output to a GCS bucket
3. **Return the GCS URI** — the wrapper passes this to Gecko for scoring

Your function **never** touches Gecko, **never** manages retries, **never** deals with refinement. It just generates when asked.

### Example: Nano banana on Vertex AI

```python
# your_generate.py
import uuid
from google import genai
from google.genai import types
from google.cloud import storage

def my_imagen_generate(reference_uris, description, attempt, failing_verdicts, **kwargs):
    """Generate a product image using Gemini's image generation."""
    sku_id = kwargs.get("sku_id", "unknown")
    user_prompt = kwargs.get("user_prompt", "")

    client = genai.Client(
        vertexai=True, project="my-project", location="us-central1",
    )

    # Build the prompt from description + any user creative direction
    prompt = f"Generate a product photo: {description}"
    if user_prompt:
        prompt += f"\nCreative direction: {user_prompt}"

    # On retries, emphasize what failed
    if failing_verdicts:
        prompt += "\n\nPay special attention to these attributes that were missed:"
        for v in failing_verdicts:
            prompt += f"\n- {v}"

    # Include reference images so the model can see the actual product
    content_parts = []
    for uri in reference_uris:
        ext = uri.lower().rsplit(".", 1)[-1]
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        content_parts.append(types.Part.from_uri(file_uri=uri, mime_type=mime))
    content_parts.append(prompt)

    # Generate
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=content_parts,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )

    # Extract image bytes and upload to GCS
    for part in response.parts:
        if part.inline_data is not None:
            image_bytes = part.inline_data.data
            gcs_uri = (
                f"gs://my-bucket/generated/{sku_id}"
                f"/attempt_{attempt}_{uuid.uuid4().hex[:8]}.png"
            )

            # Upload to GCS
            path = gcs_uri[5:]  # strip "gs://"
            bucket_name = path.split("/")[0]
            blob_path = "/".join(path.split("/")[1:])
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            bucket.blob(blob_path).upload_from_string(image_bytes)

            return gcs_uri

    raise RuntimeError("No image generated by the model")
```

### Example: Wrapping an Existing System

If your generation logic already lives inside a larger system, write a thin adapter:

```python
# my_adapter.py
from my_existing_system import run_generation_pipeline

def generate_via_my_system(reference_uris, description, attempt, failing_verdicts, **kwargs):
    """Adapter that calls an existing generation pipeline."""
    result = run_generation_pipeline(
        images=reference_uris,
        prompt=description,
        retry_hints=failing_verdicts,
    )
    return result.output_gcs_uri
```

### Example: Minimal Stub (for testing)

```python
# test_generate.py
def stub_generate(reference_uris, description, attempt, failing_verdicts, **kwargs):
    """Returns a fixed URI — useful for testing the pipeline without actual generation."""
    return "gs://my-bucket/test-fixtures/sample-candidate.png"
```

---

## Step 3: Run the Pipeline

Pass your generation function to `EvalPipeline` and call `run()`:

```python
# your_app.py
from evaluation_wrapper import EvalPipeline, EvalConfig  # evaluation_wrapper/__init__.py

config = EvalConfig.from_settings()  # reads from evaluation_wrapper/settings.py
pipeline = EvalPipeline(generate_fn=my_imagen_generate, config=config)  # evaluation_wrapper/pipeline.py

result = pipeline.run(
    reference_uris=[
        "gs://my-bucket/product/front.png",
        "gs://my-bucket/product/side.png",
    ],
    sku_id="SKU-1234",
    user_prompt="product on a wooden table in warm lighting",  # optional
)

print(result["passed"])       # True / False
print(result["final_score"])  # 0.85
print(result["attempts"])     # list of per-attempt details
```

The result dict:

```python
{
    "sku_id": "SKU-1234",
    "passed": True,
    "final_score": 0.85,
    "attempts": [
        {
            "attempt": 1,
            "score": 0.55,
            "candidate_uri": "gs://my-bucket/generated/SKU-1234/attempt_1_a1b2c3d4.png",
            "passing_verdicts": ["Color is correct", "Shape matches"],
            "failing_verdicts": ["Logo text 'COACH' is missing", "Handle material is wrong"],
        },
        {
            "attempt": 2,
            "score": 0.85,
            "candidate_uri": "gs://my-bucket/generated/SKU-1234/attempt_2_e5f6g7h8.png",
            "passing_verdicts": ["Color is correct", "Shape matches", "Logo present", "Handle correct"],
            "failing_verdicts": [],
        },
    ],
    "ground_truth_description": "A red leather handbag with gold 'COACH' lettering...",
}
```

---

## Integrating with a Multi-Agent System

There are two architectural patterns for integrating this wrapper into a multi-agent system.

### Option 1: The "Tool" Approach (Simpler)

In this approach, you write a standard Python `my_generate()` function. Your agent pauses, calls the wrapper as a single **tool**, and the wrapper runs its entire Generate → Score → Refine loop internally using your Python function.

- **Who generates?** Your pure Python `my_generate()` function.
- **Who loops?** The `EvalPipeline` handles the retry loop internally.
- **Best for:** Most users. It treats the complex evaluation loop as a single black box.

### Option 2: The "ADK-Native" Approach (Advanced)

In this approach, you do *not* write a `my_generate()` Python function. Instead, you build a fully-fledged "Generation Agent" (which likely has its own tools for calling Vertex AI, Midjourney, etc.). You hand that entire Agent over to the wrapper, which orchestrates it alongside its own Evaluation and Refinement Agents in a shared ADK tree.

- **Who generates?** Your "Generation Agent" (using its own tools).
- **Who loops?** The ADK orchestrator (passing state between your generation agent and the wrapper's eval agents).
- **Best for:** Complex generation workflows where your generation step requires its own multi-agent logic or extensive state management.

---

## Option 1: Using as a Tool

If you want to use the simpler "Tool" approach, wrap `pipeline.run()` in a tool function. The pipeline is a regular Python function call — it doesn't require ADK to run.

### As an ADK tool

```python
# your_adk_app.py
from google.adk.tools.tool_context import ToolContext
from evaluation_wrapper import EvalPipeline, EvalConfig  # evaluation_wrapper/__init__.py

config = EvalConfig.from_settings()  # reads from evaluation_wrapper/settings.py
pipeline = EvalPipeline(generate_fn=my_imagen_generate, config=config)  # evaluation_wrapper/pipeline.py

def evaluate_product(image_uris: str, sku_id: str, tool_context: ToolContext) -> dict:
    """Evaluate product fidelity for a generated image.

    Args:
        image_uris: Comma-separated GCS URIs of product reference images.
        sku_id: Product SKU identifier.

    Returns:
        Evaluation result with pass/fail, score, and per-attempt details.
    """
    uris = [u.strip() for u in image_uris.split(",") if u.strip()]
    user_prompt = tool_context.state.get("user_prompt", "")

    result = pipeline.run(
        reference_uris=uris,
        sku_id=sku_id,
        user_prompt=user_prompt,
    )

    # Optionally store results in agent state
    tool_context.state["eval_result"] = result
    return result

# Use it in your agent
from google.adk.agents.llm_agent import LlmAgent

root = LlmAgent(
    name="MyAgent",
    model="gemini-3-pro-preview",
    tools=[evaluate_product],
    instruction="When the user provides product images, evaluate their fidelity...",
)
```

Your multi-agent system calls `evaluate_product` as a tool, the pipeline runs synchronously inside it, and the result dict comes back. No ADK agent tree required inside the wrapper.

### As a plain function call

If your multi-agent system isn't ADK-based, just call `pipeline.run()` directly from wherever your orchestration logic lives:

```python
# works from any Python code — LangChain, CrewAI, custom orchestrator, etc.
result = pipeline.run(
    reference_uris=["gs://bucket/front.png"],
    sku_id="SKU-1234",
)
if not result["passed"]:
    # route to human review, retry with different model, etc.
    ...
```

---

## Option 2: ADK-Native Integration

If your generation logic is itself an ADK agent and you want the eval loop to run **inside** ADK's agent orchestration (with ADK tracing, state management, and agent tree composition), use `build_eval_pipeline()`.

You provide your Generation Agent. The wrapper builds an agent tree around it:

```
SequentialAgent (EvalPipeline)
  ├── DescriptionAgent              ← wrapper's agent
  └── LoopAgent (RefinementLoop)    ← wrapper's loop
        ├── YOUR GenerationAgent    ← your agent, plugged in here
        ├── EvaluationAgent         ← wrapper's agent (runs Gecko)
        └── RefinementAgent         ← wrapper's agent (refines description)
```

```python
# your_adk_app.py
from evaluation_wrapper.adk import build_eval_pipeline  # evaluation_wrapper/adk/pipeline.py
from evaluation_wrapper import EvalConfig               # evaluation_wrapper/config.py
from google.adk.agents.llm_agent import LlmAgent

# Your generation agent — must read/write tool_context.state
# See "ADK State Contract" below for required keys
my_gen_agent = LlmAgent(
    name="MyImageGenAgent",
    model="gemini-3-pro-preview",
    tools=[my_generate_tool],
    instruction="Generate a product image using the reference images in {image_uris}...",
)

# Build the eval pipeline agent tree
eval_pipeline = build_eval_pipeline(
    generate_agent=my_gen_agent,
    config=EvalConfig(project_id="my-project"),
)

# Wire into your own agent tree
root = LlmAgent(
    name="Root",
    sub_agents=[eval_pipeline],
    instruction="...",
)
```

### ADK State Contract

Your generation agent communicates with the wrapper's agents via `tool_context.state`. The wrapper reads and writes these keys automatically. Your agent only needs to handle the ones marked below.

**Keys your generation agent should READ:**

| Key | Type | Description |
|-----|------|-------------|
| `image_uris` | `str` | Comma-separated GCS URIs of reference images |
| `current_description` | `str` | Current description (original on attempt 1, refined on retries) |
| `attempt` | `int` | Current attempt number |
| `failing_verdicts_text` | `str` | Newline-separated failing verdicts (empty on attempt 1) |
| `user_prompt` | `str` | Optional creative direction from the user |
| `sku_id` | `str` | Product SKU identifier |

**Keys your generation agent must WRITE:**

| Key | Type | Description |
|-----|------|-------------|
| `candidate_image_uri` | `str` | GCS URI of generated image (when `media_type="image"`) |
| `candidate_video_uri` | `str` | GCS URI of generated video (when `media_type="video"`) |

---

## How the Pipeline Works Internally

The pipeline runs a loop: generate a candidate, score it, and if it fails, refine the description and try again.

```
pipeline.run(reference_uris, sku_id)       (evaluation_wrapper/pipeline.py)
│
│  Step 1: DESCRIBE
│  Calls Gemini to analyze the reference product photos and generate a
│  detailed ground-truth text description.
│  (evaluation_wrapper/tools/description.py → generate_description)
│
│  Step 2–5: RETRY LOOP (up to max_retries)
│  ┌─────────────────────────────────────────────────────────────────┐
│  │                                                                 │
│  │  Step 2: GENERATE — YOUR function is called here                │
│  │  Input:  reference URIs, description, attempt, failing verdicts │
│  │  Output: GCS URI of the generated candidate                     │
│  │                                                                 │
│  │  Step 3: EVALUATE                                               │
│  │  Gecko scores the candidate against the description.            │
│  │  Returns a 0.0–1.0 score and per-attribute pass/fail verdicts.  │
│  │  (evaluation_wrapper/tools/gecko.py → evaluate)                 │
│  │                                                                 │
│  │  Step 4: THRESHOLD CHECK                                        │
│  │  score >= threshold? → return {passed: True}                    │
│  │  score <  threshold? → continue to Step 5                       │
│  │                                                                 │
│  │  Step 5: REFINE                                                 │
│  │  Rewrites the description to emphasize the failing attributes.  │
│  │  This refined description is passed to YOUR function on the     │
│  │  next attempt so the generation model can correct what it       │
│  │  missed.                                                        │
│  │  (evaluation_wrapper/tools/description.py → refine_description) │
│  │                                                                 │
│  └──────────────────────── loops back to Step 2 ───────────────────┘
│
│  All retries exhausted → return {passed: False}
```

---

## Package Structure

```
evaluation_wrapper/
├── __init__.py           # Public API: EvalPipeline, EvalConfig
├── config.py             # EvalConfig dataclass + from_settings()
├── settings.py           # ← EDIT THIS: your GCP project settings
├── pipeline.py           # Pure Python pipeline (no ADK)
├── tools/
│   ├── gecko.py          # Standalone Gecko evaluation (image + video)
│   └── description.py    # Standalone description gen + refinement
├── adk/
│   ├── __init__.py       # ADK public API: build_eval_pipeline
│   ├── tools.py          # ADK tool wrappers (ToolContext bridges)
│   ├── agents.py         # ADK agent definitions (factories)
│   ├── callbacks.py      # Generic callbacks
│   └── pipeline.py       # build_eval_pipeline() factory
└── prompts/
    ├── description_system.txt   # System prompt for description generation
    └── description_user.txt     # User prompt for description generation
```

---

## Syncing with product_fidelity_agent

The standalone tools in `evaluation_wrapper/tools/` are derived from `product_fidelity_agent/tools/`. They contain the same core logic but stripped of ADK `ToolContext` dependencies and hardcoded config.

**Source files and their standalone counterparts:**

| product_fidelity_agent/tools/ | evaluation_wrapper/tools/ | Notes |
|-------------------------------|--------------------------|-------|
| `gecko.py` (`run_gecko_evaluation`, `run_gecko_video_evaluation`, `check_threshold`) | `gecko.py` (`evaluate`) | Unified into one function with `media_type` param. No ToolContext, no state writes. |
| `gemini.py` (`generate_description`, `refine_description`) | `description.py` (`generate_description`, `refine_description`) | Takes `list[str]` instead of comma-separated string. Returns plain strings instead of dicts. |
| `gcs.py` | Not included | Not needed — Gecko SDK and GenAI SDK handle GCS URIs natively. |
| `image_gen.py`, `video_gen.py` | Not included | Generation is the external repo's responsibility. |
| `reporting.py` | Not included | Replaced by structured logging + result dicts. |

**When to sync:**

After updating logic in `product_fidelity_agent/tools/gecko.py` or `product_fidelity_agent/tools/gemini.py`, review and propagate changes to the corresponding `evaluation_wrapper/tools/` files. Each standalone file has a "Synced from" header noting the source file and last sync date.

**What typically changes:**

- Gecko retry logic, rubric parsing, verdict extraction → sync to `evaluation_wrapper/tools/gecko.py`
- Description/refinement prompts, model parameters → sync to `evaluation_wrapper/tools/description.py`
- Prompt templates → sync to `evaluation_wrapper/prompts/`

**What doesn't need syncing:**

- ToolContext state writes (only in the ADK wrappers in `evaluation_wrapper/adk/tools.py`)
- Image/video generation tools (external repo's domain)
- Reporting tools (replaced by structured logging)
- Callbacks (ADK-specific, reimplemented in `evaluation_wrapper/adk/callbacks.py`)

---

## Logging

The pure Python pipeline (`evaluation_wrapper/pipeline.py`) uses structured logging instead of a report agent:

```
INFO  | evaluation_wrapper | sku=SKU-1234 | stage=description | generating
INFO  | evaluation_wrapper | sku=SKU-1234 | stage=generation  | attempt=1 | candidate=gs://...
INFO  | evaluation_wrapper | sku=SKU-1234 | stage=evaluation  | attempt=1 | score=0.55 | pass=3 | fail=4
INFO  | evaluation_wrapper | sku=SKU-1234 | stage=refinement  | attempt=1 | done
INFO  | evaluation_wrapper | sku=SKU-1234 | stage=generation  | attempt=2 | candidate=gs://...
INFO  | evaluation_wrapper | sku=SKU-1234 | stage=evaluation  | attempt=2 | score=0.85 | pass=6 | fail=1
INFO  | evaluation_wrapper | sku=SKU-1234 | stage=threshold   | attempt=2 | action=pass
```

Configure with standard Python logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```
