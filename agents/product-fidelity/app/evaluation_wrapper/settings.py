"""User-editable settings for the evaluation wrapper.

Vendored from behardja/product-fidelity-eval (eval_wrapper branch) for the
A2UI product-fidelity agent. All values are env-driven so the same code runs
locally, under `adk web`, and (later) on Agent Engine. Set them in .env.
"""

import os

# --- GCP Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("LOCATION", "us-central1")  # Gecko eval service region
# Endpoint for the describe/refine model (gemini-3.x live on `global`, not regional).
DESCRIPTION_LOCATION = os.environ.get("DESCRIPTION_LOCATION", "global")
# Bucket where Gecko-readable candidate media is written (see generate.py).
BUCKET_NAME = os.environ.get("CANDIDATE_BUCKET") or os.environ.get("BUCKET_NAME", "")

# --- Model IDs ---
DESCRIPTION_MODEL = os.environ.get("DESCRIPTION_MODEL", "gemini-3.5-flash")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")

# --- Evaluation Thresholds ---
PASSING_THRESHOLD = float(os.environ.get("PASSING_THRESHOLD", "0.7"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# --- Media Type ---
MEDIA_TYPE = os.environ.get("MEDIA_TYPE", "image")  # "image" or "video"
