"""Product fidelity evaluation wrapper.

Two integration paths:

  1. Pure Python (recommended for most use cases):
     from evaluation_wrapper import EvalPipeline, EvalConfig

  2. ADK-native (for ADK-based multi-agent systems):
     from evaluation_wrapper.adk import build_eval_pipeline
"""

from .config import EvalConfig
from .pipeline import EvalPipeline
