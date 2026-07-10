"""ADK agent factories for the evaluation pipeline.

Each make_*() factory closes over an EvalConfig and returns a configured
ADK agent. These are wired together by build_eval_pipeline() in pipeline.py.
"""

from google.adk.agents.llm_agent import LlmAgent

from ..config import EvalConfig
from .callbacks import normalize_tool_args, cleanup_image_data
from .tools import (
    make_description_tool,
    make_gecko_tool,
    make_threshold_tool,
    make_refinement_tool,
)


def make_description_agent(config: EvalConfig, agent_model: str) -> LlmAgent:
    """Create the description generation agent."""
    return LlmAgent(
        name="DescriptionAgent",
        model=agent_model,
        include_contents="default",
        before_tool_callback=normalize_tool_args,
        instruction="""You are a product description generation coordinator.

Your task is to generate a ground-truth description of the product from its
reference images.

Call the generate_description tool with the image URIs: {image_uris}

Output only the generated description text, nothing else.""",
        tools=[make_description_tool(config)],
        output_key="ground_truth_description",
        description=(
            "Generates a ground-truth product description from reference "
            "images using Gemini."
        ),
    )


def make_evaluation_agent(config: EvalConfig, agent_model: str) -> LlmAgent:
    """Create the Gecko evaluation + threshold check agent."""
    media_word = "image" if config.media_type == "image" else "video"
    media_uri_key = (
        "candidate_image_uri"
        if config.media_type == "image"
        else "candidate_video_uri"
    )

    return LlmAgent(
        name="EvaluationAgent",
        model=agent_model,
        include_contents="default",
        before_model_callback=cleanup_image_data,
        before_tool_callback=normalize_tool_args,
        instruction=f"""You are a product {media_word} evaluation coordinator.

Step 1: Call run_gecko_evaluation with:
  - prompt: {{ground_truth_description}}
  - media_uri: {{{media_uri_key}}}

Step 2: After evaluation completes, call check_threshold to determine the result.

The check_threshold tool makes the pass/retry/fail decision deterministically.
If it returns "pass" or "fail", the pipeline will exit the loop automatically.
If it returns "retry", output a brief summary and stop — the refinement agent
will handle the next step.

Output a brief summary of the evaluation result.""",
        tools=[make_gecko_tool(config), make_threshold_tool(config)],
        description=(
            f"Evaluates candidate {media_word}s using Gecko and checks the "
            "fidelity threshold."
        ),
    )


def make_refinement_agent(config: EvalConfig, agent_model: str) -> LlmAgent:
    """Create the description refinement agent."""
    return LlmAgent(
        name="RefinementAgent",
        model=agent_model,
        include_contents="default",
        before_model_callback=cleanup_image_data,
        before_tool_callback=normalize_tool_args,
        instruction="""You are a description refinement coordinator.

The candidate did not pass the fidelity threshold. Refine the product
description to better emphasize the attributes that failed evaluation.

Call refine_description with:
  - description: {ground_truth_description}
  - failing_verdicts: the failing verdicts listed below

Failing verdicts from the latest evaluation:
{failing_verdicts_text}

Output only the refined description.""",
        tools=[make_refinement_tool(config)],
        description=(
            "Refines the product description based on failing evaluation "
            "verdicts, always deriving from the original to prevent drift."
        ),
    )
