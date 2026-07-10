"""ADK-native evaluation pipeline builder.

Constructs an ADK agent tree that wires together description generation,
an external generation agent (provided by the consumer), Gecko evaluation,
and iterative refinement.
"""

from google.adk.agents import LoopAgent, SequentialAgent
from google.adk.agents.llm_agent import LlmAgent

from ..config import EvalConfig
from .. import settings
from .agents import (
    make_description_agent,
    make_evaluation_agent,
    make_refinement_agent,
)
from .callbacks import save_product_results


def build_eval_pipeline(
    generate_agent: LlmAgent,
    config: EvalConfig | None = None,
    agent_model: str | None = None,
) -> SequentialAgent:
    """Build an ADK agent tree for the evaluation pipeline.

    Args:
        generate_agent: An ADK Agent that handles media generation.
            It should read reference URIs and description from
            tool_context.state and write the candidate URI to state:
              - Read:  state["image_uris"], state["current_description"],
                       state["attempt"], state["failing_verdicts_text"]
              - Write: state["candidate_image_uri"] (for images)
                    or state["candidate_video_uri"] (for videos)

        config: EvalConfig. If None, builds from settings.py.

        agent_model: Model ID for the orchestration agents (description,
            evaluation, refinement). If None, reads from settings.AGENT_MODEL.

    Returns:
        A SequentialAgent that runs: description → [generate → eval → refine] loop.
        Wire this into your own agent tree or run it directly.

    Usage:
        from evaluation_wrapper.adk import build_eval_pipeline
        from evaluation_wrapper import EvalConfig

        pipeline = build_eval_pipeline(
            generate_agent=my_image_gen_agent,
            config=EvalConfig(project_id="my-project"),
        )

        # Use as a sub-agent in your own root agent:
        root = LlmAgent(
            name="Root",
            sub_agents=[pipeline],
            ...
        )
    """
    if config is None:
        config = EvalConfig.from_settings()
    if agent_model is None:
        agent_model = settings.AGENT_MODEL

    description_agent = make_description_agent(config, agent_model)
    evaluation_agent = make_evaluation_agent(config, agent_model)
    refinement_agent = make_refinement_agent(config, agent_model)

    refinement_loop = LoopAgent(
        name="RefinementLoop",
        sub_agents=[generate_agent, evaluation_agent, refinement_agent],
        max_iterations=config.max_retries,
    )

    pipeline = SequentialAgent(
        name="EvalPipeline",
        sub_agents=[description_agent, refinement_loop],
        after_agent_callback=save_product_results,
        description=(
            "Generates description and runs iterative evaluation/refinement "
            "for one product."
        ),
    )

    return pipeline
