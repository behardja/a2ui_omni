"""ADK tool wrappers — bridge ToolContext.state to standalone functions.

Each make_*() factory closes over an EvalConfig and returns an ADK-compatible
tool function (with ToolContext parameter). ADK introspects the returned
function's name, docstring, and type hints to build the tool schema.
"""

from google.adk.tools.tool_context import ToolContext

from ..config import EvalConfig
from ..tools.gecko import evaluate as gecko_evaluate
from ..tools.description import (
    generate_description as gen_desc_standalone,
    refine_description as ref_desc_standalone,
)


def make_description_tool(config: EvalConfig):
    """Create an ADK tool for generating ground-truth descriptions."""

    def generate_description(image_uris: str, tool_context: ToolContext) -> dict:
        """Generate a ground-truth product description from reference images.

        Args:
            image_uris: Comma-separated GCS URIs of product reference images.

        Returns:
            dict with 'description' containing the generated ground-truth text.
        """
        uris = [u.strip() for u in image_uris.split(",") if u.strip()]
        description = gen_desc_standalone(
            image_uris=uris,
            project_id=config.project_id,
            location=config.location,
            model=config.description_model,
        )
        tool_context.state["ground_truth_description"] = description
        tool_context.state["current_description"] = description
        tool_context.state["attempt"] = 1
        return {"status": "success", "description": description}

    return generate_description


def make_refinement_tool(config: EvalConfig):
    """Create an ADK tool for refining descriptions based on failing verdicts."""

    def refine_description(
        description: str, failing_verdicts: str, tool_context: ToolContext
    ) -> dict:
        """Refine a product description to emphasize attributes that failed evaluation.

        Args:
            description: The original ground-truth description to refine.
            failing_verdicts: Newline-separated list of failing rubric verdicts.

        Returns:
            dict with 'refined_description' containing the updated text.
        """
        verdicts_list = [
            v.lstrip("- ").strip()
            for v in failing_verdicts.split("\n")
            if v.strip()
        ]
        refined = ref_desc_standalone(
            description=description,
            failing_verdicts=verdicts_list,
            project_id=config.project_id,
            location=config.location,
            model=config.description_model,
        )
        tool_context.state["current_description"] = refined
        attempt = tool_context.state.get("attempt", 1)
        tool_context.state["attempt"] = attempt + 1
        return {"status": "success", "refined_description": refined}

    return refine_description


def make_gecko_tool(config: EvalConfig):
    """Create an ADK tool for running Gecko evaluation."""

    def run_gecko_evaluation(
        prompt: str, media_uri: str, tool_context: ToolContext
    ) -> dict:
        """Run Gecko evaluation on a candidate image or video.

        Args:
            prompt: The ground-truth description to evaluate against.
            media_uri: GCS URI of the candidate media to evaluate.

        Returns:
            dict with score, verdict counts, and lists of passing/failing verdicts.
        """
        result = gecko_evaluate(
            prompt=prompt,
            media_uri=media_uri,
            media_type=config.media_type,
            project_id=config.project_id,
            location=config.location,
        )

        if result.get("status") == "error":
            return result

        # Write to state — same keys as product_fidelity_agent
        tool_context.state["gecko_score"] = result["score"]
        tool_context.state["rubric_verdicts"] = {
            "passing": result["passing"],
            "failing": result["failing"],
        }
        tool_context.state["failing_verdicts_text"] = "\n".join(
            f"- {v}" for v in result["failing"]
        )

        # Track history across attempts
        history = tool_context.state.get("evaluation_history", [])
        history.append(
            {
                "attempt": tool_context.state.get("attempt", 1),
                "score": result["score"],
                "passing_verdicts": result["passing"],
                "failing_verdicts": result["failing"],
                "media_uri": media_uri,
            }
        )
        tool_context.state["evaluation_history"] = history

        return {
            "status": "success",
            "score": result["score"],
            "total_verdicts": result["total_verdicts"],
            "passing_count": result["passing_count"],
            "failing_count": result["failing_count"],
            "failing_verdicts": result["failing"],
        }

    return run_gecko_evaluation


def make_threshold_tool(config: EvalConfig):
    """Create an ADK tool for checking the evaluation threshold."""

    def check_threshold(tool_context: ToolContext, **kwargs) -> dict:
        """Check if the current Gecko score meets the passing threshold.

        This is a deterministic check — no LLM reasoning is involved in the
        pass/retry/fail decision. Call this after run_gecko_evaluation.

        Returns:
            dict with 'action' (pass, retry, or fail), score, and context.
        """
        score = tool_context.state.get("gecko_score", 0.0)
        attempt = tool_context.state.get("attempt", 1)

        if score >= config.threshold:
            tool_context.state["evaluation_passed"] = True
            tool_context.actions.escalate = True
            return {
                "action": "pass",
                "score": score,
                "threshold": config.threshold,
                "attempt": attempt,
                "message": (
                    f"Score {score:.2f} meets threshold {config.threshold}. "
                    "Evaluation passed."
                ),
            }

        if attempt >= config.max_retries:
            tool_context.state["evaluation_passed"] = False
            tool_context.actions.escalate = True
            return {
                "action": "fail",
                "score": score,
                "threshold": config.threshold,
                "attempt": attempt,
                "max_attempts": config.max_retries,
                "message": (
                    f"Score {score:.2f} below threshold after {attempt} "
                    "attempts. Flagged for HITL review."
                ),
            }

        # Retry needed
        failing_verdicts = tool_context.state.get("rubric_verdicts", {}).get(
            "failing", []
        )
        return {
            "action": "retry",
            "score": score,
            "threshold": config.threshold,
            "attempt": attempt,
            "max_attempts": config.max_retries,
            "failing_verdicts": failing_verdicts,
            "message": (
                f"Score {score:.2f} below threshold {config.threshold}. "
                f"Attempt {attempt}/{config.max_retries}. Refinement needed."
            ),
        }

    return check_threshold
