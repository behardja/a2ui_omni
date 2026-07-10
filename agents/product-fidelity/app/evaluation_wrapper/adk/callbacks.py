"""Generic ADK callbacks for the evaluation pipeline.

These are adapted from product_fidelity_agent/callbacks.py but stripped of
UI-specific concerns (base64 image injection, uploaded image extraction).
Only the structurally necessary callbacks are included.
"""

import inspect
import logging
import re

logger = logging.getLogger("evaluation_wrapper.adk")


def normalize_tool_args(tool, args, tool_context):
    """before_tool_callback: remap unrecognized argument names to valid ones.

    LLMs sometimes hallucinate slightly wrong parameter names. This callback
    inspects the tool function's signature and attempts to match any
    unrecognized arg to a valid parameter by substring containment.
    """
    func = getattr(tool, "func", None)
    if func is None:
        return None

    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys()) - {"tool_context"}

    unknown_keys = [k for k in args if k not in valid_params]
    if not unknown_keys:
        return None

    for key in unknown_keys:
        matched = None
        for param in valid_params:
            if key in param or param in key:
                if param not in args:
                    matched = param
                    break
        if matched:
            logger.info(
                "normalize_tool_args: remapped '%s' -> '%s' for tool %s",
                key,
                matched,
                getattr(tool, "name", "?"),
            )
            args[matched] = args.pop(key)
        else:
            logger.warning(
                "normalize_tool_args: unknown arg '%s' for tool %s (valid: %s)",
                key,
                getattr(tool, "name", "?"),
                valid_params,
            )

    return None


def cleanup_image_data(callback_context, llm_request):
    """before_model_callback: strip base64 image data from request contents.

    Replaces inline base64 markdown images with lightweight placeholder tags
    to prevent token bloat.
    """
    pattern = (
        r"!\[[^\]]*\]\(data:image/"
        r"(?:jpeg|png|gif|bmp|webp);base64,[A-Za-z0-9+/=\s]+\)"
    )
    for content in llm_request.contents:
        for part in content.parts:
            if hasattr(part, "text") and part.text:
                part.text = re.sub(pattern, "[image]", part.text)
    return None


def save_product_results(callback_context):
    """after_agent_callback: save evaluation results and reset per-product state.

    Called after the eval pipeline completes for one product. Appends results
    to the all_products accumulator and resets per-product state.
    """
    all_products = callback_context.state.get("all_products", [])
    all_products.append(
        {
            "sku_id": callback_context.state.get("sku_id"),
            "image_uris": callback_context.state.get("image_uris"),
            "ground_truth_description": callback_context.state.get(
                "ground_truth_description"
            ),
            "evaluation_history": callback_context.state.get(
                "evaluation_history", []
            ),
            "evaluation_passed": callback_context.state.get(
                "evaluation_passed", False
            ),
        }
    )
    callback_context.state["all_products"] = all_products

    # Reset per-product state for next iteration
    for key in [
        "sku_id",
        "image_uris",
        "ground_truth_description",
        "current_description",
        "candidate_image_uri",
        "candidate_video_uri",
        "gecko_score",
        "rubric_verdicts",
        "failing_verdicts_text",
    ]:
        callback_context.state[key] = None

    callback_context.state["evaluation_history"] = []
    callback_context.state["evaluation_passed"] = False
    callback_context.state["attempt"] = 1

    return None
