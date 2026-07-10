"""Pure Python evaluation pipeline — no ADK dependency.

Owns the eval loop: description → generate (plugin) → gecko → threshold → refine → retry.
The external repo provides only the generate_fn callable.
"""

import logging
from typing import Callable, Protocol

from .config import EvalConfig
from .tools.gecko import evaluate as gecko_evaluate
from .tools.description import generate_description, refine_description

logger = logging.getLogger("evaluation_wrapper")


class GenerateFn(Protocol):
    """Protocol for the generation function the external repo provides.

    The wrapper calls this function at each attempt. The function should:
      1. Use the reference images + description to generate media
      2. Upload the result to GCS (or any accessible URI)
      3. Return the URI of the generated media

    Additional context is passed via kwargs (sku_id, user_prompt).
    """

    def __call__(
        self,
        reference_uris: list[str],
        description: str,
        attempt: int,
        failing_verdicts: list[str],
        **kwargs,
    ) -> str: ...


class EvalPipeline:
    """Pure Python product fidelity evaluation pipeline.

    Usage:
        from evaluation_wrapper import EvalPipeline, EvalConfig

        def my_generate(reference_uris, description, attempt, failing_verdicts, **kwargs):
            # ... generate media, upload to GCS ...
            return "gs://my-bucket/output/candidate.png"

        config = EvalConfig.from_settings()
        pipeline = EvalPipeline(generate_fn=my_generate, config=config)

        result = pipeline.run(
            reference_uris=["gs://bucket/front.png", "gs://bucket/side.png"],
            sku_id="SKU-1234",
        )
    """

    def __init__(self, generate_fn: GenerateFn, config: EvalConfig):
        self.generate_fn = generate_fn
        self.config = config

    def run(
        self,
        reference_uris: list[str],
        sku_id: str,
        user_prompt: str = "",
        image_model: str = "",
    ) -> dict:
        """Run the full evaluation pipeline for one product.

        Args:
            reference_uris: GCS URIs of product reference images.
            sku_id: Product SKU identifier.
            user_prompt: Optional creative direction for generation.
            image_model: Optional per-run image model id passed to the generate_fn.

        Returns:
            dict with keys:
              - sku_id: str
              - passed: bool
              - final_score: float
              - attempts: list[dict]  (per-attempt details)
              - ground_truth_description: str
        """
        cfg = self.config

        # Step 1: Generate ground-truth description
        logger.info("sku=%s | stage=description | generating", sku_id)
        description = generate_description(
            image_uris=reference_uris,
            project_id=cfg.project_id,
            location=cfg.description_location,
            model=cfg.description_model,
        )
        original_description = description
        logger.info("sku=%s | stage=description | done", sku_id)

        attempts = []

        for attempt in range(1, cfg.max_retries + 1):
            failing = attempts[-1]["failing_verdicts"] if attempts else []

            # Step 2: Generate candidate (THE PLUGIN CALL)
            logger.info(
                "sku=%s | stage=generation | attempt=%d", sku_id, attempt
            )
            candidate_uri = self.generate_fn(
                reference_uris=reference_uris,
                description=description,
                attempt=attempt,
                failing_verdicts=failing,
                sku_id=sku_id,
                user_prompt=user_prompt,
                image_model=image_model,
            )
            logger.info(
                "sku=%s | stage=generation | attempt=%d | candidate=%s",
                sku_id,
                attempt,
                candidate_uri,
            )

            # Step 3: Gecko evaluation
            logger.info(
                "sku=%s | stage=evaluation | attempt=%d", sku_id, attempt
            )
            eval_result = gecko_evaluate(
                prompt=description,
                media_uri=candidate_uri,
                media_type=cfg.media_type,
                project_id=cfg.project_id,
                location=cfg.location,
            )

            # Handle evaluation infrastructure errors
            if eval_result.get("status") == "error":
                logger.error(
                    "sku=%s | stage=evaluation | attempt=%d | error=%s",
                    sku_id,
                    attempt,
                    eval_result.get("message"),
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "score": 0.0,
                        "candidate_uri": candidate_uri,
                        "passing_verdicts": [],
                        "failing_verdicts": [],
                        "error": eval_result.get("message"),
                    }
                )
                # Treat infra errors as a failed attempt — continue to retry
                if attempt < cfg.max_retries:
                    continue
                break

            score = eval_result["score"]
            attempts.append(
                {
                    "attempt": attempt,
                    "score": score,
                    "candidate_uri": candidate_uri,
                    "passing_verdicts": eval_result.get("passing", []),
                    "failing_verdicts": eval_result.get("failing", []),
                }
            )

            logger.info(
                "sku=%s | stage=evaluation | attempt=%d | score=%.2f | "
                "pass=%d | fail=%d",
                sku_id,
                attempt,
                score,
                eval_result.get("passing_count", 0),
                eval_result.get("failing_count", 0),
            )

            # Step 4: Threshold check
            if score >= cfg.threshold:
                logger.info(
                    "sku=%s | stage=threshold | attempt=%d | action=pass",
                    sku_id,
                    attempt,
                )
                return {
                    "sku_id": sku_id,
                    "passed": True,
                    "final_score": score,
                    "attempts": attempts,
                    "ground_truth_description": original_description,
                }

            # Step 5: Refine description for next attempt
            if attempt < cfg.max_retries:
                logger.info(
                    "sku=%s | stage=refinement | attempt=%d", sku_id, attempt
                )
                description = refine_description(
                    description=original_description,
                    failing_verdicts=eval_result.get("failing", []),
                    project_id=cfg.project_id,
                    location=cfg.description_location,
                    model=cfg.description_model,
                )
                logger.info(
                    "sku=%s | stage=refinement | attempt=%d | done",
                    sku_id,
                    attempt,
                )
            else:
                logger.warning(
                    "sku=%s | stage=threshold | attempt=%d | action=fail",
                    sku_id,
                    attempt,
                )

        return {
            "sku_id": sku_id,
            "passed": False,
            "final_score": attempts[-1]["score"] if attempts else 0.0,
            "attempts": attempts,
            "ground_truth_description": original_description,
        }
