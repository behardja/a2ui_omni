from dataclasses import dataclass


@dataclass
class EvalConfig:
    """Configuration for the evaluation pipeline.

    Two ways to use:
      1. Edit settings.py and call EvalConfig.from_settings()
      2. Pass values directly: EvalConfig(project_id="my-project", ...)
    """

    project_id: str
    location: str = "us-central1"  # region for the Gecko eval service
    bucket_name: str = ""
    threshold: float = 0.7
    max_retries: int = 3
    description_model: str = "gemini-3.5-flash"
    # Endpoint for the describe/refine model. Newer Gemini (3.x) models are served
    # on the `global` endpoint, not regional — keep this separate from `location`
    # (which the Gecko eval service needs as a real region like us-central1).
    description_location: str = "global"
    media_type: str = "image"  # "image" or "video"

    @classmethod
    def from_settings(cls) -> "EvalConfig":
        """Build an EvalConfig from evaluation_wrapper/settings.py values."""
        from . import settings

        return cls(
            project_id=settings.PROJECT_ID,
            location=settings.LOCATION,
            bucket_name=settings.BUCKET_NAME,
            threshold=settings.PASSING_THRESHOLD,
            max_retries=settings.MAX_RETRIES,
            description_model=settings.DESCRIPTION_MODEL,
            description_location=settings.DESCRIPTION_LOCATION,
            media_type=settings.MEDIA_TYPE,
        )
