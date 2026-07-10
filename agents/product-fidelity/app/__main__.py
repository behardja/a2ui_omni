"""Local A2A server for the product-fidelity agent.

Serves the ADK agent over the A2A protocol so a local A2UI renderer (the
dev_client) can consume and render its A2UI output — the local stand-in for
Gemini Enterprise. Run:  python -m app   (from agents/product-fidelity/, defaults to port 10002)

The agent card advertises the A2UI extension so renderers know to treat the
`application/json+a2ui` DataParts as UI.
"""

import logging
import os
import warnings

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from starlette.middleware.cors import CORSMiddleware

from app.executor import ProductFidelityExecutor

logging.basicConfig(level=logging.INFO)

# --- Quiet known-harmless noise so real log lines are visible ---
# 1. Python 3.10 EOL + google-cloud-storage<3 deprecation notices.
warnings.filterwarnings("ignore", category=FutureWarning)
# 2. ADK's OpenTelemetry tracing leaks "Failed to detach context" tracebacks when
#    async generators close (GeneratorExit) — non-fatal; requests still return 200.
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

A2UI_EXTENSION = {
    "uri": "https://a2ui.org/a2a-extension/a2ui/v0.8",
    "description": "Ability to render A2UI",
    "required": False,
    "params": {
        "supportedCatalogIds": [
            "https://a2ui.org/specification/v0_8/standard_catalog_definition.json"
        ]
    },
}


def build_agent_card(base_url: str) -> AgentCard:
    skill = AgentSkill(
        id="product-fidelity-eval",
        name="Product Fidelity Evaluation",
        description=(
            "Browse GCS product images, generate a candidate with "
            "gemini-3.1-flash-image, score fidelity with Gecko, and render "
            "the results as A2UI."
        ),
        tags=["product", "fidelity", "eval", "image", "a2ui"],
        examples=[
            "List images in gs://my-bucket/products/",
            "Evaluate gs://my-bucket/products/sneaker.png",
        ],
    )
    capabilities = AgentCapabilities(streaming=False, extensions=[A2UI_EXTENSION])
    return AgentCard(
        name="Product Fidelity Agent",
        description="Generate new product images with fidelity scoring. Type 'start' to pick or upload a reference image, add optional creative direction, and get a generated image plus a fidelity report.",
        url=base_url,
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=capabilities,
        skills=[skill],
    )


@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=int(os.environ.get("PORT", 10002)))
def main(host: str, port: int):
    base_url = os.environ.get("AGENT_BASE_URL", f"http://localhost:{port}")
    request_handler = DefaultRequestHandler(
        agent_executor=ProductFidelityExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=build_agent_card(base_url),
        http_handler=request_handler,
    )
    app = server.build()
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://localhost:\d+",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logging.getLogger(__name__).info("A2A server on %s:%d (card at %s)", host, port, base_url)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
