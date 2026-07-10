"""Deploy the product-fidelity agent to Vertex AI Agent Engine and (optionally)
register it on Gemini Enterprise (Discovery Engine), injecting the A2UI v0.8
extension into the agent card (GE's built-in renderer supports A2UI v0.8 only).
Adapted from the in-workspace invoice-auditor sample.

Prereqs (set in .env):
  PROJECT_ID, LOCATION, STORAGE_BUCKET (gs://), plus the model/bucket vars used
  at runtime. Set GEMINI_ENTERPRISE_APP_ID (and optional AGENT_AUTHORIZATION) to
  also register on Gemini Enterprise; if unset, deployment stops after Agent
  Engine so you can register on GE later.
Run:  python deploy.py   (from the repo root)
"""

import json
import os
import sys

from a2a.types import AgentSkill
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request
from google.genai import types
import httpx
import requests
import vertexai
from vertexai.preview.reasoning_engines import A2aAgent
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

# The agent code lives in its own package under agents/<name>/app. Put the agent
# directory on sys.path so `import app.*` resolves, and ship that whole `app`
# package to Agent Engine via extra_packages.
AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "product-fidelity")
sys.path.insert(0, AGENT_DIR)

from app.executor import ProductFidelityExecutor

A2UI_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.8"
A2UI_CATALOG_ID = "https://a2ui.org/specification/v0_8/standard_catalog_definition.json"


def _get_bearer_token():
    try:
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error getting credentials: {e}")
        print("Run 'gcloud auth application-default login' first.")
    return None


def _patch_agent_card_serialization():
    """Work around an incompatibility in google-cloud-aiplatform 1.154.x.

    Agent Engine's spec generator serializes the A2A agent card with protobuf's
    `json_format.MessageToJson`, but a2a-sdk's `AgentCard` is a *pydantic* model
    (no `.DESCRIPTOR`), so it raises `AttributeError`. The platform later rebuilds
    the card via `AgentCard(**json.loads(<the string>))`, so it only needs a JSON
    string. We intercept that one call: emit canonical A2A JSON for pydantic
    models, and defer to the original for real protobuf messages.
    """
    from google.protobuf import json_format
    from pydantic import BaseModel

    original = json_format.MessageToJson
    if getattr(original, "_a2ui_patched", False):
        return
    def patched(message, *args, **kwargs):
        if isinstance(message, BaseModel):
            return message.model_dump_json(by_alias=True, exclude_none=True)
        return original(message, *args, **kwargs)
    patched._a2ui_patched = True
    json_format.MessageToJson = patched


def _register_agent_on_gemini_enterprise(
    project_id, app_id, agent_card, agent_name, display_name, description,
    agent_authorization=None,
):
    api_endpoint = (
        f"https://discoveryengine.googleapis.com/v1alpha/projects/{project_id}/"
        f"locations/global/collections/default_collection/engines/{app_id}/"
        "assistants/default_assistant/agents"
    )
    payload = {
        "name": agent_name,
        "displayName": display_name,
        "description": description,
        "a2aAgentDefinition": {"jsonAgentCard": agent_card},
    }
    if agent_authorization:
        payload["authorization_config"] = {"agent_authorization": agent_authorization}

    bearer_token = _get_bearer_token()
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": project_id,
    }
    response = requests.post(api_endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        print("✓ Agent registered successfully!")
        return response.json()
    print(f"✗ Registration failed: {response.status_code}\n{response.text}")
    response.raise_for_status()


def main():
    load_dotenv()
    project_id = os.environ.get("PROJECT_ID")
    location = os.environ.get("LOCATION")
    storage = os.environ.get("STORAGE_BUCKET")
    app_id = os.environ.get("GEMINI_ENTERPRISE_APP_ID")
    api_endpoint = f"{location}-aiplatform.googleapis.com"

    # Run the deployed container AS this service account. Default: the project's
    # Compute Engine default SA, which we grant roles/iam.serviceAccountTokenCreator
    # (on itself) + storage read so it can mint the V4 signed URLs A2UI Images
    # need. Without this, Agent Engine runs as a Google-managed service agent that
    # lacks signBlob, so image URLs come back empty and GE renders nothing.
    project_number = os.environ.get("PROJECT_NUMBER", "757654702990")
    service_account = os.environ.get(
        "RUNTIME_SERVICE_ACCOUNT",
        f"{project_number}-compute@developer.gserviceaccount.com",
    )

    vertexai.init(project=project_id, location=location,
                  api_endpoint=api_endpoint, staging_bucket=storage)

    skill = AgentSkill(
        id="product-fidelity-eval",
        name="Product Fidelity Evaluation",
        description=(
            "Browse GCS product images, generate a candidate with "
            "gemini-3.1-flash-image, score fidelity with Gecko, render as A2UI."
        ),
        tags=["product", "fidelity", "eval", "image", "chart"],
        examples=[
            "List images in gs://my-bucket/products/",
            "Evaluate gs://my-bucket/products/sneaker.png",
        ],
    )
    agent_card = create_agent_card(
        agent_name="Product Fidelity Agent",
        description="Generate new product images with fidelity scoring. Type 'start' to pick or upload a reference image, add optional creative direction, and get a generated image plus a fidelity report.",
        skills=[skill],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )

    a2ui_agent = A2aAgent(
        agent_card=agent_card,
        agent_executor_builder=ProductFidelityExecutor,
    )
    a2ui_agent.set_up()

    config = {
        "display_name": "product_fidelity_a2ui",
        "description": "A2UI product-fidelity agent (Gecko eval + Nano Banana 2).",
        "agent_framework": "google-adk",
        "staging_bucket": storage,
        "gcs_dir_name": "product_fidelity_a2ui",
        # Pins MUST match the locally-installed versions: the A2aAgent is shipped
        # as a cloudpickle, so the remote runtime has to unpickle it with the same
        # cloudpickle/pydantic/SDK versions or it fails on load.
        "requirements": [
            "google-cloud-aiplatform[agent_engines,adk,evaluation]==1.154.0",
            # google-genai intentionally unpinned: aiplatform[adk] resolves a
            # consistent version (google-adk needs >=2.9); it's a runtime dep,
            # not part of the pickled object graph, so it needn't match local.
            "python-dotenv",
            "uvicorn==0.34.1",
            "a2a-sdk==0.3.26",
            "cloudpickle==3.0.0",
            "pydantic==2.12.5",
            "jsonschema==4.23.0",
            "a2ui-agent-sdk==0.2.4",
            "fastapi==0.134.0",
            "pandas==2.1.4",
            "Pillow==10.2.0",
            "google-cloud-storage==2.18.2",
        ],
        "http_options": {"api_version": "v1beta1"},
        "max_instances": 1,
        # Ship the whole agent package (agent/executor/tools/generate + examples +
        # evaluation_wrapper) as one importable `app` package.
        "extra_packages": [os.path.join(AGENT_DIR, "app")],
        "env_vars": {
            "NUM_WORKERS": "1",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "PROJECT_ID": project_id,
            "LOCATION": location,
            # Model calls must go to the *global* endpoint: gemini-3.x models
            # (orchestrator gemini-3.5-flash) are not served regionally and 404
            # in us-central1. LOCATION above stays regional for Gecko eval.
            "GOOGLE_CLOUD_LOCATION": "global",
            "CANDIDATE_BUCKET": os.environ.get("CANDIDATE_BUCKET", ""),
            "IMAGE_MODEL": os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image"),
            "GENERATION_LOCATION": os.environ.get("GENERATION_LOCATION", "global"),
            "DESCRIPTION_MODEL": os.environ.get("DESCRIPTION_MODEL", "gemini-3.5-flash"),
            "GOOGLE_GENAI_MODEL": os.environ.get("GOOGLE_GENAI_MODEL", "gemini-3.5-flash"),
            "PASSING_THRESHOLD": os.environ.get("PASSING_THRESHOLD", "0.7"),
            "MAX_RETRIES": os.environ.get("MAX_RETRIES", "3"),
        },
    }

    # Deploy via the classic Agent Engine API (the pairing the preview A2aAgent
    # template is built for). The newer genai `client.agent_engines.create` path
    # tries to protobuf-serialize the pydantic a2a AgentCard and crashes on 1.154.
    from vertexai import agent_engines

    _patch_agent_card_serialization()  # 1.154.x pydantic-agent-card workaround

    # Upsert by display name: updating an existing engine in place keeps the
    # SAME resource name, so a Gemini Enterprise registration pointing at it
    # stays valid across code/env changes.
    existing = [
        e for e in agent_engines.list()
        if (getattr(e, "display_name", "") or "") == config["display_name"]
    ]
    common = dict(
        agent_engine=a2ui_agent,
        requirements=config["requirements"],
        display_name=config["display_name"],
        description=config["description"],
        gcs_dir_name=config["gcs_dir_name"],
        extra_packages=config["extra_packages"],
        env_vars=config["env_vars"],
        max_instances=config["max_instances"],
        service_account=service_account,
    )
    print(f"  runtime service account: {service_account}")
    if existing:
        remote_agent = agent_engines.update(
            resource_name=existing[0].resource_name, **common
        )
        remote_engine_resource = remote_agent.resource_name
        print(f"✓ Remote agent UPDATED in place: {remote_engine_resource}")
    else:
        remote_agent = agent_engines.create(**common)
        remote_engine_resource = getattr(remote_agent, "resource_name", None) or (
            remote_agent.api_resource.name
        )
        print(f"✓ Remote agent created: {remote_engine_resource}")

    # Gemini Enterprise registration is optional: only run it when an app ID is
    # configured. Without it, we stop after the Agent Engine deployment so you
    # can register the agent on your GE app later (set GEMINI_ENTERPRISE_APP_ID
    # and re-run, or register via the console using the resource name above).
    if not app_id:
        print(
            "\nℹ GEMINI_ENTERPRISE_APP_ID not set — skipping Gemini Enterprise "
            "registration.\n  Agent Engine deployment complete. To register on "
            "GE later, set GEMINI_ENTERPRISE_APP_ID in .env and re-run, or add "
            "this agent in the GE console.\n"
            f"  Resource name: {remote_engine_resource}"
        )
        return

    # Fetch the deployed agent card and inject the A2UI v0.8 extension so any
    # consumer (including Gemini Enterprise) knows this agent renders A2UI.
    a2a_endpoint = f"https://{api_endpoint}/v1beta1/{remote_engine_resource}/a2a/v1/card"
    headers = {"Authorization": f"Bearer {_get_bearer_token()}", "Content-Type": "application/json"}
    response = httpx.get(a2a_endpoint, headers=headers)
    response.raise_for_status()
    card_json = response.json()

    card_json["capabilities"] = {
        "streaming": False,
        "extensions": [{
            "uri": A2UI_EXTENSION_URI,
            "description": "Ability to render A2UI",
            "required": False,
            "params": {"supportedCatalogIds": [A2UI_CATALOG_ID]},
        }],
    }

    _register_agent_on_gemini_enterprise(
        project_id=project_id,
        app_id=app_id,
        agent_card=json.dumps(card_json),
        agent_name="product_fidelity_a2ui",
        display_name="product_fidelity_a2ui",
        description="A2UI product-fidelity agent with Gecko evaluation.",
        agent_authorization=os.environ.get("AGENT_AUTHORIZATION"),
    )


if __name__ == "__main__":
    main()
