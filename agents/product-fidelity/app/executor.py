"""A2A executor: bridges the A2A protocol and the ADK agent, and turns the
agent's <a2ui-json> output into A2UI DataParts. Also maps inbound A2UI user
actions (e.g. a "select_reference" button click) back into a query.
"""

import json
import logging
import re
import time
import uuid
from typing import List

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    Message,
    Part,
    Role,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError

try:
    from a2a.utils import new_agent_parts_message
except ImportError:  # pragma: no cover
    def new_agent_parts_message(parts, context_id, task_id):
        return Message(message_id=str(uuid.uuid4()), role=Role.agent, parts=parts)

from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.agent import get_agent

logger = logging.getLogger(__name__)

APP_NAME = "ProductFidelityAgent"
A2UI_MIME_TYPE = "application/json+a2ui"
A2UI_OPEN_TAG = "<a2ui-json>"
A2UI_CLOSE_TAG = "</a2ui-json>"

_A2UI_BLOCK_RE = re.compile(
    f"{re.escape(A2UI_OPEN_TAG)}(.*?){re.escape(A2UI_CLOSE_TAG)}", re.DOTALL
)


def _sanitize_json(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[len("```json"):]
    elif s.startswith("```"):
        s = s[len("```"):]
    if s.endswith("```"):
        s = s[:-len("```")]
    return s.strip()


def _loads_lenient(text: str):
    """Parse LLM-emitted JSON, tolerating the errors LLMs commonly make.

    1. json.loads(strict=False) — allows literal control chars (newlines/tabs)
       inside strings, the single most common failure.
    2. json5 (if installed) — additionally tolerates trailing commas, single
       quotes, and unquoted keys. Soft import so it's not a hard dependency.
    Raises the original json error if both fail.
    """
    try:
        return json.loads(text, strict=False)
    except Exception as primary:
        try:
            import json5  # type: ignore
        except Exception:
            raise primary
        return json5.loads(text)


A2UI_MSG_KEYS = ("beginRendering", "surfaceUpdate", "dataModelUpdate", "deleteSurface")


def _expand_a2ui(payload):
    """Yield individual single-key A2UI (v0.8) messages.

    A v0.8 payload is a bare JSON array of messages, each containing exactly ONE
    of the action keys (beginRendering / surfaceUpdate / dataModelUpdate /
    deleteSurface) and no version field. Accepts a list, a single message, or —
    a common LLM mistake — one object that merges several messages (e.g.
    {"beginRendering": ..., "surfaceUpdate": ...}); the merged form is split
    back into separate messages (preserving canonical order), otherwise the
    renderer rejects it and nothing draws. Stray extra keys next to an action
    key (e.g. a v0.9-habit "version") are dropped — v0.8 messages allow no
    additional properties.
    """
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if isinstance(item, dict):
            present = [k for k in A2UI_MSG_KEYS if k in item]
            if present and len(item) > len(present):
                # Strip anything that isn't an action key (version, name, …).
                for k in present:
                    yield {k: item[k]}
                continue
            if len(present) > 1:
                for k in present:
                    yield {k: item[k]}
                continue
        yield item


def _create_a2ui_part(data: dict) -> Part:
    return Part(root=DataPart(data=data, metadata={"mimeType": A2UI_MIME_TYPE}))


_V08_CATALOG = "https://a2ui.org/specification/v0_8/standard_catalog_definition.json"


def _pop_webframe():
    """Read + clear the HTML surface (+ chat-side summary) a tool stashed."""
    from app import tools as _t
    html = getattr(_t, "LAST_WEBFRAME_HTML", None)
    height = getattr(_t, "LAST_WEBFRAME_HEIGHT", 1000)
    text = getattr(_t, "LAST_WEBFRAME_TEXT", None)
    _t.LAST_WEBFRAME_HTML = None
    _t.LAST_WEBFRAME_TEXT = None
    return html, height, text


def _pop_native():
    """Read + clear native v0.8 A2UI messages a tool stashed (inline in chat)."""
    from app import tools as _t
    msgs = getattr(_t, "LAST_NATIVE_A2UI", None)
    _t.LAST_NATIVE_A2UI = None
    return msgs


def _webframe_parts(html: str, height: int = 1000) -> List[Part]:
    """Render a server-built HTML surface as an A2UI v0.8 WebFrameSrcdoc.

    Uses the CORRECT v0.8 schema — `htmlContent: {literalString}` + numeric `height`
    (NOT `srcdoc`/`sandbox`/`style`, which GE ignores → blank). The sandbox CSP is
    `connect-src 'none'`, which blocks fetch/XHR but NOT <img>/<script>/<style>, so
    signed-URL images load, inline CSS applies, and inline JS can postMessage
    actions back to GE. All data is baked in server-side (no client fetch). Matches
    the working a2ui-seed-agent shape.
    """
    msgs = [
        {"beginRendering": {
            "surfaceId": "fidelity-surface", "catalogId": _V08_CATALOG,
            "root": "root", "styles": {"primaryColor": "#135bec"}}},
        {"surfaceUpdate": {"surfaceId": "fidelity-surface", "components": [
            {"id": "root", "component": {"WebFrameSrcdoc": {
                "htmlContent": {"literalString": html}, "height": int(height)}}}]}},
    ]
    return [_create_a2ui_part(m) for m in msgs]


def parse_response_to_parts(content: str) -> List[Part]:
    matches = list(_A2UI_BLOCK_RE.finditer(content))
    if not matches:
        clean = content.strip()
        return [Part(root=TextPart(text=clean))] if clean else []
    parts: List[Part] = []
    last_end = 0
    for match in matches:
        start, end = match.span()
        before = content[last_end:start].strip()
        if before:
            parts.append(Part(root=TextPart(text=before)))
        try:
            payload = _loads_lenient(_sanitize_json(match.group(1)))
            for msg in _expand_a2ui(payload):
                parts.append(_create_a2ui_part(msg))
        except Exception as e:
            logger.error("Failed to parse A2UI JSON block: %s", e)
        last_end = end
    trailing = content[last_end:].strip()
    if trailing:
        parts.append(Part(root=TextPart(text=trailing)))
    return parts


def _unwrap_value(v):
    """Unwrap a v0.8 value-object to its scalar.

    Action context values may arrive as v0.8 value objects
    ({"literalString": "x"} / {"literalNumber": 3} / {"literalBoolean": true})
    or already resolved to plain scalars (renderers resolve {"path": ...}
    bindings before dispatching the action). Handle both.
    """
    if isinstance(v, dict):
        for k in ("literalString", "literalNumber", "literalBoolean"):
            if k in v:
                return v[k]
    return v


def _query_from_user_action(action: dict) -> str:
    """Map an inbound A2UI userAction into a natural-language query."""
    name = action.get("name")
    ctx = action.get("context", {}) or {}
    if isinstance(ctx, list):  # v0.8 context is a list of {key, value}
        ctx = {c.get("key"): c.get("value") for c in ctx if isinstance(c, dict)}
    ctx = {k: _unwrap_value(v) for k, v in ctx.items()}
    uri = ctx.get("referenceUri") or ctx.get("gs_uri", "")

    # Entry-screen choices.
    if name == "choose_gcs":
        return "Open the GCS image browser by calling list_gcs_images (no arguments)."
    if name == "choose_upload":
        return "Open the upload screen by calling open_upload_panel."

    # Web-app "Browse" button (or GCS-path field): (re)list a bucket/prefix.
    if name == "browse":
        prefix = ctx.get("prefix") or ctx.get("gcs_prefix") or ""
        if prefix:
            return f"List the product images in {prefix} using list_gcs_images."
        return "List the product images using list_gcs_images (default bucket)."

    # Grid "Configure & Evaluate" (select_reference): DON'T run the eval yet — show
    # the settings panel so the user can add creative direction + tune the run.
    # This is how the optional creative-direction field surfaces inside GE (which
    # has no local shell around the agent).
    if name == "select_reference":
        return (
            f"The user picked reference image {uri}. Call get_eval_defaults, then "
            f"render the Evaluation settings panel (see UI rules) pre-filled with "
            f"those defaults and referenceUri='{uri}', so they can optionally enter "
            f"creative direction and adjust the passing threshold and max attempts. "
            f"Do NOT run the evaluation yet — wait for the run_eval action."
        )

    # Settings-panel / web-app "Run evaluation" (run_eval): actually run the eval
    # with the values the user set (threshold / max_retries / creative direction /
    # model). If the upload panel sent a base64 image, ingest it to GCS first.
    if name == "run_eval":
        upload_data = ctx.get("uploadData")
        if upload_data and not uri:
            from app import tools as _t
            try:
                uri = _t.ingest_base64(upload_data, ctx.get("uploadName", "upload.jpg"))
            except Exception as e:
                logger.error("upload ingest failed: %s", e)
                return "Tell the user the image upload failed; ask them to try a smaller image."
        prompt = ctx.get("userPrompt") or ctx.get("creativeDirection", "")
        threshold = ctx.get("threshold")
        retries = ctx.get("maxRetries") or ctx.get("max_retries")
        image_model = ctx.get("imageModel") or ctx.get("image_model")
        bits = [f"Run the fidelity evaluation on {uri} using run_fidelity_eval."]
        if prompt:
            bits.append(f"Creative direction: {prompt}.")
        if threshold not in (None, ""):
            bits.append(f"Use threshold={threshold}.")
        if retries not in (None, ""):
            bits.append(f"Use max_retries={retries}.")
        if image_model not in (None, ""):
            bits.append(f"Use image_model='{image_model}' (exactly this string).")
        bits.append("Then show the results UI.")
        return " ".join(bits)
    return f"The user submitted action '{name}' with context: {json.dumps(ctx)}"


class ProductFidelityExecutor(AgentExecutor):
    def __init__(self):
        self.agent = None
        self.runner = None

    def _init_agent(self):
        if self.agent is None:
            # Force model calls onto the global endpoint. The orchestrator model
            # (gemini-3.5-flash) is global-only, but Agent Engine's A2aAgent
            # template stomps GOOGLE_CLOUD_LOCATION with the engine's region
            # (us-central1) during set_up() — which runs after our module
            # import. _init_agent runs at the first request, after set_up, so
            # setting it here wins; ADK's genai client reads the env at the
            # first LLM call. Gecko eval stays regional via the LOCATION var.
            import os
            os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get(
                "MODEL_LOCATION", "global"
            )
            self.agent = get_agent()
            self.runner = Runner(
                app_name=APP_NAME,
                agent=self.agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )
            logger.info("ProductFidelityExecutor initialized runner")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self._init_agent()

        # 1. Inbound A2UI user action (button click) takes priority.
        runner_parts = []
        action = None
        if context.message and context.message.parts:
            for part in context.message.parts:
                root = getattr(part, "root", part)
                if isinstance(root, DataPart) and "userAction" in (root.data or {}):
                    action = root.data["userAction"]
                    break

        if action is not None:
            runner_parts.append(types.Part(text=_query_from_user_action(action)))
        else:
            # 2. Otherwise convert message parts (handles uploaded files) to ADK parts.
            try:
                from a2a.contrib.tasks.vertex_task_converter import to_stored_part

                if context.message and context.message.parts:
                    for part in context.message.parts:
                        try:
                            runner_parts.append(to_stored_part(part))
                        except Exception as e:
                            logger.warning("Failed to convert part: %s", e)
            except Exception as e:
                logger.warning("vertex_task_converter unavailable: %s", e)

            if not runner_parts:
                runner_parts.append(types.Part(text=context.get_user_input()))

        # Human-readable progress trace (watch these in the server.py terminal).
        t0 = time.time()
        req_preview = (
            _query_from_user_action(action) if action is not None
            else (context.get_user_input() or "(uploaded file)")
        )
        logger.info("─" * 60)
        logger.info("▶ REQUEST: %s", req_preview[:120])
        logger.info("  …running agent (this is the ~20–40s wait: LLM → tool → LLM→A2UI)")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.submit()
        await updater.start_work()

        try:
            session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name,
                user_id="user",
                session_id=context.context_id,
            )
            if session is None:
                session = await self.runner.session_service.create_session(
                    app_name=self.runner.app_name,
                    user_id="user",
                    state={},
                    session_id=context.context_id,
                )

            content = types.Content(role="user", parts=runner_parts)

            async for event in self.runner.run_async(
                session_id=session.id, user_id="user", new_message=content
            ):
                if hasattr(event, "is_final_response") and event.is_final_response():
                    answer_text = ""
                    if event.content and event.content.parts:
                        answer_text = "\n".join(
                            p.text for p in event.content.parts if p.text
                        )
                    final_parts = (
                        parse_response_to_parts(answer_text)
                        if answer_text
                        else [Part(root=TextPart(text="No response generated."))]
                    )
                    # If a tool produced a server-built HTML surface (the interactive
                    # browse web-app or the fidelity report), render it as a
                    # WebFrameSrcdoc (custom CSS + inline JS action bridge) instead of
                    # the LLM's native widgets. Built server-side → no LLM transcription;
                    # correct v0.8 `htmlContent` schema so GE renders it.
                    wf_html, wf_height, wf_text = _pop_webframe()
                    native_msgs = _pop_native()
                    if native_msgs:
                        # Native inline surface (e.g. the entry choice) — renders in
                        # the chat; its buttons post standard A2UI actions.
                        parts = [_create_a2ui_part(m) for m in native_msgs]
                        final_parts = ([Part(root=TextPart(text=wf_text))] if wf_text else []) + parts
                    elif wf_html:
                        summary = wf_text or next(
                            (p.root.text for p in final_parts
                             if isinstance(getattr(p, "root", None), TextPart) and p.root.text),
                            "Here you go.",
                        )
                        final_parts = [Part(root=TextPart(text=summary))] + \
                            _webframe_parts(wf_html, wf_height)
                    n_ui = sum(1 for p in final_parts if isinstance(getattr(p, "root", None), DataPart))
                    logger.info(
                        "✔ DONE in %.0fs — sending %d part(s) to the browser (%d A2UI, %d text)",
                        time.time() - t0, len(final_parts), n_ui, len(final_parts) - n_ui,
                    )
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_parts_message(
                            final_parts, context.context_id, context.task_id
                        ),
                        final=True,
                    )
                    break
        except Exception as e:
            logger.error("Error in ProductFidelityExecutor: %s", e, exc_info=True)
            await updater.update_status(
                TaskState.failed,
                message=Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=f"An error occurred: {e}")],
                ),
            )
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())
