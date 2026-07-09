"""Tools for the A2UI product-fidelity agent.

- list_gcs_images:   browse a GCS prefix, return image URIs + signed URLs
- ingest_uploaded_image_tool: persist a user-uploaded image to GCS -> gs:// URI
- run_fidelity_eval: run the full Gecko eval loop via the vendored
  evaluation_wrapper, using gemini-3.1-flash-image for generation.

Tools return plain JSON strings; the LLM turns that JSON into A2UI UI.
"""

import json
import logging
import os
import re
import uuid
from datetime import timedelta

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")


# --- GCP helpers ---------------------------------------------------------

def _project() -> str:
    project = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("Missing PROJECT_ID / GOOGLE_CLOUD_PROJECT env var.")
    return project


def _default_bucket() -> str:
    bucket = os.environ.get("CANDIDATE_BUCKET") or os.environ.get("BUCKET_NAME") or ""
    return bucket.replace("gs://", "").split("/", 1)[0]


def _split_gs(uri: str):
    path = uri.replace("gs://", "", 1)
    parts = path.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _mime_for(name: str) -> str:
    ext = name.lower().rsplit(".", 1)[-1]
    return "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"


def _signed_url(blob, minutes: int = 60) -> str:
    """Best-effort V4 signed URL so A2UI's Image component can display GCS objects.

    Works directly with service-account key credentials; falls back to IAM
    SignBlob when running under compute/ADC credentials (e.g. a GCP notebook).
    Returns "" if signing is unavailable — the flow still works, the image
    just won't render.
    """
    try:
        return blob.generate_signed_url(
            version="v4", expiration=timedelta(minutes=minutes), method="GET"
        )
    except Exception:
        try:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(google.auth.transport.requests.Request())
            email = getattr(creds, "service_account_email", None) or os.environ.get(
                "SIGNING_SA_EMAIL"
            )
            return blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=minutes),
                method="GET",
                service_account_email=email,
                access_token=creds.token,
            )
        except Exception as e:  # pragma: no cover - environment dependent
            logger.warning("Signed URL unavailable for %s: %s", blob.name, e)
            return ""


def _signed_url_for_uri(gs_uri: str) -> str:
    if not gs_uri.startswith("gs://"):
        return ""
    from google.cloud import storage

    bucket_name, blob_name = _split_gs(gs_uri)
    if not bucket_name or not blob_name:
        return ""
    blob = storage.Client(project=_project()).bucket(bucket_name).blob(blob_name)
    return _signed_url(blob)


# --- Tools ---------------------------------------------------------------

def list_gcs_images(gcs_prefix: str = "", max_results: int = 6) -> str:
    """List product images under a Google Cloud Storage prefix.

    Args:
        gcs_prefix: A gs:// prefix, e.g. "gs://my-bucket/products/". If no
            bucket scheme is given, the default CANDIDATE_BUCKET is used.
        max_results: Maximum number of images to return.

    Returns a JSON string: {"images": [{"name","gs_uri","url"}, ...]}.
    Each image's gs_uri can be passed to run_fidelity_eval as a reference;
    url is a signed URL for display in an A2UI Image component.
    """
    from google.cloud import storage

    gcs_prefix = (gcs_prefix or "").strip()
    if not gcs_prefix:  # "start"/greeting with no path → default browse bucket
        gcs_prefix = os.environ.get(
            "BROWSE_PREFIX", "gs://sandbox-401718-product-fidelity-evals/"
        )
    if gcs_prefix.startswith("gs://"):
        bucket_name, prefix = _split_gs(gcs_prefix)
    else:
        bucket_name, prefix = _default_bucket(), gcs_prefix.lstrip("/")
    display_prefix = gcs_prefix if gcs_prefix.startswith("gs://") else f"gs://{bucket_name}/{prefix}"
    logger.info("🔎 list_gcs_images | bucket=%s prefix=%r max=%d", bucket_name, prefix, max_results)
    if not bucket_name:
        logger.warning("🔎 list_gcs_images | no bucket in prefix and no default set")
        return json.dumps({"error": "No bucket specified and no default bucket set."})

    try:
        client = storage.Client(project=_project())
        images, scanned, signed = [], 0, 0
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            scanned += 1
            if blob.name.endswith("/") or not blob.name.lower().endswith(_IMAGE_EXTS):
                continue
            url = _signed_url(blob)
            if url:
                signed += 1
            images.append(
                {
                    "name": blob.name.rsplit("/", 1)[-1],
                    "gs_uri": f"gs://{bucket_name}/{blob.name}",
                    "url": url,
                }
            )
            if len(images) >= max_results:
                break
    except Exception as e:
        logger.error("🔎 list_gcs_images | GCS error on bucket=%s: %s", bucket_name, e, exc_info=True)
        return json.dumps({"error": f"Could not list gs://{bucket_name}/{prefix}: {e}"})

    logger.info(
        "🔎 list_gcs_images | scanned=%d images=%d signed_urls=%d%s",
        scanned, len(images), signed,
        " ⚠️ signed URLs failed — images may not display" if images and signed == 0 else "",
    )
    if not images:
        logger.warning("🔎 list_gcs_images | no images found under gs://%s/%s", bucket_name, prefix)

    # Stash the interactive browse web-app (WebFrameSrcdoc) for the executor.
    global LAST_WEBFRAME_HTML, LAST_WEBFRAME_HEIGHT, LAST_WEBFRAME_TEXT
    try:
        if images:
            thr = os.environ.get("PASSING_THRESHOLD", "0.7")
            ret = os.environ.get("MAX_RETRIES", "3")
            mdl = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image")
            LAST_WEBFRAME_HTML, LAST_WEBFRAME_HEIGHT = _build_browser_html(
                images, thr, ret, mdl, prefix=display_prefix)
            LAST_WEBFRAME_TEXT = (
                f"Product Generation with Fidelity: {len(images)} reference image"
                f"{'s' if len(images) != 1 else ''} loaded. Pick one, add optional "
                f"creative direction, then Generate & Evaluate."
            )
        else:
            LAST_WEBFRAME_HTML = None
    except Exception as e:  # pragma: no cover - never block browse on cosmetics
        logger.warning("browser HTML build failed: %s", e)
        LAST_WEBFRAME_HTML = None

    return json.dumps({"images": images})


async def ingest_uploaded_image_tool(tool_context=None) -> str:
    """Persist the user's uploaded image to GCS and return its gs:// URI.

    Call this when the user has uploaded/dragged an image into the chat and
    wants it evaluated. The returned gs_uri is then passed to run_fidelity_eval
    as a reference. Returns JSON: {"gs_uri": "..."} or {"error": "..."}.
    """
    logger.info("📤 ingest_uploaded_image | start")
    if not tool_context:
        logger.warning("📤 ingest_uploaded_image | no tool_context")
        return json.dumps({"error": "No tool context available."})
    from google.cloud import storage

    part, filename, source = None, None, None

    # 1. Formal artifacts (staged uploads)
    artifact_keys = await tool_context.list_artifacts()
    logger.info("📤 ingest_uploaded_image | artifacts=%s", list(artifact_keys or []))
    if artifact_keys:
        filename = artifact_keys[-1]
        part = await tool_context.load_artifact(filename)
        source = "artifact"

    # 2. Fallback: scan session history for an inline/file image part
    if not part:
        for event in reversed(tool_context.session.events):
            if event.author == "user" and event.content and event.content.parts:
                for p in event.content.parts:
                    if p.inline_data or p.file_data:
                        part, filename, source = p, "uploaded_image", "session-scan"
                        break
            if part:
                break

    if not part:
        logger.warning("📤 ingest_uploaded_image | no image part found (artifacts or session)")
        return json.dumps({"error": "No uploaded image found. Please attach one."})
    logger.info("📤 ingest_uploaded_image | source=%s filename=%s", source, filename)

    data_bytes, mime_type = None, "image/png"
    if part.inline_data:
        mime_type = part.inline_data.mime_type or mime_type
        data_bytes = part.inline_data.data
    elif part.file_data and part.file_data.file_uri:
        # Already in GCS — just hand back the URI.
        logger.info("📤 ingest_uploaded_image | already in GCS -> %s", part.file_data.file_uri)
        return json.dumps({"gs_uri": part.file_data.file_uri})

    if not data_bytes:
        logger.warning("📤 ingest_uploaded_image | could not read bytes for %s", filename)
        return json.dumps({"error": f"Could not read bytes for {filename}."})

    bucket_name = _default_bucket()
    if not bucket_name:
        logger.warning("📤 ingest_uploaded_image | no CANDIDATE_BUCKET/BUCKET_NAME set")
        return json.dumps({"error": "No CANDIDATE_BUCKET/BUCKET_NAME set for uploads."})
    ext = (mime_type.split("/")[-1] or "png").replace("jpeg", "jpg")
    blob_name = f"uploads/{uuid.uuid4().hex[:8]}_{filename or 'image'}.{ext}"
    logger.info(
        "📤 ingest_uploaded_image | uploading %d bytes (%s) -> gs://%s/%s",
        len(data_bytes), mime_type, bucket_name, blob_name,
    )
    try:
        blob = storage.Client(project=_project()).bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(data_bytes, content_type=mime_type)
    except Exception as e:
        logger.error("📤 ingest_uploaded_image | upload failed to bucket=%s: %s", bucket_name, e, exc_info=True)
        return json.dumps({"error": f"Upload to gs://{bucket_name} failed: {e}"})
    gs_uri = f"gs://{bucket_name}/{blob_name}"
    logger.info("📤 ingest_uploaded_image | ✔ uploaded -> %s", gs_uri)
    return json.dumps({"gs_uri": gs_uri, "url": _signed_url(blob)})


def get_eval_defaults() -> str:
    """Return the current server-side evaluation defaults for pre-filling the
    settings UI. Call this before rendering the "Evaluation settings" panel so
    the sliders/fields show the true configured values.

    Returns a JSON string: {"threshold", "max_retries", "media_type",
    "description_model", "image_model"}.
    """
    try:
        from evaluation_wrapper import EvalConfig
    except ImportError:
        from .evaluation_wrapper import EvalConfig  # type: ignore
    cfg = EvalConfig.from_settings()
    return json.dumps(
        {
            "threshold": cfg.threshold,
            "max_retries": cfg.max_retries,
            "media_type": cfg.media_type,
            "description_model": cfg.description_model,
            "image_model": os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image"),
        }
    )


_IMAGE_MODELS = (
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-image",
    "gemini-3-pro-image",
)


# Server-built HTML surface (report or browser) stashed for the executor to render
# as a WebFrameSrcdoc — custom CSS, built here so the LLM never transcribes big HTML.
LAST_WEBFRAME_HTML = None
LAST_WEBFRAME_HEIGHT = 1000
LAST_WEBFRAME_TEXT = None  # short chat-side (left) summary shown next to the surface
LAST_NATIVE_A2UI = None  # native v0.8 A2UI messages (rendered inline in the chat)

_V08_CATALOG_ID = "https://a2ui.org/specification/v0_8/standard_catalog_definition.json"

_IMAGE_MODEL_CHOICES = [
    ("gemini-3.1-flash-image", "Nano Banana 2 · balanced"),
    ("gemini-3.1-flash-lite-image", "Nano Banana 2 Lite · fast & cheap"),
    ("gemini-3-pro-image", "Nano Banana Pro · highest quality"),
]

# Shared CSS + the tiny postMessage bridge, reused by both HTML surfaces so the
# report and the browser web-app look like one product inside GE.
_A2UI_CSS = """
html,body{height:100%}
*{box-sizing:border-box} body{margin:0;background:#f6f6f8;color:#101622;padding:0;overflow:hidden;
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Inter,Arial,sans-serif}
.scroll{height:100%;overflow-y:auto;-webkit-overflow-scrolling:touch;padding:20px}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:22px;margin:0 0 2px;font-weight:700}
.sub{color:#5b6472;font-size:13px;margin:0 0 18px}
.pathrow{display:flex;gap:8px;margin-bottom:16px}
.gp{flex:1;padding:9px 12px;border:1px solid #e3e6ec;border-radius:10px;font:inherit;font-size:13px;background:#fff}
.btn2{padding:9px 18px;border:1px solid #135bec;background:#fff;color:#135bec;border-radius:10px;font:inherit;font-weight:600;cursor:pointer;white-space:nowrap}
.btn2:hover{background:#eef4ff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin-bottom:20px}
.tile{background:#fff;border:2px solid #e3e6ec;border-radius:14px;padding:10px;cursor:pointer;
transition:border-color .12s,box-shadow .12s,transform .12s;box-shadow:0 1px 3px rgba(16,22,34,.06)}
.tile:hover{border-color:#c3cad6;transform:translateY(-1px)}
.tile.on{border-color:#135bec;box-shadow:0 0 0 3px rgba(19,91,236,.18)}
.tile img{width:100%;height:150px;object-fit:cover;border-radius:8px;background:#f0f1f4;display:block}
.tile .nm{font-size:12px;color:#101622;margin-top:8px;text-align:center;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.fl{display:block;font-size:12px;color:#5b6472;font-weight:600;margin:14px 0 6px}
textarea{width:100%;min-height:64px;border:1px solid #e3e6ec;border-radius:10px;padding:10px 12px;
font:inherit;font-size:13px;resize:vertical;background:#fff}
.row2{display:flex;gap:18px;flex-wrap:wrap;margin-top:6px}
.row2>div{flex:1;min-width:180px}
input[type=range]{width:100%;accent-color:#135bec}
select{width:100%;padding:8px 10px;border:1px solid #e3e6ec;border-radius:10px;font:inherit;font-size:13px;background:#fff}
.btn{margin-top:18px;width:100%;background:#135bec;color:#fff;border:none;border-radius:12px;
padding:13px;font:inherit;font-size:15px;font-weight:600;cursor:pointer}
.btn:hover{background:#0d4ac9} .btn:disabled{opacity:.5;cursor:default}
.hint{color:#5b6472;font-size:12px;text-align:center;margin-top:10px}
.choice{display:flex;gap:16px;margin-top:8px}
.choicebtn{flex:1;background:#fff;border:2px solid #e3e6ec;border-radius:16px;padding:22px 20px;cursor:pointer;text-align:left;font:inherit;transition:border-color .12s,box-shadow .12s}
.choicebtn:hover{border-color:#135bec;box-shadow:0 0 0 3px rgba(19,91,236,.15)}
.choicebtn .ct{font-size:16px;font-weight:700;margin-bottom:6px}
.choicebtn .cd{font-size:13px;color:#5b6472}
.uprow{display:flex;gap:10px;align-items:center;margin-bottom:14px}
.uprow input[type=file]{flex:1;font:inherit;font-size:13px}
.preview{display:none;max-width:100%;max-height:280px;border-radius:12px;border:1px solid #e3e6ec;margin-bottom:12px;object-fit:contain}
/* report */
.head{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;padding:6px 15px;border-radius:999px;color:#fff;font-weight:700;font-size:14px}
.imgs{display:flex;gap:16px;margin:6px 0 22px}
.card{flex:1;background:#fff;border:1px solid #e3e6ec;border-radius:16px;padding:12px;box-shadow:0 1px 3px rgba(16,22,34,.06)}
.cap{font-size:11px;color:#5b6472;text-align:center;margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em;font-weight:600}
.card .big{width:100%;height:340px;object-fit:contain;border-radius:10px;background:#f0f1f4}
.cols{display:flex;gap:16px;flex-wrap:wrap}
.col{flex:1;min-width:240px;background:#fff;border:1px solid #e3e6ec;border-radius:16px;padding:16px 18px}
.col h2{font-size:14px;margin:0 0 10px;font-weight:700}
.pass h2{color:#16a34a}.fail h2{color:#dc2626}
ul{list-style:none;margin:0;padding:0}
li{font-size:13px;padding:7px 0;border-bottom:1px solid #eef0f4;line-height:1.4}
li:last-child{border-bottom:none} li.none{color:#9aa2ad}
.pass li::before{content:'\\2713  ';color:#16a34a;font-weight:800}
.fail li::before{content:'\\2715  ';color:#dc2626;font-weight:800}
.scores{background:#fff;border:1px solid #e3e6ec;border-radius:16px;padding:16px 18px;margin-top:16px}
.scores h2{font-size:14px;margin:0 0 12px;font-weight:700}
.bar{display:flex;align-items:center;gap:12px;margin:9px 0;font-size:13px}
.bar .l{width:82px;color:#5b6472}.t{flex:1;height:9px;background:#eef0f4;border-radius:99px;overflow:hidden}
.f{display:block;height:100%;background:#135bec}.bar .v{width:46px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
"""

_A2UI_BRIDGE_JS = (
    "function a2uiAction(name,data){try{parent.postMessage("
    "{type:'a2ui_action',action:name,data:data},'*');}catch(e){}}"
)


def _esc(s) -> str:
    import html as _h
    return _h.escape(str(s if s is not None else ""))


def _build_browser_html(images, threshold, max_retries, image_model, prefix="") -> str:
    """Interactive browse + config web-app (WebFrameSrcdoc): pick an image, add
    creative direction, tune the run, then post run_eval back to the agent.
    A GCS-path field + Browse button re-lists a different bucket/prefix."""
    tiles = "".join(
        f'<div class="tile" data-uri="{_esc(im.get("gs_uri"))}" onclick="pick(this)">'
        f'<img src="{_esc(im.get("url"))}" alt=""><div class="nm">{_esc(im.get("name"))}</div></div>'
        for im in images
    )
    opts = "".join(
        f'<option value="{mid}"{" selected" if mid == image_model else ""}>{_esc(label)}</option>'
        for mid, label in _IMAGE_MODEL_CHOICES
    )
    height = 760  # fixed viewport; the .scroll container handles overflow
    js = _A2UI_BRIDGE_JS + (
        "var sel=null;"
        "function pick(el){sel=el.getAttribute('data-uri');"
        "document.querySelectorAll('.tile').forEach(function(t){t.classList.remove('on')});"
        "el.classList.add('on');document.getElementById('go').disabled=false;"
        "document.getElementById('hint').textContent='Selected: '+sel;}"
        "function browseNow(){a2uiAction('browse',{prefix:document.getElementById('gp').value});"
        "document.getElementById('hint').textContent='Loading images…';}"
        "function go(){if(!sel)return;"
        "a2uiAction('run_eval',{referenceUri:sel,"
        "userPrompt:document.getElementById('cd').value,"
        "threshold:document.getElementById('thr').value,"
        "maxRetries:document.getElementById('ret').value,"
        "imageModel:document.getElementById('mdl').value});"
        "var b=document.getElementById('go');b.textContent='Generating…';b.disabled=true;"
        "document.getElementById('hint').textContent='Running the fidelity evaluation… the report will appear shortly.';}"
    )
    body = (
        '<div class="wrap">'
        '<h1>Product Generation with Fidelity</h1>'
        '<p class="sub">Pick a reference image, add optional creative direction, then generate &amp; evaluate.</p>'
        f'<div class="pathrow"><input id="gp" class="gp" value="{_esc(prefix)}" placeholder="gs://bucket/prefix/">'
        '<button class="btn2" onclick="browseNow()">Browse</button></div>'
        f'<div class="grid">{tiles}</div>'
        '<label class="fl">Creative direction (optional)</label>'
        '<textarea id="cd" placeholder="e.g. a model wearing the product on a rooftop at sunset"></textarea>'
        '<div class="row2">'
        f'<div><label class="fl">Passing threshold: <b id="thrv">{threshold}</b></label>'
        f'<input id="thr" type="range" min="0" max="1" step="0.05" value="{threshold}" '
        "oninput=\"document.getElementById('thrv').textContent=this.value\"></div>"
        f'<div><label class="fl">Max attempts: <b id="retv">{max_retries}</b></label>'
        f'<input id="ret" type="range" min="1" max="5" step="1" value="{max_retries}" '
        "oninput=\"document.getElementById('retv').textContent=this.value\"></div>"
        f'<div><label class="fl">Image model</label><select id="mdl">{opts}</select></div>'
        '</div>'
        '<button id="go" class="btn" disabled onclick="go()">Generate &amp; Evaluate</button>'
        '<div id="hint" class="hint">Select an image to begin.</div>'
        '</div>'
    )
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + _A2UI_CSS
            + "</style></head><body><div class='scroll'>" + body + "</div><script>"
            + js + "</script></body></html>"), height


def _wrap_html(body: str, js: str = "") -> str:
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + _A2UI_CSS
            + "</style></head><body><div class='scroll'>" + body + "</div>"
            + ("<script>" + js + "</script>" if js else "") + "</body></html>")


def _config_controls_html(threshold, max_retries, image_model) -> str:
    """The shared creative-direction + threshold + attempts + model controls."""
    opts = "".join(
        f'<option value="{mid}"{" selected" if mid == image_model else ""}>{_esc(label)}</option>'
        for mid, label in _IMAGE_MODEL_CHOICES
    )
    return (
        '<label class="fl">Creative direction (optional)</label>'
        '<textarea id="cd" placeholder="e.g. a model wearing the product on a rooftop at sunset"></textarea>'
        '<div class="row2">'
        f'<div><label class="fl">Passing threshold: <b id="thrv">{threshold}</b></label>'
        f'<input id="thr" type="range" min="0" max="1" step="0.05" value="{threshold}" '
        "oninput=\"document.getElementById('thrv').textContent=this.value\"></div>"
        f'<div><label class="fl">Max attempts: <b id="retv">{max_retries}</b></label>'
        f'<input id="ret" type="range" min="1" max="5" step="1" value="{max_retries}" '
        "oninput=\"document.getElementById('retv').textContent=this.value\"></div>"
        f'<div><label class="fl">Image model</label><select id="mdl">{opts}</select></div>'
        '</div>'
    )


def _build_choice_html() -> str:
    """Entry screen: choose Upload-your-own vs Browse-GCS-bucket."""
    body = (
        '<div class="wrap">'
        '<h1>Product Generation with Fidelity</h1>'
        '<p class="sub">How would you like to provide your reference product image?</p>'
        '<div class="choice">'
        '<button class="choicebtn" onclick="a2uiAction(\'choose_gcs\',{})">'
        '<div class="ct">Browse GCS bucket</div>'
        '<div class="cd">Pick a reference image from a Cloud Storage bucket.</div></button>'
        '<button class="choicebtn" onclick="a2uiAction(\'choose_upload\',{})">'
        '<div class="ct">Upload your own</div>'
        '<div class="cd">Select an image file from your device.</div></button>'
        '</div></div>'
    )
    return _wrap_html(body, _A2UI_BRIDGE_JS)


def _build_upload_html(threshold, max_retries, image_model) -> str:
    """Upload panel: choose a local file, add creative direction + config, run."""
    body = (
        '<div class="wrap">'
        '<h1>Upload a reference image</h1>'
        '<p class="sub">Choose an image file, add optional creative direction, then generate &amp; evaluate.</p>'
        '<div class="uprow"><input type="file" id="file" accept="image/*" onchange="prev()"></div>'
        '<img id="pv" class="preview" alt="">'
        + _config_controls_html(threshold, max_retries, image_model) +
        '<button id="go" class="btn" disabled onclick="go()">Generate &amp; Evaluate</button>'
        '<div id="uhint" class="hint">Choose an image file to begin.</div>'
        '</div>'
    )
    js = _A2UI_BRIDGE_JS + (
        "function prev(){var f=document.getElementById('file').files[0];if(!f)return;"
        "var img=new Image();img.onload=function(){var m=1024,"
        "s=Math.min(1,m/Math.max(img.width,img.height));var c=document.createElement('canvas');"
        "c.width=Math.round(img.width*s);c.height=Math.round(img.height*s);"
        "c.getContext('2d').drawImage(img,0,0,c.width,c.height);"
        "window._d=c.toDataURL('image/jpeg',0.85);window._n=f.name;"
        "var p=document.getElementById('pv');p.src=window._d;p.style.display='block';"
        "document.getElementById('go').disabled=false;"
        "document.getElementById('uhint').textContent=f.name+' ready — set options and evaluate.';};"
        "img.src=URL.createObjectURL(f);}"
        "function go(){if(!window._d)return;"
        "a2uiAction('run_eval',{uploadData:window._d,uploadName:window._n||'upload.jpg',"
        "userPrompt:document.getElementById('cd').value,threshold:document.getElementById('thr').value,"
        "maxRetries:document.getElementById('ret').value,imageModel:document.getElementById('mdl').value});"
        "var b=document.getElementById('go');b.textContent='Generating…';b.disabled=true;"
        "document.getElementById('uhint').textContent='Uploading & running the evaluation… the report will appear shortly.';}"
    )
    return _wrap_html(body, js)


def _build_choice_native():
    """Native v0.8 A2UI for the entry choice (rendered inline in the GE chat).

    Two Buttons — "Browse GCS bucket" (choose_gcs) and "Upload your own"
    (choose_upload) — each with a caption. Native so it renders in the chat and
    the buttons post standard A2UI actions (no iframe bridge needed here).
    """
    comps = [
        {"id": "root", "component": {"Card": {"child": "col"}}},
        {"id": "col", "component": {"Column": {"children": {"explicitList": [
            "title", "sub", "choices"]}}}},
        {"id": "title", "component": {"Text": {
            "text": {"literalString": "Product Generation with Fidelity"}, "usageHint": "h2"}}},
        {"id": "sub", "component": {"Text": {
            "text": {"literalString": "How would you like to provide your reference product image?"},
            "usageHint": "body"}}},
        {"id": "choices", "component": {"Row": {
            "children": {"explicitList": ["gcs-col", "up-col"]}, "distribution": "spaceAround"}}},
        # GCS choice
        {"id": "gcs-col", "component": {"Column": {"children": {"explicitList": ["gcs-btn", "gcs-desc"]}}}},
        {"id": "gcs-btn", "component": {"Button": {"child": "gcs-txt", "primary": True,
            "action": {"name": "choose_gcs", "context": []}}}},
        {"id": "gcs-txt", "component": {"Text": {"text": {"literalString": "Browse GCS bucket"}}}},
        {"id": "gcs-desc", "component": {"Text": {
            "text": {"literalString": "Pick a reference image from a Cloud Storage bucket."},
            "usageHint": "caption"}}},
        # Upload choice
        {"id": "up-col", "component": {"Column": {"children": {"explicitList": ["up-btn", "up-desc"]}}}},
        {"id": "up-btn", "component": {"Button": {"child": "up-txt",
            "action": {"name": "choose_upload", "context": []}}}},
        {"id": "up-txt", "component": {"Text": {"text": {"literalString": "Upload your own"}}}},
        {"id": "up-desc", "component": {"Text": {
            "text": {"literalString": "Select an image file from your device."},
            "usageHint": "caption"}}},
    ]
    return [
        {"beginRendering": {"surfaceId": "pf-choice", "catalogId": _V08_CATALOG_ID,
                            "root": "root", "styles": {"primaryColor": "#135bec"}}},
        {"surfaceUpdate": {"surfaceId": "pf-choice", "components": comps}},
    ]


def open_evaluator() -> str:
    """Open the Product Generation with Fidelity entry screen (choose upload vs GCS bucket).

    Renders a NATIVE inline choice in the chat; its buttons open the HTML panels.
    Call this for a greeting, "start", or any vague first message.
    """
    global LAST_NATIVE_A2UI, LAST_WEBFRAME_HTML, LAST_WEBFRAME_TEXT
    LAST_NATIVE_A2UI = _build_choice_native()
    LAST_WEBFRAME_HTML = None
    LAST_WEBFRAME_TEXT = "How would you like to provide your reference image?"
    return json.dumps({"status": "ok"})


def open_upload_panel() -> str:
    """Open the upload panel (file browse + creative direction + settings)."""
    global LAST_WEBFRAME_HTML, LAST_WEBFRAME_HEIGHT, LAST_WEBFRAME_TEXT
    thr = os.environ.get("PASSING_THRESHOLD", "0.7")
    ret = os.environ.get("MAX_RETRIES", "3")
    mdl = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image")
    LAST_WEBFRAME_HTML = _build_upload_html(thr, ret, mdl)
    LAST_WEBFRAME_HEIGHT = 700
    LAST_WEBFRAME_TEXT = "Upload a reference image, add options, then Generate & Evaluate."
    return json.dumps({"status": "ok"})


def ingest_base64(data_url: str, name: str = "upload.jpg") -> str:
    """Decode a data: URL (from the upload panel) to GCS; return its gs:// URI."""
    import base64 as _b64
    from google.cloud import storage
    header, _, b64 = (data_url or "").partition(",")
    mime = "image/png" if "image/png" in header else "image/jpeg"
    ext = "png" if mime == "image/png" else "jpg"
    data = _b64.b64decode(b64)
    bucket = _default_bucket()
    if not bucket:
        raise ValueError("No CANDIDATE_BUCKET/BUCKET_NAME set for uploads.")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name or "upload")[:40]
    blob_name = f"uploads/{uuid.uuid4().hex[:8]}_{safe}.{ext}"
    blob = storage.Client(project=_project()).bucket(bucket).blob(blob_name)
    blob.upload_from_string(data, content_type=mime)
    gs = f"gs://{bucket}/{blob_name}"
    logger.info("📤 ingest_base64 | %d bytes (%s) -> %s", len(data), mime, gs)
    return gs


def _build_report_html(result: dict, ref_uri: str = "", user_prompt: str = "") -> str:
    """Render the generation + fidelity result as a self-contained styled HTML doc.

    ref_uri / user_prompt let the "Regenerate" button re-post run_eval with the
    same reference + settings (so the user can retry generation).
    """
    passed = bool(result.get("passed"))
    score = float(result.get("final_score") or 0)
    attempts = [a for a in (result.get("attempts") or [])]
    n = len(attempts)
    ok_attempts = [a for a in attempts if not a.get("error")]
    best = None
    for a in ok_attempts:
        if a.get("candidate_url") and (best is None or (a.get("score") or 0) >= (best.get("score") or 0)):
            best = a
    src = best or (ok_attempts[-1] if ok_attempts else {})
    ref_url = ((result.get("reference_display") or [{}])[0] or {}).get("url", "")
    cand_url = src.get("candidate_url", "")
    cand_uri = src.get("candidate_uri", "")  # gs:// destination of the generated image
    passing = src.get("passing_verdicts", []) or []
    failing = src.get("failing_verdicts", []) or []
    settings = result.get("settings_used", {}) or {}
    chip_bg = "#16a34a" if passed else "#dc2626"
    status = "PASS" if passed else "FAIL"
    # Regenerate re-posts run_eval with the SAME reference/settings, and folds the
    # current report's failing verdicts into the creative direction so the next
    # generation explicitly addresses what it missed ("report feedback").
    feedback = ""
    if failing:
        feedback = "\n\nAlso correct these issues found in the previous result:\n" + \
            "\n".join(f"- {v}" for v in failing)
    retry = json.dumps({
        "referenceUri": ref_uri, "userPrompt": (user_prompt or "") + feedback,
        "threshold": settings.get("threshold"), "maxRetries": settings.get("max_retries"),
        "imageModel": settings.get("image_model"),
    })
    regen_caption = "using report feedback" if failing else "fresh attempt"
    retry_js = _A2UI_BRIDGE_JS + (
        "function regen(){a2uiAction('run_eval'," + retry + ");"
        "var b=document.getElementById('regen');if(b){b.innerHTML='Regenerating…';b.disabled=true;}}"
        "function newimg(){a2uiAction('choose_gcs',{});}"
    )
    pass_li = "".join(f"<li>{_esc(v)}</li>" for v in passing) or '<li class="none">None</li>'
    fail_li = "".join(f"<li>{_esc(v)}</li>" for v in failing) or '<li class="none">None</li>'
    bars = "".join(
        f'<div class="bar"><span class="l">Attempt {_esc(a.get("attempt"))}</span>'
        f'<span class="t"><span class="f" style="width:{round(float(a.get("score") or 0)*100)}%"></span></span>'
        f'<span class="v">{float(a.get("score") or 0):.2f}</span></div>'
        for a in ok_attempts
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{height:100%}}
*{{box-sizing:border-box}}
body{{margin:0;background:#f6f6f8;color:#101622;padding:0;overflow:hidden;
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Inter,Arial,sans-serif}}
.scroll{{height:100%;overflow-y:auto;-webkit-overflow-scrolling:touch;padding:20px}}
.wrap{{max-width:920px;margin:0 auto}}
.head{{display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
h1{{font-size:22px;margin:0;font-weight:700}}
.chip{{display:inline-flex;align-items:center;padding:6px 15px;border-radius:999px;
color:#fff;font-weight:700;font-size:14px;background:{chip_bg}}}
.sub{{color:#5b6472;font-size:13px;margin:6px 0 20px}}
.imgs{{display:flex;gap:16px;margin-bottom:22px}}
.card{{flex:1;background:#fff;border:1px solid #e3e6ec;border-radius:16px;padding:12px;
box-shadow:0 1px 3px rgba(16,22,34,.06)}}
.cap{{font-size:11px;color:#5b6472;text-align:center;margin-bottom:8px;
text-transform:uppercase;letter-spacing:.06em;font-weight:600}}
.card img{{width:100%;height:340px;object-fit:contain;border-radius:10px;background:#f0f1f4}}
.cols{{margin-top:4px}}
.col{{background:#fff;border:1px solid #e3e6ec;border-radius:16px;padding:14px 18px;margin-bottom:14px}}
.col summary{{font-size:14px;font-weight:700;cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px}}
.col summary::-webkit-details-marker{{display:none}}
.col summary::before{{content:'\\25B8';color:#9aa2ad;transition:transform .15s}}
.col[open] summary::before{{transform:rotate(90deg)}}
.col[open] summary{{margin-bottom:8px}}
.pass summary{{color:#16a34a}} .fail summary{{color:#dc2626}}
ul{{list-style:none;margin:0;padding:0}}
li{{font-size:13px;padding:7px 0;border-bottom:1px solid #eef0f4;line-height:1.4}}
li:last-child{{border-bottom:none}} li.none{{color:#9aa2ad}}
.pass li::before{{content:'✓  ';color:#16a34a;font-weight:800}}
.fail li::before{{content:'✕  ';color:#dc2626;font-weight:800}}
.scores{{background:#fff;border:1px solid #e3e6ec;border-radius:16px;padding:16px 18px;margin-top:16px}}
.scores h2{{font-size:14px;margin:0 0 12px;font-weight:700}}
.bar{{display:flex;align-items:center;gap:12px;margin:9px 0;font-size:13px}}
.bar .l{{width:82px;color:#5b6472}}
.t{{flex:1;height:9px;background:#eef0f4;border-radius:99px;overflow:hidden}}
.f{{display:block;height:100%;background:#135bec}}
.bar .v{{width:46px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}}
.dest{{font-size:12px;color:#5b6472;margin:-14px 0 18px;word-break:break-all}}
.dest b{{color:#101622;font-weight:600}}
.actions{{display:flex;gap:12px;margin-top:18px;padding-bottom:8px}}
.abtn{{padding:10px 18px;border-radius:10px;font:inherit;font-size:14px;font-weight:600;cursor:pointer;
border:1px solid #d5d9e0;background:#fff;color:#135bec;display:inline-flex;flex-direction:column;align-items:center;line-height:1.2}}
.abtn:hover{{background:#eef4ff}}
.abtn.primary{{background:#135bec;color:#fff;border-color:#135bec}}
.abtn.primary:hover{{background:#0d4ac9}}
.abtn:disabled{{opacity:.5;cursor:default}}
.ab-cap{{font-size:11px;font-weight:400;opacity:.85;margin-top:2px}}
</style></head><body><div class="scroll"><div class="wrap">
<div class="head"><h1>Generated Image · Fidelity Report</h1><span class="chip">{status} · {score:.2f}</span></div>
<div class="sub">{n} attempt{'s' if n != 1 else ''} · passing threshold {_esc(settings.get('threshold'))} · model {_esc(settings.get('image_model'))}</div>
<div class="imgs">
<div class="card"><div class="cap">Reference</div><img src="{_esc(ref_url)}" alt="reference"></div>
<div class="card"><div class="cap">Generated image · attempt {_esc(src.get('attempt'))}</div><img src="{_esc(cand_url)}" alt="generated"></div>
</div>
{f'<div class="dest">Generated image saved to <b>{_esc(cand_uri)}</b></div>' if cand_uri else ''}
<div class="cols">
<details class="col pass"><summary>Passing ({len(passing)})</summary><ul>{pass_li}</ul></details>
<details class="col fail"><summary>Failing ({len(failing)})</summary><ul>{fail_li}</ul></details>
</div>
<div class="scores"><h2>Scores by attempt</h2>{bars}</div>
<div class="actions">
<button id="regen" class="abtn primary" onclick="regen()"><span>Regenerate</span><span class="ab-cap">{regen_caption}</span></button>
<button class="abtn" onclick="newimg()"><span>New image</span></button></div>
</div></div><script>{retry_js}</script></body></html>"""
    # Fixed iframe height; the inner .scroll container handles overflow so nothing
    # is clipped (GE's iframe itself doesn't add a scrollbar). Dropdowns collapsed
    # by default — expanding scrolls within the panel.
    return html, 720


def run_fidelity_eval(
    reference_uris,
    sku_id: str = "",
    user_prompt: str = "",
    threshold: float = 0.0,
    max_retries: int = 0,
    image_model: str = "",
) -> str:
    """Run the full product-fidelity evaluation loop on reference image(s).

    Loop: describe reference -> generate candidate (gemini-3.1-flash-image)
    -> Gecko score vs description -> threshold -> refine -> retry.

    Args:
        reference_uris: One or more gs:// URIs of the product reference image(s).
            Accepts a list or a comma/space-separated string.
        sku_id: Optional product identifier (auto-generated if empty).
        user_prompt: Optional creative direction for generation.
        threshold: Optional passing-score override in (0, 1]. 0 = use the
            server default (from the settings widget or .env).
        max_retries: Optional max-attempts override (>= 1). 0 = use the default.
        image_model: Optional image model id for candidate generation. One of
            gemini-3.1-flash-lite-image (fastest/cheapest),
            gemini-3.1-flash-image (balanced default),
            gemini-3-pro-image (highest quality). Invalid/blank = env default.

    Returns a JSON string with keys: sku_id, passed, final_score, attempts[]
    (each with score, candidate_uri, candidate_url, passing/failing verdicts),
    ground_truth_description, reference_display[], settings_used.
    """
    logger.info("🧪 run_fidelity_eval | raw refs=%r sku=%r thr=%s retries=%s", reference_uris, sku_id, threshold, max_retries)
    if isinstance(reference_uris, str):
        reference_uris = [u for u in re.split(r"[,\s]+", reference_uris) if u]
    reference_uris = [u for u in (reference_uris or []) if str(u).startswith("gs://")]
    if not reference_uris:
        logger.warning("🧪 run_fidelity_eval | no valid gs:// reference URIs")
        return json.dumps({"error": "Provide at least one gs:// reference URI."})
    if not sku_id:
        sku_id = "sku-" + uuid.uuid4().hex[:6]

    try:
        from evaluation_wrapper import EvalConfig, EvalPipeline
    except ImportError:
        from .evaluation_wrapper import EvalConfig, EvalPipeline  # type: ignore
    try:
        from generate import generate_candidate_image
    except ImportError:
        from .generate import generate_candidate_image  # type: ignore

    config = EvalConfig.from_settings()
    # Apply per-run overrides from the settings widget (0 = keep default).
    try:
        if threshold and 0 < float(threshold) <= 1:
            config.threshold = float(threshold)
        if max_retries and int(max_retries) >= 1:
            config.max_retries = int(max_retries)
    except (TypeError, ValueError):
        pass
    # Validate the image model against the known set; blank/unknown -> env default.
    env_image_model = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image")
    chosen_image_model = image_model if image_model in _IMAGE_MODELS else env_image_model
    if image_model and image_model not in _IMAGE_MODELS:
        logger.warning("🧪 run_fidelity_eval | unknown image_model %r → using %s", image_model, env_image_model)
    if not config.bucket_name:
        logger.warning("🧪 run_fidelity_eval | no CANDIDATE_BUCKET/BUCKET_NAME set")
        return json.dumps({"error": "CANDIDATE_BUCKET/BUCKET_NAME must be set."})

    logger.info(
        "🧪 run_fidelity_eval | sku=%s refs=%d | threshold=%.2f max_retries=%d | "
        "creative_direction=%r | desc_model=%s@%s image_model=%s@global "
        "gecko_region=%s bucket=%s",
        sku_id, len(reference_uris), config.threshold, config.max_retries,
        user_prompt or "(none)",
        config.description_model, config.description_location,
        chosen_image_model, config.location, config.bucket_name,
    )
    pipeline = EvalPipeline(generate_fn=generate_candidate_image, config=config)
    try:
        result = pipeline.run(
            reference_uris=reference_uris, sku_id=sku_id, user_prompt=user_prompt,
            image_model=chosen_image_model,
        )
    except Exception as e:
        logger.error("🧪 run_fidelity_eval | pipeline failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Evaluation failed: {e}"})

    # Attach signed URLs so the LLM can render images in A2UI.
    result["reference_display"] = [
        {"gs_uri": u, "url": _signed_url_for_uri(u)} for u in reference_uris
    ]
    ref_signed = sum(1 for r in result["reference_display"] if r["url"])
    cand_signed = 0
    for attempt in result.get("attempts", []):
        attempt["candidate_url"] = _signed_url_for_uri(attempt.get("candidate_uri", ""))
        if attempt["candidate_url"]:
            cand_signed += 1

    result["settings_used"] = {
        "threshold": config.threshold,
        "max_retries": config.max_retries,
        "image_model": chosen_image_model,
    }
    attempts = result.get("attempts", [])
    logger.info(
        "🧪 run_fidelity_eval | ✔ sku=%s passed=%s final_score=%s attempts=%d | "
        "signed refs=%d/%d candidates=%d/%d",
        sku_id, result.get("passed"), result.get("final_score"), len(attempts),
        ref_signed, len(result["reference_display"]), cand_signed, len(attempts),
    )
    for a in attempts:
        if a.get("error"):
            logger.warning("🧪   attempt %s: ERROR %s", a.get("attempt"), a.get("error"))
        else:
            logger.info(
                "🧪   attempt %s: score=%s pass=%d fail=%d",
                a.get("attempt"), a.get("score"),
                len(a.get("passing_verdicts", [])), len(a.get("failing_verdicts", [])),
            )

    # Stash a server-built HTML report for the executor to render as a
    # WebFrameSrcdoc surface (nicer than GE's native v0.8 widgets).
    global LAST_WEBFRAME_HTML, LAST_WEBFRAME_HEIGHT, LAST_WEBFRAME_TEXT
    try:
        ref0 = reference_uris[0] if reference_uris else ""
        LAST_WEBFRAME_HTML, LAST_WEBFRAME_HEIGHT = _build_report_html(
            result, ref_uri=ref0, user_prompt=user_prompt)
        _status = "✅ PASS" if result.get("passed") else "❌ FAIL"
        _score = float(result.get("final_score") or 0)
        _n = len(attempts)
        # best (highest-scoring) candidate's GCS destination for the chat summary
        _best = max(
            (a for a in attempts if not a.get("error") and a.get("candidate_uri")),
            key=lambda a: a.get("score") or 0, default={},
        )
        _dest = _best.get("candidate_uri", "")
        LAST_WEBFRAME_TEXT = (
            f"{_status} · Score {_score:.2f} · {_n} attempt{'s' if _n != 1 else ''} "
            f"· model {result.get('settings_used', {}).get('image_model', '')}"
            + (f"\nGenerated image saved to {_dest}" if _dest else "")
        )
    except Exception as e:  # pragma: no cover - never block the eval on cosmetics
        logger.warning("report HTML build failed: %s", e)
        LAST_WEBFRAME_HTML = None

    return json.dumps(result)
