# Dev client — local A2UI renderer

A minimal Vite app that renders the agent's A2UI output locally, so you can see
real widgets before deploying to Gemini Enterprise. It uses the self-contained
`@a2ui/lit` **v0.9** renderer (`A2uiSurface` + `basicCatalog` via
`@a2ui/web_core/v0_9`'s `MessageProcessor`) and talks to the agent over raw A2A
JSON-RPC (`message/send`) via `fetch`. Button actions arrive through the
processor's action handler and are forwarded back as `userAction` data parts.

## Run

### Recommended: one command (best on a GCP VM)

```bash
cd dev_client && yarn install && cd ..   # once, from the repo root
python server.py                         # from the repo root
```

`server.py` starts BOTH the agent (:10002) and this renderer (:5173), binds them
to `0.0.0.0`, and prints a single **external-IP URL** to open from your laptop.
Vite proxies the browser's A2A calls (`/a2a`) to the agent, so there's one origin
and no CORS to worry about. Ctrl-C stops both.

### Alternative: two terminals (localhost only)

```bash
# 1. Start the agent as an A2A server (from the agent dir, agents/product-fidelity/)
python -m app                  # serves on http://localhost:10002

# 2. Start this renderer (in a second terminal, from the repo root)
cd dev_client
yarn install      # or: npm install
yarn dev          # opens http://localhost:5173 (still uses the /a2a proxy)
```

## Use

- **Browse**: enter a `gs://bucket/prefix/` and click *Browse* → thumbnails +
  "Evaluate this" buttons. Clicking one runs the eval loop on that reference.
- **Evaluate**: paste a `gs://` reference URI (+ optional creative direction).
- **Upload & Evaluate**: pick a local image; it's sent to the agent, stored in
  GCS, then evaluated.

## Notes

- Version must match the agent: this client uses A2UI **v0.9** (the agent emits
  v0.9: createSurface/updateComponents/updateDataModel). The v0.9 `A2uiSurface`
  bundles the catalog + theme, so no theme context needs to be provided.
- Signed URLs: images render only if the agent can mint GCS signed URLs (see
  `tools.py::_signed_url`). In a notebook/compute environment you may need a
  service account with `iam.serviceAccounts.signBlob` (set `SIGNING_SA_EMAIL`).
- The button round-trip listens for the `a2ui.action` DOM event; if your
  installed `@a2ui/lit` uses a different event/detail shape, adjust the listener
  in `src/main.js`.
