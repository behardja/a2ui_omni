"""product-fidelity — one-command dev launcher.

Starts BOTH local processes with a single command and prints a single URL to
open from your laptop browser:

  1. The agent A2A server  — `python -m app` from agents/product-fidelity/
     (uvicorn + Starlette, :10002), bound to 0.0.0.0 so it's reachable on the VM.
  2. The dev renderer      — Vite (:5173), bound to 0.0.0.0 (`--host`).

Vite proxies the browser's A2A calls (`/a2a/*`) to the agent on :10002, so the
browser only ever talks to ONE origin (:5173). That means no CORS is needed for
this flow, and — on a GCP VM — you open a single external-IP link.

Usage from the repo root:
    python server.py            # no reload
    python server.py --dev      # uvicorn --reload (agent hot-reloads on edits)

Prereqs: `cd dev_client && yarn install` (or npm install) once, and a populated
`.env` (PROJECT_ID, CANDIDATE_BUCKET). Ctrl-C stops both processes.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))          # repo root
AGENT_DIR = os.path.join(ROOT, "agents", "product-fidelity")  # holds the `app` package (for `-m app`)
DEV_CLIENT = os.path.join(ROOT, "dev_client")
AGENT_PORT = int(os.environ.get("PORT", 10002))
VITE_PORT = int(os.environ.get("VITE_PORT", 5173))

BLUE = "\033[38;5;75m"
GREEN = "\033[38;5;114m"
AMBER = "\033[38;5;215m"
GRAY = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _metadata(path):
    """Fetch a value from the GCP metadata server, or None."""
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/" + path,
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().decode().strip()
    except Exception:
        return None


def get_external_ip():
    return _metadata("instance/network-interfaces/0/access-configs/0/external-ip")


def get_workbench_proxy_url():
    """Workbench's authenticated proxy host (…notebooks.googleusercontent.com)."""
    return _metadata("instance/attributes/proxy-url")


def get_vm_identity():
    """Return (instance_name, zone, project_id) for the SSH-forward command."""
    name = _metadata("instance/name")
    zone = _metadata("instance/zone")  # e.g. projects/NNN/zones/us-central1-a
    if zone:
        zone = zone.rsplit("/", 1)[-1]
    project = _metadata("project/project-id")
    return name, zone, project


def stream(proc, prefix, color):
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"{color}{prefix}{RESET} {text}")
    proc.stdout.close()


def _load_dotenv():
    """Load the repo-root .env into os.environ so the spawned agent + Vite inherit it.

    Uses python-dotenv if present; otherwise a minimal KEY=VALUE fallback so the
    launcher has no hard dependency. Existing env vars are NOT overridden.
    """
    env_path = os.path.join(ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    for line in open(env_path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="Enable uvicorn --reload on the agent")
    args = ap.parse_args()

    # Line-buffer stdout so the URL block appears immediately even when the
    # launcher's output is redirected to a file/log (not just a TTY).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    _load_dotenv()  # so `python server.py` picks up .env without manual exports

    if not (os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")):
        print(f"{AMBER}[warn]{RESET} PROJECT_ID not set — GCS/eval tools will fail until you set it (see .env).")
    if not (os.environ.get("CANDIDATE_BUCKET") or os.environ.get("BUCKET_NAME")):
        print(f"{AMBER}[warn]{RESET} CANDIDATE_BUCKET not set — uploads and candidate generation will fail.")
    if not os.path.isdir(os.path.join(DEV_CLIENT, "node_modules")):
        print(f"{AMBER}[warn]{RESET} dev_client/node_modules missing — run `cd dev_client && yarn install` first.")

    agent_proc = None
    vite_proc = None
    try:
        # ── Start the agent A2A server (python -m app, from the agent dir) ──
        agent_cmd = [
            sys.executable, "-m", "app",
            "--host", "0.0.0.0", "--port", str(AGENT_PORT),
        ]
        print(f"{BLUE}[agent]{RESET} Starting on :{AGENT_PORT} (uvicorn + Starlette)...")
        agent_env = dict(os.environ)
        if args.dev:
            agent_env["UVICORN_RELOAD"] = "1"  # informational; --reload wiring lives in __main__ if added
        agent_proc = subprocess.Popen(
            agent_cmd, cwd=AGENT_DIR, env=agent_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        threading.Thread(target=stream, args=(agent_proc, "[agent]", BLUE), daemon=True).start()

        # ── Serve the renderer ──
        # Default: BUILD the client (relative base) and serve it with `vite preview`
        # so it works behind the authenticated Workbench proxy (/proxy/<port>/) —
        # the reliable, no-firewall path. `--dev` runs the HMR dev server instead
        # (root/external-IP access only; breaks behind the proxy's path prefix).
        if args.dev:
            print(f"{GREEN}[vite] {RESET} DEV server on :{VITE_PORT} (proxies /a2a → :{AGENT_PORT})...")
            vite_cmd = ["npx", "vite", "--host", "--port", str(VITE_PORT), "--strictPort"]
        else:
            print(f"{GREEN}[vite] {RESET} Building client (base=./)…")
            build = subprocess.run(
                ["npx", "vite", "build", "--base=./"],
                cwd=DEV_CLIENT, capture_output=True, text=True,
            )
            if build.returncode != 0:
                print(f"{AMBER}[vite] {RESET} build failed:\n{build.stdout}\n{build.stderr}")
                raise SystemExit(1)
            print(f"{GREEN}[vite] {RESET} Serving built app on :{VITE_PORT} (proxies /a2a → :{AGENT_PORT})...")
            vite_cmd = ["npx", "vite", "preview", "--host", "--port", str(VITE_PORT), "--strictPort"]
        vite_proc = subprocess.Popen(
            vite_cmd, cwd=DEV_CLIENT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        threading.Thread(target=stream, args=(vite_proc, "[vite] ", GREEN), daemon=True).start()

        # Give both a moment to bind before printing the URL block
        time.sleep(2.0)

        proxy_host = get_workbench_proxy_url()
        external_ip = get_external_ip()
        print()
        print(f"  {BOLD}product-fidelity — dev renderer ready{RESET}")
        print()
        print(f"  {BOLD}▶ 1. Open one of these in your browser:{RESET}")
        if proxy_host:
            print(f"  {GRAY}Workbench proxy (recommended — same Google login as JupyterLab):{RESET}")
            print(f"  {BOLD}{GREEN}     https://{proxy_host}/proxy/{VITE_PORT}/{RESET}")
            print(f"  {GRAY}     (keep the trailing slash){RESET}")
        if external_ip:
            print(f"  {GRAY}External IP (needs firewall rule for tcp:{VITE_PORT}; on some networks the")
            print(f"  ~20-40s agent POST gets dropped — see findings.md):{RESET}")
            print(f"  {BOLD}     http://{external_ip}:{VITE_PORT}{RESET}")
        if not proxy_host and not external_ip:
            print(f"  {BOLD}{GREEN}     http://localhost:{VITE_PORT}{RESET}  {GRAY}(no proxy/external IP detected){RESET}")
        print()
        print(f"  {BOLD}▶ 2. Click Browse (bucket is pre-filled) — then WAIT ~20–40s.{RESET}")
        print(f"  {GRAY}The stage shows \"Generating UI…\" while the agent runs 2 LLM calls,")
        print(f"  then the image grid — or a clear error message if something failed.{RESET}")
        print()

        # Block until either subprocess exits
        while True:
            if agent_proc.poll() is not None:
                print(f"{BLUE}[agent]{RESET} exited with code {agent_proc.returncode}")
                raise SystemExit(agent_proc.returncode or 1)
            if vite_proc.poll() is not None:
                print(f"{GREEN}[vite] {RESET} exited with code {vite_proc.returncode}")
                raise SystemExit(vite_proc.returncode or 1)
            time.sleep(0.5)

    except (KeyboardInterrupt, SystemExit):
        print(f"\n{GRAY}Shutting down...{RESET}")
        for proc in (agent_proc, vite_proc):
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
