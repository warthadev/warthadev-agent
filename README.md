# warthadev-agent

A minimal plain-text chat gateway for an OpenAI-compatible 9Router instance. It does not run an agent and has no workspace, host-action, filesystem, process, package, model-control, web-search, image/vision, tool-execution, or function-calling functionality.

## Run locally

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a; source .env; set +a
python gateway.py
```

The service listens on `PORT` (default `8788`) and exposes only:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

`ROUTER_BASE_URL` defaults to `http://9router:20128/v1`. The gateway forwards the caller's `Authorization` header to 9Router, or uses `ROUTER_API_KEY` when no caller header is supplied. Request bodies are never logged.

Chat requests use a strict, standard OpenAI-compatible JSON shape. Messages must have string text content and `system`, `user`, or `assistant` roles. Tools, functions, tool-role messages, multimodal content, and unknown request fields are rejected.

Example:

```bash
curl http://127.0.0.1:8788/v1/chat/completions \
  -H 'Authorization: Bearer your-router-token' \
  -H 'Content-Type: application/json' \
  -d '{"model":"mimo-v2.6:latest","messages":[{"role":"user","content":"Hello"}]}'
```

## Docker

```bash
docker build -t warthadev-chat-gateway .
docker run --env-file .env -p 127.0.0.1:8788:8788 warthadev-chat-gateway
```
