import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

ROUTER_MODELS_URL = "http://9router:20128/v1/models"
ROUTER_CHAT_URL = "http://9router:20128/v1/chat/completions"
AGENT_RUN_URL = "http://warthadev-agent:8788/v1/agent/run"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        follow_redirects=False,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(
    title="Wartha restricted gateway",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

app.router.redirect_slashes = False


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def models(request: Request):
    headers = {}
    authorization = request.headers.get("authorization")
    if authorization is not None:
        headers["Authorization"] = authorization

    try:
        upstream = await request.app.state.http.get(
            ROUTER_MODELS_URL,
            headers=headers,
            params=request.query_params,
        )
    except httpx.HTTPError:
        return JSONResponse(status_code=502, content={"detail": "Model catalog unavailable"})

    if upstream.status_code < 400:
        try:
            payload = upstream.json()
            models = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(models, list) and not any(
                isinstance(model, dict) and model.get("id") == "mimo-v2.6:latest"
                for model in models
            ):
                models.append({"id": "mimo-v2.6:latest", "object": "model", "created": 0, "owned_by": "local"})
                return JSONResponse(content=payload, status_code=upstream.status_code)
        except ValueError:
            pass

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    headers = {}
    for name in ("authorization", "content-type"):
        value = request.headers.get(name)
        if value is not None:
            headers[name.title()] = value

    body = await request.body()
    try:
        upstream = await request.app.state.http.post(
            ROUTER_CHAT_URL,
            headers=headers,
            content=body,
        )
    except httpx.HTTPError:
        return JSONResponse(status_code=502, content={"detail": "Router chat unavailable"})

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@app.post("/v1/agent/run")
async def run_agent(request: Request):
    headers = {}
    internal_key = request.headers.get("x-agent-internal-key")
    if internal_key is not None:
        headers["X-Agent-Internal-Key"] = internal_key
    content_type = request.headers.get("content-type")
    if content_type is not None:
        headers["Content-Type"] = content_type

    body = await request.body()
    try:
        upstream = await request.app.state.http.post(
            AGENT_RUN_URL,
            headers=headers,
            content=body,
        )
    except httpx.HTTPError:
        return JSONResponse(status_code=502, content={"detail": "Agent runtime unavailable"})

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gateway:app", host="0.0.0.0", port=int(os.getenv("PORT", "8789")), workers=1)
