"""Minimal text-only OpenAI-compatible gateway for 9Router."""

from contextlib import asynccontextmanager
import os
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROUTER_BASE_URL = os.getenv("ROUTER_BASE_URL", "http://9router:20128/v1").rstrip("/")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("ROUTER_REQUEST_TIMEOUT", "180"))


class TextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=256)
    messages: list[TextMessage] = Field(min_length=1, max_length=1000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    n: int | None = Field(default=None, ge=1, le=128)
    stream: bool | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None
    response_format: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        follow_redirects=False,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(
    title="Wartha text chat gateway",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.router.redirect_slashes = False


def _authorization(request: Request) -> str | None:
    """Use the caller's bearer token, falling back to the configured router key."""
    supplied = request.headers.get("authorization")
    if supplied:
        return supplied
    if ROUTER_API_KEY:
        return f"Bearer {ROUTER_API_KEY}"
    return None


def _proxy_headers(request: Request) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    authorization = _authorization(request)
    if authorization:
        headers["Authorization"] = authorization
    return headers


def _error(detail: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/v1/models")
async def models(request: Request) -> Response:
    try:
        upstream = await request.app.state.http.get(
            f"{ROUTER_BASE_URL}/models",
            headers={"Authorization": _authorization(request)} if _authorization(request) else {},
            params=request.query_params,
        )
    except httpx.HTTPError:
        return _error("Model catalog unavailable", 502)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        raw = await request.json()
    except ValueError:
        return _error("Request body must be valid JSON")
    if not isinstance(raw, dict):
        return _error("Request body must be a JSON object")
    if any(field in raw for field in ("tools", "tool_choice", "functions", "function_call")):
        return _error("Tools and function calling are not supported")
    try:
        payload = ChatCompletionRequest.model_validate(raw)
    except ValidationError as exc:
        messages = exc.errors()
        if any(error.get("loc", ())[0:1] == ("messages",) for error in messages):
            return _error("Only text messages with roles system, user, or assistant are supported")
        return _error("Invalid chat completion request")
    if any(message.role not in {"system", "user", "assistant"} for message in payload.messages):
        return _error("Tool and other message roles are not supported")

    try:
        upstream = await request.app.state.http.post(
            f"{ROUTER_BASE_URL}/chat/completions",
            headers=_proxy_headers(request),
            json=payload.model_dump(exclude_none=True),
        )
    except httpx.HTTPError:
        return _error("Router chat unavailable", 502)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gateway:app", host="0.0.0.0", port=int(os.getenv("PORT", "8788")), workers=1)
