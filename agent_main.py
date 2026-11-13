# /warthadev-agent/agent_main.py
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from agent_config import AgentConfig
from agent_runtime import AgentRuntime, VisionUnsupportedError

config = AgentConfig.from_env()
runtime = AgentRuntime(config)


class AgentImagePayload(BaseModel):
    mime_type: str = Field(min_length=1, max_length=32)
    data: str = Field(min_length=1)


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = Field(default=None, max_length=200)
    provider_api_key: str = Field(default="", max_length=500)
    user_email: str = Field(min_length=3, max_length=320)
    user_uid: str = Field(min_length=1, max_length=128)
    web_search: bool = False
    search_query: str = Field(default="", max_length=500)
    timezone: str = Field(default="Asia/Jakarta", max_length=64)
    history: list[dict] = Field(default_factory=list, max_length=40)
    images: list[AgentImagePayload] = Field(default_factory=list)


class WorkspacePayload(BaseModel):
    uid: str = Field(min_length=1, max_length=128)
    path: str = Field(default=".", max_length=2048)
    content: str | None = None
    source: str | None = None
    destination: str | None = None
    package: str | None = None
    target_uid: str | None = None
    request_id: str | None = None
    decision: str | None = None
    model: str | None = None
    action: str | None = None
    action_id: str | None = None


class WorkspaceCommandPayload(BaseModel):
    command: str = Field(min_length=1, max_length=20_000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.internal_api_key:
        raise RuntimeError("AGENT_INTERNAL_API_KEY is required")
    yield


app = FastAPI(title="WarthaDev Agent Runtime", version="1.1.0", lifespan=lifespan)


def _verify_internal(internal_key: str | None) -> None:
    if not config.internal_api_key or internal_key != config.internal_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _verify_user_identity(uid: str, header_uid: str | None) -> None:
    if not header_uid or uid != header_uid:
        raise HTTPException(status_code=403, detail="Verified user identity is required")


def _owner_for_request(uid: str, role: str | None, owner_uid: str | None) -> bool:
    return bool(role == "owner" and owner_uid and uid == owner_uid)


@app.get("/")
async def status_page():
    return {"status": "online", "health": "/health"}


@app.get("/health")
async def health():
    return {"status": "healthy", "router": config.router_base_url, "local_model": config.local_model}


@app.head("/")
@app.head("/health")
async def status_head():
    return Response(status_code=200)


@app.get("/v1/models")
async def list_models(x_agent_internal_key: str | None = Header(default=None)):
    _verify_internal(x_agent_internal_key)
    if not config.router_api_key:
        raise HTTPException(status_code=500, detail="Router API key is not configured")
    try:
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.get(f"{config.router_base_url}/models", headers={"Authorization": f"Bearer {config.router_api_key}"})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Router unavailable") from exc
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="Router models request failed")
    return response.json()


@app.post("/v1/agent/run")
async def run_agent(
    payload: AgentRequest,
    x_agent_internal_key: str | None = Header(default=None),
    x_agent_user_uid: str | None = Header(default=None),
    x_agent_user_role: str | None = Header(default=None),
    x_agent_owner_uid: str | None = Header(default=None),
):
    _verify_internal(x_agent_internal_key)
    _verify_user_identity(payload.user_uid, x_agent_user_uid)
    is_owner = _owner_for_request(payload.user_uid, x_agent_user_role, x_agent_owner_uid)
    try:
        result = await runtime.run(
            payload.prompt,
            payload.model,
            payload.provider_api_key,
            payload.user_email,
            payload.history,
            [image.model_dump() for image in payload.images],
            web_search=payload.web_search,
            search_query=payload.search_query,
            timezone_name=payload.timezone,
            user_uid=payload.user_uid,
            user_is_owner=is_owner,
            owner_uid=x_agent_owner_uid or "",
        )
        return {"success": True, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except VisionUnsupportedError as exc:
        raise HTTPException(status_code=422, detail={"code": "vision_unsupported", "message": str(exc)}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def _workspace_call(
    operation: str,
    payload: WorkspacePayload,
    internal_key: str | None,
    header_uid: str | None,
    role: str | None,
    owner_uid: str | None,
) -> dict[str, Any]:
    _verify_internal(internal_key)
    _verify_user_identity(payload.uid, header_uid)
    body = payload.model_dump(exclude={"uid"}, exclude_none=True)
    try:
        return await runtime.workspace_api(
            operation,
            uid=payload.uid,
            user_role=role or "user",
            owner_uid=owner_uid or "",
            payload=body,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/workspace/list")
async def workspace_list(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("list", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/read")
async def workspace_read(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("read", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/write")
async def workspace_write(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("write", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/move")
async def workspace_move(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("move", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/delete")
async def workspace_delete(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("delete", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/command")
async def workspace_command(payload: WorkspaceCommandPayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    _verify_internal(x_agent_internal_key)
    if not x_agent_user_uid:
        raise HTTPException(status_code=403, detail="Verified user identity is required")
    try:
        return await runtime.workspace_api(
            "command",
            uid=x_agent_user_uid,
            user_role=x_agent_user_role or "user",
            owner_uid=x_agent_owner_uid or "",
            payload={"command": payload.command},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/workspace/packages")
async def workspace_packages(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("packages", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/packages/request")
async def workspace_package_request(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("package_request", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/admin/packages")
async def workspace_admin_packages(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("admin_packages", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/admin/packages/review")
async def workspace_admin_review(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("admin_package_review", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/models/local")
async def workspace_local_models(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("local_models", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/admin/models/local")
async def workspace_admin_local_models(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("admin_local_models", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/owner/list")
async def workspace_owner_list(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("owner_list", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/workspace/owner/read")
async def workspace_owner_read(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    return await _workspace_call("owner_read", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


@app.post("/v1/agent/host-actions/confirm")
async def confirm_host_action(payload: WorkspacePayload, x_agent_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    if not _owner_for_request(payload.uid, x_agent_user_role, x_agent_owner_uid):
        raise HTTPException(status_code=403, detail="Owner access required")
    return await _workspace_call("owner_confirm", payload, x_agent_internal_key, x_agent_user_uid, x_agent_user_role, x_agent_owner_uid)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent_main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8788")), workers=1)
