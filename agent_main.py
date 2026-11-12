# /warthadev-agent/agent_main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
import httpx
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
    history: list[dict] = Field(default_factory=list, max_length=40)
    images: list[AgentImagePayload] = Field(default_factory=list)
@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.internal_api_key:
        raise RuntimeError("AGENT_INTERNAL_API_KEY is required")
    yield
app = FastAPI(title="WarthaDev Agent Runtime", version="1.0.0", lifespan=lifespan)
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
    if x_agent_internal_key != config.internal_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
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
async def run_agent(payload: AgentRequest, x_agent_internal_key: str | None = Header(default=None)):
    if x_agent_internal_key != config.internal_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        result = await runtime.run(payload.prompt, payload.model, payload.provider_api_key, payload.user_email, payload.history, [image.model_dump() for image in payload.images])
        return {"success": True, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except VisionUnsupportedError as exc:
        raise HTTPException(status_code=422, detail={"code": "vision_unsupported", "message": str(exc)}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8788")), workers=1)
