# /warthadev-agent/agent_main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from agent_config import AgentConfig
from agent_runtime import AgentRuntime

config = AgentConfig.from_env()
runtime = AgentRuntime(config)

class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=24000)
    model: str | None = Field(default=None, max_length=200)
    provider_api_key: str = Field(min_length=1, max_length=500)
    history: list[dict] = Field(default_factory=list, max_length=40)

@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.internal_api_key:
        raise RuntimeError("AGENT_INTERNAL_API_KEY is required")
    yield

app = FastAPI(title="WarthaDev Agent Runtime", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy", "router": config.router_base_url}

@app.post("/v1/agent/run")
async def run_agent(payload: AgentRequest, x_agent_internal_key: str | None = Header(default=None)):
    if x_agent_internal_key != config.internal_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        result = await runtime.run(payload.prompt, payload.model, payload.provider_api_key, payload.history)
        return {"success": True, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8788")), workers=1)
