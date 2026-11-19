# /warthadev-agent/agent_config.py
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentConfig:
    router_base_url: str
    router_api_key: str
    internal_api_key: str
    default_model: str
    request_timeout: float
    workspace: str
    workspace_supervisor_url: str
    workspace_supervisor_key: str
    max_tool_calls: int
    tool_timeout: int

    @classmethod
    def from_env(cls):
        return cls(
            router_base_url=os.getenv("AGENT_ROUTER_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/"),
            router_api_key=os.getenv("AGENT_ROUTER_API_KEY", ""),
            internal_api_key=os.getenv("AGENT_INTERNAL_API_KEY", ""),
            default_model=os.getenv("AGENT_DEFAULT_MODEL", "gemini/gemini-3.6-flash"),
            request_timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", "180")),
            workspace=os.getenv("AGENT_WORKSPACE", "/data/workspace"),
            workspace_supervisor_url=os.getenv("WORKSPACE_SUPERVISOR_URL", "http://wartha-workspace-supervisor:8790").rstrip("/"),
            workspace_supervisor_key=os.getenv("WORKSPACE_SUPERVISOR_INTERNAL_KEY", ""),
            max_tool_calls=int(os.getenv("AGENT_MAX_TOOL_CALLS", "8")),
            tool_timeout=int(os.getenv("AGENT_TOOL_TIMEOUT", "120")),
        )
