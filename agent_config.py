# /warthadev-agent/agent_config.py
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentConfig:
    router_base_url: str
    router_api_key: str
    local_base_url: str
    internal_api_key: str
    default_model: str
    local_model: str
    local_allowed_email: str
    request_timeout: float
    max_prompt_chars: int
    workspace: str
    max_tool_calls: int
    tool_timeout: int

    @classmethod
    def from_env(cls):
        return cls(
            router_base_url=os.getenv("AGENT_ROUTER_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/"),
            router_api_key=os.getenv("AGENT_ROUTER_API_KEY", ""),
            local_base_url=os.getenv("AGENT_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/"),
            internal_api_key=os.getenv("AGENT_INTERNAL_API_KEY", ""),
            default_model=os.getenv("AGENT_DEFAULT_MODEL", "gemini/gemini-3.6-flash"),
            local_model=os.getenv("AGENT_LOCAL_MODEL", "mimo-v2.6:latest").strip("'\""),
            local_allowed_email=os.getenv("AGENT_LOCAL_ALLOWED_EMAIL", "warthadev@gmail.com").strip().lower(),
            request_timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", "180")),
            max_prompt_chars=int(os.getenv("AGENT_MAX_PROMPT_CHARS", "24000")),
            workspace=os.getenv("AGENT_WORKSPACE", "/data/workspace"),
            max_tool_calls=int(os.getenv("AGENT_MAX_TOOL_CALLS", "8")),
            tool_timeout=int(os.getenv("AGENT_TOOL_TIMEOUT", "120")),
        )
