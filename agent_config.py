# /warthadev-agent/agent_config.py
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentConfig:
    router_base_url: str
    internal_api_key: str
    default_model: str
    request_timeout: float
    max_prompt_chars: int

    @classmethod
    def from_env(cls):
        return cls(
            router_base_url=os.getenv("AGENT_ROUTER_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/"),
            internal_api_key=os.getenv("AGENT_INTERNAL_API_KEY", ""),
            default_model=os.getenv("AGENT_DEFAULT_MODEL", "openrouter/free"),
            request_timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", "180")),
            max_prompt_chars=int(os.getenv("AGENT_MAX_PROMPT_CHARS", "24000")),
        )
