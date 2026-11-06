# /warthadev-agent/agent_config.py
import os
from dataclasses import dataclass
@dataclass(frozen=True)
class AgentConfig:
    router_base_url: str
    local_base_url: str
    internal_api_key: str
    default_model: str
    local_model: str
    local_allowed_email: str
    request_timeout: float
    max_prompt_chars: int
    @classmethod
    def from_env(cls):
        return cls(
            router_base_url=os.getenv("AGENT_ROUTER_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/"),
            local_base_url=os.getenv("AGENT_LOCAL_BASE_URL", "http://127.0.0.1:11436/v1").rstrip("/"),
            internal_api_key=os.getenv("AGENT_INTERNAL_API_KEY", ""),
            default_model=os.getenv("AGENT_DEFAULT_MODEL", "openrouter/free"),
            local_model=os.getenv("AGENT_LOCAL_MODEL", "mimo-v2.6:latest"),
            local_allowed_email=os.getenv("AGENT_LOCAL_ALLOWED_EMAIL", "warthadev@gmail.com").strip().lower(),
            request_timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", "180")),
            max_prompt_chars=int(os.getenv("AGENT_MAX_PROMPT_CHARS", "24000")),
        )
