# /warthadev-agent/agent_runtime.py
from typing import Any
import httpx
from agent_config import AgentConfig
class AgentRuntime:
    def __init__(self, config: AgentConfig):
        self.config = config
    def _is_local_model(self, model: str) -> bool:
        return model.strip().lower() == self.config.local_model.strip().lower()
    async def run(self, prompt: str, model: str | None, provider_api_key: str, user_email: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("Prompt is required")
        if len(prompt) > self.config.max_prompt_chars:
            raise ValueError("Prompt is too long")
        selected_model = model or self.config.default_model
        local = self._is_local_model(selected_model)
        if local and user_email.strip().lower() != self.config.local_allowed_email:
            raise PermissionError("Local agent access is restricted")
        if not local and not provider_api_key:
            raise ValueError("Provider API key is required")
        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})
        headers = {"Content-Type": "application/json"}
        if not local:
            headers["Authorization"] = f"Bearer {provider_api_key}"
        base_url = self.config.local_base_url if local else self.config.router_base_url
        payload = {"model": selected_model, "messages": messages, "temperature": 0.2}
        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            response = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"Model request failed: {response.status_code}")
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Model returned no choices")
        message = choices[0].get("message") or {}
        return {"id": data.get("id"), "model": data.get("model"), "content": message.get("content", ""), "usage": data.get("usage") or {}}
