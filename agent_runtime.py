# /warthadev-agent/agent_runtime.py
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
import httpx
from agent_config import AgentConfig

logger = logging.getLogger(__name__)

CLOUD_SYSTEM_INSTRUCTION = (
    "Answer the user's request directly and honestly, in the user's language. "
    "Do not invent facts or claim to have used tools or accessed information you did not. "
    "If a request needs current web information and no web-search tool is available, say so; do not invent results or sources. "
    "Do not include internal notes or reasoning; provide code only when requested."
)

class AgentRuntime:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.workspace = Path(config.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _is_local_model(self, model: str) -> bool:
        return model.strip().lower() == self.config.local_model.strip().lower()

    def _is_mimo_model(self, model: str) -> bool:
        model_id = model.strip().lower()
        local_id = self.config.local_model.strip().lower()
        return bool(local_id) and (model_id == local_id or model_id.endswith(f"/{local_id}"))

    def _safe_content_type(self, response: httpx.Response) -> str:
        content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
        safe_types = {
            "application/json",
            "application/problem+json",
            "application/x-ndjson",
            "application/octet-stream",
            "text/event-stream",
            "text/html",
            "text/plain",
        }
        if not content_type:
            return "missing"
        return content_type if content_type in safe_types else "other"

    async def _post_model_request(
        self,
        client: httpx.AsyncClient,
        request_url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        is_mimo: bool,
        attempt_number: int,
    ) -> httpx.Response:
        started_at = time.monotonic()
        try:
            response = await client.post(request_url, headers=headers, json=payload)
        except httpx.TransportError as exc:
            if is_mimo:
                logger.info(
                    "mimo_request_attempt attempt=%d status=none elapsed_ms=%d "
                    "content_type=missing transport_exception=%s",
                    attempt_number,
                    int((time.monotonic() - started_at) * 1000),
                    type(exc).__name__,
                )
            raise

        if is_mimo:
            logger.info(
                "mimo_request_attempt attempt=%d status=%d elapsed_ms=%d content_type=%s",
                attempt_number,
                response.status_code,
                int((time.monotonic() - started_at) * 1000),
                self._safe_content_type(response),
            )
        return response

    def _path(self, value: str) -> Path:
        path = (self.workspace / value).resolve() if not os.path.isabs(value) else Path(value).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise PermissionError("Path is outside the agent workspace")
        return path

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "list_files", "description": "List files in the agent workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
            {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file inside the agent workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write a UTF-8 text file inside the agent workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "run_command", "description": "Run a non-interactive shell command inside the agent workspace.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
        ]

    def _tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "list_files":
            path = self._path(arguments.get("path", "."))
            if not path.is_dir():
                raise ValueError("Directory not found")
            items = sorted(str(p.relative_to(self.workspace)) for p in path.iterdir() if p.name not in {".git", "node_modules", "dist", "__pycache__"})
            return "\n".join(items[:300]) or "(empty)"
        if name == "read_file":
            path = self._path(arguments["path"])
            if not path.is_file():
                raise ValueError("File not found")
            limit = max(1, min(int(arguments.get("max_chars", 20000)), 100000))
            return path.read_text(encoding="utf-8", errors="replace")[:limit]
        if name == "write_file":
            path = self._path(arguments["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(arguments["content"]), encoding="utf-8")
            return f"Wrote {path.relative_to(self.workspace)}"
        if name == "run_command":
            command = str(arguments["command"]).strip()
            blocked = ["mkfs", "shutdown", "reboot", "curl | sh", "wget | sh"]
            if any(token in command.lower() for token in blocked):
                raise PermissionError("Command blocked by safety policy")
            import subprocess
            result = subprocess.run(command, shell=True, cwd=self.workspace, capture_output=True, text=True, timeout=self.config.tool_timeout)
            output = (result.stdout + result.stderr).strip()
            return f"exit_code={result.returncode}\n{output[:30000]}"
        raise ValueError("Unknown tool")

    def _response_data(self, response: httpx.Response) -> dict[str, Any]:
        try:
            return response.json()
        except json.JSONDecodeError:
            content_type = response.headers.get("content-type", "").lower()
            if "text/event-stream" not in content_type and not response.text.lstrip().startswith("data:"):
                raise

        message: dict[str, Any] = {"role": "assistant", "content": ""}
        tool_calls: dict[int, dict[str, Any]] = {}
        response_id = None
        model = None
        usage: dict[str, Any] = {}
        finish_reason = None

        for line in response.text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            response_id = chunk.get("id", response_id)
            model = chunk.get("model", model)
            usage.update(chunk.get("usage") or {})
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("role"):
                    message["role"] = delta["role"]
                if delta.get("content") is not None:
                    message["content"] += str(delta["content"])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                for call_delta in delta.get("tool_calls") or []:
                    index = int(call_delta.get("index", 0))
                    call = tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    call["id"] = call_delta.get("id", call["id"])
                    call["type"] = call_delta.get("type", call["type"])
                    function = call_delta.get("function") or {}
                    call["function"]["name"] += str(function.get("name") or "")
                    call["function"]["arguments"] += str(function.get("arguments") or "")

        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return {
            "id": response_id,
            "model": model,
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }

    async def run(self, prompt: str, model: str | None, provider_api_key: str, user_email: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("Prompt is required")
        if len(prompt) > self.config.max_prompt_chars:
            raise ValueError("Prompt is too long")
        selected_model = model or self.config.default_model
        local = self._is_local_model(selected_model)
        if local and not self.config.local_access_all_users and user_email.strip().lower() != self.config.local_allowed_email:
            raise PermissionError("Local agent access is restricted")
        if not local:
            provider_api_key = provider_api_key or self.config.router_api_key
            if not provider_api_key:
                raise ValueError("Provider API key is required")
        messages = list(history or [])
        if not self._is_mimo_model(selected_model):
            messages.insert(0, {"role": "system", "content": CLOUD_SYSTEM_INSTRUCTION})
        messages.append({"role": "user", "content": prompt})
        headers = {"Content-Type": "application/json"}
        if not local:
            headers["Authorization"] = f"Bearer {provider_api_key}"
        base_url = self.config.local_base_url if local else self.config.router_base_url
        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            attempt_number = 0
            for _ in range(self.config.max_tool_calls + 1):
                payload = {"model": selected_model, "messages": messages, "temperature": 0.2, "tools": self._tool_specs(), "tool_choice": "auto", "stream": True}
                if local:
                    payload.update({"think": False, "max_tokens": 128, "num_predict": 128})
                request_url = f"{base_url}/chat/completions"
                attempt_number += 1
                is_mimo = self._is_mimo_model(selected_model)
                response = await self._post_model_request(
                    client,
                    request_url,
                    headers,
                    payload,
                    is_mimo=is_mimo,
                    attempt_number=attempt_number,
                )
                if response.status_code == 502:
                    if is_mimo:
                        attempt_number += 1
                        response = await self._post_model_request(
                            client,
                            request_url,
                            headers,
                            payload,
                            is_mimo=True,
                            attempt_number=attempt_number,
                        )
                    else:
                        payload["stream"] = False
                        response = await client.post(request_url, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise RuntimeError(f"Model request failed: {response.status_code} {response.text[:500]}")
                data = self._response_data(response)
                message = (data.get("choices") or [{}])[0].get("message") or {}
                calls = message.get("tool_calls") or []
                if not calls:
                    return {"id": data.get("id"), "model": data.get("model"), "content": message.get("content", ""), "usage": data.get("usage") or {}}
                messages.append(message)
                for call in calls:
                    try:
                        args = json.loads(call.get("function", {}).get("arguments") or "{}")
                        result = self._tool(call.get("function", {}).get("name", ""), args)
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": call.get("function", {}).get("name", ""), "content": result})
        raise RuntimeError("Tool call limit reached")
