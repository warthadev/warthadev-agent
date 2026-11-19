import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_runtime import AgentRuntime


class _Response:
    status_code = 200
    headers = {"content-type": "application/json"}
    payload = {
        "id": "cloud-response",
        "model": "mimo-v2.6:latest",
        "choices": [{"message": {"role": "assistant", "content": "cloud reply"}}],
        "usage": {},
    }

    def json(self):
        return self.payload


class CloudOnlyRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def _config(self, router_api_key=""):
        return SimpleNamespace(
            router_base_url="https://router.example/v1",
            router_api_key=router_api_key,
            default_model="cloud/default-model",
            request_timeout=5,
            max_tool_calls=1,
        )

    async def test_legacy_local_named_model_is_sent_only_to_cloud_router(self):
        requests = []

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def post(self, url, *, headers, json):
                requests.append({"url": url, "headers": headers, "body": json})
                return _Response()

        runtime = AgentRuntime(self._config())
        inventory = AsyncMock(side_effect=AssertionError("runtime must not inspect local inventory"))
        runtime.workspace_api = inventory
        with patch("agent_runtime.httpx.AsyncClient", FakeClient):
            result = await runtime.run(
                prompt="hello",
                model="mimo-v2.6:latest",
                provider_api_key="account-router-key",
                user_uid="verified-user-uid",
            )

        self.assertEqual(result["content"], "cloud reply")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["url"], "https://router.example/v1/chat/completions")
        self.assertEqual(requests[0]["headers"]["Authorization"], "Bearer account-router-key")
        self.assertEqual(requests[0]["body"]["model"], "mimo-v2.6:latest")
        self.assertNotIn("think", requests[0]["body"])
        inventory.assert_not_awaited()

    async def test_legacy_local_named_model_without_cloud_key_is_rejected(self):
        runtime = AgentRuntime(self._config())
        inventory = AsyncMock(side_effect=AssertionError("runtime must not inspect local inventory"))
        runtime.workspace_api = inventory
        with self.assertRaisesRegex(ValueError, "Provider API key is required"):
            await runtime.run(
                prompt="hello",
                model="mimo-v2.6:latest",
                provider_api_key="",
                user_uid="verified-user-uid",
            )
        inventory.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
