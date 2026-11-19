import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import gateway


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(gateway.app)

    def tearDown(self):
        self.client.close()

    def test_route_set_excludes_agent_and_workspace_paths(self):
        paths = {route.path for route in gateway.app.routes}
        self.assertEqual(paths, {"/health", "/v1/models", "/v1/chat/completions"})

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    def test_chat_rejects_tools_and_tool_messages_without_upstream_call(self):
        upstream = AsyncMock()
        gateway.app.state.http = SimpleNamespace(post=upstream)
        for body in (
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": []},
            {"model": "m", "messages": [{"role": "tool", "content": "hi"}]},
            {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        ):
            with self.subTest(body=body):
                response = self.client.post("/v1/chat/completions", json=body)
                self.assertEqual(response.status_code, 400)
        upstream.assert_not_awaited()

    def test_chat_proxies_text_payload_and_authorization(self):
        response = SimpleNamespace(
            content=b'{"id":"chat-1"}',
            status_code=200,
            headers={"content-type": "application/json"},
        )
        upstream = AsyncMock(return_value=response)
        gateway.app.state.http = SimpleNamespace(post=upstream)
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
        }
        result = self.client.post(
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {"id": "chat-1"})
        upstream.assert_awaited_once()
        call = upstream.await_args
        self.assertEqual(call.args[0], f"{gateway.ROUTER_BASE_URL}/chat/completions")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(call.kwargs["json"], body)

    def test_models_proxies_authorization(self):
        response = SimpleNamespace(
            content=b'{"data":[]}',
            status_code=200,
            headers={"content-type": "application/json"},
        )
        upstream = AsyncMock(return_value=response)
        gateway.app.state.http = SimpleNamespace(get=upstream)
        result = self.client.get("/v1/models", headers={"Authorization": "Bearer token"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {"data": []})
        upstream.assert_awaited_once()
        self.assertEqual(upstream.await_args.kwargs["headers"], {"Authorization": "Bearer token"})


if __name__ == "__main__":
    unittest.main()
