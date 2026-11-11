import tempfile
import unittest
from unittest.mock import patch

import httpx

from agent_config import AgentConfig
from agent_runtime import AgentRuntime


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []
        self.urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, *, headers, json):
        if not url.endswith("/chat/completions"):
            raise AssertionError(f"Unexpected request URL: {url}")
        self.urls.append(url)
        self.payloads.append(dict(json))
        return self.responses.pop(0)


def make_runtime(workspace):
    config = AgentConfig(
        router_base_url="http://router.invalid/v1",
        router_api_key="mock-router-key",
        local_base_url="http://ollama.invalid/v1",
        internal_api_key="mock-internal-key",
        default_model="gemini/gemini-3.6-flash",
        local_model="mimo-v2.6:latest",
        local_allowed_email="user@example.test",
        local_access_all_users=True,
        request_timeout=5,
        max_prompt_chars=1000,
        workspace=workspace,
        max_tool_calls=2,
        tool_timeout=5,
    )
    return AgentRuntime(config)


def ok_response(content="mock-success"):
    return httpx.Response(
        200,
        json={
            "id": "mock-id",
            "model": "mock-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 1},
        },
    )


class AgentRuntimeRequestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.runtime = make_runtime(self.tempdir.name)

    async def test_cloud_retries_once_non_stream_after_502(self):
        fake = FakeClient(
            [httpx.Response(502, text="mock gateway"), ok_response("fallback-ok")]
        )
        with patch("agent_runtime.httpx.AsyncClient", return_value=fake):
            result = await self.runtime.run(
                "mock-only",
                "gemini/gemini-3.6-flash",
                "mock-provider-key",
                "user@example.test",
            )

        self.assertEqual(result["content"], "fallback-ok")
        self.assertEqual([payload["stream"] for payload in fake.payloads], [True, False])
        self.assertEqual(len(fake.payloads), 2)
        self.assertEqual(fake.urls, ["http://router.invalid/v1/chat/completions"] * 2)
        self.assertFalse(self.runtime._is_mimo_model("gemini/gemini-3.6-flash"))
        system_message = fake.payloads[0]["messages"][0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn("directly and honestly", system_message["content"])
        self.assertIn("provide code only when requested", system_message["content"])

    async def test_local_payload_keeps_think_and_token_caps_on_retry(self):
        fake = FakeClient(
            [httpx.Response(502, text="mock gateway"), ok_response("local-fallback-ok")]
        )
        with patch("agent_runtime.httpx.AsyncClient", return_value=fake):
            result = await self.runtime.run(
                "mock-only", "mimo-v2.6:latest", "", "user@example.test"
            )

        self.assertEqual(result["content"], "local-fallback-ok")
        self.assertEqual([payload["stream"] for payload in fake.payloads], [True, True])
        self.assertEqual(len(fake.payloads), 2)
        for payload in fake.payloads:
            self.assertIs(payload["think"], False)
            self.assertEqual(payload["max_tokens"], 128)
            self.assertEqual(payload["num_predict"], 128)
            self.assertEqual(payload["messages"][0]["role"], "user")

    async def test_prefixed_mimo_catalog_id_retries_streaming_through_router(self):
        model_id = "openai-compatible-chat-test/mimo-v2.6:latest"
        fake = FakeClient(
            [httpx.Response(502, text="mock gateway"), ok_response("mimo-retry-ok")]
        )
        with patch("agent_runtime.httpx.AsyncClient", return_value=fake):
            result = await self.runtime.run(
                "mock-only", model_id, "mock-provider-key", "user@example.test"
            )

        self.assertEqual(result["content"], "mimo-retry-ok")
        self.assertEqual([payload["stream"] for payload in fake.payloads], [True, True])
        self.assertEqual(len(fake.payloads), 2)
        self.assertEqual(fake.urls, ["http://router.invalid/v1/chat/completions"] * 2)
        self.assertTrue(all(payload["model"] == model_id for payload in fake.payloads))
        self.assertTrue(all(payload["messages"] == [{"role": "user", "content": "mock-only"}] for payload in fake.payloads))
        self.assertFalse(self.runtime._is_local_model(model_id))

    async def test_persistent_502_for_prefixed_mimo_stops_after_one_retry(self):
        model_id = "openai-compatible-chat-test/mimo-v2.6:latest"
        fake = FakeClient(
            [httpx.Response(502, text="mock gateway")] * 2 + [ok_response("must-not-be-used")]
        )
        with patch("agent_runtime.httpx.AsyncClient", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "502"):
                await self.runtime.run(
                    "mock-only", model_id, "mock-provider-key", "user@example.test"
                )

        self.assertEqual(len(fake.payloads), 2)
        self.assertEqual([payload["stream"] for payload in fake.payloads], [True, True])

    async def test_successful_sse_is_parsed_without_retry(self):
        sse = (
            'data: {"id":"sse-id","model":"sse-model","choices":[{"delta":'
            '{"role":"assistant","content":"stream-ok"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        fake = FakeClient(
            [
                httpx.Response(
                    200,
                    text=sse,
                    headers={"content-type": "text/event-stream"},
                )
            ]
        )
        with patch("agent_runtime.httpx.AsyncClient", return_value=fake):
            result = await self.runtime.run(
                "mock-only",
                "gemini/gemini-3.6-flash",
                "mock-provider-key",
                "user@example.test",
            )

        self.assertEqual(result["content"], "stream-ok")
        self.assertEqual(len(fake.payloads), 1)
        self.assertIs(fake.payloads[0]["stream"], True)

    async def test_non_502_error_is_not_retried(self):
        fake = FakeClient([httpx.Response(500, text="mock failure")])
        with patch("agent_runtime.httpx.AsyncClient", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "500"):
                await self.runtime.run(
                    "mock-only",
                    "gemini/gemini-3.6-flash",
                    "mock-provider-key",
                    "user@example.test",
                )

        self.assertEqual(len(fake.payloads), 1)
        self.assertIs(fake.payloads[0]["stream"], True)


if __name__ == "__main__":
    unittest.main()
