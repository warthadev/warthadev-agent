import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient

_import_workspace = tempfile.TemporaryDirectory(prefix="agent-main-test-")
with patch.dict(os.environ, {"AGENT_WORKSPACE": _import_workspace.name}):
    import agent_main


class AgentMainRouteTests(unittest.TestCase):
    def test_root_is_plain_public_status_json(self):
        test_config = replace(agent_main.config, internal_api_key="offline-test-key")
        with patch.object(agent_main, "config", test_config):
            with TestClient(agent_main.app) as client:
                response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        self.assertEqual(response.json(), {"status": "online", "health": "/health"})
        self.assertNotIn(test_config.router_base_url, response.text)
        self.assertNotIn(test_config.local_model, response.text)
        self.assertNotIn(test_config.internal_api_key, response.text)
        self.assertNotIn("model", response.text.lower())

    def test_root_head_returns_200_with_empty_body(self):
        test_config = replace(agent_main.config, internal_api_key="offline-test-key")
        with patch.object(agent_main, "config", test_config):
            with TestClient(agent_main.app) as client:
                response = client.head("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"")
        self.assertEqual(response.headers["content-length"], "0")

    def test_health_head_returns_200_with_empty_body(self):
        test_config = replace(agent_main.config, internal_api_key="offline-test-key")
        with patch.object(agent_main, "config", test_config):
            with TestClient(agent_main.app) as client:
                response = client.head("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"")
        self.assertEqual(response.headers["content-length"], "0")

    def test_health_response_contract_is_unchanged(self):
        test_config = replace(agent_main.config, internal_api_key="offline-test-key")
        with patch.object(agent_main, "config", test_config):
            with TestClient(agent_main.app) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "healthy",
                "router": test_config.router_base_url,
                "local_model": test_config.local_model,
            },
        )


if __name__ == "__main__":
    unittest.main()
