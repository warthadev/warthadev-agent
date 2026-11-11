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
    def test_root_is_a_static_public_status_page(self):
        test_config = replace(agent_main.config, internal_api_key="offline-test-key")
        with patch.object(agent_main, "config", test_config):
            with TestClient(agent_main.app) as client:
                response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("WarthaDev Agent Runtime Online", response.text)
        self.assertIn('href="/health"', response.text)
        self.assertIn("Model lists are account-specific", response.text)
        self.assertIn("shown after sign-in", response.text)
        self.assertNotIn(test_config.router_base_url, response.text)
        self.assertNotIn(test_config.local_model, response.text)
        self.assertNotIn(test_config.internal_api_key, response.text)

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
