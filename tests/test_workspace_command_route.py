import sys
import inspect
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

try:
    import httpx  # noqa: F401
except ModuleNotFoundError:
    httpx_stub = types.ModuleType("httpx")

    class HTTPError(Exception):
        pass

    class Response:
        pass

    class UnusedAsyncClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("No HTTP client should be created in these tests")

    def unused_get(*args, **kwargs):
        raise AssertionError("No HTTP request should be made in these tests")

    httpx_stub.HTTPError = HTTPError
    httpx_stub.TransportError = type("TransportError", (HTTPError,), {})
    httpx_stub.Response = Response
    httpx_stub.AsyncClient = UnusedAsyncClient
    httpx_stub.Timeout = object
    httpx_stub.get = unused_get
    sys.modules["httpx"] = httpx_stub

from fastapi import HTTPException

import agent_main
import agent_runtime
import gateway
import workspace_supervisor


class WorkspaceCommandRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_route_uses_trusted_headers_not_body_identity(self):
        payload = agent_main.WorkspaceCommandPayload.model_validate(
            {
                "command": "printf safe",
                "uid": "body-attacker-uid",
                "user_role": "owner",
                "owner_uid": "body-attacker-owner",
            }
        )
        self.assertFalse(hasattr(payload, "uid"))
        self.assertFalse(hasattr(payload, "user_role"))
        self.assertFalse(hasattr(payload, "owner_uid"))

        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock) as workspace_api,
        ):
            workspace_api.return_value = {"success": True}
            result = await agent_main.workspace_command(
                payload,
                x_agent_internal_key="test-internal-key",
                x_agent_user_uid="verified-header-uid",
                x_agent_user_role="user",
                x_agent_owner_uid="",
            )

        self.assertEqual(result, {"success": True})
        workspace_api.assert_awaited_once_with(
            "command",
            uid="verified-header-uid",
            user_role="user",
            owner_uid="",
            payload={"command": "printf safe"},
        )

    async def test_body_uid_cannot_replace_missing_verified_uid_header(self):
        payload = agent_main.WorkspaceCommandPayload.model_validate(
            {"command": "printf safe", "uid": "body-attacker-uid"}
        )
        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock) as workspace_api,
        ):
            with self.assertRaises(HTTPException) as raised:
                await agent_main.workspace_command(
                    payload,
                    x_agent_internal_key="test-internal-key",
                    x_agent_user_uid=None,
                    x_agent_user_role="owner",
                    x_agent_owner_uid="body-or-header-owner",
                )

        self.assertEqual(raised.exception.status_code, 403)
        workspace_api.assert_not_awaited()

    async def test_internal_key_is_required_before_command_dispatch(self):
        payload = agent_main.WorkspaceCommandPayload(command="printf safe")
        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock) as workspace_api,
        ):
            with self.assertRaises(HTTPException) as raised:
                await agent_main.workspace_command(
                    payload,
                    x_agent_internal_key="wrong-key",
                    x_agent_user_uid="verified-header-uid",
                    x_agent_user_role="user",
                    x_agent_owner_uid="",
                )

        self.assertEqual(raised.exception.status_code, 401)
        workspace_api.assert_not_awaited()

    async def test_run_and_every_workspace_route_reject_missing_internal_key(self):
        ws = agent_main.WorkspacePayload(uid="verified-test-uid", path=".")
        calls = [
            ("/v1/agent/run", lambda: agent_main.run_agent(agent_main.AgentRequest(prompt="test", user_email="test@example.invalid", user_uid="verified-test-uid"), None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/list", lambda: agent_main.workspace_list(ws, None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/read", lambda: agent_main.workspace_read(ws, None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/write", lambda: agent_main.workspace_write(agent_main.WorkspacePayload(uid="verified-test-uid", path="x", content="x"), None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/move", lambda: agent_main.workspace_move(agent_main.WorkspacePayload(uid="verified-test-uid", source="a", destination="b"), None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/delete", lambda: agent_main.workspace_delete(ws, None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/command", lambda: agent_main.workspace_command(agent_main.WorkspaceCommandPayload(command="printf safe"), None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/packages", lambda: agent_main.workspace_packages(ws, None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/packages/request", lambda: agent_main.workspace_package_request(agent_main.WorkspacePayload(uid="verified-test-uid", package="safe-package"), None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/admin/packages", lambda: agent_main.workspace_admin_packages(ws, None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/admin/packages/review", lambda: agent_main.workspace_admin_review(agent_main.WorkspacePayload(uid="verified-test-uid", target_uid="target-uid", request_id="request-1", decision="approve"), None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/models/local", lambda: agent_main.workspace_local_models(ws, None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/admin/models/local", lambda: agent_main.workspace_admin_local_models(agent_main.WorkspacePayload(uid="verified-test-uid", model="test-model", action="add"), None, "verified-test-uid", "user", "owner-test-uid")),
            ("/v1/workspace/owner/list", lambda: agent_main.workspace_owner_list(ws, None, "verified-test-uid", "owner", "owner-test-uid")),
            ("/v1/workspace/owner/read", lambda: agent_main.workspace_owner_read(ws, None, "verified-test-uid", "owner", "owner-test-uid")),
        ]
        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock) as workspace_api,
        ):
            for route, call in calls:
                with self.subTest(route=route):
                    with self.assertRaises(HTTPException) as raised:
                        await call()
                    self.assertEqual(raised.exception.status_code, 401)
            workspace_api.assert_not_awaited()

    async def test_workspace_body_uid_mismatch_is_rejected_before_runtime_dispatch(self):
        payload = agent_main.WorkspacePayload(uid="forged-body-uid", path=".")
        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock) as workspace_api,
        ):
            with self.assertRaises(HTTPException) as raised:
                await agent_main.workspace_list(
                    payload,
                    x_agent_internal_key="test-internal-key",
                    x_agent_user_uid="trusted-header-uid",
                    x_agent_user_role="user",
                    x_agent_owner_uid="backend-owner-uid",
                )

        self.assertEqual(raised.exception.status_code, 403)
        workspace_api.assert_not_awaited()

    async def test_owner_browse_routes_use_trusted_headers_and_reject_body_uid_spoof(self):
        owner_uid = "verified-owner-uid"
        cases = (
            (agent_main.workspace_owner_list, "owner_list", "/"),
            (agent_main.workspace_owner_read, "owner_read", "/etc/hostname"),
        )
        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock, return_value={"success": True}) as workspace_api,
        ):
            for route, operation, path in cases:
                with self.subTest(operation=operation):
                    payload = agent_main.WorkspacePayload.model_validate({
                        "uid": owner_uid,
                        "path": path,
                        "user_role": "forged-user-role",
                        "owner_uid": "forged-owner-uid",
                    })
                    self.assertFalse(hasattr(payload, "user_role"))
                    self.assertFalse(hasattr(payload, "owner_uid"))
                    await route(
                        payload,
                        x_agent_internal_key="test-internal-key",
                        x_agent_user_uid=owner_uid,
                        x_agent_user_role="owner",
                        x_agent_owner_uid=owner_uid,
                    )
                    workspace_api.assert_awaited_with(
                        operation,
                        uid=owner_uid,
                        user_role="owner",
                        owner_uid=owner_uid,
                        payload={"path": path},
                    )

                spoofed = agent_main.WorkspacePayload(uid="body-spoof-uid", path=path)
                with self.assertRaises(HTTPException) as raised:
                    await route(
                        spoofed,
                        x_agent_internal_key="test-internal-key",
                        x_agent_user_uid=owner_uid,
                        x_agent_user_role="owner",
                        x_agent_owner_uid=owner_uid,
                    )
                self.assertEqual(raised.exception.status_code, 403)

            self.assertEqual(workspace_api.await_count, 2)

    async def test_runtime_sends_uid_header_and_command_only_body_to_supervisor(self):
        sent = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"success": True}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, *, headers, json):
                sent.update(url=url, headers=headers, body=json)
                return FakeResponse()

        config = SimpleNamespace(
            workspace_supervisor_url="http://supervisor:8790",
            workspace_supervisor_key="test-supervisor-key",
            request_timeout=5,
        )
        runtime = agent_runtime.AgentRuntime(config)
        with patch.object(agent_runtime.httpx, "AsyncClient", FakeClient):
            result = await runtime.workspace_api(
                "command",
                uid="verified-header-uid",
                user_role="user",
                owner_uid="",
                payload={"command": "printf safe", "uid": "body-attacker-uid", "role": "owner"},
            )

        self.assertEqual(result, {"success": True})
        self.assertEqual(sent["url"], "http://supervisor:8790/internal/v1/workspace/command")
        self.assertEqual(sent["body"], {"command": "printf safe"})
        headers = {name.lower(): value for name, value in sent["headers"].items()}
        self.assertEqual(headers["x-agent-user-uid"], "verified-header-uid")
        self.assertEqual(headers["x-agent-user-role"], "user")
        self.assertEqual(headers["x-agent-owner-uid"], "")
        self.assertEqual(headers["x-workspace-internal-key"], "test-supervisor-key")

    async def test_runtime_sends_trusted_uid_header_and_data_only_body_for_workspace_file(self):
        sent = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"success": True}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, *, headers, json):
                sent.update(url=url, headers=headers, body=json)
                return FakeResponse()

        runtime = agent_runtime.AgentRuntime(SimpleNamespace(
            workspace_supervisor_url="http://supervisor:8790",
            workspace_supervisor_key="test-supervisor-key",
            request_timeout=5,
        ))
        with patch.object(agent_runtime.httpx, "AsyncClient", FakeClient):
            await runtime.workspace_api(
                "write",
                uid="verified-user-uid",
                user_role="user",
                owner_uid="backend-owner-uid",
                payload={"path": "notes.txt", "content": "safe", "uid": "body-spoof", "owner_uid": "body-spoof", "role": "owner"},
            )

        self.assertEqual(sent["body"], {"path": "notes.txt", "content": "safe"})
        headers = {name.lower(): value for name, value in sent["headers"].items()}
        self.assertEqual(headers["x-agent-user-uid"], "verified-user-uid")
        self.assertEqual(headers["x-agent-owner-uid"], "backend-owner-uid")

    async def test_runtime_fetches_shared_local_models_with_backend_owner_uid_header(self):
        sent = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"models": ["mimo-v2.6:latest"]}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, *, headers, json):
                sent.update(url=url, headers=headers, body=json)
                return FakeResponse()

        runtime = agent_runtime.AgentRuntime(SimpleNamespace(
            workspace_supervisor_url="http://supervisor:8790",
            workspace_supervisor_key="test-supervisor-key",
            request_timeout=5,
        ))
        with patch.object(agent_runtime.httpx, "AsyncClient", FakeClient):
            result = await runtime.workspace_api(
                "local_models",
                uid="regular-user-uid",
                user_role="user",
                owner_uid="backend-verified-owner-uid",
                payload={"uid": "body-spoof", "owner_uid": "body-spoof"},
            )

        self.assertEqual(result["models"], ["mimo-v2.6:latest"])
        self.assertEqual(sent["url"], "http://supervisor:8790/internal/v1/models/local")
        self.assertEqual(sent["body"], {})
        headers = {name.lower(): value for name, value in sent["headers"].items()}
        self.assertEqual(headers["x-agent-user-uid"], "regular-user-uid")
        self.assertEqual(headers["x-agent-owner-uid"], "backend-verified-owner-uid")

    async def test_runtime_routes_owner_browse_to_supervisor_with_trusted_identity(self):
        sent = []

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"path": "/", "entries": []}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, *, headers, json):
                sent.append({"url": url, "headers": headers, "body": json})
                return FakeResponse()

        runtime = agent_runtime.AgentRuntime(SimpleNamespace(
            workspace_supervisor_url="http://supervisor:8790",
            workspace_supervisor_key="test-supervisor-key",
            request_timeout=5,
        ))
        with patch.object(agent_runtime.httpx, "AsyncClient", FakeClient):
            for operation, path in (("owner_list", "/internal/v1/owner/list"), ("owner_read", "/internal/v1/owner/read")):
                await runtime.workspace_api(
                    operation,
                    uid="verified-owner-uid",
                    user_role="owner",
                    owner_uid="verified-owner-uid",
                    payload={"path": "/", "uid": "body-spoof", "role": "user", "owner_uid": "body-spoof"},
                )
                request = sent[-1]
                self.assertEqual(request["url"], f"http://supervisor:8790{path}")
                self.assertEqual(request["body"], {"path": "/"})
                headers = {name.lower(): value for name, value in request["headers"].items()}
                self.assertEqual(headers["x-agent-user-uid"], "verified-owner-uid")
                self.assertEqual(headers["x-agent-user-role"], "owner")
                self.assertEqual(headers["x-agent-owner-uid"], "verified-owner-uid")

    async def test_gateway_allowlists_and_routes_command(self):
        self.assertIn("command", gateway.WORKSPACE_ENDPOINTS)
        request = object()
        with patch.object(gateway, "_forward_agent_request", new_callable=AsyncMock) as forward:
            forward.return_value = "forwarded"
            result = await gateway.workspace_operation("command", request)

        self.assertEqual(result, "forwarded")
        forward.assert_awaited_once_with(request, "/v1/workspace/command")

    async def test_gateway_allows_only_owner_list_and_read_browse_routes(self):
        self.assertIn("owner/list", gateway.WORKSPACE_ENDPOINTS)
        self.assertIn("owner/read", gateway.WORKSPACE_ENDPOINTS)
        self.assertNotIn("owner/write", gateway.WORKSPACE_ENDPOINTS)
        self.assertNotIn("owner/delete", gateway.WORKSPACE_ENDPOINTS)
        request = object()
        with patch.object(gateway, "_forward_agent_request", new_callable=AsyncMock, return_value="forwarded") as forward:
            for operation in ("owner/list", "owner/read"):
                self.assertEqual(await gateway.workspace_operation(operation, request), "forwarded")
                forward.assert_awaited_with(request, f"/v1/workspace/{operation}")

    async def test_gateway_forwards_trusted_identity_headers(self):
        class FakeUpstream:
            status_code = 200
            content = b'{"success":true}'
            headers = {"content-type": "application/json"}

        class FakeHttp:
            def __init__(self):
                self.request = None

            async def post(self, url, *, headers, content):
                self.request = {"url": url, "headers": headers, "content": content}
                return FakeUpstream()

        fake_http = FakeHttp()

        class FakeRequest:
            def __init__(self):
                self.headers = {
                    "x-agent-internal-key": "test-agent-key",
                    "x-agent-user-uid": "verified-header-uid",
                    "x-agent-user-role": "user",
                    "x-agent-owner-uid": "",
                    "content-type": "application/json",
                }
                self.app = SimpleNamespace(state=SimpleNamespace(http=fake_http))

            @staticmethod
            async def body():
                return b'{"command":"printf safe"}'

        response = await gateway._forward_agent_request(FakeRequest(), "/v1/workspace/command")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_http.request["url"], "http://warthadev-agent:8788/v1/workspace/command")
        self.assertEqual(fake_http.request["content"], b'{"command":"printf safe"}')
        headers = {name.lower(): value for name, value in fake_http.request["headers"].items()}
        self.assertEqual(headers["x-agent-user-uid"], "verified-header-uid")
        self.assertEqual(headers["x-agent-user-role"], "user")
        self.assertEqual(headers["x-agent-internal-key"], "test-agent-key")

    def test_agent_registers_post_command_route(self):
        route = next(
            route for route in agent_main.app.routes
            if getattr(route, "path", None) == "/v1/workspace/command"
        )
        self.assertIn("POST", route.methods)

    async def test_owner_browse_route_maps_non_owner_permission_error_to_403(self):
        payload = agent_main.WorkspacePayload(uid="regular-user-uid", path="/")
        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock, side_effect=PermissionError("Owner access required")) as workspace_api,
        ):
            with self.assertRaises(HTTPException) as raised:
                await agent_main.workspace_owner_list(
                    payload,
                    x_agent_internal_key="test-internal-key",
                    x_agent_user_uid="regular-user-uid",
                    x_agent_user_role="user",
                    x_agent_owner_uid="verified-owner-uid",
                )

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(raised.exception.detail, "Owner access required")
        workspace_api.assert_awaited_once()

    async def test_owner_browse_route_maps_supervisor_http_errors(self):
        class FakeResponse:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self.body = body

            def json(self):
                return self.body

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, *, headers, json):
                raise AssertionError("AsyncClient.post should be mocked")

        cases = (
            (403, {"detail": "Owner access required"}, 403, "Owner access required"),
            (503, {"detail": "Supervisor temporarily unavailable"}, 502, "Supervisor temporarily unavailable"),
        )
        config = SimpleNamespace(
            workspace_supervisor_url="http://supervisor:8790",
            workspace_supervisor_key="test-supervisor-key",
            request_timeout=5,
        )

        for upstream_status, body, expected_status, expected_detail in cases:
            with self.subTest(upstream_status=upstream_status):
                runtime = agent_runtime.AgentRuntime(config)
                payload = agent_main.WorkspacePayload(uid="verified-owner-uid", path="/")
                with (
                    patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
                    patch.object(agent_main, "runtime", runtime),
                    patch.object(agent_runtime.httpx, "AsyncClient", FakeClient),
                    patch.object(
                        FakeClient,
                        "post",
                        new_callable=AsyncMock,
                        return_value=FakeResponse(upstream_status, body),
                    ) as post,
                ):
                    with self.assertRaises(HTTPException) as raised:
                        await agent_main.workspace_owner_list(
                            payload,
                            x_agent_internal_key="test-internal-key",
                            x_agent_user_uid="verified-owner-uid",
                            x_agent_user_role="owner",
                            x_agent_owner_uid="verified-owner-uid",
                        )

                self.assertEqual(raised.exception.status_code, expected_status)
                self.assertEqual(raised.exception.detail, expected_detail)
                post.assert_awaited_once()

    async def test_workspace_route_maps_invalid_and_runtime_errors(self):
        payload = agent_main.WorkspacePayload(uid="verified-owner-uid", path="/")
        cases = (
            (ValueError("Invalid host path"), 400),
            (RuntimeError("Supervisor unavailable"), 502),
        )
        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock) as workspace_api,
        ):
            for error, status_code in cases:
                with self.subTest(status_code=status_code):
                    workspace_api.side_effect = error
                    with self.assertRaises(HTTPException) as raised:
                        await agent_main.workspace_owner_list(
                            payload,
                            x_agent_internal_key="test-internal-key",
                            x_agent_user_uid="verified-owner-uid",
                            x_agent_user_role="owner",
                            x_agent_owner_uid="verified-owner-uid",
                        )

                    self.assertEqual(raised.exception.status_code, status_code)
                    self.assertEqual(raised.exception.detail, str(error))

    async def test_workspace_command_maps_supervisor_permission_error_to_403(self):
        payload = agent_main.WorkspaceCommandPayload(command="printf safe")
        with (
            patch.object(agent_main, "config", SimpleNamespace(internal_api_key="test-internal-key")),
            patch.object(agent_main.runtime, "workspace_api", new_callable=AsyncMock, side_effect=PermissionError("Owner access required")) as workspace_api,
        ):
            with self.assertRaises(HTTPException) as raised:
                await agent_main.workspace_command(
                    payload,
                    x_agent_internal_key="test-internal-key",
                    x_agent_user_uid="regular-user-uid",
                    x_agent_user_role="user",
                    x_agent_owner_uid="verified-owner-uid",
                )

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(raised.exception.detail, "Owner access required")
        workspace_api.assert_awaited_once()


class WorkspaceSupervisorCommandIdentityTests(unittest.TestCase):
    def test_supervisor_uses_verified_uid_header_and_ignores_body_identity(self):
        payload = workspace_supervisor.WorkspaceCommandRequest.model_validate(
            {
                "command": "printf safe",
                "uid": "body-attacker-uid",
                "role": "owner",
            }
        )
        self.assertFalse(hasattr(payload, "uid"))
        self.assertFalse(hasattr(payload, "role"))

        with (
            patch.object(workspace_supervisor, "SUPERVISOR_KEY", "test-supervisor-key"),
            patch.object(workspace_supervisor, "_run_workspace_command", return_value={"success": True}) as run_command,
        ):
            result = workspace_supervisor.workspace_command(
                payload,
                x_workspace_internal_key="test-supervisor-key",
                x_agent_user_uid="verified-header-uid",
            )

        self.assertEqual(result, {"success": True})
        run_command.assert_called_once_with("verified-header-uid", "printf safe")

    def test_supervisor_rejects_missing_verified_uid_even_if_body_has_uid(self):
        payload = workspace_supervisor.WorkspaceCommandRequest.model_validate(
            {"command": "printf safe", "uid": "body-attacker-uid"}
        )
        with (
            patch.object(workspace_supervisor, "SUPERVISOR_KEY", "test-supervisor-key"),
            patch.object(workspace_supervisor, "_run_workspace_command") as run_command,
        ):
            with self.assertRaises(HTTPException) as raised:
                workspace_supervisor.workspace_command(
                    payload,
                    x_workspace_internal_key="test-supervisor-key",
                    x_agent_user_uid=None,
                )

        self.assertEqual(raised.exception.status_code, 403)
        run_command.assert_not_called()

    def test_supervisor_authenticates_internal_key_before_command_dispatch(self):
        payload = workspace_supervisor.WorkspaceCommandRequest(command="printf safe")
        with (
            patch.object(workspace_supervisor, "SUPERVISOR_KEY", "test-supervisor-key"),
            patch.object(workspace_supervisor, "_run_workspace_command") as run_command,
        ):
            with self.assertRaises(HTTPException) as raised:
                workspace_supervisor.workspace_command(
                    payload,
                    x_workspace_internal_key="wrong-key",
                    x_agent_user_uid="verified-header-uid",
                )

        self.assertEqual(raised.exception.status_code, 401)
        run_command.assert_not_called()

    def test_workspace_routes_use_verified_uid_header_and_ignore_body_uid(self):
        payload = workspace_supervisor.WorkspaceRequest.model_validate({"uid": "body-spoof-uid", "path": "."})
        self.assertFalse(hasattr(payload, "uid"))
        with (
            patch.object(workspace_supervisor, "SUPERVISOR_KEY", "test-supervisor-key"),
            patch.object(workspace_supervisor, "_list_workspace", return_value={"entries": []}) as list_workspace,
        ):
            result = workspace_supervisor.workspace_list(
                payload,
                x_workspace_internal_key="test-supervisor-key",
                x_agent_user_uid="trusted-header-uid",
            )

        self.assertEqual(result, {"entries": []})
        list_workspace.assert_called_once_with("trusted-header-uid", ".")

    def test_workspace_route_rejects_body_uid_when_trusted_uid_header_missing(self):
        payload = workspace_supervisor.WorkspaceRequest.model_validate({"uid": "body-spoof-uid", "path": "."})
        with (
            patch.object(workspace_supervisor, "SUPERVISOR_KEY", "test-supervisor-key"),
            patch.object(workspace_supervisor, "_list_workspace") as list_workspace,
        ):
            with self.assertRaises(HTTPException) as raised:
                workspace_supervisor.workspace_list(
                    payload,
                    x_workspace_internal_key="test-supervisor-key",
                    x_agent_user_uid=None,
                )

        self.assertEqual(raised.exception.status_code, 403)
        list_workspace.assert_not_called()

    def test_owner_body_uid_cannot_authorize_without_trusted_user_uid_header(self):
        payload = workspace_supervisor.HostReadRequest.model_validate({"uid": "body-owner-uid", "path": "/etc/passwd"})
        self.assertFalse(hasattr(payload, "uid"))
        with (
            patch.object(workspace_supervisor, "SUPERVISOR_KEY", "test-supervisor-key"),
            patch.object(workspace_supervisor, "_list_host") as list_host,
        ):
            with self.assertRaises(HTTPException) as raised:
                workspace_supervisor.owner_list(
                    payload,
                    x_workspace_internal_key="test-supervisor-key",
                    x_agent_user_uid=None,
                    x_agent_user_role="owner",
                    x_agent_owner_uid="body-owner-uid",
                )

        self.assertEqual(raised.exception.status_code, 403)
        list_host.assert_not_called()

    def test_shared_local_model_inventory_uses_backend_owner_uid_header(self):
        payload = workspace_supervisor.EmptyRequest.model_validate({"uid": "body-user-uid", "owner_uid": "body-owner-uid"})
        with (
            patch.object(workspace_supervisor, "SUPERVISOR_KEY", "test-supervisor-key"),
            patch.object(workspace_supervisor, "_local_models", return_value=["mimo-v2.6:latest"]) as local_models,
        ):
            result = workspace_supervisor.local_models(
                payload,
                x_workspace_internal_key="test-supervisor-key",
                x_agent_user_uid="regular-user-uid",
                x_agent_owner_uid="backend-verified-owner-uid",
            )

        self.assertEqual(result, {"models": ["mimo-v2.6:latest"]})
        local_models.assert_called_once_with("backend-verified-owner-uid")

    def test_every_supervisor_workspace_admin_and_owner_route_requires_user_uid_header(self):
        expected_paths = {
            "/internal/v1/workspace/list",
            "/internal/v1/workspace/read",
            "/internal/v1/workspace/write",
            "/internal/v1/workspace/move",
            "/internal/v1/workspace/delete",
            "/internal/v1/workspace/command",
            "/internal/v1/workspace/packages",
            "/internal/v1/workspace/packages/request",
            "/internal/v1/admin/packages",
            "/internal/v1/admin/packages/review",
            "/internal/v1/models/local",
            "/internal/v1/admin/models/local",
            "/internal/v1/owner/list",
            "/internal/v1/owner/read",
            "/internal/v1/owner/propose/write",
            "/internal/v1/owner/propose/move",
            "/internal/v1/owner/propose/delete",
            "/internal/v1/owner/propose/command",
            "/internal/v1/owner/confirm",
        }
        endpoints = {
            route.path: route.endpoint
            for route in workspace_supervisor.app.routes
            if getattr(route, "path", None) in expected_paths
        }
        self.assertEqual(set(endpoints), expected_paths)
        for path, endpoint in endpoints.items():
            with self.subTest(path=path):
                self.assertIn("x_agent_user_uid", inspect.signature(endpoint).parameters)


if __name__ == "__main__":
    unittest.main()
