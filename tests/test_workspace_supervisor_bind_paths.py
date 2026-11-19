import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from fastapi import HTTPException

import workspace_supervisor as supervisor


HOST_DATA = Path("/host/home/ubuntu/warthadev-datauser")
DOCKER_DATA = PurePosixPath("/home/ubuntu/warthadev-datauser")


class _FakeContainer:
    def start(self):
        pass

    def wait(self, timeout):
        return {"StatusCode": 0}

    def logs(self, stdout=True, stderr=True, tail=1000):
        return b"stubbed command output"

    def remove(self, force=True):
        pass


class _FakeDockerClient:
    def __init__(self):
        self.containers = self
        self.create_kwargs = None
        self.container = _FakeContainer()
        self.closed = False

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return self.container

    def close(self):
        self.closed = True


class WorkspaceSupervisorBindPathTests(unittest.TestCase):
    def _stub_filesystem(self, stack: ExitStack, *, samefile=True, symlink_paths=()):
        symlink_paths = set(symlink_paths)
        stack.enter_context(patch.object(supervisor, "DATA_ROOT", Path("/data/users")))
        stack.enter_context(patch.object(supervisor, "HOST_ROOT", Path("/host")))
        stack.enter_context(patch.object(supervisor, "HOST_DATA_ROOT", HOST_DATA))
        stack.enter_context(patch.object(supervisor, "DOCKER_HOST_DATA_ROOT", DOCKER_DATA))
        stack.enter_context(patch.object(supervisor, "_host_root", return_value=Path("/host")))
        stack.enter_context(patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path in symlink_paths))
        stack.enter_context(patch.object(Path, "is_dir", autospec=True, return_value=True))
        stack.enter_context(patch.object(Path, "resolve", autospec=True, side_effect=lambda path, *args, **kwargs: path))
        mkdir_mock = stack.enter_context(patch.object(Path, "mkdir", autospec=True))
        stack.enter_context(patch.object(Path, "chmod", autospec=True))
        samefile_mock = stack.enter_context(patch.object(supervisor.os.path, "samefile", return_value=samefile))
        return mkdir_mock, samefile_mock

    def test_dockerfile_documents_both_mount_views_and_host_daemon_source(self):
        dockerfile = Path(__file__).parent.parent / "Dockerfile.workspace-supervisor"
        content = dockerfile.read_text(encoding="utf-8")
        self.assertIn("/home/ubuntu/warthadev-datauser:/data/users:rw", content)
        self.assertIn("/:/host:rw", content)
        self.assertIn("WORKSPACE_HOST_DATA_ROOT=/host/home/ubuntu/warthadev-datauser", content)
        self.assertIn("WORKSPACE_DOCKER_HOST_DATA_ROOT=/home/ubuntu/warthadev-datauser", content)

    def test_host_path_is_created_through_host_root_and_docker_path_is_host_absolute(self):
        uid = "firebase_uid-42"
        api_workspace = Path(f"/data/users/{uid}/workspace")
        with ExitStack() as stack:
            mkdir_mock, samefile_mock = self._stub_filesystem(stack)
            host_workspace, docker_workspace = supervisor._host_workspace_bind_paths(uid, api_workspace)

        self.assertEqual(host_workspace, HOST_DATA / uid / "workspace")
        self.assertEqual(docker_workspace, DOCKER_DATA / uid / "workspace")
        mkdir_paths = [call.args[0] for call in mkdir_mock.call_args_list]
        self.assertEqual(mkdir_paths, [HOST_DATA / uid, HOST_DATA / uid / "workspace"])
        self.assertEqual(samefile_mock.call_count, 2)
        self.assertEqual(samefile_mock.call_args_list[1].args, (api_workspace, host_workspace))

    def test_docker_api_receives_only_the_real_host_source_path(self):
        uid = "firebase_uid-42"
        api_workspace = Path(f"/data/users/{uid}/workspace")
        client = _FakeDockerClient()
        docker_module = types.ModuleType("docker")
        docker_errors = types.ModuleType("docker.errors")
        class DockerException(Exception):
            pass
        class NotFound(Exception):
            pass
        docker_errors.DockerException = DockerException
        docker_errors.NotFound = NotFound
        docker_module.errors = docker_errors
        docker_module.from_env = lambda timeout=15: client

        with ExitStack() as stack:
            self._stub_filesystem(stack)
            stack.enter_context(patch.object(supervisor, "_workspace_root", return_value=api_workspace))
            stack.enter_context(patch.object(supervisor, "SUPERVISOR_KEY", "test-only-key"))
            stack.enter_context(patch.dict(sys.modules, {"docker": docker_module, "docker.errors": docker_errors}))
            command = "printf 'smoke output' > smoke.txt && cat smoke.txt"
            result = supervisor._run_workspace_command(uid, command)

        expected_source = str(DOCKER_DATA / uid / "workspace")
        self.assertEqual(list(client.create_kwargs["volumes"]), [expected_source])
        self.assertEqual(client.create_kwargs["volumes"][expected_source], {"bind": "/data/workspace", "mode": "rw"})
        self.assertEqual(client.create_kwargs["command"], [command])
        self.assertEqual(client.create_kwargs["entrypoint"], ["/bin/sh", "-lc"])
        self.assertTrue(result["success"])
        self.assertTrue(client.closed)

    def test_symlinked_uid_workspace_is_rejected_without_creating_directories(self):
        uid = "firebase_uid-42"
        symlink_uid = HOST_DATA / uid
        with ExitStack() as stack:
            mkdir_mock, _ = self._stub_filesystem(stack, symlink_paths={symlink_uid})
            with self.assertRaises(HTTPException) as raised:
                supervisor._host_workspace_bind_paths(uid, Path(f"/data/users/{uid}/workspace"))

        self.assertEqual(raised.exception.status_code, 403)
        mkdir_mock.assert_not_called()

    def test_invalid_uid_and_mismatched_mounts_fail_closed_without_directory_creation(self):
        with self.assertRaises(HTTPException) as invalid_uid:
            supervisor._host_workspace_bind_paths("../escape", Path("/data/users/escape/workspace"))
        self.assertEqual(invalid_uid.exception.status_code, 400)

        with ExitStack() as stack:
            mkdir_mock, _ = self._stub_filesystem(stack, samefile=False)
            with self.assertRaises(HTTPException) as mismatched_mount:
                supervisor._host_workspace_bind_paths("valid_uid", Path("/data/users/valid_uid/workspace"))

        self.assertEqual(mismatched_mount.exception.status_code, 503)
        mkdir_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
