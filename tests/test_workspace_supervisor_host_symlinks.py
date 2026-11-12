import stat
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

    httpx_stub.HTTPError = HTTPError
    httpx_stub.TransportError = type("TransportError", (HTTPError,), {})
    httpx_stub.Response = Response
    httpx_stub.AsyncClient = UnusedAsyncClient
    httpx_stub.Timeout = object
    sys.modules["httpx"] = httpx_stub

from fastapi import HTTPException

import workspace_supervisor as supervisor


HOST_ROOT = Path("/host")


def _fake_lstat(symlink_paths=(), *, occurrence_symlinks=None):
    symlink_paths = set(symlink_paths)
    occurrence_symlinks = occurrence_symlinks or {}
    counts = {}

    def lstat(path):
        counts[path] = counts.get(path, 0) + 1
        if path in symlink_paths or counts[path] in occurrence_symlinks.get(path, set()):
            return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777)
        mode = stat.S_IFDIR if path == HOST_ROOT or path.name not in {"source.txt", "target.txt"} else stat.S_IFREG
        return SimpleNamespace(st_mode=mode | 0o755)

    return lstat


class WorkspaceSupervisorHostSymlinkTests(unittest.TestCase):
    def test_host_path_rejects_symlinked_parent_and_leaf_without_resolving_target(self):
        cases = (
            ("/tmp/link/secret.txt", HOST_ROOT / "tmp" / "link"),
            ("/tmp/shortcut", HOST_ROOT / "tmp" / "shortcut"),
        )
        for value, symlink in cases:
            with self.subTest(value=value):
                with (
                    patch.object(supervisor, "_host_root", return_value=HOST_ROOT),
                    patch.object(Path, "lstat", autospec=True, side_effect=_fake_lstat({symlink})),
                    patch.object(Path, "resolve", autospec=True, side_effect=AssertionError("host paths must not be resolved through symlinks")) as resolve,
                ):
                    with self.assertRaises(HTTPException) as raised:
                        supervisor._host_path(value)
                self.assertEqual(raised.exception.status_code, 403)
                resolve.assert_not_called()

    def test_confirmed_write_delete_and_move_reject_symlink_paths_before_mutation(self):
        cases = (
            ({"type": "write", "path": "/tmp/link/new.txt", "content": "x"}, {HOST_ROOT / "tmp" / "link"}),
            ({"type": "delete", "path": "/tmp/link"}, {HOST_ROOT / "tmp" / "link"}),
            ({"type": "move", "source": "/tmp/link", "destination": "/tmp/moved"}, {HOST_ROOT / "tmp" / "link"}),
            ({"type": "move", "source": "/tmp/source.txt", "destination": "/tmp/link/moved"}, {HOST_ROOT / "tmp" / "link"}),
        )
        for action, symlinks in cases:
            with self.subTest(action=action["type"], symlinks=symlinks):
                with (
                    patch.object(supervisor, "_host_root", return_value=HOST_ROOT),
                    patch.object(Path, "lstat", autospec=True, side_effect=_fake_lstat(symlinks)),
                    patch.object(supervisor.os, "open") as open_file,
                    patch.object(supervisor.os, "replace") as replace,
                    patch.object(supervisor.os, "rename") as rename,
                    patch.object(supervisor.os, "unlink") as unlink,
                    patch.object(supervisor.shutil, "rmtree") as rmtree,
                ):
                    with self.assertRaises(HTTPException) as raised:
                        supervisor._perform_host_action(action)
                self.assertEqual(raised.exception.status_code, 403)
                open_file.assert_not_called()
                replace.assert_not_called()
                rename.assert_not_called()
                unlink.assert_not_called()
                rmtree.assert_not_called()

    def test_host_write_rechecks_components_before_opening_parent(self):
        parent = HOST_ROOT / "tmp"
        lstat = _fake_lstat(occurrence_symlinks={parent: {2}})
        with (
            patch.object(supervisor, "_host_root", return_value=HOST_ROOT),
            patch.object(Path, "lstat", autospec=True, side_effect=lstat),
            patch.object(supervisor.os, "open") as open_file,
        ):
            with self.assertRaises(HTTPException) as raised:
                supervisor._perform_host_action({"type": "write", "path": "/tmp/target.txt", "content": "x"})

        self.assertEqual(raised.exception.status_code, 403)
        open_file.assert_not_called()

    def test_host_directory_open_uses_nofollow_for_each_component(self):
        path = HOST_ROOT / "var" / "log"
        returned_fds = iter((101, 102, 103))
        with (
            patch.object(supervisor, "_host_root", return_value=HOST_ROOT),
            patch.object(Path, "lstat", autospec=True, side_effect=_fake_lstat()),
            patch.object(supervisor.os, "open", side_effect=lambda *args, **kwargs: next(returned_fds)) as open_file,
            patch.object(supervisor.os, "close") as close_file,
        ):
            result = supervisor._open_host_directory(path)

        self.assertEqual(result, 103)
        self.assertEqual(open_file.call_count, 3)
        for call in open_file.call_args_list:
            flags = call.args[1]
            self.assertTrue(flags & supervisor.os.O_NOFOLLOW)
            self.assertTrue(flags & supervisor.os.O_DIRECTORY)
        self.assertEqual(close_file.call_args_list[0].args, (101,))
        self.assertEqual(close_file.call_args_list[1].args, (102,))


if __name__ == "__main__":
    unittest.main()
