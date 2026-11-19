import stat
import sys
import os
import tempfile
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

    def test_owner_list_and_read_require_owner_headers_not_body_identity(self):
        payload = supervisor.HostReadRequest.model_validate({"uid": "body-owner-uid", "path": "/"})
        self.assertFalse(hasattr(payload, "uid"))
        with (
            patch.object(supervisor, "SUPERVISOR_KEY", "test-supervisor-key"),
            patch.object(supervisor, "_list_host") as list_host,
            patch.object(supervisor, "_read_host") as read_host,
        ):
            identities = (
                ("regular-user-uid", "owner", "verified-owner-uid"),
                ("verified-owner-uid", "user", "verified-owner-uid"),
            )
            for endpoint in (supervisor.owner_list, supervisor.owner_read):
                for user_uid, role, owner_uid in identities:
                    with self.subTest(endpoint=endpoint.__name__, role=role, user_uid=user_uid):
                        with self.assertRaises(HTTPException) as raised:
                            endpoint(
                                payload,
                                x_workspace_internal_key="test-supervisor-key",
                                x_agent_user_uid=user_uid,
                                x_agent_user_role=role,
                                x_agent_owner_uid=owner_uid,
                            )
                        self.assertEqual(raised.exception.status_code, 403)

        list_host.assert_not_called()
        read_host.assert_not_called()

    def test_host_list_uses_directory_fd_and_skips_special_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "folder").mkdir()
            (root / "notes.txt").write_text("safe", encoding="utf-8")
            (root / "link").symlink_to(root / "folder", target_is_directory=True)
            os.mkfifo(root / "pipe")
            with patch.object(supervisor, "_host_root", return_value=root):
                result = supervisor._list_host("/")

        entries = {entry["name"]: entry for entry in result["entries"]}
        self.assertEqual(result["path"], "/")
        self.assertEqual(entries["folder"]["type"], "directory")
        self.assertEqual(entries["notes.txt"]["type"], "file")
        self.assertEqual(entries["link"]["type"], "symlink")
        self.assertNotIn("pipe", entries)

    def test_host_listing_rejects_directory_symlink_swap_before_scanning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tmp").mkdir()
            victim = root / "tmp" / "victim"
            victim.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.txt").write_text("must not list", encoding="utf-8")
            real_open = os.open
            swapped = False

            def race_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == "tmp" and "dir_fd" in kwargs and not swapped:
                    fd = real_open(path, flags, *args, **kwargs)
                    victim.rmdir()
                    victim.symlink_to(outside, target_is_directory=True)
                    swapped = True
                    return fd
                return real_open(path, flags, *args, **kwargs)

            with (
                patch.object(supervisor, "_host_root", return_value=root),
                patch.object(supervisor.os, "open", side_effect=race_open),
                patch.object(supervisor.os, "scandir", wraps=os.scandir) as scandir,
            ):
                with self.assertRaises(HTTPException) as raised:
                    supervisor._list_host("/tmp/victim")

        self.assertTrue(swapped)
        self.assertEqual(raised.exception.status_code, 403)
        scandir.assert_not_called()

    def test_host_read_is_bounded_utf8_only_and_never_returns_binary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "notes.txt").write_text("safe text", encoding="utf-8")
            (root / "binary.dat").write_bytes(b"\xff\x00")
            (root / "large.txt").write_text("12345", encoding="utf-8")
            with patch.object(supervisor, "_host_root", return_value=root):
                result = supervisor._read_host("/notes.txt")
                with self.assertRaises(HTTPException) as binary_error:
                    supervisor._read_host("/binary.dat")
                with (
                    patch.object(supervisor, "MAX_HOST_TEXT_READ_BYTES", 4),
                    self.assertRaises(HTTPException) as large_error,
                ):
                    supervisor._read_host("/large.txt")

        self.assertEqual(result["text"], "safe text")
        self.assertTrue(result["isText"])
        self.assertNotIn("dataBase64", result)
        self.assertEqual(binary_error.exception.status_code, 415)
        self.assertEqual(large_error.exception.status_code, 413)

    def test_host_read_blocks_proc_pseudofiles_and_leaf_symlink_swap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proc = root / "proc"
            proc.mkdir()
            (proc / "cpuinfo").write_text("pseudo", encoding="utf-8")
            victim = root / "notes.txt"
            victim.write_text("original", encoding="utf-8")
            outside = root / "outside.txt"
            outside.write_text("must not read", encoding="utf-8")
            real_open = os.open
            swapped = False

            def race_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == "notes.txt" and "dir_fd" in kwargs and not swapped:
                    victim.unlink()
                    victim.symlink_to(outside)
                    swapped = True
                return real_open(path, flags, *args, **kwargs)

            with patch.object(supervisor, "_host_root", return_value=root):
                with self.assertRaises(HTTPException) as pseudo_error:
                    supervisor._read_host("/proc/cpuinfo")
                with (
                    patch.object(supervisor.os, "open", side_effect=race_open),
                    self.assertRaises(HTTPException) as race_error,
                ):
                    supervisor._read_host("/notes.txt")

        self.assertEqual(pseudo_error.exception.status_code, 403)
        self.assertTrue(swapped)
        self.assertEqual(race_error.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
