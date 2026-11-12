"""Private host-side workspace supervisor.

Run only on the private Docker network. The supervisor may receive the host
workspace-data bind, the Docker socket (for isolated task containers), and the
host-root bind at /host. None of those mounts belong in the ordinary agent
runtime or task containers. No listener should be published to the public host.
"""
from __future__ import annotations

import base64
import errno
import hmac
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")

DATA_ROOT = Path(os.getenv("WORKSPACE_DATA_ROOT", "/data/users")).resolve()
HOST_ROOT = Path(os.getenv("WORKSPACE_HOST_ROOT", "/host")).resolve()
HOST_DATA_ROOT = Path(os.getenv("WORKSPACE_HOST_DATA_ROOT", str(HOST_ROOT / "home/ubuntu/warthadev-datauser")))
DOCKER_HOST_DATA_ROOT = PurePosixPath(os.getenv("WORKSPACE_DOCKER_HOST_DATA_ROOT", "/home/ubuntu/warthadev-datauser"))
SUPERVISOR_KEY = os.getenv("WORKSPACE_SUPERVISOR_INTERNAL_KEY", "")
TASK_IMAGE = os.getenv("WORKSPACE_TASK_IMAGE", "warthadev-agent:local")
OLLAMA_TAGS_URL = os.getenv("WORKSPACE_OLLAMA_TAGS_URL", "http://ollama:11434/api/tags")
OWNER_EMAIL = "warthadev@gmail.com"
UID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
PACKAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@/+:-]{0,199}$")
MAX_COMMAND_SECONDS = 120
MAX_OUTPUT_CHARS = 30_000
MAX_HOST_TEXT_READ_BYTES = 1024 * 1024
HOST_PSEUDO_FILESYSTEMS = frozenset({"proc", "sys", "dev"})
ACTION_TTL_SECONDS = 600

app = FastAPI(
    title="Private Wartha workspace supervisor",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.router.redirect_slashes = False


class WorkspaceRequest(BaseModel):
    path: str = Field(default=".", max_length=2048)


class WorkspaceWriteRequest(WorkspaceRequest):
    content: str


class WorkspaceMoveRequest(BaseModel):
    source: str = Field(min_length=1, max_length=2048)
    destination: str = Field(min_length=1, max_length=2048)


class WorkspaceCommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=20_000)


class PackageRequest(BaseModel):
    package: str = Field(min_length=1, max_length=200)


class PackageReviewRequest(BaseModel):
    target_uid: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=64)
    decision: str = Field(pattern="^(approve|reject)$")


class LocalModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    action: str = Field(pattern="^(add|remove)$")


class EmptyRequest(BaseModel):
    pass


class HostReadRequest(WorkspaceRequest):
    pass


class HostWriteRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2048)
    content: str


class HostMoveRequest(BaseModel):
    source: str = Field(min_length=1, max_length=2048)
    destination: str = Field(min_length=1, max_length=2048)


class HostDeleteRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2048)


class HostCommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=20_000)


class HostConfirmRequest(BaseModel):
    action_id: str = Field(min_length=20, max_length=128)


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_PENDING_HOST_ACTIONS: dict[str, dict[str, Any]] = {}
_PENDING_GUARD = threading.Lock()


def _uid(value: str) -> str:
    if not UID_PATTERN.fullmatch(value or ""):
        raise HTTPException(status_code=400, detail="Invalid Firebase UID")
    return value


def _lock(uid: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(uid, threading.RLock())


def _authorize(internal_key: str | None) -> None:
    if not SUPERVISOR_KEY or not internal_key or not hmac.compare_digest(internal_key, SUPERVISOR_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_owner(uid: str, role: str | None, owner_uid: str | None) -> str:
    normalized_uid = _uid(uid)
    normalized_owner = _uid(owner_uid or "")
    if role != "owner" or not hmac.compare_digest(normalized_uid, normalized_owner):
        raise HTTPException(status_code=403, detail="Owner access required")
    return normalized_owner


def _trusted_user_uid(internal_key: str | None, user_uid: str | None) -> str:
    _authorize(internal_key)
    if not user_uid:
        raise HTTPException(status_code=403, detail="Verified user identity is required")
    return _uid(user_uid)


def _user_data_root(uid: str, *, create: bool = True) -> Path:
    normalized = _uid(uid)
    DATA_ROOT.mkdir(parents=True, exist_ok=True, mode=0o750)
    root = DATA_ROOT / normalized
    if root.is_symlink():
        raise HTTPException(status_code=403, detail="User data root is not a directory")
    if create:
        root.mkdir(mode=0o700, exist_ok=True)
        try:
            root.chmod(0o700)
        except OSError:
            pass
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail="User data not found")
    resolved_root = root.resolve()
    if resolved_root.parent != DATA_ROOT:
        raise HTTPException(status_code=403, detail="User data path is invalid")
    return resolved_root


def _workspace_root(uid: str, *, create: bool = True) -> Path:
    user_root = _user_data_root(uid, create=create)
    root = user_root / "workspace"
    if root.is_symlink():
        raise HTTPException(status_code=403, detail="Workspace path is not a directory")
    if create:
        root.mkdir(mode=0o700, exist_ok=True)
        try:
            os.chown(root, 1000, 1000)
            root.chmod(0o700)
        except OSError as exc:
            if root.stat().st_uid != 1000:
                raise HTTPException(status_code=500, detail="Workspace permissions are not ready for the isolated task user") from exc
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail="Workspace not found")
    resolved_root = root.resolve()
    if resolved_root.parent != user_root:
        raise HTTPException(status_code=403, detail="Workspace path is invalid")
    return resolved_root


def _metadata_dir(uid: str, *, create: bool = True) -> Path:
    user_root = _user_data_root(uid, create=create)
    metadata = user_root / ".wartha"
    if metadata.is_symlink():
        raise HTTPException(status_code=403, detail="Workspace metadata path is invalid")
    if create:
        metadata.mkdir(mode=0o700, exist_ok=True)
        try:
            metadata.chmod(0o700)
        except OSError:
            pass
    if not metadata.exists() or not metadata.is_dir():
        raise HTTPException(status_code=404, detail="Workspace metadata not found")
    if metadata.resolve().parent != user_root:
        raise HTTPException(status_code=403, detail="Workspace metadata path is invalid")
    return metadata


def _relative_path(value: str, *, allow_root: bool = True) -> PurePosixPath:
    raw = (value or ".").strip()
    if "\x00" in raw or raw.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="Workspace paths must be relative")
    path = PurePosixPath(raw)
    if any(part in {"..", ""} for part in path.parts):
        raise HTTPException(status_code=400, detail="Workspace path traversal is not allowed")
    if path.parts and path.parts[0] == ".wartha":
        raise HTTPException(status_code=403, detail="Workspace metadata is private")
    if not allow_root and (not path.parts or str(path) == "."):
        raise HTTPException(status_code=400, detail="A file path is required")
    return path


def _workspace_path(uid: str, value: str, *, allow_root: bool = True, allow_missing: bool = False) -> Path:
    root = _workspace_root(uid)
    relative = _relative_path(value, allow_root=allow_root)
    candidate = root.joinpath(*relative.parts) if relative.parts else root
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise HTTPException(status_code=403, detail="Symbolic links are not allowed in workspace operations")
    resolved = candidate.resolve(strict=not allow_missing)
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=403, detail="Workspace path escapes its UID directory")
    return resolved


def _json_read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp_name = tempfile.mkstemp(prefix=".wartha-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _file_item(path: Path, root: Path) -> dict[str, Any]:
    info = path.lstat()
    if path.is_symlink():
        kind = "symlink"
    elif path.is_dir():
        kind = "directory"
    elif path.is_file():
        kind = "file"
    else:
        kind = "other"
    return {
        "name": path.name,
        "path": path.relative_to(root).as_posix() if path != root else ".",
        "type": kind,
        "size": info.st_size,
        "modified": int(info.st_mtime),
    }


def _list_workspace(uid: str, value: str) -> dict[str, Any]:
    root = _workspace_root(uid)
    target = _workspace_path(uid, value)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")
    entries = []
    for item in sorted(target.iterdir(), key=lambda entry: entry.name.casefold()):
        if target == root and item.name == ".wartha":
            continue
        entries.append(_file_item(item, root))
        if len(entries) >= 2000:
            break
    return {"path": "." if target == root else target.relative_to(root).as_posix(), "entries": entries}


def _read_workspace_file(uid: str, value: str) -> dict[str, Any]:
    target = _workspace_path(uid, value, allow_root=False)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    data = target.read_bytes()
    try:
        text: str | None = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    return {
        "path": target.relative_to(_workspace_root(uid)).as_posix(),
        "name": target.name,
        "mimeType": mimetypes.guess_type(target.name)[0] or "application/octet-stream",
        "isText": text is not None,
        "text": text,
        "dataBase64": base64.b64encode(data).decode("ascii"),
        "size": len(data),
    }


def _write_workspace_file(uid: str, value: str, content: str) -> dict[str, Any]:
    root = _workspace_root(uid)
    relative = _relative_path(value, allow_root=False)
    parent_value = PurePosixPath(*relative.parts[:-1]) if len(relative.parts) > 1 else PurePosixPath(".")
    parent = _workspace_path(uid, parent_value.as_posix())
    if not parent.is_dir():
        raise HTTPException(status_code=404, detail="Parent directory not found")
    target = parent / relative.parts[-1]
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise HTTPException(status_code=400, detail="File path is not a regular file")
    fd, temp_name = tempfile.mkstemp(prefix=".wartha-write-", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return {"success": True, "path": target.relative_to(root).as_posix(), "size": target.stat().st_size}


def _move_workspace_file(uid: str, source: str, destination: str) -> dict[str, Any]:
    src = _workspace_path(uid, source, allow_root=False)
    dst = _workspace_path(uid, destination, allow_root=False, allow_missing=True)
    if not src.exists() or src.is_symlink():
        raise HTTPException(status_code=404, detail="Source not found")
    if dst.exists() or dst.is_symlink():
        raise HTTPException(status_code=409, detail="Destination already exists")
    dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Revalidate after parent creation; only the supervisor can create parent directories.
    dst = _workspace_path(uid, destination, allow_root=False, allow_missing=True)
    src.rename(dst)
    root = _workspace_root(uid)
    return {"success": True, "source": src.relative_to(root).as_posix(), "destination": dst.relative_to(root).as_posix()}


def _delete_workspace_file(uid: str, value: str) -> dict[str, Any]:
    target = _workspace_path(uid, value, allow_root=False, allow_missing=True)
    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    elif target.is_file():
        target.unlink()
    else:
        raise HTTPException(status_code=404, detail="File not found")
    return {"success": True, "path": value}


def _shared_packages() -> set[str]:
    raw = os.getenv("WORKSPACE_SHARED_PACKAGES", "")
    try:
        parsed = json.loads(raw) if raw.strip().startswith("[") else [item.strip() for item in raw.split(",")]
    except json.JSONDecodeError:
        parsed = []
    if not isinstance(parsed, list):
        return set()
    return {str(item).strip().casefold() for item in parsed if isinstance(item, (str, int, float)) and str(item).strip()}


def _package_store(uid: str) -> Path:
    return _metadata_dir(uid) / "packages.json"


def _validate_package(value: str) -> str:
    normalized = (value or "").strip()
    if not PACKAGE_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="Package name/version contains unsupported characters")
    return normalized


def _list_packages(uid: str) -> dict[str, Any]:
    store = _package_store(uid)
    saved = _json_read(store, {"requests": []})
    records = saved.get("requests", []) if isinstance(saved, dict) else []
    shared = _shared_packages()
    result = []
    for record in records:
        if not isinstance(record, dict):
            continue
        item = dict(record)
        item["status"] = "available" if str(item.get("package", "")).casefold() in shared else item.get("status", "pending")
        result.append(item)
    return {"requests": result, "sharedPackages": sorted(shared)}


def _request_package(uid: str, package: str) -> dict[str, Any]:
    normalized = _validate_package(package)
    store = _package_store(uid)
    saved = _json_read(store, {"requests": []})
    records = saved.get("requests", []) if isinstance(saved, dict) else []
    shared = _shared_packages()
    match = next((item for item in records if isinstance(item, dict) and item.get("package", "").casefold() == normalized.casefold()), None)
    status = "available" if normalized.casefold() in shared else "pending"
    if match:
        match["status"] = status if status == "available" else ("approved_for_image_build" if match.get("status") == "approved_for_image_build" else "pending")
        match["updatedAt"] = int(time.time())
        result = match
    else:
        result = {"id": uuid.uuid4().hex, "package": normalized, "status": status, "requestedAt": int(time.time()), "updatedAt": int(time.time())}
        records.append(result)
    _json_write(store, {"requests": records})
    return {"request": result}


def _all_package_requests() -> list[dict[str, Any]]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True, mode=0o750)
    output = []
    shared = _shared_packages()
    for child in DATA_ROOT.iterdir():
        if child.is_symlink() or not child.is_dir() or not UID_PATTERN.fullmatch(child.name):
            continue
        data = _json_read(child / ".wartha" / "packages.json", {"requests": []})
        for record in data.get("requests", []) if isinstance(data, dict) else []:
            if isinstance(record, dict):
                item = dict(record)
                item["uid"] = child.name
                if str(item.get("package", "")).casefold() in shared:
                    item["status"] = "available"
                output.append(item)
    return sorted(output, key=lambda item: item.get("requestedAt", 0), reverse=True)


def _review_package(uid: str, request_id: str, decision: str) -> dict[str, Any]:
    store = _package_store(uid)
    saved = _json_read(store, {"requests": []})
    records = saved.get("requests", []) if isinstance(saved, dict) else []
    request = next((item for item in records if isinstance(item, dict) and item.get("id") == request_id), None)
    if not request:
        raise HTTPException(status_code=404, detail="Package request not found")
    if str(request.get("package", "")).casefold() in _shared_packages():
        request["status"] = "available"
    else:
        request["status"] = "approved_for_image_build" if decision == "approve" else "rejected"
    request["reviewedAt"] = int(time.time())
    _json_write(store, {"requests": records})
    return {"request": request, "sharedImageChanged": False}


def _default_local_models() -> list[str]:
    configured = os.getenv("WORKSPACE_DEFAULT_LOCAL_MODELS", os.getenv("AGENT_LOCAL_MODEL", "mimo-v2.6:latest"))
    values = [item.strip() for item in configured.split(",") if item.strip()]
    return values or ["mimo-v2.6:latest"]


def _local_model_store(owner_uid: str) -> Path:
    return _metadata_dir(owner_uid) / "local-models.json"


def _local_models(owner_uid: str) -> list[str]:
    defaults = _default_local_models()
    try:
        store = _local_model_store(owner_uid)
        saved = _json_read(store, {"models": defaults})
    except HTTPException:
        return defaults
    models = saved.get("models", defaults) if isinstance(saved, dict) else defaults
    if not isinstance(models, list):
        models = defaults
    normalized = [item.strip() for item in models if isinstance(item, str) and item.strip()]
    # The currently shared Mimo model stays available to every account.
    for model in defaults:
        if model not in normalized:
            normalized.insert(0, model)
    return list(dict.fromkeys(normalized))


def _installed_ollama_models() -> set[str]:
    try:
        response = httpx.get(OLLAMA_TAGS_URL, timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return set()
    models = payload.get("models", []) if isinstance(payload, dict) else []
    return {str(item.get("name", "")).strip() for item in models if isinstance(item, dict) and item.get("name")}


def _update_local_models(owner_uid: str, model: str, action: str) -> dict[str, Any]:
    normalized = model.strip()
    if not normalized or len(normalized) > 200 or any(char.isspace() for char in normalized):
        raise HTTPException(status_code=400, detail="Invalid local model ID")
    models = _local_models(owner_uid)
    defaults = set(_default_local_models())
    if action == "add":
        if normalized not in _installed_ollama_models():
            raise HTTPException(status_code=409, detail="Only models already installed in shared Ollama can be registered; pull/install is not exposed here")
        if normalized not in models:
            models.append(normalized)
    else:
        if normalized in defaults:
            raise HTTPException(status_code=409, detail="The shared default Mimo model remains available to all users")
        models = [item for item in models if item != normalized]
    _json_write(_local_model_store(owner_uid), {"models": models, "updatedAt": int(time.time())})
    return {"models": models, "ollamaChanged": False}


def _host_root() -> Path:
    if not HOST_ROOT.is_dir() or not (HOST_ROOT / "etc" / "os-release").is_file():
        raise HTTPException(status_code=503, detail="Privileged host root is not mounted in the private supervisor")
    return HOST_ROOT


def _reject_host_symlink_components(root: Path, candidate: Path) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Host workspace path escapes the mounted Sandbox root") from exc
    if any(part in {"..", ""} for part in relative.parts):
        raise HTTPException(status_code=403, detail="Host workspace path is invalid")
    current = root
    if current.is_symlink():
        raise HTTPException(status_code=403, detail="Host root must not be a symbolic link")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise HTTPException(status_code=403, detail="Host workspace paths must not contain symbolic links")


def _validate_host_path_components(root: Path, candidate: Path, *, allow_missing_leaf: bool = False) -> None:
    try:
        relative = candidate.relative_to(root)
        root_info = root.lstat()
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Host path escapes the mounted Sandbox root") from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Host root cannot be verified") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise HTTPException(status_code=403, detail="Host root must be a real directory")
    current = root
    parts = relative.parts
    for index, part in enumerate(parts):
        if part in {"..", ""}:
            raise HTTPException(status_code=403, detail="Host path is invalid")
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise HTTPException(status_code=404, detail="Host path not found") from exc
        except OSError as exc:
            raise HTTPException(status_code=403, detail="Host path cannot be verified") from exc
        if stat.S_ISLNK(info.st_mode):
            raise HTTPException(status_code=403, detail="Host paths must not contain symbolic links")


def _open_host_directory(path: Path) -> int:
    root = _host_root()
    _validate_host_path_components(root, path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise HTTPException(status_code=503, detail="Safe host path operations are unavailable")
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    fd: int | None = None
    try:
        fd = os.open(root, flags)
        for part in path.relative_to(root).parts:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if exc.errno == errno.ENOENT:
            raise HTTPException(status_code=404, detail="Host directory not found") from exc
        if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EACCES}:
            raise HTTPException(status_code=403, detail="Host path changed or contains a symbolic link") from exc
        raise HTTPException(status_code=503, detail="Host directory cannot be opened safely") from exc


def _host_workspace_bind_paths(uid: str, api_workspace: Path) -> tuple[Path, PurePosixPath]:
    """Create/verify via the host-root bind; return the Docker daemon's host path separately."""
    normalized_uid = _uid(uid)
    host_root = _host_root()
    docker_data_root = DOCKER_HOST_DATA_ROOT
    if not docker_data_root.is_absolute() or docker_data_root == PurePosixPath("/") or ".." in docker_data_root.parts:
        raise HTTPException(status_code=503, detail="Docker host workspace root is invalid")

    expected_access_root = host_root.joinpath(*docker_data_root.parts[1:])
    if HOST_DATA_ROOT != expected_access_root:
        raise HTTPException(status_code=503, detail="Host-root and Docker workspace roots are inconsistent")
    _reject_host_symlink_components(host_root, HOST_DATA_ROOT)
    if not HOST_DATA_ROOT.is_dir():
        raise HTTPException(status_code=503, detail="Host workspace data root is not mounted")
    try:
        if not os.path.samefile(DATA_ROOT, HOST_DATA_ROOT):
            raise HTTPException(status_code=503, detail="/data/users and host-root workspace mounts do not reference the same host directory")
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Workspace data mounts cannot be verified") from exc

    host_user_root = HOST_DATA_ROOT / normalized_uid
    host_workspace = host_user_root / "workspace"
    for directory in (host_user_root, host_workspace):
        if directory.is_symlink():
            raise HTTPException(status_code=403, detail="Host workspace paths must not be symbolic links")
        try:
            directory.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=503, detail="Host workspace directory could not be created") from exc
        if directory.is_symlink() or not directory.is_dir():
            raise HTTPException(status_code=403, detail="Host workspace path is not a real directory")
        _reject_host_symlink_components(HOST_DATA_ROOT, directory)

    try:
        resolved_data_root = HOST_DATA_ROOT.resolve(strict=True)
        resolved_user_root = host_user_root.resolve(strict=True)
        resolved_workspace = host_workspace.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Host workspace path could not be verified") from exc
    if resolved_user_root.parent != resolved_data_root or resolved_workspace.parent != resolved_user_root:
        raise HTTPException(status_code=403, detail="Host workspace path escapes its UID data directory")
    try:
        if not os.path.samefile(api_workspace, resolved_workspace):
            raise HTTPException(status_code=503, detail="API and Docker workspace mounts do not reference the same UID workspace")
    except OSError as exc:
        raise HTTPException(status_code=503, detail="UID workspace bind source cannot be verified") from exc

    docker_workspace = docker_data_root / normalized_uid / "workspace"
    if docker_workspace.parent.parent != docker_data_root:
        raise HTTPException(status_code=403, detail="Docker workspace source escapes its host data root")
    return resolved_workspace, docker_workspace


def _host_path(value: str, *, allow_missing: bool = False) -> Path:
    if not value or "\x00" in value or not value.startswith("/"):
        raise HTTPException(status_code=400, detail="Host paths must be absolute")
    raw = PurePosixPath(value)
    if ".." in raw.parts:
        raise HTTPException(status_code=400, detail="Host path traversal is not allowed")
    root = _host_root()
    candidate = root.joinpath(*[part for part in raw.parts if part not in {"/", ""}])
    _validate_host_path_components(root, candidate, allow_missing_leaf=allow_missing)
    return candidate


def _list_host(value: str) -> dict[str, Any]:
    target = _host_path(value)
    directory_fd = _open_host_directory(target)
    root = _host_root()
    relative = target.relative_to(root)
    pseudo_filesystem = relative.parts[0] if relative.parts else ""
    items = []
    try:
        with os.scandir(os.dup(directory_fd)) as entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                mode = info.st_mode
                if stat.S_ISLNK(mode):
                    kind = "symlink"
                elif stat.S_ISDIR(mode):
                    kind = "directory"
                elif stat.S_ISREG(mode):
                    kind = "file"
                else:
                    continue
                if pseudo_filesystem in HOST_PSEUDO_FILESYSTEMS and kind != "directory":
                    continue
                if len(items) >= 2000:
                    break
                entry_relative = relative / entry.name
                entry_path = "/" + entry_relative.as_posix()
                items.append({
                    "name": entry.name,
                    "path": entry_path,
                    "type": kind,
                    "size": info.st_size,
                    "modified": int(info.st_mtime),
                })
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.ELOOP, errno.ENOTDIR}:
            raise HTTPException(status_code=403, detail="Host directory cannot be listed safely") from exc
        raise HTTPException(status_code=503, detail="Host directory listing is unavailable") from exc
    finally:
        os.close(directory_fd)
    items.sort(key=lambda item: item["name"].casefold())
    display_path = "/" if not relative.parts else "/" + relative.as_posix()
    return {"path": display_path, "entries": items}


def _read_host(value: str) -> dict[str, Any]:
    target = _host_path(value)
    root = _host_root()
    relative = target.relative_to(root)
    if not relative.parts:
        raise HTTPException(status_code=400, detail="Host root is not a file")
    if relative.parts[0] in HOST_PSEUDO_FILESYSTEMS:
        raise HTTPException(status_code=403, detail="Reading host pseudo-filesystem files is disabled")
    parent_fd = _open_host_directory(target.parent)
    file_fd: int | None = None
    try:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        nonblock = getattr(os, "O_NONBLOCK", 0)
        if not nofollow or not nonblock:
            raise HTTPException(status_code=503, detail="Safe host file reads are unavailable")
        flags = os.O_RDONLY | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0)
        file_fd = os.open(target.name, flags, dir_fd=parent_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise HTTPException(status_code=415, detail="Only regular text files can be read")
        if info.st_size > MAX_HOST_TEXT_READ_BYTES:
            raise HTTPException(status_code=413, detail="Host text file exceeds the 1 MiB read limit")
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_HOST_TEXT_READ_BYTES:
            chunk = os.read(file_fd, min(64 * 1024, MAX_HOST_TEXT_READ_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > MAX_HOST_TEXT_READ_BYTES:
            raise HTTPException(status_code=413, detail="Host text file exceeds the 1 MiB read limit")
        data = b"".join(chunks)
        text = data.decode("utf-8")
        if "\x00" in text:
            raise HTTPException(status_code=415, detail="Only UTF-8 text files can be read")
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="Only UTF-8 text files can be read") from None
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise HTTPException(status_code=404, detail="Host file not found") from exc
        if exc.errno in {errno.EACCES, errno.ELOOP, errno.ENOTDIR}:
            raise HTTPException(status_code=403, detail="Host file changed or cannot be opened safely") from exc
        raise HTTPException(status_code=503, detail="Host file cannot be read safely") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)
    return {
        "path": "/" + relative.as_posix(),
        "name": target.name,
        "mimeType": mimetypes.guess_type(target.name)[0] or "text/plain",
        "isText": True,
        "text": text,
        "size": len(data),
    }


def _perform_host_action(action: dict[str, Any]) -> dict[str, Any]:
    kind = action.get("type")
    if kind == "write":
        target = _host_path(str(action["path"]), allow_missing=True)
        root = _host_root()
        if target == root:
            raise HTTPException(status_code=400, detail="Refusing to replace the host root")
        parent_fd = _open_host_directory(target.parent)
        temp_name = f".wartha-host-write-{secrets.token_hex(12)}"
        try:
            _validate_host_path_components(root, target, allow_missing_leaf=True)
            try:
                existing = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing and stat.S_ISLNK(existing.st_mode):
                raise HTTPException(status_code=400, detail="Refusing to replace a symbolic link")
            if existing and stat.S_ISDIR(existing.st_mode):
                raise HTTPException(status_code=400, detail="Refusing to replace a host directory")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(str(action["content"]))
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o644)
            _validate_host_path_components(root, target, allow_missing_leaf=True)
            try:
                existing = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing and stat.S_ISLNK(existing.st_mode):
                raise HTTPException(status_code=400, detail="Refusing to replace a symbolic link")
            if existing and stat.S_ISDIR(existing.st_mode):
                raise HTTPException(status_code=400, detail="Refusing to replace a host directory")
            os.replace(temp_name, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        finally:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.close(parent_fd)
        return {"success": True, "message": f"Wrote {action['path']}"}
    if kind == "delete":
        target = _host_path(str(action["path"]))
        root = _host_root()
        if target == root:
            raise HTTPException(status_code=400, detail="Refusing to delete the host root")
        parent_fd = _open_host_directory(target.parent)
        try:
            _validate_host_path_components(root, target)
            try:
                target_info = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Host path not found") from exc
            if stat.S_ISLNK(target_info.st_mode):
                raise HTTPException(status_code=400, detail="Refusing to delete a symbolic link")
            _validate_host_path_components(root, target)
            if stat.S_ISDIR(target_info.st_mode):
                if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
                    raise HTTPException(status_code=503, detail="Safe recursive host deletion is unavailable")
                shutil.rmtree(target.name, dir_fd=parent_fd)
            else:
                os.unlink(target.name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        return {"success": True, "message": f"Deleted {action['path']}"}
    if kind == "move":
        source = _host_path(str(action["source"]))
        destination = _host_path(str(action["destination"]), allow_missing=True)
        root = _host_root()
        if source == root or destination == root:
            raise HTTPException(status_code=400, detail="Unsafe host move path")
        source_parent_fd = _open_host_directory(source.parent)
        destination_parent_fd = _open_host_directory(destination.parent)
        try:
            _validate_host_path_components(root, source)
            _validate_host_path_components(root, destination, allow_missing_leaf=True)
            try:
                source_info = os.stat(source.name, dir_fd=source_parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Host source not found") from exc
            if stat.S_ISLNK(source_info.st_mode):
                raise HTTPException(status_code=400, detail="Refusing to move a symbolic link")
            try:
                os.stat(destination.name, dir_fd=destination_parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise HTTPException(status_code=400, detail="Host move destination already exists")
            _validate_host_path_components(root, source)
            _validate_host_path_components(root, destination, allow_missing_leaf=True)
            os.rename(source.name, destination.name, src_dir_fd=source_parent_fd, dst_dir_fd=destination_parent_fd)
        finally:
            os.close(source_parent_fd)
            os.close(destination_parent_fd)
        return {"success": True, "message": f"Moved {action['source']} to {action['destination']}"}
    if kind == "command":
        command = str(action["command"])
        root = _host_root()
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            def enter_host_root() -> None:
                os.chroot(str(root))
                os.chdir("/")
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_CPU, (MAX_COMMAND_SECONDS, MAX_COMMAND_SECONDS))
                    resource.setrlimit(resource.RLIMIT_FSIZE, (5 * 1024 * 1024, 5 * 1024 * 1024))
                    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
                except (ImportError, OSError, ValueError):
                    pass

            try:
                completed = subprocess.run(
                    ["/bin/sh", "-lc", command],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    cwd="/",
                    env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root"},
                    timeout=MAX_COMMAND_SECONDS,
                    check=False,
                    preexec_fn=enter_host_root,
                )
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(status_code=504, detail="Confirmed host command timed out") from exc
            stdout_file.seek(0)
            stderr_file.seek(0)
            output = (stdout_file.read(128_000) + stderr_file.read(128_000)).decode("utf-8", errors="replace")
        return {"success": completed.returncode == 0, "exitCode": completed.returncode, "output": output[:MAX_OUTPUT_CHARS]}
    raise HTTPException(status_code=400, detail="Unknown host action")


def _propose_host_action(uid: str, action: dict[str, Any]) -> dict[str, Any]:
    action_id = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + ACTION_TTL_SECONDS
    preview = {key: value for key, value in action.items() if key != "uid"}
    with _PENDING_GUARD:
        _PENDING_HOST_ACTIONS[action_id] = {"uid": uid, "action": action, "expiresAt": expires_at}
    return {"confirmationRequired": True, "actionId": action_id, "action": preview, "expiresAt": expires_at}


def _prune_pending_actions() -> None:
    now = int(time.time())
    expired = [key for key, value in _PENDING_HOST_ACTIONS.items() if value.get("expiresAt", 0) < now]
    for key in expired:
        _PENDING_HOST_ACTIONS.pop(key, None)


def _run_workspace_command(uid: str, command: str) -> dict[str, Any]:
    if not command.strip():
        raise HTTPException(status_code=400, detail="Command is required")
    if len(command) > 20_000:
        raise HTTPException(status_code=413, detail="Command is too long")
    if not SUPERVISOR_KEY:
        raise HTTPException(status_code=503, detail="Workspace supervisor is not configured")
    normalized_uid = _uid(uid)
    workspace = _workspace_root(normalized_uid)
    host_workspace, docker_workspace = _host_workspace_bind_paths(normalized_uid, workspace)
    try:
        host_workspace.chmod(0o700)
    except OSError:
        pass
    try:
        import docker
        from docker.errors import DockerException, NotFound
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Docker client is not available in the supervisor") from exc
    try:
        client = docker.from_env(timeout=15)
        container = client.containers.create(
            image=TASK_IMAGE,
            # Docker SDK tokenizes string commands; preserve the complete shell script as one argv item.
            command=[command],
            entrypoint=["/bin/sh", "-lc"],
            user="1000:1000",
            working_dir="/data/workspace",
            volumes={str(docker_workspace): {"bind": "/data/workspace", "mode": "rw"}},
            network_mode="none",
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            pids_limit=128,
            mem_limit="1g",
            nano_cpus=1_000_000_000,
            tmpfs={"/tmp": "rw,noexec,nosuid,nodev,size=64m"},
            log_config={"type": "json-file", "config": {"max-size": "1m", "max-file": "1"}},
            labels={"com.wartha.workspace.uid": normalized_uid, "com.wartha.workspace.task": "isolated"},
            detach=True,
            auto_remove=False,
        )
        try:
            container.start()
            try:
                status = container.wait(timeout=MAX_COMMAND_SECONDS)
            except Exception as exc:
                try:
                    container.kill()
                except Exception:
                    pass
                if "timeout" in str(exc).lower():
                    raise HTTPException(status_code=504, detail="Workspace command timed out") from exc
                raise
            output = container.logs(stdout=True, stderr=True, tail=1000).decode("utf-8", errors="replace")
            exit_code = int(status.get("StatusCode", 1)) if isinstance(status, dict) else 1
            return {"success": exit_code == 0, "exitCode": exit_code, "output": output[-MAX_OUTPUT_CHARS:], "workspace": str(workspace)}
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass
            client.close()
    except HTTPException:
        raise
    except (DockerException, OSError) as exc:
        logger.warning("Isolated workspace command unavailable (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Isolated workspace executor is unavailable") from exc


def _headers(internal_key: str | None) -> None:
    _authorize(internal_key)


@app.get("/health")
def health():
    return {"status": "healthy", "public": False}


@app.post("/internal/v1/workspace/list")
def workspace_list(payload: WorkspaceRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    with _lock(uid):
        return _list_workspace(uid, payload.path)


@app.post("/internal/v1/workspace/read")
def workspace_read(payload: WorkspaceRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    with _lock(uid):
        return _read_workspace_file(uid, payload.path)


@app.post("/internal/v1/workspace/write")
def workspace_write(payload: WorkspaceWriteRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    with _lock(uid):
        return _write_workspace_file(uid, payload.path, payload.content)


@app.post("/internal/v1/workspace/move")
def workspace_move(payload: WorkspaceMoveRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    with _lock(uid):
        return _move_workspace_file(uid, payload.source, payload.destination)


@app.post("/internal/v1/workspace/delete")
def workspace_delete(payload: WorkspaceRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    with _lock(uid):
        return _delete_workspace_file(uid, payload.path)


@app.post("/internal/v1/workspace/command")
def workspace_command(payload: WorkspaceCommandRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    with _lock(uid):
        return _run_workspace_command(uid, payload.command)


@app.post("/internal/v1/workspace/packages")
def workspace_packages(payload: WorkspaceRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    with _lock(uid):
        return _list_packages(uid)


@app.post("/internal/v1/workspace/packages/request")
def workspace_package_request(payload: PackageRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    with _lock(uid):
        return _request_package(uid, payload.package)


@app.post("/internal/v1/admin/packages")
def admin_packages(payload: WorkspaceRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    return {"requests": _all_package_requests(), "sharedPackages": sorted(_shared_packages())}


@app.post("/internal/v1/admin/packages/review")
def admin_review_package(payload: PackageReviewRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    owner_uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(owner_uid, x_agent_user_role, x_agent_owner_uid)
    target_uid = _uid(payload.target_uid)
    with _lock(target_uid):
        return _review_package(target_uid, payload.request_id, payload.decision)


@app.post("/internal/v1/models/local")
def local_models(payload: EmptyRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    owner_uid = _uid(x_agent_owner_uid or "")
    return {"models": _local_models(owner_uid)}


@app.post("/internal/v1/admin/models/local")
def admin_local_models(payload: LocalModelRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    owner_uid = _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    return _update_local_models(owner_uid, payload.model, payload.action)


@app.post("/internal/v1/owner/list")
def owner_list(payload: HostReadRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    return _list_host(payload.path)


@app.post("/internal/v1/owner/read")
def owner_read(payload: HostReadRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    return _read_host(payload.path)


@app.post("/internal/v1/owner/propose/write")
def owner_propose_write(payload: HostWriteRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    target = _host_path(payload.path, allow_missing=True)
    if target == _host_root() or target.is_symlink() or not target.parent.is_dir():
        raise HTTPException(status_code=400, detail="Unsafe host write target")
    return _propose_host_action(uid, {"type": "write", "path": "/" + target.relative_to(_host_root()).as_posix(), "content": payload.content})


@app.post("/internal/v1/owner/propose/move")
def owner_propose_move(payload: HostMoveRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    source = _host_path(payload.source)
    destination = _host_path(payload.destination, allow_missing=True)
    if source == _host_root() or source.is_symlink() or destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise HTTPException(status_code=400, detail="Unsafe host move target")
    return _propose_host_action(uid, {"type": "move", "source": "/" + source.relative_to(_host_root()).as_posix(), "destination": "/" + destination.relative_to(_host_root()).as_posix()})


@app.post("/internal/v1/owner/propose/delete")
def owner_propose_delete(payload: HostDeleteRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    target = _host_path(payload.path)
    if target == _host_root() or target.is_symlink():
        raise HTTPException(status_code=400, detail="Unsafe host delete target")
    return _propose_host_action(uid, {"type": "delete", "path": "/" + target.relative_to(_host_root()).as_posix()})


@app.post("/internal/v1/owner/propose/command")
def owner_propose_command(payload: HostCommandRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    _host_root()
    return _propose_host_action(uid, {"type": "command", "command": payload.command})


@app.post("/internal/v1/owner/confirm")
def owner_confirm(payload: HostConfirmRequest, x_workspace_internal_key: str | None = Header(default=None), x_agent_user_uid: str | None = Header(default=None), x_agent_user_role: str | None = Header(default=None), x_agent_owner_uid: str | None = Header(default=None)):
    uid = _trusted_user_uid(x_workspace_internal_key, x_agent_user_uid)
    _require_owner(uid, x_agent_user_role, x_agent_owner_uid)
    with _PENDING_GUARD:
        _prune_pending_actions()
        pending = _PENDING_HOST_ACTIONS.pop(payload.action_id, None)
    if not pending or pending.get("uid") != uid:
        raise HTTPException(status_code=404, detail="Confirmation is expired or invalid")
    return _perform_host_action(pending["action"])
