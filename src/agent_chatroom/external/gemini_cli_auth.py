import asyncio
import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"

# Module-level caches
_cached_client_credentials: Optional[Tuple[str, str]] = None
_cached_access_token: Optional[str] = None
_cached_token_expiry: float = 0.0
_token_lock: Optional[asyncio.Lock] = None
_token_lock_init = threading.Lock()


def _get_token_lock() -> asyncio.Lock:
    global _token_lock
    if _token_lock is None:
        with _token_lock_init:
            if _token_lock is None:
                _token_lock = asyncio.Lock()
    return _token_lock


def find_gemini_cli() -> Optional[str]:
    return shutil.which("gemini")


def _resolve_gemini_cli_dirs(gemini_path: str) -> list[str]:
    resolved = os.path.realpath(gemini_path)
    bin_dir = os.path.dirname(gemini_path)
    candidates = [
        os.path.dirname(os.path.dirname(resolved)),
        os.path.join(os.path.dirname(resolved), "node_modules", "@google", "gemini-cli"),
        os.path.join(bin_dir, "node_modules", "@google", "gemini-cli"),
        os.path.join(os.path.dirname(bin_dir), "node_modules", "@google", "gemini-cli"),
        os.path.join(os.path.dirname(bin_dir), "lib", "node_modules", "@google", "gemini-cli"),
    ]
    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _find_file_recursive(directory: str, name: str, max_depth: int, _root: Optional[str] = None) -> Optional[str]:
    if max_depth <= 0:
        return None
    if _root is None:
        _root = os.path.realpath(directory)
    try:
        for entry in os.scandir(directory):
            real_path = os.path.realpath(entry.path)
            if not real_path.startswith(_root + os.sep) and real_path != _root:
                continue
            if entry.is_file() and entry.name == name:
                return real_path
            if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                result = _find_file_recursive(entry.path, name, max_depth - 1, _root)
                if result:
                    return result
    except (PermissionError, OSError):
        pass
    return None


def extract_client_credentials() -> Tuple[str, str]:
    global _cached_client_credentials
    if _cached_client_credentials:
        return _cached_client_credentials

    env_id = os.environ.get("GEMINI_CLI_OAUTH_CLIENT_ID", "").strip()
    env_secret = os.environ.get("GEMINI_CLI_OAUTH_CLIENT_SECRET", "").strip()
    if env_id and env_secret:
        _cached_client_credentials = (env_id, env_secret)
        return _cached_client_credentials

    gemini_path = find_gemini_cli()
    if not gemini_path:
        raise RuntimeError("Gemini CLI not found in PATH.")

    cli_dirs = _resolve_gemini_cli_dirs(gemini_path)
    content = None

    for cli_dir in cli_dirs:
        search_paths = [
            os.path.join(cli_dir, "node_modules", "@google", "gemini-cli-core", "dist", "src", "code_assist", "oauth2.js"),
        ]
        for p in search_paths:
            if os.path.isfile(p):
                content = Path(p).read_text(encoding="utf-8")
                break
        if content:
            break
        found = _find_file_recursive(cli_dir, "oauth2.js", 10)
        if found:
            content = Path(found).read_text(encoding="utf-8")
            break

    if not content:
        raise RuntimeError("Could not find oauth2.js in Gemini CLI installation.")

    id_match = re.search(r"(\d+-[a-z0-9]+\.apps\.googleusercontent\.com)", content)
    secret_match = re.search(r"(GOCSPX-[A-Za-z0-9_-]+)", content)

    if not id_match or not secret_match:
        raise RuntimeError("Could not extract client_id/client_secret from oauth2.js.")

    _cached_client_credentials = (id_match.group(1), secret_match.group(1))
    return _cached_client_credentials


def read_oauth_creds() -> dict:
    if not OAUTH_CREDS_PATH.exists():
        raise RuntimeError(f"OAuth credentials not found at {OAUTH_CREDS_PATH}.")
    data = json.loads(OAUTH_CREDS_PATH.read_text(encoding="utf-8"))
    if "refresh_token" not in data:
        raise RuntimeError(f"No refresh_token in {OAUTH_CREDS_PATH}.")
    return data


async def refresh_access_token(refresh_token: str, client_id: str, client_secret: str) -> Tuple[str, float]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Token refresh failed ({resp.status})")
            data = await resp.json()
            access_token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)
            expires_at = time.time() + expires_in - 300
            return access_token, expires_at


async def invalidate_cached_token() -> None:
    global _cached_access_token, _cached_token_expiry
    async with _get_token_lock():
        _cached_access_token = None
        _cached_token_expiry = 0.0


async def get_valid_access_token() -> str:
    global _cached_access_token, _cached_token_expiry
    if _cached_access_token and time.time() < _cached_token_expiry:
        return _cached_access_token

    async with _get_token_lock():
        if _cached_access_token and time.time() < _cached_token_expiry:
            return _cached_access_token

        creds = read_oauth_creds()
        client_id, client_secret = extract_client_credentials()

        existing_token = creds.get("access_token")
        expiry_date = creds.get("expiry_date", 0)
        if existing_token and expiry_date > 0 and (expiry_date / 1000) > (time.time() + 300):
            _cached_access_token = existing_token
            _cached_token_expiry = expiry_date / 1000 - 300
            return _cached_access_token

        new_token, expires_at = await refresh_access_token(creds["refresh_token"], client_id, client_secret)
        _cached_access_token = new_token
        _cached_token_expiry = expires_at
        return _cached_access_token
