import asyncio
import hashlib
import json
import logging
import os
import uuid
from typing import Optional

import aiohttp

from .gemini_cli_auth import get_valid_access_token
from ..api_throttle import wait_after_gemini_response, wait_for_gemini_slot

logger = logging.getLogger(__name__)

# Cloud Code Assist endpoint
API_BASE = "https://cloudcode-pa.googleapis.com"
API_VERSION = "v1internal"

# Reusable aiohttp session for connection pooling
_aio_session: Optional[aiohttp.ClientSession] = None


def _get_aio_session() -> aiohttp.ClientSession:
    global _aio_session
    if _aio_session is None or _aio_session.closed:
        _aio_session = aiohttp.ClientSession()
    return _aio_session


# Code Assist endpoint model names
CLI_PRO = "gemini-3.1-pro-preview"
CLI_FLASH = "gemini-3-flash-preview"
CLI_PRO_25 = "gemini-2.5-pro"
CLI_FLASH_25 = "gemini-2.5-flash"
CLI_DEFAULT = CLI_FLASH

CLI_PRO_FALLBACK_CHAIN = [CLI_PRO, CLI_FLASH, CLI_FLASH_25]
CLI_FLASH_FALLBACK_CHAIN = [CLI_FLASH, CLI_FLASH_25]

# Map standard gemini_client model names -> Code Assist model names
_MODEL_MAP = {
    "gemini-3-flash": CLI_FLASH,
    "gemini-3.0-flash": CLI_FLASH,
    "gemini-3.1-pro-preview": CLI_PRO,
    "gemini-3.1-flash-lite-preview": CLI_FLASH_25,
}


def _map_model(model: str) -> str:
    return _MODEL_MAP.get(model, model)


def _supports_thinking(model: str) -> bool:
    return model != CLI_FLASH_25


# Cached project ID (discovered via loadCodeAssist)
_cached_project_id: Optional[str] = None
_project_lock: Optional[asyncio.Lock] = None
_broker_lock: Optional[asyncio.Lock] = None
_inflight_requests: dict[str, asyncio.Task[str]] = {}
_request_semaphore: Optional[asyncio.Semaphore] = None


def _get_project_lock() -> asyncio.Lock:
    global _project_lock
    if _project_lock is None:
        _project_lock = asyncio.Lock()
    return _project_lock


def _get_broker_lock() -> asyncio.Lock:
    global _broker_lock
    if _broker_lock is None:
        _broker_lock = asyncio.Lock()
    return _broker_lock


def _get_request_semaphore() -> asyncio.Semaphore:
    global _request_semaphore
    if _request_semaphore is None:
        max_concurrency = max(
            1,
            int(os.environ.get("GEMINI_BROKER_MAX_CONCURRENCY", "1") or "1"),
        )
        _request_semaphore = asyncio.Semaphore(max_concurrency)
    return _request_semaphore


async def _build_headers() -> dict:
    token = await get_valid_access_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def _invalidate_and_rebuild_headers() -> dict:
    from .gemini_cli_auth import invalidate_cached_token
    await invalidate_cached_token()
    return await _build_headers()


async def _discover_project_id(max_retries: int = 3) -> str:
    global _cached_project_id
    if _cached_project_id:
        return _cached_project_id

    # Check env vars first
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
    if env_project:
        _cached_project_id = env_project
        logger.info(f"[GeminiCLI] Using project from env: {env_project}")
        return _cached_project_id

    async with _get_project_lock():
        if _cached_project_id:
            return _cached_project_id

        url = f"{API_BASE}/{API_VERSION}:loadCodeAssist"
        body = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }

        session = _get_aio_session()
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            headers = await _build_headers()
            try:
                await wait_for_gemini_slot()
                async with session.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status in (401, 403):
                        logger.warning(
                            "[GeminiCLI] loadCodeAssist got %s, refreshing token and retrying...",
                            resp.status,
                        )
                        headers = await _invalidate_and_rebuild_headers()
                        await wait_for_gemini_slot()
                        async with session.post(
                            url,
                            headers=headers,
                            json=body,
                            timeout=aiohttp.ClientTimeout(total=20),
                        ) as retry_resp:
                            if retry_resp.status != 200:
                                error_body = await retry_resp.text()
                                raise RuntimeError(
                                    f"loadCodeAssist failed ({retry_resp.status}) after token refresh: "
                                    f"{error_body[:500]}"
                                )
                            data = await retry_resp.json()
                    else:
                        if resp.status in (429,) or resp.status >= 500:
                            error_body = await resp.text()
                            raise RuntimeError(
                                f"loadCodeAssist failed ({resp.status}): {error_body[:500]}"
                            )
                        if resp.status != 200:
                            error_body = await resp.text()
                            raise RuntimeError(
                                f"loadCodeAssist failed ({resp.status}): {error_body[:500]}"
                            )
                        data = await resp.json()
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries - 1:
                    break
                wait_time = 2 ** attempt
                logger.warning(
                    "[GeminiCLI] Project discovery attempt %s/%s failed: %s. Retrying in %ss.",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
                continue

            project_id = data.get("cloudaicompanionProject")
            if project_id:
                _cached_project_id = project_id
                logger.info(f"[GeminiCLI] Discovered project: {project_id}")
                return _cached_project_id

            last_error = RuntimeError("Could not discover project ID from loadCodeAssist.")
            if attempt >= max_retries - 1:
                break

        raise RuntimeError("Could not discover project ID from loadCodeAssist.") from last_error


def _request_cache_key(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_instruction: Optional[str],
    thinking_level: str,
    use_google_search: bool,
    enable_fallback: bool,
) -> str:
    payload = {
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_instruction": system_instruction or "",
        "thinking_level": thinking_level,
        "use_google_search": use_google_search,
        "enable_fallback": enable_fallback,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


async def warmup_gemini_cli() -> None:
    """Warm up token/session/project discovery for the long-lived worker process."""
    await _build_headers()
    await _discover_project_id()


async def close_gemini_cli_client() -> None:
    global _aio_session
    if _aio_session is not None and not _aio_session.closed:
        await _aio_session.close()
    _aio_session = None

    async with _get_broker_lock():
        tasks = list(_inflight_requests.values())
        _inflight_requests.clear()

    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _query_gemini_cli_uncached(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_instruction: Optional[str],
    thinking_level: str,
    use_google_search: bool,
    enable_fallback: bool,
) -> str:
    async with _get_request_semaphore():
        # Build fallback chain
        if enable_fallback:
            if model in (CLI_PRO, CLI_PRO_25):
                models_to_try = [model] + [m for m in CLI_PRO_FALLBACK_CHAIN if m != model]
            elif model in (CLI_FLASH, CLI_FLASH_25):
                models_to_try = [model] + [m for m in CLI_FLASH_FALLBACK_CHAIN if m != model]
            else:
                models_to_try = [model] + CLI_PRO_FALLBACK_CHAIN
        else:
            models_to_try = [model]

        last_error = None
        for current_model in models_to_try:
            try:
                logger.info(
                    "[GeminiCLI] Trying model=%s use_google_search=%s",
                    current_model,
                    use_google_search,
                )
                result = await _call_gemini_rest(
                    prompt=prompt,
                    model=current_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=system_instruction,
                    thinking_level=thinking_level,
                    use_google_search=use_google_search,
                )
                if current_model != model:
                    logger.info(
                        "[GeminiCLI] Fallback success original_model=%s resolved_model=%s",
                        model,
                        current_model,
                    )
                else:
                    logger.info("[GeminiCLI] Primary model succeeded model=%s", current_model)
                return result
            except Exception as e:
                error_str = str(e)
                is_retryable = (
                    "429" in error_str
                    or "503" in error_str
                    or "RESOURCE_EXHAUSTED" in error_str
                )
                if is_retryable and enable_fallback:
                    logger.warning(
                        "[GeminiCLI] %s failed (%s), trying next model...",
                        current_model,
                        error_str[:100],
                    )
                    last_error = e
                    continue
                logger.error(
                    "[GeminiCLI] Non-retryable model failure model=%s error=%s",
                    current_model,
                    error_str[:200],
                )
                raise

        raise RuntimeError("All models in fallback chain failed") from last_error


async def query_gemini_cli(
    prompt: str,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    system_instruction: Optional[str] = None,
    thinking_level: str = "NONE",
    use_google_search: bool = False,
    enable_fallback: bool = True,
) -> str:
    """
    Query Gemini API using OAuth credentials from Gemini CLI.
    """
    if not model:
        model = CLI_DEFAULT
    else:
        model = _map_model(model)

    logger.info(
        "[GeminiCLI] Request start model=%s use_google_search=%s prompt_chars=%s system_chars=%s max_tokens=%s",
        model,
        use_google_search,
        len(prompt or ""),
        len(system_instruction or ""),
        max_tokens,
    )

    request_key = _request_cache_key(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_instruction=system_instruction,
        thinking_level=thinking_level,
        use_google_search=use_google_search,
        enable_fallback=enable_fallback,
    )

    created_task = False
    async with _get_broker_lock():
        task = _inflight_requests.get(request_key)
        if task is None:
            task = asyncio.create_task(
                _query_gemini_cli_uncached(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=system_instruction,
                    thinking_level=thinking_level,
                    use_google_search=use_google_search,
                    enable_fallback=enable_fallback,
                )
            )
            _inflight_requests[request_key] = task
            created_task = True
        else:
            logger.info("[GeminiCLI] Coalescing duplicate in-flight Gemini request.")

    try:
        result = await asyncio.shield(task)
        logger.info(
            "[GeminiCLI] Request success model=%s use_google_search=%s text_chars=%s",
            model,
            use_google_search,
            len(result or ""),
        )
        return result
    finally:
        if created_task:
            async with _get_broker_lock():
                if _inflight_requests.get(request_key) is task:
                    _inflight_requests.pop(request_key, None)


async def _call_gemini_rest(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_instruction: Optional[str],
    thinking_level: str,
    use_google_search: bool,
    max_retries: int = 5,
) -> str:
    import asyncio
    project_id = await _discover_project_id()
    url = f"{API_BASE}/{API_VERSION}:generateContent"

    inner_request: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    if system_instruction:
        inner_request["systemInstruction"] = {
            "role": "user",
            "parts": [{"text": system_instruction}],
        }

    if thinking_level and thinking_level.upper() != "NONE" and _supports_thinking(model):
        inner_request["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": thinking_level.upper(),
        }

    if use_google_search:
        inner_request["tools"] = [{"googleSearch": {}}]

    body = {
        "model": model,
        "project": project_id,
        "user_prompt_id": str(uuid.uuid4()),
        "request": inner_request,
    }

    session = _get_aio_session()
    
    for attempt in range(max_retries):
        headers = await _build_headers()
        await wait_for_gemini_slot()
        async with session.post(
            url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status in (401, 403):
                logger.warning(f"[GeminiCLI] Got {resp.status}, refreshing token and retrying...")
                headers = await _invalidate_and_rebuild_headers()
                await wait_for_gemini_slot()
                async with session.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as retry_resp:
                    if retry_resp.status != 200:
                        error_body = await retry_resp.text()
                        raise RuntimeError(
                            f"Gemini API error {retry_resp.status} after token refresh: {error_body[:500]}"
                        )
                    data = await retry_resp.json()
                    await wait_after_gemini_response()
                    return _extract_text(data)

            if resp.status == 429 or resp.status >= 500:
                error_body = await resp.text()
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"[GeminiCLI] Rate limited/Server Error ({resp.status}). Retrying in {wait_time}s... Error: {error_body[:100]}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Gemini API error {resp.status} after {max_retries} retries: {error_body[:500]}")

            if resp.status != 200:
                error_body = await resp.text()
                raise RuntimeError(f"Gemini API error {resp.status}: {error_body[:500]}")

            data = await resp.json()
            await wait_after_gemini_response()
            return _extract_text(data)
            
    raise RuntimeError("Unexpected end of retry loop")


def _extract_text(response_data: dict) -> str:
    inner_response = response_data.get("response", response_data)
    candidates = inner_response.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Empty response from Gemini API: {json.dumps(response_data)[:500]}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = []
    for part in parts:
        if "text" in part:
            text_parts.append(part["text"])

    if not text_parts:
        raise RuntimeError(f"No text in Gemini response: {json.dumps(response_data)[:500]}")

    return "\n".join(text_parts)
