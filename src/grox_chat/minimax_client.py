import os
import httpx
import logging
import re
import contextvars
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
import asyncio
from .api_throttle import wait_after_minimax_response, wait_for_minimax_slot

load_dotenv()

TIMEOUT = 300.0

logger = logging.getLogger(__name__)

# Reusable httpx client for connection pooling
_http_client: Optional[httpx.AsyncClient] = None
_main_semaphore: Optional[asyncio.Semaphore] = None
_daemon_semaphore: Optional[asyncio.Semaphore] = None

# ContextVar: set to True inside daemon tasks so they use the dedicated daemon semaphore
is_daemon_channel: contextvars.ContextVar[bool] = contextvars.ContextVar("minimax_daemon_channel", default=False)
MINIMAX_TOOL_BLOCK_RE = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
MINIMAX_INVOKE_RE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.DOTALL)
MINIMAX_PARAM_RE = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.DOTALL)
ENGLISH_ONLY_INSTRUCTION = "Respond in English only."


def _get_minimax_api_key() -> Optional[str]:
    return os.getenv("MINIMAX_API_KEY")


def _use_international_minimax() -> bool:
    return os.getenv("MINIMAX_EN", "0") == "1"


def _get_minimax_api_host() -> str:
    return "https://api.minimax.io" if _use_international_minimax() else "https://api.minimaxi.com"


def _get_minimax_message_url() -> str:
    return f"{_get_minimax_api_host()}/anthropic/v1/messages"


def _get_minimax_coding_plan_base() -> str:
    return _get_minimax_api_host()

def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=TIMEOUT)
    return _http_client


def _get_request_semaphore() -> asyncio.Semaphore:
    global _main_semaphore, _daemon_semaphore
    max_concurrent = int(os.getenv("MINIMAX_MAX_CONCURRENT", "4"))
    if is_daemon_channel.get():
        if _daemon_semaphore is None:
            _daemon_semaphore = asyncio.Semaphore(1)
            logger.info("[MiniMax] Daemon semaphore initialized: max_concurrent=1")
        return _daemon_semaphore
    if _main_semaphore is None:
        main_slots = max(1, max_concurrent - 1)
        _main_semaphore = asyncio.Semaphore(main_slots)
        logger.info("[MiniMax] Main semaphore initialized: max_concurrent=%d", main_slots)
    return _main_semaphore


async def close_minimax_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


def _reinforce_minimax_prompt(system_prompt: str, question: str) -> tuple[str, str]:
    base_system_prompt = (system_prompt or "").strip()
    if ENGLISH_ONLY_INSTRUCTION not in base_system_prompt:
        base_system_prompt = (
            f"{base_system_prompt}\n\n{ENGLISH_ONLY_INSTRUCTION}".strip()
            if base_system_prompt
            else ENGLISH_ONLY_INSTRUCTION
        )

    body = (question or "").strip()
    if base_system_prompt:
        body = f"{base_system_prompt}\n\n{body}\n\n{base_system_prompt}".strip()
    return base_system_prompt, body


def _extract_pseudo_tool_markup(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Strip MiniMax pseudo-tool XML and surface any embedded tool metadata."""
    raw_text = text or ""
    if "<minimax:tool_call>" not in raw_text:
        return raw_text.strip(), []

    recovered_tools: List[Dict[str, Any]] = []

    def _replace_block(match: re.Match[str]) -> str:
        block = match.group(1)
        for tool_name, body in MINIMAX_INVOKE_RE.findall(block):
            params: Dict[str, Any] = {}
            for param_name, param_value in MINIMAX_PARAM_RE.findall(body):
                cleaned_value = param_value.strip()
                if cleaned_value:
                    params[param_name] = cleaned_value
            recovered_tools.append({
                "type": "tool_use",
                "name": tool_name,
                "input": params,
            })
        return ""

    stripped_text = MINIMAX_TOOL_BLOCK_RE.sub(_replace_block, raw_text).strip()
    if stripped_text:
        return stripped_text, recovered_tools

    return "", recovered_tools


def _recover_queries_from_tools(tools: List[Dict[str, Any]]) -> str:
    recovered_queries = []
    for tool in tools:
        query = tool.get("input", {}).get("query")
        if isinstance(query, str) and query.strip():
            recovered_queries.append(query.strip())
    return "\n".join(recovered_queries)


def _extract_text_and_tools(data: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts text and tool_use blocks from the response content."""
    content_blocks = data.get("content", [])
    text_parts = []
    tools = []
    
    for block in content_blocks:
        if block.get("type") == "text":
            cleaned_text, pseudo_tools = _extract_pseudo_tool_markup(block.get("text", ""))
            if cleaned_text:
                text_parts.append(cleaned_text)
            tools.extend(pseudo_tools)
        elif block.get("type") == "tool_use":
            tools.append(block)
            
    return "\n".join(text_parts).strip(), tools

async def query_minimax(
    system_prompt: str,
    question: str,
    model: str = "MiniMax-M2.5",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 3,
    recover_pseudo_tool_query: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Query MiniMax API and return a tuple of (raw_text, tool_calls).
    Filters out the thinking blocks automatically. Includes throttling.
    The Anthropic-compatible MiniMax endpoint is used as text generation only;
    the separate Coding Plan APIs handle search/VLM capabilities.
    """
    api_key = _get_minimax_api_key()
    if not api_key:
        logger.error("No MiniMax API key found in .env")
        return "Error: No API key.", []

    system_prompt, question = _reinforce_minimax_prompt(system_prompt, question)
    logger.info(
        "[MiniMax] Request start model=%s prompt_chars=%s system_chars=%s max_tokens=%s recover_pseudo_tool_query=%s",
        model,
        len(question or ""),
        len(system_prompt or ""),
        max_tokens,
        recover_pseudo_tool_query,
    )

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": question}],
    }

    async with _get_request_semaphore():
        for attempt in range(max_retries):
            await wait_for_minimax_slot()
            try:
                client = _get_http_client()
                resp = await client.post(_get_minimax_message_url(), headers=headers, json=payload)

                if resp.status_code == 429:
                    logger.warning("[MiniMax] Rate limited (429). Retrying %s/%s...", attempt + 1, max_retries)
                    await asyncio.sleep(2 ** attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()

                text, tool_calls = _extract_text_and_tools(data)
                if not text and tool_calls and recover_pseudo_tool_query:
                    recovered_query = _recover_queries_from_tools(tool_calls)
                    if recovered_query:
                        await wait_after_minimax_response()
                        logger.info(
                            "[MiniMax] Request success via pseudo-tool recovery model=%s recovered_query_chars=%s tool_calls=%s",
                            model,
                            len(recovered_query),
                            len(tool_calls),
                        )
                        return recovered_query, tool_calls
                if not text and tool_calls:
                    return "Error: MiniMax emitted pseudo-tool markup in text-only mode", tool_calls
                if not text and not tool_calls:
                    return "Error: Empty response", []
                await wait_after_minimax_response()
                logger.info(
                    "[MiniMax] Request success model=%s text_chars=%s tool_calls=%s",
                    model,
                    len(text or ""),
                    len(tool_calls),
                )
                return text, tool_calls

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    logger.warning("[MiniMax] Server error (%s). Retrying %s/%s...", e.response.status_code, attempt + 1, max_retries)
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error("[MiniMax] HTTP error %s (response body omitted for safety)", e.response.status_code)
                return f"Error: HTTP {e.response.status_code}", []
            except Exception as e:
                logger.error("[MiniMax] Request error: %s", type(e).__name__)
                if attempt == max_retries - 1:
                    return f"Error: request failed after {max_retries} attempts", []
                await asyncio.sleep(2 ** attempt)

    return "Error: Max retries exceeded.", []

async def minimax_search(query: str, timeout: float = 60.0, max_retries: int = 3) -> dict:
    """Call MiniMax Coding Plan web search API."""
    api_key = _get_minimax_api_key()
    if not api_key:
        return {"error": "No MiniMax API key."}
    logger.info("[MiniMax Search] Request start query=%r timeout=%s", query, timeout)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "MM-API-Source": "custom-chatroom",
        "Content-Type": "application/json",
    }

    async with _get_request_semaphore():
        for attempt in range(max_retries):
            await wait_for_minimax_slot()
            try:
                client = _get_http_client()
                resp = await client.post(
                    f"{_get_minimax_coding_plan_base()}/v1/coding_plan/search",
                    headers=headers,
                    json={"q": query},
                    timeout=timeout
                )

                if resp.status_code == 429:
                    logger.warning("[MiniMax Search] Rate limited (429). Retrying %s/%s...", attempt + 1, max_retries)
                    await asyncio.sleep(2 ** attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()
                await wait_after_minimax_response()
                logger.info(
                    "[MiniMax Search] Request success query=%r organic_results=%s",
                    query,
                    len(data.get("organic", []) or []),
                )
                return data
            except Exception as e:
                logger.error("[MiniMax Search] Request error: %s", type(e).__name__)
                if attempt == max_retries - 1:
                    return {"error": f"request failed after {max_retries} attempts"}
                await asyncio.sleep(2 ** attempt)

    return {"error": "Max retries exceeded"}
