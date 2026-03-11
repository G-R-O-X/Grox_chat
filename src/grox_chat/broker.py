from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass

from .external.gemini_cli_client import (
    close_gemini_cli_client,
    query_gemini_cli,
)
from .minimax_client import (
    close_minimax_client,
    minimax_search,
    query_minimax,
)
from .prompts import PROMPTS

logger = logging.getLogger(__name__)

SEARCH_QUERY_SENTINEL = "NO_SEARCH"


@dataclass(frozen=True)
class BrokerRequest:
    prompt: str
    system_instruction: str = ""
    provider: str = "minimax"
    strategy: str = "direct"
    allow_web: bool = False
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 8192
    fallback_role: str = "agent"
    recover_pseudo_tool_query: bool = False

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=True)


_broker_lock = asyncio.Lock()
_broker_semaphore = asyncio.Semaphore(8)
_inflight_requests: dict[str, asyncio.Task[str]] = {}


def _raise_on_minimax_error(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("Error:"):
        raise RuntimeError(stripped)
    return stripped


def _normalize_search_query(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    if first_line.startswith("Error:"):
        return ""
    if first_line.upper() == SEARCH_QUERY_SENTINEL:
        return ""
    if first_line.startswith('"') and first_line.endswith('"') and len(first_line) >= 2:
        first_line = first_line[1:-1].strip()
    if first_line.upper() == SEARCH_QUERY_SENTINEL:
        return ""
    if first_line.startswith("{") or first_line.startswith("["):
        return ""
    if len(first_line) > 200:
        first_line = first_line[:200].strip()
    if first_line.upper().startswith(f"{SEARCH_QUERY_SENTINEL}:"):
        return ""
    return first_line


def _render_search_results(search_res: dict) -> str:
    rendered = "=== WEB SEARCH RESULTS ===\n"
    if "organic" in search_res:
        for org in search_res["organic"][:3]:
            rendered += f"Title: {org.get('title')}\nSnippet: {org.get('snippet')}\n\n"
    else:
        rendered += "No useful results found.\n\n"
    return rendered


def _build_search_decision_prompt(initial_prompt: str, rendered_results: list[str]) -> str:
    prompt = initial_prompt
    if rendered_results:
        prompt += "\n\n" + "\n".join(rendered_results)
    prompt += (
        "\n\nBased on the current context and any search results above, either output a new search query "
        f"if more evidence is required, or output {SEARCH_QUERY_SENTINEL} if you have enough information."
    )
    return prompt


def _build_final_answer_prompt(initial_prompt: str, rendered_results: list[str]) -> str:
    if not rendered_results:
        return initial_prompt
    return initial_prompt + "\n\n" + "\n".join(rendered_results)


async def _decide_search_query(agent_role: str, current_prompt: str, system_prompt: str) -> str:
    decision_system_prompt = (
        f"{system_prompt}\n\n"
        "You are deciding whether web search is necessary before answering.\n"
        f"Reply with exactly one short search query string, or `{SEARCH_QUERY_SENTINEL}` if search is unnecessary.\n"
        "Do not output JSON. Do not explain your reasoning. Do not add extra text."
    )
    raw_text, _ = await query_minimax(
        system_prompt=decision_system_prompt,
        question=current_prompt,
        max_tokens=8192,
        recover_pseudo_tool_query=True,
    )
    return _normalize_search_query(raw_text)


async def collect_search_evidence(
    agent_role: str,
    initial_prompt: str,
    max_iter: int = 2,
    system_prompt: str | None = None,
) -> tuple[list[str], bool]:
    system_prompt = system_prompt or PROMPTS.get(agent_role, "")
    rendered_results: list[str] = []
    search_failed = False

    for i in range(max_iter):
        logger.info("[%s] ReAct Loop Iteration %s/%s", agent_role, i + 1, max_iter)
        decision_prompt = _build_search_decision_prompt(initial_prompt, rendered_results)
        query = await _decide_search_query(agent_role, decision_prompt, system_prompt)
        if not query:
            return rendered_results, search_failed

        logger.info("[%s] Executing web search for: '%s'", agent_role, query)
        search_res = await minimax_search(query)
        if "error" in search_res:
            search_failed = True
        rendered_results.append(_render_search_results(search_res))

    logger.warning("[%s] Max iterations (%s) reached in search loop.", agent_role, max_iter)
    return rendered_results, search_failed


async def react_search_loop(
    agent_role: str,
    initial_prompt: str,
    max_iter: int = 2,
    system_prompt: str | None = None,
) -> tuple[str, bool]:
    system_prompt = system_prompt or PROMPTS.get(agent_role, "")
    rendered_results, search_failed = await collect_search_evidence(
        agent_role,
        initial_prompt,
        max_iter=max_iter,
        system_prompt=system_prompt,
    )
    final_prompt = _build_final_answer_prompt(initial_prompt, rendered_results)
    final_text, _ = await query_minimax(
        system_prompt=system_prompt,
        question=final_prompt,
    )
    return final_text, search_failed


async def _dispatch_request(request: BrokerRequest) -> str:
    if request.provider == "gemini":
        try:
            return await query_gemini_cli(
                request.prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                system_instruction=request.system_instruction,
                use_google_search=request.allow_web,
                enable_fallback=True,
            )
        except Exception as exc:
            logger.warning(
                "[Broker] Gemini failed for %s (%s). Falling back to MiniMax.",
                request.fallback_role,
                exc,
            )
            if request.allow_web or request.strategy == "react":
                text, _ = await react_search_loop(
                    request.fallback_role,
                    request.prompt,
                    max_iter=2,
                    system_prompt=request.system_instruction,
                )
                return _raise_on_minimax_error(text)

            text, _ = await query_minimax(
                system_prompt=request.system_instruction,
                question=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                recover_pseudo_tool_query=request.recover_pseudo_tool_query,
            )
            return _raise_on_minimax_error(text)

    if request.allow_web or request.strategy == "react":
        text, _ = await react_search_loop(
            request.fallback_role,
            request.prompt,
            max_iter=2,
            system_prompt=request.system_instruction,
        )
        return _raise_on_minimax_error(text)

    text, _ = await query_minimax(
        system_prompt=request.system_instruction,
        question=request.prompt,
        model=request.model or "MiniMax-M2.5",
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        recover_pseudo_tool_query=request.recover_pseudo_tool_query,
    )
    return _raise_on_minimax_error(text)


async def _run_request(request: BrokerRequest) -> str:
    async with _broker_semaphore:
        return await _dispatch_request(request)


async def call_via_broker(request: BrokerRequest) -> str:
    key = request.key()
    owner = False

    async with _broker_lock:
        task = _inflight_requests.get(key)
        if task is None:
            task = asyncio.create_task(_run_request(request))
            _inflight_requests[key] = task
            owner = True

    try:
        return await asyncio.shield(task)
    finally:
        if owner:
            async with _broker_lock:
                if _inflight_requests.get(key) is task:
                    _inflight_requests.pop(key, None)


async def call_text(
    prompt: str,
    *,
    system_instruction: str = "",
    provider: str = "minimax",
    strategy: str = "direct",
    allow_web: bool = False,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    fallback_role: str = "agent",
    recover_pseudo_tool_query: bool = False,
) -> str:
    return await call_via_broker(
        BrokerRequest(
            prompt=prompt,
            system_instruction=system_instruction,
            provider=provider,
            strategy=strategy,
            allow_web=allow_web,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            fallback_role=fallback_role,
            recover_pseudo_tool_query=recover_pseudo_tool_query,
        )
    )


async def shutdown_broker() -> None:
    async with _broker_lock:
        tasks = list(_inflight_requests.values())
        _inflight_requests.clear()

    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    await close_gemini_cli_client()
    await close_minimax_client()


async def reset_broker_state() -> None:
    async with _broker_lock:
        tasks = list(_inflight_requests.values())
        _inflight_requests.clear()

    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
