from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Optional
from urllib.parse import urlparse

from . import api
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
from .reranker import arerank

logger = logging.getLogger(__name__)

SEARCH_QUERY_SENTINEL = "NO_SEARCH"
PROFILE_GEMINI_PRO = "gemini_pro"
PROFILE_GEMINI_FLASH = "gemini_flash"
PROFILE_MINIMAX = "minimax"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
GEMINI_FLASH_MODEL = "gemini-3.0-flash"
MINIMAX_DEFAULT_MODEL = "MiniMax-M2.5"
ENABLE_GEMINI_ENV = "ENABLE_GEMINI"
WEB_CACHE_MAX_AGE_DAYS = 30
WEB_CACHE_TOP_K = 6
WEB_CACHE_SELECTED_TOP_K = 3
WEB_CACHE_RERANK_THRESHOLD = 0.45


@dataclass(frozen=True)
class BrokerRequest:
    prompt: str
    system_instruction: str = ""
    provider_profile: str = ""
    provider: str = "minimax"
    strategy: str = "direct"
    allow_web: bool = False
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 8192
    fallback_role: str = "agent"
    recover_pseudo_tool_query: bool = False
    require_json: bool = False
    search_budget: int = 2
    topic_id: int = 0
    subtopic_id: int = 0

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=True)


@dataclass(frozen=True)
class SearchEvidenceItem:
    query: str
    rendered_results: str
    had_error: bool = False
    web_ids: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BrokerResponse:
    text: str
    provider_used: str = "unknown"
    fallback_used: bool = False
    search_used: bool = False
    search_evidence: tuple[SearchEvidenceItem, ...] = field(default_factory=tuple)
    search_failed: bool = False
    error: Optional[str] = None


LLMResult = BrokerResponse


_broker_lock = asyncio.Lock()
_broker_semaphore = asyncio.Semaphore(8)
_inflight_requests: dict[str, asyncio.Task[BrokerResponse]] = {}


def _raise_on_minimax_error(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("Error:"):
        raise RuntimeError(stripped)
    return stripped


def _clean_minimax_internal_text(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("Error:"):
        return ""
    return stripped


def is_gemini_enabled() -> bool:
    return (os.getenv(ENABLE_GEMINI_ENV, "0") or "0").strip() == "1"


def _is_gemini_profile(provider_profile: str) -> bool:
    return provider_profile in {PROFILE_GEMINI_PRO, PROFILE_GEMINI_FLASH}


def _classify_search_query_result(raw_text: str) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if not text:
        return "", "empty"

    first_line = text.splitlines()[0].strip()
    if first_line.startswith('"') and first_line.endswith('"') and len(first_line) >= 2:
        first_line = first_line[1:-1].strip()

    if not first_line:
        return "", "empty"
    if first_line.upper() == SEARCH_QUERY_SENTINEL:
        return "", "no_search"
    if first_line.upper().startswith(f"{SEARCH_QUERY_SENTINEL}:"):
        return "", "no_search"
    if first_line.startswith("Error:"):
        return "", "empty"
    if first_line.startswith("{") or first_line.startswith("["):
        return "", "empty"
    if len(first_line) > 200:
        first_line = first_line[:200].strip()
    if not first_line:
        return "", "empty"
    return first_line, "query"


def _render_search_results(search_res: dict) -> str:
    rendered = "=== WEB SEARCH RESULTS ===\n"
    if "organic" in search_res:
        for org in search_res["organic"][:3]:
            rendered += f"Title: {org.get('title')}\nSnippet: {org.get('snippet')}\n\n"
    else:
        rendered += "No useful results found.\n\n"
    return rendered


def _web_record_to_content(record: dict) -> str:
    return "\n".join(
        part.strip()
        for part in (
            record.get("title") or "",
            record.get("snippet") or "",
            record.get("query_text") or "",
            record.get("source_domain") or "",
        )
        if isinstance(part, str) and part.strip()
    )


async def _select_web_cache_rows(query: str, rows: list[dict], *, top_k: int = WEB_CACHE_SELECTED_TOP_K) -> list[dict]:
    if not rows:
        return []
    docs = [_web_record_to_content(row) for row in rows]
    try:
        ranked_indices = await arerank(query, docs, top_k=min(top_k, len(rows)))
    except Exception as exc:
        logger.warning("[Broker] Web cache rerank failed: %s", exc)
        return []
        
    selected: list[dict] = []
    for idx, score in ranked_indices:
        if score >= WEB_CACHE_RERANK_THRESHOLD:
            selected.append({**rows[idx], "score": score})
    return selected


def _render_web_records(rows: list[dict]) -> str:
    rendered = "=== WEB SEARCH RESULTS ===\n"
    if not rows:
        rendered += "No useful results found.\n\n"
        return rendered
    for row in rows[:WEB_CACHE_SELECTED_TOP_K]:
        source = row.get("source_domain") or "unknown"
        title = row.get("title") or "(untitled)"
        snippet = row.get("snippet") or ""
        url = row.get("url") or ""
        rendered += (
            f"[W{row['id']}] Title: {title}\n"
            f"Source: {source}\n"
            f"Snippet: {snippet}\n"
        )
        if url:
            rendered += f"URL: {url}\n"
        rendered += "\n"
    return rendered


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


async def _lookup_cached_web_rows(query: str, *, topic_id: int) -> list[dict]:
    if topic_id <= 0:
        return []

    topic_rows = api.search_web_evidence_same_topic(
        topic_id,
        query,
        top_k=WEB_CACHE_TOP_K,
        max_age_days=WEB_CACHE_MAX_AGE_DAYS,
    )
    selected_topic_rows = await _select_web_cache_rows(query, topic_rows)
    if selected_topic_rows:
        logger.info("[Broker] Reused same-topic web cache query=%s rows=%s", query, len(selected_topic_rows))
        return selected_topic_rows

    cross_topic_rows = api.search_web_evidence_cross_topic(
        topic_id,
        query,
        top_k=WEB_CACHE_TOP_K,
        max_age_days=WEB_CACHE_MAX_AGE_DAYS,
    )
    selected_cross_topic_rows = await _select_web_cache_rows(query, cross_topic_rows)
    if selected_cross_topic_rows:
        logger.info("[Broker] Reused cross-topic web cache query=%s rows=%s", query, len(selected_cross_topic_rows))
        return selected_cross_topic_rows

    return []


def _persist_web_search_rows(
    *,
    topic_id: int,
    subtopic_id: int,
    query: str,
    search_res: dict,
    role: str,
) -> list[dict]:
    if topic_id <= 0:
        return []

    stored_rows: list[dict] = []
    organic = search_res.get("organic") if isinstance(search_res, dict) else None
    if not isinstance(organic, list):
        return stored_rows

    for rank, item in enumerate(organic[:WEB_CACHE_SELECTED_TOP_K], start=1):
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        url = (item.get("link") or item.get("url") or "").strip()
        if not title and not snippet:
            continue
        web_id = api.insert_web_evidence(
            topic_id,
            subtopic_id or None,
            query,
            title,
            snippet,
            url,
            _extract_domain(url),
            rank,
            "minimax_search",
            role,
        )
        stored_rows.append(
            {
                "id": web_id,
                "origin_topic_id": topic_id,
                "origin_subtopic_id": subtopic_id,
                "query_text": query,
                "title": title,
                "snippet": snippet,
                "url": url,
                "source_domain": _extract_domain(url),
                "result_rank": rank,
            }
        )

    return stored_rows


async def get_or_collect_search_evidence_item(
    query: str,
    *,
    topic_id: int = 0,
    subtopic_id: int = 0,
    role: str = "agent",
) -> SearchEvidenceItem:
    cached_rows = await _lookup_cached_web_rows(query, topic_id=topic_id)
    if cached_rows:
        return SearchEvidenceItem(
            query=query,
            rendered_results=_render_web_records(cached_rows),
            had_error=False,
            web_ids=tuple(int(row["id"]) for row in cached_rows if row.get("id") is not None),
        )

    search_res = await minimax_search(query)
    had_error = "error" in search_res
    stored_rows = _persist_web_search_rows(
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        query=query,
        search_res=search_res,
        role=role,
    )

    if stored_rows:
        return SearchEvidenceItem(
            query=query,
            rendered_results=_render_web_records(stored_rows),
            had_error=had_error,
            web_ids=tuple(int(row["id"]) for row in stored_rows if row.get("id") is not None),
        )

    return SearchEvidenceItem(
        query=query,
        rendered_results=_render_search_results(search_res),
        had_error=had_error,
    )


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


def _strip_markdown_fences(text: str) -> str:
    stripped = (text or "").strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3:
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _normalize_json_text(text: str) -> Optional[str]:
    stripped = (text or "").strip()
    if not stripped:
        return None
    candidates = [stripped]
    unfenced = _strip_markdown_fences(stripped)
    if unfenced and unfenced != stripped:
        candidates.append(unfenced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return json.dumps(parsed, ensure_ascii=False)
    return None


def _build_json_repair_prompt(original_prompt: str, invalid_text: str) -> str:
    return (
        "Original task:\n"
        f"{original_prompt}\n\n"
        "Invalid response that must be repaired into strict JSON:\n"
        f"{invalid_text}\n\n"
        "Rewrite the invalid response into valid JSON that best satisfies the original task.\n"
        "Output valid JSON only. Do not use markdown fences. Do not add any explanation."
    )


async def _repair_required_json_response(
    request: BrokerRequest,
    response: BrokerResponse,
) -> BrokerResponse:
    if not request.require_json:
        return response

    normalized = _normalize_json_text(response.text)
    if normalized is not None:
        if normalized != (response.text or "").strip():
            logger.info(
                "[Broker] Normalized JSON response without repair role=%s provider_used=%s",
                request.fallback_role,
                response.provider_used,
            )
            return replace(response, text=normalized)
        return response

    logger.warning(
        "[Broker] Invalid JSON response detected role=%s provider_used=%s allow_web=%s. Attempting MiniMax repair.",
        request.fallback_role,
        response.provider_used,
        request.allow_web,
    )
    try:
        repair_text, _ = await query_minimax(
            system_prompt=(
                f"{request.system_instruction}\n\n"
                "JSON REPAIR MODE:\n"
                "You are repairing an invalid response that was required to be strict JSON.\n"
                "Preserve the original meaning and requested schema as much as possible.\n"
                "Output valid JSON only. Do not add markdown fences or commentary."
            ).strip(),
            question=_build_json_repair_prompt(request.prompt, response.text),
            model=MINIMAX_DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=request.max_tokens,
        )
        normalized_repair = _normalize_json_text(_raise_on_minimax_error(repair_text))
        if normalized_repair is not None:
            logger.info(
                "[Broker] JSON repair succeeded role=%s original_provider=%s",
                request.fallback_role,
                response.provider_used,
            )
            return replace(
                response,
                text=normalized_repair,
                fallback_used=True,
            )
    except Exception as exc:
        logger.warning(
            "[Broker] JSON repair call failed role=%s original_provider=%s (%s).",
            request.fallback_role,
            response.provider_used,
            exc,
        )

    logger.warning(
        "[Broker] JSON repair did not produce valid JSON role=%s provider_used=%s. Returning original response.",
        request.fallback_role,
        response.provider_used,
    )
    return response


async def _decide_search_query_with_retry(agent_role: str, current_prompt: str, system_prompt: str) -> str:
    for attempt in range(2):
        raw_text, _ = await query_minimax(
            system_prompt=(
                f"{system_prompt}\n\n"
                "You are deciding whether web search is necessary before answering.\n"
                f"Reply with exactly one short search query string, or `{SEARCH_QUERY_SENTINEL}` if search is unnecessary.\n"
                "Do not output JSON. Do not explain your reasoning. Do not add extra text."
            ),
            question=current_prompt,
            max_tokens=8192,
            recover_pseudo_tool_query=True,
        )
        query, status = _classify_search_query_result(raw_text)
        if status == "query":
            return query
        if status == "no_search":
            return ""
        if attempt == 0:
            logger.warning(
                "[%s] Search-decision returned empty query. Retrying once.",
                agent_role,
            )
    return ""


def _infer_provider_profile(provider: str, model: str) -> str:
    normalized_provider = (provider or "").strip().lower()
    normalized_model = (model or "").strip().lower()
    if normalized_provider in {PROFILE_GEMINI_PRO, PROFILE_GEMINI_FLASH, PROFILE_MINIMAX}:
        return normalized_provider
    if normalized_provider == "gemini":
        return PROFILE_GEMINI_PRO if "pro" in normalized_model else PROFILE_GEMINI_FLASH
    return PROFILE_MINIMAX


def _resolve_profile_model(provider_profile: str, requested_model: str) -> str:
    if requested_model:
        return requested_model
    if provider_profile == PROFILE_GEMINI_PRO:
        return GEMINI_PRO_MODEL
    if provider_profile == PROFILE_GEMINI_FLASH:
        return GEMINI_FLASH_MODEL
    return ""


def _build_minimax_deep_plan_prompt(initial_prompt: str) -> str:
    return (
        "Original task:\n"
        f"{initial_prompt}\n\n"
        "Think through the task without web access. Produce a short working plan only.\n"
        "Include the key sub-questions, important assumptions, and the intended answer structure.\n"
        "Do not answer the task yet."
    )


def _build_minimax_deep_answer_prompt(initial_prompt: str, plan_text: str, require_json: bool) -> str:
    format_instruction = (
        "Return strict JSON only. Do not use markdown fences. Do not add commentary outside the JSON object."
        if require_json
        else "Return the best direct answer for the user."
    )
    return (
        "Original task:\n"
        f"{initial_prompt}\n\n"
        "Working plan:\n"
        f"{plan_text}\n\n"
        "Using the plan above, draft the strongest answer you can without web access.\n"
        f"{format_instruction}"
    )


def _build_minimax_deep_reflection_prompt(
    initial_prompt: str,
    plan_text: str,
    draft_text: str,
    require_json: bool,
) -> str:
    format_instruction = (
        "Output only the corrected final JSON object. No markdown fences or extra text."
        if require_json
        else "Output only the revised final answer."
    )
    return (
        "Original task:\n"
        f"{initial_prompt}\n\n"
        "Working plan:\n"
        f"{plan_text}\n\n"
        "Draft answer:\n"
        f"{draft_text}\n\n"
        "Review the draft for correctness, missing caveats, weak reasoning, and formatting mistakes.\n"
        "Revise it into the best final answer you can without web access.\n"
        f"{format_instruction}"
    )


async def _query_gemini_text(
    *,
    prompt: str,
    system_instruction: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    return await query_gemini_cli(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_instruction=system_instruction,
        thinking_level="HIGH",
        use_google_search=False,
        enable_fallback=True,
    )


async def _run_minimax_deep_profile(request: BrokerRequest, *, fallback_used: bool) -> BrokerResponse:
    logger.info(
        "[Broker] MiniMax deep fallback start role=%s require_json=%s",
        request.fallback_role,
        request.require_json,
    )
    plan_text = ""
    try:
        raw_plan_text, _ = await query_minimax(
            system_prompt=(
                f"{request.system_instruction}\n\n"
                "You are preparing internal reasoning notes before answering. Keep the output concise and useful."
            ).strip(),
            question=_build_minimax_deep_plan_prompt(request.prompt),
            model=MINIMAX_DEFAULT_MODEL,
            temperature=min(request.temperature, 0.7),
            max_tokens=min(request.max_tokens, 2048),
        )
        plan_text = _clean_minimax_internal_text(raw_plan_text)
    except Exception as exc:
        logger.warning(
            "[Broker] MiniMax deep fallback planning failed for role=%s (%s). Continuing without a separate plan.",
            request.fallback_role,
            exc,
        )

    if not plan_text:
        plan_text = "Reason directly from the task, surface assumptions, and keep the answer structured."

    draft_text = ""
    try:
        raw_draft_text, _ = await query_minimax(
            system_prompt=request.system_instruction,
            question=_build_minimax_deep_answer_prompt(request.prompt, plan_text, request.require_json),
            model=MINIMAX_DEFAULT_MODEL,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            recover_pseudo_tool_query=request.recover_pseudo_tool_query,
        )
        draft_text = _raise_on_minimax_error(raw_draft_text)
    except Exception as exc:
        logger.warning(
            "[Broker] MiniMax deep fallback drafting failed for role=%s (%s). Falling back to direct MiniMax answer.",
            request.fallback_role,
            exc,
        )
        direct_text, _ = await query_minimax(
            system_prompt=request.system_instruction,
            question=request.prompt,
            model=MINIMAX_DEFAULT_MODEL,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            recover_pseudo_tool_query=request.recover_pseudo_tool_query,
        )
        return BrokerResponse(
            text=_raise_on_minimax_error(direct_text),
            provider_used=PROFILE_MINIMAX,
            fallback_used=fallback_used,
        )

    try:
        raw_final_text, _ = await query_minimax(
            system_prompt=request.system_instruction,
            question=_build_minimax_deep_reflection_prompt(
                request.prompt,
                plan_text,
                draft_text,
                request.require_json,
            ),
            model=MINIMAX_DEFAULT_MODEL,
            temperature=max(0.1, min(request.temperature, 0.5)),
            max_tokens=request.max_tokens,
            recover_pseudo_tool_query=request.recover_pseudo_tool_query,
        )
        final_text = _clean_minimax_internal_text(raw_final_text)
        if final_text:
            return BrokerResponse(
                text=final_text,
                provider_used=PROFILE_MINIMAX,
                fallback_used=fallback_used,
            )
        logger.warning(
            "[Broker] MiniMax deep fallback reflection returned empty output for role=%s. Using draft answer.",
            request.fallback_role,
        )
    except Exception as exc:
        logger.warning(
            "[Broker] MiniMax deep fallback reflection failed for role=%s (%s). Using draft answer.",
            request.fallback_role,
            exc,
        )

    return BrokerResponse(
        text=draft_text,
        provider_used=PROFILE_MINIMAX,
        fallback_used=fallback_used,
    )


async def _run_gemini_plain_profile(request: BrokerRequest, provider_profile: str) -> BrokerResponse:
    model = _resolve_profile_model(provider_profile, request.model)
    try:
        text = await _query_gemini_text(
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            model=model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return BrokerResponse(
            text=text,
            provider_used=provider_profile,
        )
    except Exception as exc:
        logger.warning(
            "[Broker] %s plain path failed for role=%s (%s). Falling back to MiniMax deep profile.",
            provider_profile,
            request.fallback_role,
            exc,
        )
        return await _run_minimax_deep_profile(request, fallback_used=True)


async def _run_minimax_web_profile(request: BrokerRequest) -> BrokerResponse:
    response = await react_search_loop_with_evidence(
        request.fallback_role,
        request.prompt,
        max_iter=request.search_budget,
        system_prompt=request.system_instruction,
        topic_id=request.topic_id,
        subtopic_id=request.subtopic_id,
    )
    return BrokerResponse(
        text=_raise_on_minimax_error(response.text),
        provider_used=PROFILE_MINIMAX,
        search_used=bool(response.search_evidence),
        search_evidence=response.search_evidence,
        search_failed=response.search_failed,
    )


async def _run_gemini_web_profile(request: BrokerRequest, provider_profile: str) -> BrokerResponse:
    rendered_results: list[str] = []
    evidence_items: list[SearchEvidenceItem] = []
    search_failed = False

    try:
        for i in range(request.search_budget):
            logger.info(
                "[Broker] %s web query-step iteration=%s/%s role=%s",
                provider_profile,
                i + 1,
                request.search_budget,
                request.fallback_role,
            )
            decision_prompt = _build_search_decision_prompt(request.prompt, rendered_results)
            query = await _decide_search_query_with_retry(
                request.fallback_role,
                decision_prompt,
                request.system_instruction,
            )
            if not query:
                break

            logger.info(
                "[Broker] %s web search executing role=%s query=%s",
                provider_profile,
                request.fallback_role,
                query,
            )
            item = await get_or_collect_search_evidence_item(
                query,
                topic_id=request.topic_id,
                subtopic_id=request.subtopic_id,
                role=request.fallback_role,
            )
            if item.had_error:
                search_failed = True
            evidence_items.append(item)
            rendered_results.append(item.rendered_results)

        final_prompt = _build_final_answer_prompt(request.prompt, rendered_results)
        final_model = _resolve_profile_model(provider_profile, request.model)
        try:
            text = await _query_gemini_text(
                prompt=final_prompt,
                system_instruction=request.system_instruction,
                model=final_model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            return BrokerResponse(
                text=text,
                provider_used=provider_profile,
                search_used=bool(evidence_items),
                search_evidence=tuple(evidence_items),
                search_failed=search_failed,
            )
        except Exception as final_exc:
            logger.warning(
                "[Broker] %s web final-analysis failed for role=%s (%s).",
                provider_profile,
                request.fallback_role,
                final_exc,
            )
            if provider_profile == PROFILE_GEMINI_PRO:
                text = await _query_gemini_text(
                    prompt=final_prompt,
                    system_instruction=request.system_instruction,
                    model=GEMINI_FLASH_MODEL,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                return BrokerResponse(
                    text=text,
                    provider_used=PROFILE_GEMINI_FLASH,
                    fallback_used=True,
                    search_used=bool(evidence_items),
                    search_evidence=tuple(evidence_items),
                    search_failed=search_failed,
                )
            raise
    except Exception as exc:
        logger.warning(
            "[Broker] %s web path failed for role=%s (%s). Falling back to MiniMax web loop.",
            provider_profile,
            request.fallback_role,
            exc,
        )
        fallback = await _run_minimax_web_profile(request)
        return BrokerResponse(
            text=fallback.text,
            provider_used=fallback.provider_used,
            fallback_used=True,
            search_used=fallback.search_used,
            search_evidence=fallback.search_evidence,
            search_failed=fallback.search_failed,
            error=fallback.error,
        )


async def collect_search_evidence_bundle(
    agent_role: str,
    initial_prompt: str,
    max_iter: int = 2,
    system_prompt: str | None = None,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> BrokerResponse:
    system_prompt = system_prompt or PROMPTS.get(agent_role, "")
    rendered_results: list[SearchEvidenceItem] = []
    search_failed = False

    for i in range(max_iter):
        logger.info("[%s] ReAct Loop Iteration %s/%s", agent_role, i + 1, max_iter)
        decision_prompt = _build_search_decision_prompt(
            initial_prompt,
            [item.rendered_results for item in rendered_results],
        )
        query = await _decide_search_query_with_retry(agent_role, decision_prompt, system_prompt)
        if not query:
            return BrokerResponse(
                text="",
                search_evidence=tuple(rendered_results),
                search_failed=search_failed,
            )

        logger.info("[%s] Executing web search for: '%s'", agent_role, query)
        item = await get_or_collect_search_evidence_item(
            query,
            topic_id=topic_id,
            subtopic_id=subtopic_id,
            role=agent_role,
        )
        if item.had_error:
            search_failed = True
        rendered_results.append(item)

    logger.warning("[%s] Max iterations (%s) reached in search loop.", agent_role, max_iter)
    return BrokerResponse(
        text="",
        search_evidence=tuple(rendered_results),
        search_failed=search_failed,
    )


async def collect_search_evidence(
    agent_role: str,
    initial_prompt: str,
    max_iter: int = 2,
    system_prompt: str | None = None,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> tuple[list[str], bool]:
    response = await collect_search_evidence_bundle(
        agent_role,
        initial_prompt,
        max_iter=max_iter,
        system_prompt=system_prompt,
        topic_id=topic_id,
        subtopic_id=subtopic_id,
    )
    return [item.rendered_results for item in response.search_evidence], response.search_failed


async def react_search_loop_with_evidence(
    agent_role: str,
    initial_prompt: str,
    max_iter: int = 2,
    system_prompt: str | None = None,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> BrokerResponse:
    system_prompt = system_prompt or PROMPTS.get(agent_role, "")
    evidence = await collect_search_evidence_bundle(
        agent_role,
        initial_prompt,
        max_iter=max_iter,
        system_prompt=system_prompt,
        topic_id=topic_id,
        subtopic_id=subtopic_id,
    )
    rendered_results = [item.rendered_results for item in evidence.search_evidence]
    final_prompt = _build_final_answer_prompt(initial_prompt, rendered_results)
    final_text, _ = await query_minimax(
        system_prompt=system_prompt,
        question=final_prompt,
    )
    return BrokerResponse(
        text=final_text,
        search_evidence=evidence.search_evidence,
        search_failed=evidence.search_failed,
    )


async def react_search_loop(
    agent_role: str,
    initial_prompt: str,
    max_iter: int = 2,
    system_prompt: str | None = None,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> tuple[str, bool]:
    response = await react_search_loop_with_evidence(
        agent_role,
        initial_prompt,
        max_iter=max_iter,
        system_prompt=system_prompt,
        topic_id=topic_id,
        subtopic_id=subtopic_id,
    )
    return response.text, response.search_failed


async def _dispatch_request(request: BrokerRequest) -> BrokerResponse:
    provider_profile = _infer_provider_profile(request.provider_profile or request.provider, request.model)
    gemini_enabled = is_gemini_enabled()
    logger.info(
        "[Broker] Dispatch start role=%s profile=%s allow_web=%s model=%s prompt_chars=%s require_json=%s recover_pseudo_tool_query=%s gemini_enabled=%s",
        request.fallback_role,
        provider_profile,
        request.allow_web,
        request.model or "(default)",
        len(request.prompt or ""),
        request.require_json,
        request.recover_pseudo_tool_query,
        gemini_enabled,
    )
    if _is_gemini_profile(provider_profile) and not gemini_enabled:
        logger.info(
            "[Broker] Gemini disabled by env. Routing role=%s allow_web=%s through MiniMax fallback.",
            request.fallback_role,
            request.allow_web,
        )
        if request.allow_web:
            fallback = await _run_minimax_web_profile(request)
            response = BrokerResponse(
                text=fallback.text,
                provider_used=fallback.provider_used,
                fallback_used=True,
                search_used=fallback.search_used,
                search_evidence=fallback.search_evidence,
                search_failed=fallback.search_failed,
                error=fallback.error,
            )
            logger.info(
                "[Broker] Web path succeeded role=%s provider_used=%s fallback_used=%s text_chars=%s search_items=%s search_failed=%s",
                request.fallback_role,
                response.provider_used,
                response.fallback_used,
                len(response.text or ""),
                len(response.search_evidence),
                response.search_failed,
            )
            return response

        response = await _run_minimax_deep_profile(request, fallback_used=True)
        logger.info(
            "[Broker] Plain path succeeded role=%s provider_used=%s fallback_used=%s text_chars=%s",
            request.fallback_role,
            response.provider_used,
            response.fallback_used,
            len(response.text or ""),
        )
        return response

    if request.allow_web:
        if _is_gemini_profile(provider_profile):
            response = await _run_gemini_web_profile(request, provider_profile)
        else:
            response = await _run_minimax_web_profile(request)
        logger.info(
            "[Broker] Web path succeeded role=%s provider_used=%s fallback_used=%s text_chars=%s search_items=%s search_failed=%s",
            request.fallback_role,
            response.provider_used,
            response.fallback_used,
            len(response.text or ""),
            len(response.search_evidence),
            response.search_failed,
        )
        return response

    if _is_gemini_profile(provider_profile):
        response = await _run_gemini_plain_profile(request, provider_profile)
        logger.info(
            "[Broker] Plain path succeeded role=%s provider_used=%s fallback_used=%s text_chars=%s",
            request.fallback_role,
            response.provider_used,
            response.fallback_used,
            len(response.text or ""),
        )
        return response

    text, _ = await query_minimax(
        system_prompt=request.system_instruction,
        question=request.prompt,
        model=request.model or MINIMAX_DEFAULT_MODEL,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        recover_pseudo_tool_query=request.recover_pseudo_tool_query,
    )
    broker_response = BrokerResponse(
        text=_raise_on_minimax_error(text),
        provider_used=PROFILE_MINIMAX,
    )
    logger.info(
        "[Broker] Plain path succeeded role=%s provider_used=%s fallback_used=%s text_chars=%s",
        request.fallback_role,
        broker_response.provider_used,
        broker_response.fallback_used,
        len(broker_response.text or ""),
    )
    return broker_response


async def _run_request(request: BrokerRequest) -> BrokerResponse:
    async with _broker_semaphore:
        response = await _dispatch_request(request)
        return await _repair_required_json_response(request, response)


async def call_via_broker(request: BrokerRequest) -> BrokerResponse:
    key = request.key()
    owner = False

    async with _broker_lock:
        task = _inflight_requests.get(key)
        if task is None:
            task = asyncio.create_task(_run_request(request))
            _inflight_requests[key] = task
            owner = True
            logger.info(
                "[Broker] Created new in-flight request role=%s provider=%s strategy=%s allow_web=%s",
                request.fallback_role,
                request.provider,
                request.strategy,
                request.allow_web,
            )
        else:
            logger.info(
                "[Broker] Coalescing duplicate in-flight request role=%s provider=%s strategy=%s allow_web=%s",
                request.fallback_role,
                request.provider,
                request.strategy,
                request.allow_web,
            )

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
    require_json: bool = False,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> str:
    provider_profile = _infer_provider_profile(provider, model)
    if allow_web or strategy == "react":
        response = await llm_call_with_web(
            prompt,
            system_prompt=system_instruction,
            provider_profile=provider_profile,
            require_json=require_json,
            role=fallback_role,
            model=model,
            search_budget=2,
            temperature=temperature,
            max_tokens=max_tokens,
            recover_pseudo_tool_query=recover_pseudo_tool_query,
            topic_id=topic_id,
            subtopic_id=subtopic_id,
        )
    else:
        response = await llm_call(
            prompt,
            system_prompt=system_instruction,
            provider_profile=provider_profile,
            require_json=require_json,
            role=fallback_role,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            recover_pseudo_tool_query=recover_pseudo_tool_query,
            topic_id=topic_id,
            subtopic_id=subtopic_id,
        )
    return response.text


async def call_text_with_search_evidence(
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
    require_json: bool = False,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> BrokerResponse:
    provider_profile = _infer_provider_profile(provider, model)
    if allow_web or strategy == "react":
        return await llm_call_with_web(
            prompt,
            system_prompt=system_instruction,
            provider_profile=provider_profile,
            require_json=require_json,
            role=fallback_role,
            model=model,
            search_budget=2,
            temperature=temperature,
            max_tokens=max_tokens,
            recover_pseudo_tool_query=recover_pseudo_tool_query,
            topic_id=topic_id,
            subtopic_id=subtopic_id,
        )
    return await llm_call(
        prompt,
        system_prompt=system_instruction,
        provider_profile=provider_profile,
        require_json=require_json,
        role=fallback_role,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        recover_pseudo_tool_query=recover_pseudo_tool_query,
        topic_id=topic_id,
        subtopic_id=subtopic_id,
    )


async def llm_call(
    prompt: str,
    *,
    system_prompt: str = "",
    provider_profile: str = PROFILE_MINIMAX,
    require_json: bool = False,
    role: str = "agent",
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    boost: str = "",
    recover_pseudo_tool_query: bool = False,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> LLMResult:
    effective_system_prompt = system_prompt
    if boost:
        effective_system_prompt = f"{effective_system_prompt}\n\nBOOST:\n{boost}".strip()
    return await call_via_broker(
        BrokerRequest(
            prompt=prompt,
            system_instruction=effective_system_prompt,
            provider_profile=provider_profile,
            model=model,
            allow_web=False,
            temperature=temperature,
            max_tokens=max_tokens,
            fallback_role=role,
            require_json=require_json,
            recover_pseudo_tool_query=recover_pseudo_tool_query,
            topic_id=topic_id,
            subtopic_id=subtopic_id,
        )
    )


async def llm_call_with_web(
    prompt: str,
    *,
    system_prompt: str = "",
    provider_profile: str = PROFILE_MINIMAX,
    require_json: bool = False,
    role: str = "agent",
    model: str = "",
    search_budget: int = 2,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    boost: str = "",
    recover_pseudo_tool_query: bool = False,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> LLMResult:
    effective_system_prompt = system_prompt
    if boost:
        effective_system_prompt = f"{effective_system_prompt}\n\nBOOST:\n{boost}".strip()
    return await call_via_broker(
        BrokerRequest(
            prompt=prompt,
            system_instruction=effective_system_prompt,
            provider_profile=provider_profile,
            model=model,
            allow_web=True,
            temperature=temperature,
            max_tokens=max_tokens,
            fallback_role=role,
            require_json=require_json,
            search_budget=search_budget,
            recover_pseudo_tool_query=recover_pseudo_tool_query,
            topic_id=topic_id,
            subtopic_id=subtopic_id,
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
