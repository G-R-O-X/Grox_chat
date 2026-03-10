import logging
from typing import Tuple

from .minimax_client import query_minimax, minimax_search
from .prompts import PROMPTS

logger = logging.getLogger(__name__)


SEARCH_QUERY_SENTINEL = "NO_SEARCH"


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


async def collect_search_evidence(
    agent_role: str,
    initial_prompt: str,
    max_iter: int = 2,
    system_prompt: str | None = None,
) -> Tuple[list[str], bool]:
    system_prompt = system_prompt or PROMPTS.get(agent_role, "")
    rendered_results: list[str] = []
    search_failed = False

    for i in range(max_iter):
        logger.info(f"[{agent_role}] ReAct Loop Iteration {i+1}/{max_iter}")
        decision_prompt = _build_search_decision_prompt(initial_prompt, rendered_results)
        query = await _decide_search_query(agent_role, decision_prompt, system_prompt)
        if not query:
            return rendered_results, search_failed

        logger.info(f"[{agent_role}] Executing web search for: '{query}'")
        search_res = await minimax_search(query)
        if "error" in search_res:
            search_failed = True
        rendered_results.append(_render_search_results(search_res))

    logger.warning(f"[{agent_role}] Max iterations ({max_iter}) reached in search loop.")
    return rendered_results, search_failed


async def react_search_loop(
    agent_role: str,
    initial_prompt: str,
    max_iter: int = 2,
    system_prompt: str | None = None,
) -> Tuple[str, bool]:
    """
    Executes a bounded explicit search-intent loop for a given agent.
    The model first decides whether to search, then search runs via Coding Plan API,
    and the final answer is generated as plain text.
    """
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
