import logging
from typing import Optional

from .external.gemini_cli_client import query_gemini_cli
from .minimax_client import query_minimax
from .tools import react_search_loop

logger = logging.getLogger(__name__)


def _raise_on_minimax_error(text: str) -> str:
    if (text or "").strip().startswith("Error:"):
        raise RuntimeError(text.strip())
    return text


async def query_with_fallback(
    prompt: str,
    *,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    system_instruction: Optional[str] = None,
    thinking_level: str = "NONE",
    use_google_search: bool = False,
    enable_fallback: bool = True,
    fallback_role: str = "audience",
) -> str:
    """
    Prefer Gemini for orchestration-style calls, then fall back to MiniMax.

    If Google Search grounding was requested, the MiniMax fallback uses the
    explicit search-intent loop so the answer can still incorporate web results.
    Otherwise the MiniMax fallback is a plain text-generation call.
    """
    try:
        return await query_gemini_cli(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
            thinking_level=thinking_level,
            use_google_search=use_google_search,
            enable_fallback=enable_fallback,
        )
    except Exception as exc:
        logger.warning(
            "[LLMRouter] Gemini failed for %s (google_search=%s): %s. Falling back to MiniMax.",
            fallback_role,
            use_google_search,
            exc,
        )

    if use_google_search:
        final_text, _ = await react_search_loop(
            fallback_role,
            prompt,
            max_iter=2,
            system_prompt=system_instruction or "",
        )
        return _raise_on_minimax_error(final_text)

    final_text, _ = await query_minimax(
        system_prompt=system_instruction or "",
        question=prompt,
        model="MiniMax-M2.5",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _raise_on_minimax_error(final_text)
