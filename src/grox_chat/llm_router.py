import logging
from typing import Optional

from .broker import call_text

logger = logging.getLogger(__name__)


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
    """Compatibility wrapper that routes orchestration calls through the unified broker."""
    if thinking_level and thinking_level.upper() != "NONE":
        logger.debug(
            "[LLMRouter] Broker path ignores explicit thinking_level=%s and uses provider defaults.",
            thinking_level,
        )
    if not enable_fallback:
        logger.debug(
            "[LLMRouter] Broker path ignores enable_fallback=False and uses broker fallback behavior."
        )

    return await call_text(
        prompt,
        system_instruction=system_instruction or "",
        provider="gemini",
        strategy="react" if use_google_search else "direct",
        allow_web=use_google_search,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback_role=fallback_role,
    )
