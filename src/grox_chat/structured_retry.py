from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")

DEFAULT_STRUCTURED_OUTPUT_ATTEMPTS = 2


async def retry_structured_output(
    *,
    stage_name: str,
    invoke: Callable[[], Awaitable[T]],
    is_usable: Callable[[T], bool],
    logger: logging.Logger,
    attempts: int = DEFAULT_STRUCTURED_OUTPUT_ATTEMPTS,
) -> Optional[T]:
    last_exception: Exception | None = None

    if attempts <= 0:
        logger.warning("[structured_retry] %s called with attempts=%d, returning None.", stage_name, attempts)
        return None
    for attempt in range(1, attempts + 1):
        try:
            result = await invoke()
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "[structured_retry] %s failed on attempt %s/%s: %s",
                stage_name,
                attempt,
                attempts,
                exc,
            )
            continue

        try:
            if is_usable(result):
                return result
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "[structured_retry] %s validator failed on attempt %s/%s: %s",
                stage_name,
                attempt,
                attempts,
                exc,
            )
            continue

        logger.warning(
            "[structured_retry] %s returned unusable output on attempt %s/%s.",
            stage_name,
            attempt,
            attempts,
        )

    if last_exception is not None:
        logger.warning(
            "[structured_retry] %s exhausted retries after exception failures.",
            stage_name,
        )
    else:
        logger.warning(
            "[structured_retry] %s exhausted retries after unusable outputs.",
            stage_name,
        )
    return None


def usable_text_output(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and not stripped.startswith("Error:")


async def generate_summary(text: str, *, max_words: int = 40) -> str | None:
    """Call Gemini Flash to produce a short summary for RAG embedding."""
    if not text or not text.strip():
        return None
    from .broker import call_text
    try:
        resp = await call_text(
            f"Summarize the following in ≤{max_words} words for semantic search indexing. "
            f"Output ONLY the summary text, no JSON, no preamble.\n\n{text[:2000]}",
            provider="gemini",
            model="gemini-3.0-flash",
            strategy="direct",
            temperature=0.2,
            max_tokens=256,
        )
        if resp and resp.strip():
            return resp.strip()[:500]
    except Exception as e:
        logging.getLogger(__name__).warning("[generate_summary] Failed: %s", e)
    return None
