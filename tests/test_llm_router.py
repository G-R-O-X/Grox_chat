from unittest.mock import AsyncMock, patch

import pytest

from grox_chat.llm_router import query_with_fallback


@pytest.mark.asyncio
async def test_query_with_fallback_returns_gemini_result_when_available():
    with patch(
        "grox_chat.llm_router.query_gemini_cli",
        new=AsyncMock(return_value="gemini answer"),
    ) as gemini_query:
        with patch("grox_chat.llm_router.react_search_loop", new=AsyncMock()) as react_loop:
            with patch("grox_chat.llm_router.query_minimax", new=AsyncMock()) as minimax_query:
                result = await query_with_fallback("prompt", model="gemini-3.0-flash", fallback_role="audience")

    assert result == "gemini answer"
    gemini_query.assert_awaited_once()
    react_loop.assert_not_awaited()
    minimax_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_with_fallback_uses_minimax_direct_for_non_grounded_calls():
    with patch(
        "grox_chat.llm_router.query_gemini_cli",
        new=AsyncMock(side_effect=RuntimeError("429")),
    ):
        with patch("grox_chat.llm_router.query_minimax", new=AsyncMock(return_value=("minimax answer", []))) as minimax_query:
            with patch("grox_chat.llm_router.react_search_loop", new=AsyncMock()) as react_loop:
                result = await query_with_fallback(
                    "prompt",
                    model="gemini-3.0-flash",
                    use_google_search=False,
                    fallback_role="audience",
                )

    assert result == "minimax answer"
    minimax_query.assert_awaited_once()
    react_loop.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_with_fallback_uses_search_loop_for_grounded_calls():
    with patch(
        "grox_chat.llm_router.query_gemini_cli",
        new=AsyncMock(side_effect=RuntimeError("RESOURCE_EXHAUSTED")),
    ):
        with patch(
            "grox_chat.llm_router.react_search_loop",
            new=AsyncMock(return_value=("grounded minimax answer", False)),
        ) as react_loop:
            with patch("grox_chat.llm_router.query_minimax", new=AsyncMock()) as minimax_query:
                result = await query_with_fallback(
                    "prompt",
                    model="gemini-3.1-pro-preview",
                    use_google_search=True,
                    fallback_role="writer",
                )

    assert result == "grounded minimax answer"
    react_loop.assert_awaited_once_with("writer", "prompt", max_iter=2, system_prompt="")
    minimax_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_with_fallback_raises_when_direct_minimax_returns_error_sentinel():
    with patch(
        "grox_chat.llm_router.query_gemini_cli",
        new=AsyncMock(side_effect=RuntimeError("429")),
    ):
        with patch(
            "grox_chat.llm_router.query_minimax",
            new=AsyncMock(return_value=("Error: No API key.", [])),
        ):
            with pytest.raises(RuntimeError, match="Error: No API key."):
                await query_with_fallback(
                    "prompt",
                    model="gemini-3.0-flash",
                    use_google_search=False,
                    fallback_role="audience",
                )


@pytest.mark.asyncio
async def test_query_with_fallback_raises_when_grounded_search_loop_returns_error_sentinel():
    with patch(
        "grox_chat.llm_router.query_gemini_cli",
        new=AsyncMock(side_effect=RuntimeError("RESOURCE_EXHAUSTED")),
    ):
        with patch(
            "grox_chat.llm_router.react_search_loop",
            new=AsyncMock(return_value=("Error: MiniMax search timed out", True)),
        ):
            with pytest.raises(RuntimeError, match="Error: MiniMax search timed out"):
                await query_with_fallback(
                    "prompt",
                    model="gemini-3.1-pro-preview",
                    use_google_search=True,
                    fallback_role="writer",
                )
