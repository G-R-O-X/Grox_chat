from unittest.mock import AsyncMock, patch

import pytest

from grox_chat.llm_router import query_with_fallback


@pytest.mark.asyncio
async def test_query_with_fallback_returns_gemini_result_when_available():
    with patch(
        "grox_chat.llm_router.call_text",
        new=AsyncMock(return_value="gemini answer"),
    ) as broker_call:
        result = await query_with_fallback("prompt", model="gemini-3.0-flash", fallback_role="audience")

    assert result == "gemini answer"
    broker_call.assert_awaited_once_with(
        "prompt",
        system_instruction="",
        provider="gemini",
        strategy="direct",
        allow_web=False,
        model="gemini-3.0-flash",
        temperature=0.7,
        max_tokens=8192,
        fallback_role="audience",
    )


@pytest.mark.asyncio
async def test_query_with_fallback_routes_non_grounded_calls_through_broker():
    with patch(
        "grox_chat.llm_router.call_text",
        new=AsyncMock(return_value="minimax answer"),
    ) as broker_call:
        result = await query_with_fallback(
            "prompt",
            model="gemini-3.0-flash",
            use_google_search=False,
            fallback_role="audience",
        )

    assert result == "minimax answer"
    broker_call.assert_awaited_once_with(
        "prompt",
        system_instruction="",
        provider="gemini",
        strategy="direct",
        allow_web=False,
        model="gemini-3.0-flash",
        temperature=0.7,
        max_tokens=8192,
        fallback_role="audience",
    )


@pytest.mark.asyncio
async def test_query_with_fallback_routes_grounded_calls_through_broker():
    with patch(
        "grox_chat.llm_router.call_text",
        new=AsyncMock(return_value="grounded minimax answer"),
    ) as broker_call:
        result = await query_with_fallback(
            "prompt",
            model="gemini-3.1-pro-preview",
            use_google_search=True,
            fallback_role="writer",
        )

    assert result == "grounded minimax answer"
    broker_call.assert_awaited_once_with(
        "prompt",
        system_instruction="",
        provider="gemini",
        strategy="react",
        allow_web=True,
        model="gemini-3.1-pro-preview",
        temperature=0.7,
        max_tokens=8192,
        fallback_role="writer",
    )


@pytest.mark.asyncio
async def test_query_with_fallback_propagates_broker_errors():
    with patch(
        "grox_chat.llm_router.call_text",
        new=AsyncMock(side_effect=RuntimeError("Error: No API key.")),
    ):
        with pytest.raises(RuntimeError, match="Error: No API key."):
            await query_with_fallback(
                "prompt",
                model="gemini-3.0-flash",
                use_google_search=False,
                fallback_role="audience",
            )


@pytest.mark.asyncio
async def test_query_with_fallback_propagates_grounded_broker_errors():
    with patch(
        "grox_chat.llm_router.call_text",
        new=AsyncMock(side_effect=RuntimeError("Error: MiniMax search timed out")),
    ):
        with pytest.raises(RuntimeError, match="Error: MiniMax search timed out"):
            await query_with_fallback(
                "prompt",
                model="gemini-3.1-pro-preview",
                use_google_search=True,
                fallback_role="writer",
            )
