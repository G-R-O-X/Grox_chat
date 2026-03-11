import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from grox_chat.broker import BrokerRequest, call_text, call_via_broker, reset_broker_state, shutdown_broker


@pytest.mark.asyncio
async def test_broker_routes_gemini_requests_through_gemini_client():
    with patch(
        "grox_chat.broker.query_gemini_cli",
        new=AsyncMock(return_value="gemini-result"),
    ) as query_gemini_cli:
        result = await call_text(
            "Prompt",
            provider="gemini",
            allow_web=True,
            system_instruction="sys",
            fallback_role="skynet",
        )

    assert result == "gemini-result"
    query_gemini_cli.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_routes_minimax_web_requests_through_react_loop():
    with patch(
        "grox_chat.broker.react_search_loop",
        new=AsyncMock(return_value=("web-result", False)),
    ) as react_search_loop:
        result = await call_text(
            "Prompt",
            provider="minimax",
            allow_web=True,
            system_instruction="sys",
            fallback_role="dreamer",
        )

    assert result == "web-result"
    react_search_loop.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_routes_minimax_direct_requests_through_query_minimax():
    with patch(
        "grox_chat.broker.query_minimax",
        new=AsyncMock(return_value=("plain-result", False)),
    ) as query_minimax:
        result = await call_text(
            "Prompt",
            provider="minimax",
            allow_web=False,
            system_instruction="sys",
            fallback_role="dreamer",
        )

    assert result == "plain-result"
    query_minimax.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_passes_pseudo_tool_recovery_flag_to_direct_minimax_calls():
    with patch(
        "grox_chat.broker.query_minimax",
        new=AsyncMock(return_value=("recovered-query", False)),
    ) as query_minimax:
        result = await call_text(
            "Prompt",
            provider="minimax",
            allow_web=False,
            recover_pseudo_tool_query=True,
        )

    assert result == "recovered-query"
    assert query_minimax.await_args.kwargs["recover_pseudo_tool_query"] is True


@pytest.mark.asyncio
async def test_broker_coalesces_identical_inflight_requests():
    await reset_broker_state()
    gate = AsyncMock(return_value="shared-result")

    with patch("grox_chat.broker.query_gemini_cli", new=gate):
        request = BrokerRequest(
            prompt="Prompt",
            system_instruction="sys",
            provider="gemini",
            allow_web=False,
            fallback_role="skynet",
        )
        first, second = await asyncio.gather(
            call_via_broker(request),
            call_via_broker(request),
        )

    assert first == "shared-result"
    assert second == "shared-result"
    gate.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_shields_shared_request_from_single_waiter_cancellation():
    await reset_broker_state()

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_query(*args, **kwargs):
        started.set()
        await release.wait()
        return "shared-result"

    request = BrokerRequest(prompt="Prompt", provider="gemini", fallback_role="skynet")

    with patch("grox_chat.broker.query_gemini_cli", new=slow_query):
        task1 = asyncio.create_task(call_via_broker(request))
        task2 = asyncio.create_task(call_via_broker(request))
        await started.wait()
        task1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task1
        release.set()
        assert await task2 == "shared-result"


@pytest.mark.asyncio
async def test_broker_raises_on_minimax_error_sentinel():
    with patch(
        "grox_chat.broker.query_minimax",
        new=AsyncMock(return_value=("Error: nope", False)),
    ):
        with pytest.raises(RuntimeError, match="Error: nope"):
            await call_text("Prompt", provider="minimax")


@pytest.mark.asyncio
async def test_shutdown_broker_closes_provider_clients():
    with patch("grox_chat.broker.close_gemini_cli_client", new=AsyncMock()) as close_gemini:
        with patch("grox_chat.broker.close_minimax_client", new=AsyncMock()) as close_minimax:
            await shutdown_broker()

    close_gemini.assert_awaited_once()
    close_minimax.assert_awaited_once()
