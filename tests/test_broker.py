import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from grox_chat.broker import (
    BrokerRequest,
    BrokerResponse,
    call_text,
    call_via_broker,
    get_or_collect_search_evidence_item,
    llm_call,
    llm_call_with_web,
    PROFILE_GEMINI_FLASH,
    PROFILE_MINIMAX,
    reset_broker_state,
    shutdown_broker,
)


@pytest.mark.asyncio
async def test_broker_routes_gemini_web_requests_through_query_search_and_final_analysis():
    with patch.dict(os.environ, {"ENABLE_GEMINI": "1"}, clear=False):
        with patch(
            "grox_chat.broker.query_minimax",
            new=AsyncMock(side_effect=[("search query", []), ("NO_SEARCH", [])]),
        ) as query_minimax:
            with patch(
                "grox_chat.broker.minimax_search",
                new=AsyncMock(return_value={"organic": [{"title": "T", "snippet": "S"}]}),
            ) as minimax_search:
                with patch(
                    "grox_chat.broker.query_gemini_cli",
                    new=AsyncMock(return_value="gemini-result"),
                ) as query_gemini_cli:
                    result = await llm_call_with_web(
                        "Prompt",
                        provider_profile=PROFILE_GEMINI_FLASH,
                        system_prompt="sys",
                        role="skynet",
                    )

    assert result.text == "gemini-result"
    assert result.search_used is True
    assert len(result.search_evidence) == 1
    assert query_minimax.await_count == 2
    query_gemini_cli.assert_awaited_once()
    minimax_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_retries_empty_minimax_search_decision_once_before_skipping():
    with patch.dict(os.environ, {"ENABLE_GEMINI": "1"}, clear=False):
        with patch(
            "grox_chat.broker.query_minimax",
            new=AsyncMock(
                side_effect=[
                    ("Error: Empty response", []),
                    ("search query", []),
                    ("NO_SEARCH", []),
                ]
            ),
        ) as query_minimax:
            with patch(
                "grox_chat.broker.minimax_search",
                new=AsyncMock(return_value={"organic": [{"title": "T", "snippet": "S"}]}),
            ) as minimax_search:
                with patch(
                    "grox_chat.broker.query_gemini_cli",
                    new=AsyncMock(return_value="gemini-result"),
                ) as query_gemini_cli:
                    result = await llm_call_with_web(
                        "Prompt",
                        provider_profile=PROFILE_GEMINI_FLASH,
                        system_prompt="sys",
                        role="skynet",
                    )

    assert result.text == "gemini-result"
    assert result.search_used is True
    assert query_minimax.await_count == 3
    minimax_search.assert_awaited_once()
    query_gemini_cli.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_routes_minimax_web_requests_through_react_loop():
    with patch(
        "grox_chat.broker.react_search_loop_with_evidence",
        new=AsyncMock(return_value=BrokerResponse(text="web-result", search_failed=False)),
    ) as react_search_loop:
        result = await llm_call_with_web(
            "Prompt",
            provider_profile=PROFILE_MINIMAX,
            system_prompt="sys",
            role="dreamer",
        )

    assert result.text == "web-result"
    react_search_loop.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_routes_minimax_direct_requests_through_query_minimax():
    with patch(
        "grox_chat.broker.query_minimax",
        new=AsyncMock(return_value=("plain-result", False)),
    ) as query_minimax:
        result = await llm_call(
            "Prompt",
            provider_profile=PROFILE_MINIMAX,
            system_prompt="sys",
            role="dreamer",
        )

    assert result.text == "plain-result"
    query_minimax.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_reuses_cached_web_evidence_before_running_fresh_search():
    cached_rows = [
        {
            "id": 17,
            "query_text": "query",
            "title": "Cached title",
            "snippet": "Cached snippet",
            "url": "https://example.com/a",
            "source_domain": "example.com",
            "content": "Cached title Cached snippet query example.com",
        }
    ]

    with patch("grox_chat.broker.api.search_web_evidence_same_topic", return_value=cached_rows):
        with patch("grox_chat.broker.api.search_web_evidence_cross_topic", return_value=[]):
            with patch("grox_chat.broker.arerank", new=AsyncMock(return_value=[(0, 0.9)])):
                with patch("grox_chat.broker.minimax_search", new=AsyncMock()) as minimax_search:
                    item = await get_or_collect_search_evidence_item(
                        "query",
                        topic_id=1,
                        subtopic_id=2,
                        role="dreamer",
                    )

    assert item.web_ids == (17,)
    assert "[W17]" in item.rendered_results
    minimax_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_broker_ignores_low_scoring_cached_web_evidence_and_runs_fresh_search():
    cached_rows = [
        {
            "id": 17,
            "query_text": "query",
            "title": "Cached title",
            "snippet": "Cached snippet",
            "url": "https://example.com/a",
            "source_domain": "example.com",
            "content": "Cached title Cached snippet query example.com",
        }
    ]

    with patch("grox_chat.broker.api.search_web_evidence_same_topic", return_value=cached_rows):
        with patch("grox_chat.broker.api.search_web_evidence_cross_topic", return_value=[]):
            with patch("grox_chat.broker.arerank", new=AsyncMock(return_value=[(0, 0.35)])):
                with patch(
                    "grox_chat.broker.minimax_search",
                    new=AsyncMock(return_value={"organic": [{"title": "Fresh", "snippet": "Fresh snippet", "link": "https://fresh.example.com"}]}),
                ) as minimax_search:
                    with patch("grox_chat.broker.api.insert_web_evidence", return_value=21):
                        item = await get_or_collect_search_evidence_item(
                            "query",
                            topic_id=1,
                            subtopic_id=2,
                            role="dreamer",
                        )

    minimax_search.assert_awaited_once()
    assert item.web_ids == (21,)
    assert "[W21]" in item.rendered_results


@pytest.mark.asyncio
async def test_broker_gemini_plain_fallback_uses_minimax_model_id():
    with patch.dict(os.environ, {"ENABLE_GEMINI": "1"}, clear=False):
        with patch(
            "grox_chat.broker.query_gemini_cli",
            new=AsyncMock(side_effect=RuntimeError("429")),
        ):
            with patch(
                "grox_chat.broker.query_minimax",
                new=AsyncMock(side_effect=[("plan", []), ("draft", []), ("plain-result", [])]),
            ) as query_minimax:
                result = await llm_call(
                    "Prompt",
                    provider_profile=PROFILE_GEMINI_FLASH,
                    model="gemini-3.0-flash",
                    system_prompt="sys",
                    role="skynet",
                )

    assert result.text == "plain-result"
    assert result.provider_used == PROFILE_MINIMAX
    assert result.fallback_used is True
    assert query_minimax.await_count == 3
    assert {call.kwargs["model"] for call in query_minimax.await_args_list} == {"MiniMax-M2.5"}


@pytest.mark.asyncio
async def test_broker_passes_pseudo_tool_recovery_flag_to_direct_minimax_calls():
    with patch(
        "grox_chat.broker.query_minimax",
        new=AsyncMock(return_value=("recovered-query", False)),
    ) as query_minimax:
        result = await llm_call(
            "Prompt",
            provider_profile=PROFILE_MINIMAX,
            recover_pseudo_tool_query=True,
        )

    assert result.text == "recovered-query"
    assert query_minimax.await_args.kwargs["recover_pseudo_tool_query"] is True


@pytest.mark.asyncio
async def test_broker_coalesces_identical_inflight_requests():
    await reset_broker_state()
    gate = AsyncMock(return_value="shared-result")

    with patch.dict(os.environ, {"ENABLE_GEMINI": "1"}, clear=False):
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

    assert first.text == "shared-result"
    assert second.text == "shared-result"
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

    with patch.dict(os.environ, {"ENABLE_GEMINI": "1"}, clear=False):
        with patch("grox_chat.broker.query_gemini_cli", new=slow_query):
            task1 = asyncio.create_task(call_via_broker(request))
            task2 = asyncio.create_task(call_via_broker(request))
            await started.wait()
            task1.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task1
            release.set()
            assert (await task2).text == "shared-result"


@pytest.mark.asyncio
async def test_broker_gemini_plain_requests_use_minimax_deep_fallback_when_disabled():
    with patch.dict(os.environ, {"ENABLE_GEMINI": "0"}, clear=False):
        with patch("grox_chat.broker.query_gemini_cli", new=AsyncMock()) as query_gemini_cli:
            with patch(
                "grox_chat.broker.query_minimax",
                new=AsyncMock(side_effect=[("plan", []), ("draft", []), ("final", [])]),
            ) as query_minimax:
                result = await llm_call(
                    "Prompt",
                    provider_profile=PROFILE_GEMINI_FLASH,
                    system_prompt="sys",
                    role="skynet",
                )

    assert result.text == "final"
    assert result.provider_used == PROFILE_MINIMAX
    assert result.fallback_used is True
    assert query_minimax.await_count == 3
    query_gemini_cli.assert_not_awaited()


@pytest.mark.asyncio
async def test_broker_gemini_web_requests_use_minimax_web_profile_when_disabled():
    with patch.dict(os.environ, {"ENABLE_GEMINI": "0"}, clear=False):
        with patch("grox_chat.broker.query_gemini_cli", new=AsyncMock()) as query_gemini_cli:
            with patch(
                "grox_chat.broker.react_search_loop_with_evidence",
                new=AsyncMock(return_value=BrokerResponse(text="web-result", search_failed=False)),
            ) as react_search_loop:
                result = await llm_call_with_web(
                    "Prompt",
                    provider_profile=PROFILE_GEMINI_FLASH,
                    system_prompt="sys",
                    role="skynet",
                )

    assert result.text == "web-result"
    assert result.provider_used == PROFILE_MINIMAX
    assert result.fallback_used is True
    query_gemini_cli.assert_not_awaited()
    react_search_loop.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_repairs_invalid_json_from_no_web_deep_fallback():
    with patch.dict(os.environ, {"ENABLE_GEMINI": "0"}, clear=False):
        with patch(
            "grox_chat.broker.query_minimax",
            new=AsyncMock(
                side_effect=[
                    ("plan", []),
                    ("draft", []),
                    ("not valid json", []),
                    ('{"ok": true}', []),
                ]
            ),
        ) as query_minimax:
            result = await llm_call(
                "Prompt",
                provider_profile=PROFILE_GEMINI_FLASH,
                system_prompt="sys",
                role="skynet",
                require_json=True,
            )

    assert result.text == '{"ok": true}'
    assert result.provider_used == PROFILE_MINIMAX
    assert result.fallback_used is True
    assert query_minimax.await_count == 4


@pytest.mark.asyncio
async def test_broker_repairs_invalid_json_from_web_final_response():
    with patch.dict(os.environ, {"ENABLE_GEMINI": "1"}, clear=False):
        with patch(
            "grox_chat.broker.query_minimax",
            new=AsyncMock(side_effect=[("NO_SEARCH", []), ('{"ok": true}', [])]),
        ) as query_minimax:
            with patch(
                "grox_chat.broker.query_gemini_cli",
                new=AsyncMock(return_value="not valid json"),
            ) as query_gemini_cli:
                result = await llm_call_with_web(
                    "Prompt",
                    provider_profile=PROFILE_GEMINI_FLASH,
                    system_prompt="sys",
                    role="skynet",
                    require_json=True,
                )

    assert result.text == '{"ok": true}'
    assert result.provider_used == PROFILE_GEMINI_FLASH
    assert result.fallback_used is True
    assert query_minimax.await_count == 2
    query_gemini_cli.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_repairs_invalid_json_from_direct_minimax_plain_path():
    with patch(
        "grox_chat.broker.query_minimax",
        new=AsyncMock(side_effect=[("not valid json", []), ('{"ok": true}', [])]),
    ) as query_minimax:
        result = await llm_call(
            "Prompt",
            provider_profile=PROFILE_MINIMAX,
            system_prompt="sys",
            role="dreamer",
            require_json=True,
        )

    assert result.text == '{"ok": true}'
    assert result.provider_used == PROFILE_MINIMAX
    assert result.fallback_used is True
    assert query_minimax.await_count == 2


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
