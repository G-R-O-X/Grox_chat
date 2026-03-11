from unittest.mock import AsyncMock, patch

import pytest

from grox_chat.tools import react_search_loop


@pytest.mark.asyncio
async def test_react_search_loop_skips_search_when_not_needed():
    mock_query = AsyncMock(side_effect=[
        ("NO_SEARCH", []),
        ("final answer", []),
    ])

    with patch("grox_chat.broker.query_minimax", new=mock_query):
        with patch("grox_chat.broker.minimax_search", new=AsyncMock()) as mock_search:
            result, search_failed = await react_search_loop("agent1", "prompt", max_iter=2)

    assert result == "final answer"
    assert search_failed is False
    assert mock_query.await_count == 2
    mock_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_react_search_loop_runs_one_search_then_answers():
    mock_query = AsyncMock(side_effect=[
        ("test query", []),
        ("NO_SEARCH", []),
        ("final answer after search", []),
    ])
    mock_search = AsyncMock(return_value={"organic": [{"title": "t", "snippet": "s"}]})

    with patch("grox_chat.broker.query_minimax", new=mock_query):
        with patch("grox_chat.broker.minimax_search", new=mock_search):
            result, search_failed = await react_search_loop("agent1", "prompt", max_iter=2)

    assert result == "final answer after search"
    assert search_failed is False
    assert mock_query.await_count == 3
    mock_search.assert_awaited_once_with("test query")


@pytest.mark.asyncio
async def test_react_search_loop_respects_max_iterations_before_final_answer():
    mock_query = AsyncMock(side_effect=[
        ("query one", []),
        ("query two", []),
        ("final answer after max iterations", []),
    ])
    mock_search = AsyncMock(return_value={"organic": []})

    with patch("grox_chat.broker.query_minimax", new=mock_query):
        with patch("grox_chat.broker.minimax_search", new=mock_search):
            result, search_failed = await react_search_loop("agent1", "prompt", max_iter=2)

    assert result == "final answer after max iterations"
    assert search_failed is False
    assert mock_query.await_count == 3
    assert mock_search.await_count == 2


@pytest.mark.asyncio
async def test_react_search_loop_marks_failed_search_and_continues():
    mock_query = AsyncMock(side_effect=[
        ("test query", []),
        ("NO_SEARCH", []),
        ("final answer after failed search", []),
    ])
    mock_search = AsyncMock(return_value={"error": "search backend unavailable"})

    with patch("grox_chat.broker.query_minimax", new=mock_query):
        with patch("grox_chat.broker.minimax_search", new=mock_search):
            result, search_failed = await react_search_loop("agent1", "prompt", max_iter=2)

    assert result == "final answer after failed search"
    assert search_failed is True
    mock_search.assert_awaited_once_with("test query")


@pytest.mark.asyncio
async def test_react_search_loop_preserves_original_prompt_for_final_answer_after_search():
    captured_questions = []

    async def fake_query_minimax(*, system_prompt, question, max_tokens=8192, **kwargs):
        captured_questions.append(question)
        if len(captured_questions) == 1:
            return "test query", []
        return "final answer after search", []

    mock_search = AsyncMock(return_value={"organic": [{"title": "t", "snippet": "s"}]})

    with patch("grox_chat.broker.query_minimax", new=fake_query_minimax):
        with patch("grox_chat.broker.minimax_search", new=mock_search):
            result, search_failed = await react_search_loop("agent1", "ORIGINAL JSON TASK", max_iter=1)

    assert result == "final answer after search"
    assert search_failed is False
    assert len(captured_questions) == 2
    assert captured_questions[-1].startswith("ORIGINAL JSON TASK")
    assert "=== WEB SEARCH RESULTS ===" in captured_questions[-1]
    assert "output a new search query" not in captured_questions[-1]


@pytest.mark.asyncio
async def test_react_search_loop_normalizes_quoted_no_search():
    mock_query = AsyncMock(side_effect=[
        ('"NO_SEARCH"', []),
        ("final answer", []),
    ])

    with patch("grox_chat.broker.query_minimax", new=mock_query):
        with patch("grox_chat.broker.minimax_search", new=AsyncMock()) as mock_search:
            result, search_failed = await react_search_loop("agent1", "prompt", max_iter=2)

    assert result == "final answer"
    assert search_failed is False
    mock_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_react_search_loop_falls_back_to_final_answer_on_query_decision_error():
    mock_query = AsyncMock(side_effect=[
        ("Error: 400", []),
        ("final answer after fallback", []),
    ])

    with patch("grox_chat.broker.query_minimax", new=mock_query):
        with patch("grox_chat.broker.minimax_search", new=AsyncMock()) as mock_search:
            result, search_failed = await react_search_loop("agent1", "prompt", max_iter=2)

    assert result == "final answer after fallback"
    assert search_failed is False
    mock_search.assert_not_awaited()
