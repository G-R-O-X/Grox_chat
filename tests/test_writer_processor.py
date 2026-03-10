import pytest
from unittest.mock import AsyncMock, patch

from agent_chatroom.writer_processor import process_writer_output


@pytest.mark.asyncio
async def test_process_writer_output_respects_explicit_empty_facts():
    with patch("agent_chatroom.writer_processor.api.fact_exists", return_value=False):
        with patch("agent_chatroom.writer_processor.aget_embedding", new=AsyncMock()) as embedding:
            with patch("agent_chatroom.writer_processor.api.insert_fact_with_embedding") as insert_fact:
                await process_writer_output(
                    1,
                    "Writer quoted earlier notes:\nFACT: old quoted line",
                    structured_facts=[],
                )

    embedding.assert_not_awaited()
    insert_fact.assert_not_called()


@pytest.mark.asyncio
async def test_process_writer_output_deduplicates_duplicate_structured_facts():
    with patch("agent_chatroom.writer_processor.api.fact_exists", return_value=False):
        with patch("agent_chatroom.writer_processor.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)) as embedding:
            with patch("agent_chatroom.writer_processor.api.insert_fact_with_embedding", return_value=5) as insert_fact:
                await process_writer_output(
                    1,
                    "unused",
                    structured_facts=["A   verified fact", "A verified fact"],
                )

    embedding.assert_awaited_once_with("A verified fact")
    insert_fact.assert_called_once_with(1, "A verified fact", source="Writer", embedding=[0.1] * 384)


@pytest.mark.asyncio
async def test_process_writer_output_skips_existing_fact():
    with patch("agent_chatroom.writer_processor.api.fact_exists", return_value=True):
        with patch("agent_chatroom.writer_processor.aget_embedding", new=AsyncMock()) as embedding:
            with patch("agent_chatroom.writer_processor.api.insert_fact_with_embedding") as insert_fact:
                await process_writer_output(
                    1,
                    "unused",
                    structured_facts=["Already stored fact"],
                )

    embedding.assert_not_awaited()
    insert_fact.assert_not_called()


@pytest.mark.asyncio
async def test_process_writer_output_ignores_non_string_fact_entries():
    with patch("agent_chatroom.writer_processor.api.fact_exists", return_value=False):
        with patch("agent_chatroom.writer_processor.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)) as embedding:
            with patch("agent_chatroom.writer_processor.api.insert_fact_with_embedding", return_value=7) as insert_fact:
                await process_writer_output(
                    1,
                    "unused",
                    structured_facts=[{"bad": 1}, 99, "Valid fact"],
                )

    embedding.assert_awaited_once_with("Valid fact")
    insert_fact.assert_called_once_with(1, "Valid fact", source="Writer", embedding=[0.1] * 384)
