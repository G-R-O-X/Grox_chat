import pytest
from unittest.mock import AsyncMock, patch

from grox_chat import api


@pytest.mark.asyncio
async def test_persist_message_embeds_standard_messages():
    with patch("grox_chat.api.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
        with patch("grox_chat.api.insert_message_with_embedding", return_value=42) as insert_message:
            with patch("grox_chat.api.post_message") as post_message:
                result = await api.persist_message(1, 2, "dreamer", "historical message", "standard")

    assert result == 42
    insert_message.assert_called_once()
    assert insert_message.call_args.args == (
        1,
        2,
        "dreamer",
        "historical message",
        "standard",
        [0.1] * 384,
        None,
        None,
        None,
    )
    post_message.assert_not_called()


@pytest.mark.asyncio
async def test_persist_message_falls_back_when_embedding_fails():
    with patch("grox_chat.api.aget_embedding", new=AsyncMock(return_value=None)):
        with patch("grox_chat.api.insert_message_with_embedding") as insert_message:
            with patch("grox_chat.api.post_message", return_value=11) as post_message:
                result = await api.persist_message(1, 2, "writer", "fallback message", "standard")

    assert result == 11
    insert_message.assert_not_called()
    post_message.assert_called_once()
    assert post_message.call_args.args == (
        1,
        2,
        "writer",
        "fallback message",
        "standard",
        None,
        None,
        None,
    )


def test_search_messages_hybrid_merges_dense_and_lexical_results():
    dense = [{"id": 1, "content": "Dense result 1"}, {"id": 2, "content": "Dense result 2"}]
    lexical = [{"id": 3, "content": "Lexical result"}, {"id": 1, "content": "Dense result 1"}]

    with patch("grox_chat.api.search_messages", return_value=dense):
        with patch("grox_chat.api.search_messages_lexical", return_value=lexical):
            result = api.search_messages_hybrid(1, "query", [0.1] * 384, msg_type="standard", top_k=2)

    assert [row["id"] for row in result] == [1, 3]


def test_search_facts_hybrid_merges_lexical_hits_before_truncating():
    dense = [{"id": 10, "content": "Dense fact 1"}, {"id": 11, "content": "Dense fact 2"}]
    lexical = [{"id": 12, "content": "Lexical fact"}, {"id": 10, "content": "Dense fact 1"}]

    with patch("grox_chat.api.search_facts", return_value=dense):
        with patch("grox_chat.api.search_facts_lexical", return_value=lexical):
            result = api.search_facts_hybrid(1, "query", [0.1] * 384, top_k=2)

    assert [row["id"] for row in result] == [10, 12]


def test_insert_claim_delegates_to_db_insert_claim():
    with patch("grox_chat.api.db_insert_claim", return_value=88) as insert_claim:
        claim_id = api.insert_claim(
            1,
            2,
            "Claim text",
            support_fact_ids_json="[1,2]",
            rationale_short="Facts align.",
            claim_score=7.5,
            status="active",
            candidate_id=3,
        )

    assert claim_id == 88
    insert_claim.assert_called_once()


def test_get_facts_by_ids_delegates_to_db_layer():
    with patch("grox_chat.api.db_get_facts_by_ids", return_value=[{"id": 1}, {"id": 2}]) as get_facts_by_ids:
        rows = api.get_facts_by_ids(1, [1, 2])

    assert rows == [{"id": 1}, {"id": 2}]
    get_facts_by_ids.assert_called_once_with(1, [1, 2])
