import pytest
from unittest.mock import patch, AsyncMock

from grox_chat.broker import BrokerResponse, SearchEvidenceItem
from grox_chat.rag import assemble_rag_context

@pytest.mark.asyncio
async def test_assemble_rag_context_empty():
    res, degraded = await assemble_rag_context(1, 1, [], "dreamer")
    assert res == ""
    assert degraded is False

@pytest.mark.asyncio
async def test_assemble_rag_context():
    recent_messages = [{"id": 9, "content": "This is a test message", "sender": "critic"}]
    
    mock_llm_call = AsyncMock(return_value=BrokerResponse(text='{"query":"test query"}', provider_used="minimax"))
    mock_embedding = AsyncMock(return_value=[0.1] * 384)
    
    mock_facts = [
        {"id": 1, "content": "Fact 1", "source": "Writer"},
        {"id": 2, "content": "Fact 2", "source": "Writer"}
    ]
    mock_claims = [
        {"id": 7, "content": "Claim 1", "source": "Librarian"},
        {"id": 8, "content": "Claim 2", "source": "Librarian"},
    ]
    mock_summaries = [
        {"id": 3, "content": "Summary 1", "source": "Skynet"},
        {"id": 4, "content": "Summary 2", "source": "Skynet"},
    ]
    mock_messages = [
        {"id": 5, "content": "Historical message 1", "source": "engineer"},
        {"id": 6, "content": "Historical message 2", "source": "critic"},
    ]
    
    # Reranker returns indices and scores
    mock_rerank = AsyncMock(side_effect=[
        [(0, 0.9), (1, 0.1)],
        [(0, 0.8), (1, 0.1)],
        [(0, 0.8), (1, 0.1)],
        [(0, 0.7), (1, 0.1)],
    ])
    
    with patch("grox_chat.rag.llm_call", new=mock_llm_call):
        with patch("grox_chat.rag.aget_embedding", new=mock_embedding):
            with patch("grox_chat.rag.api.search_facts_hybrid", return_value=mock_facts):
                with patch("grox_chat.rag.api.search_claims_hybrid", return_value=mock_claims):
                    with patch("grox_chat.rag.api.search_messages_hybrid", side_effect=[mock_summaries, mock_messages]):
                        with patch("grox_chat.rag.arerank", new=mock_rerank):
                            res, degraded = await assemble_rag_context(1, 1, recent_messages, "dreamer")

                            assert "RAG KNOWLEDGE INJECTION" in res
                            assert degraded is False
                            assert "[Related Claims]" in res
                            assert "[F1]" in res
                            assert "[C7]" in res
                            assert "Fact 1" in res
                            assert "[Summary: 3]" in res
                            assert "Summary 1" in res
                            assert "[Message: 5]" in res
                            assert "Historical message 1" in res
                            # Fact 2 is dropped because score 0.1 < 0.3
                            assert "[F2]" not in res


@pytest.mark.asyncio
async def test_assemble_rag_context_falls_back_to_last_message_when_query_distillation_raises():
    recent_messages = [{"id": 9, "content": "Fallback me", "sender": "critic"}]

    with patch("grox_chat.rag.retry_structured_output", new=AsyncMock(return_value=None)):
        with patch(
            "grox_chat.rag._collect_local_rag_records",
            new=AsyncMock(return_value=({"facts": (), "claims": (), "summaries": (), "messages": ()}, False)),
        ) as collect_rag:
            res, degraded = await assemble_rag_context(1, 1, recent_messages, "dreamer")

    assert res == ""
    assert degraded is True
    assert collect_rag.await_args.args[1] == "Fallback me"


@pytest.mark.asyncio
async def test_assemble_rag_context_uses_web_backup_when_local_memory_is_empty():
    recent_messages = [{"id": 9, "content": "Fallback me", "sender": "critic"}]

    with patch("grox_chat.rag.retry_structured_output", new=AsyncMock(return_value=BrokerResponse(text='{"query":"backup query"}'))):
        with patch(
            "grox_chat.rag._collect_local_rag_records",
            new=AsyncMock(return_value=({"facts": (), "claims": (), "summaries": (), "messages": ()}, False)),
        ):
            with patch(
                "grox_chat.rag.get_or_collect_search_evidence_item",
                new=AsyncMock(
                    return_value=SearchEvidenceItem(
                        query="backup query",
                        rendered_results="=== WEB SEARCH RESULTS ===\n[W11] Title: T\nSource: example.com\nSnippet: S\n\n",
                        had_error=False,
                        web_ids=(11,),
                    )
                ),
            ):
                res, degraded = await assemble_rag_context(
                    1,
                    1,
                    recent_messages,
                    "dreamer",
                    allow_web_backup=True,
                )

    assert degraded is False
    assert "[Related Web Evidence]" in res
    assert "[W11]" in res
